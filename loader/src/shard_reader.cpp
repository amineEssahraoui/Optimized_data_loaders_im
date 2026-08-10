/**
 * @file shard_reader.cpp
 * @brief Implementation of the VLM binary shard reader.
 *
 * Parses the shard file format documented in docs/shard_format.md:
 *   [HEADER 64 B] [SAMPLE RECORDS …] [OFFSET TABLE] [FOOTER 12 B]
 *
 * All multi-byte integers are stored little-endian.  This implementation
 * reads them via std::memcpy, which is correct on little-endian hosts
 * (x86, x86_64, ARM-LE) and avoids undefined-behaviour type-punning.
 * A runtime endianness check is performed in the constructor.
 *
 * CRC32 verification uses the same ISO 3309 polynomial as Python's
 * zlib.crc32 (reflected polynomial 0xEDB88320).
 */

#include "shard_reader.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vlm {

// ---------------------------------------------------------------------------
// CRC32 (ISO 3309 / ITU-T V.42, same as Python zlib.crc32)
// ---------------------------------------------------------------------------
namespace {

/** Generate the CRC32 lookup table at program startup. */
struct CRC32Table {
    uint32_t entries[256];

    constexpr CRC32Table() : entries{} {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                if (crc & 1u)
                    crc = (crc >> 1u) ^ 0xEDB88320u;
                else
                    crc >>= 1u;
            }
            entries[i] = crc;
        }
    }
};

static constexpr CRC32Table crc32_table{};

/** Incremental CRC32 update over a byte buffer. */
inline uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        crc = crc32_table.entries[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
    }
    return crc;
}

/** Compute CRC32 of a complete buffer. */
inline uint32_t crc32_compute(const uint8_t* data, size_t len) {
    return crc32_update(0xFFFFFFFFu, data, len) ^ 0xFFFFFFFFu;
}

/** Runtime little-endian check.  Throws if the host is big-endian. */
void assert_little_endian() {
    const uint32_t probe = 1u;
    const auto* byte = reinterpret_cast<const uint8_t*>(&probe);
    if (*byte != 1u) {
        throw std::runtime_error(
            "vlm::ShardReader requires a little-endian host (x86, ARM-LE)."
        );
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Magic constants
// ---------------------------------------------------------------------------
static constexpr char MAGIC_START_STR[9] = "VLMSHARD";  // 8 chars + NUL
static constexpr char MAGIC_END_STR[9]   = "SHARDEND";  // 8 chars + NUL

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
ShardReader::ShardReader(const std::string& path) : path_(path) {
    assert_little_endian();

    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open shard file: " + path);
    }

    parse_header();
    parse_offset_table();
    verify_footer();
}

ShardReader::~ShardReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

// ---------------------------------------------------------------------------
// Header parsing (64 bytes)
// ---------------------------------------------------------------------------
void ShardReader::parse_header() {
    char raw[HEADER_SIZE];
    file_.seekg(0);
    file_.read(raw, HEADER_SIZE);

    if (static_cast<size_t>(file_.gcount()) < HEADER_SIZE) {
        throw std::runtime_error(
            "Shard file too small for header: " + path_
        );
    }

    // Validate magic bytes (first 8 bytes)
    if (std::memcmp(raw, MAGIC_START_STR, 8) != 0) {
        throw std::runtime_error(
            "Invalid shard magic in: " + path_
        );
    }

    // Parse header fields (all little-endian, native on LE hosts)
    // Layout: magic(8) version(4) sample_count(4) offset_table_pos(8)
    //         channels(4) height(4) width(4) token_length(4) reserved(24)
    std::memcpy(&header_.version,          raw +  8, 4);
    std::memcpy(&header_.sample_count,     raw + 12, 4);
    std::memcpy(&header_.offset_table_pos, raw + 16, 8);
    std::memcpy(&header_.image_channels,   raw + 24, 4);
    std::memcpy(&header_.image_height,     raw + 28, 4);
    std::memcpy(&header_.image_width,      raw + 32, 4);
    std::memcpy(&header_.token_length,     raw + 36, 4);

    if (header_.version != FORMAT_VERSION) {
        throw std::runtime_error(
            "Unsupported shard version " + std::to_string(header_.version)
            + " (expected " + std::to_string(FORMAT_VERSION) + ")"
        );
    }
}

// ---------------------------------------------------------------------------
// Offset table parsing
// ---------------------------------------------------------------------------
void ShardReader::parse_offset_table() {
    file_.seekg(static_cast<std::streamoff>(header_.offset_table_pos));

    offsets_.resize(header_.sample_count);
    for (uint32_t i = 0; i < header_.sample_count; ++i) {
        uint64_t offset = 0;
        uint64_t length = 0;
        file_.read(reinterpret_cast<char*>(&offset), 8);
        file_.read(reinterpret_cast<char*>(&length), 8);

        if (!file_.good()) {
            throw std::runtime_error("Truncated offset table in: " + path_);
        }
        offsets_[i] = {offset, length};
    }
}

// ---------------------------------------------------------------------------
// Footer verification (CRC32 + end magic)
// ---------------------------------------------------------------------------
void ShardReader::verify_footer() {
    // Footer starts right after the offset table
    const uint64_t footer_pos =
        header_.offset_table_pos
        + static_cast<uint64_t>(header_.sample_count) * 16u;

    // Read footer: 4-byte CRC32 + 8-byte magic
    file_.seekg(static_cast<std::streamoff>(footer_pos));
    char footer_buf[12];
    file_.read(footer_buf, 12);
    if (static_cast<size_t>(file_.gcount()) < 12) {
        throw std::runtime_error("Truncated footer in: " + path_);
    }

    uint32_t stored_crc = 0;
    std::memcpy(&stored_crc, footer_buf, 4);

    if (std::memcmp(footer_buf + 4, MAGIC_END_STR, 8) != 0) {
        throw std::runtime_error("Invalid end magic in: " + path_);
    }

    // Recompute CRC32 over all bytes before the footer
    file_.seekg(0);
    uint32_t running_crc = 0xFFFFFFFFu;
    const size_t chunk_size = 65536;
    std::vector<uint8_t> buf(chunk_size);
    uint64_t remaining = footer_pos;

    while (remaining > 0) {
        const auto to_read = static_cast<std::streamsize>(
            std::min(remaining, static_cast<uint64_t>(chunk_size))
        );
        file_.read(reinterpret_cast<char*>(buf.data()), to_read);
        const auto got = static_cast<size_t>(file_.gcount());
        running_crc = crc32_update(running_crc, buf.data(), got);
        remaining -= got;
    }
    const uint32_t computed_crc = running_crc ^ 0xFFFFFFFFu;

    if (stored_crc != computed_crc) {
        throw std::runtime_error(
            "CRC32 mismatch in " + path_
            + ": stored=0x" + std::to_string(stored_crc)
            + " computed=0x" + std::to_string(computed_crc)
        );
    }
}

// ---------------------------------------------------------------------------
// Public accessors
// ---------------------------------------------------------------------------
const ShardFileHeader& ShardReader::header() const {
    return header_;
}

uint32_t ShardReader::sample_count() const {
    return header_.sample_count;
}

// ---------------------------------------------------------------------------
// Sample reading
// ---------------------------------------------------------------------------
Sample ShardReader::read_sample(uint32_t index) {
    if (index >= header_.sample_count) {
        throw std::out_of_range(
            "Sample index " + std::to_string(index)
            + " out of range [0, " + std::to_string(header_.sample_count) + ")"
        );
    }

    const auto [offset, length] = offsets_[index];
    file_.seekg(static_cast<std::streamoff>(offset));

    Sample sample;

    // --- Image tensor: C * H * W float32 values ---
    const size_t img_elems =
        static_cast<size_t>(header_.image_channels)
        * header_.image_height
        * header_.image_width;
    sample.image_tensor.resize(img_elems);
    file_.read(
        reinterpret_cast<char*>(sample.image_tensor.data()),
        static_cast<std::streamsize>(img_elems * sizeof(float))
    );

    // --- Question token IDs: T int32 values ---
    const size_t tok_len = header_.token_length;
    sample.question_ids.resize(tok_len);
    file_.read(
        reinterpret_cast<char*>(sample.question_ids.data()),
        static_cast<std::streamsize>(tok_len * sizeof(int32_t))
    );

    // --- Question attention mask: T int32 values ---
    sample.question_mask.resize(tok_len);
    file_.read(
        reinterpret_cast<char*>(sample.question_mask.data()),
        static_cast<std::streamsize>(tok_len * sizeof(int32_t))
    );

    // --- Answer token IDs: T int32 values ---
    sample.answer_ids.resize(tok_len);
    file_.read(
        reinterpret_cast<char*>(sample.answer_ids.data()),
        static_cast<std::streamsize>(tok_len * sizeof(int32_t))
    );

    // --- Answer attention mask: T int32 values ---
    sample.answer_mask.resize(tok_len);
    file_.read(
        reinterpret_cast<char*>(sample.answer_mask.data()),
        static_cast<std::streamsize>(tok_len * sizeof(int32_t))
    );

    // --- Metadata: uint32 length prefix + UTF-8 JSON ---
    uint32_t meta_len = 0;
    file_.read(reinterpret_cast<char*>(&meta_len), sizeof(uint32_t));

    sample.metadata_json.resize(meta_len);
    file_.read(sample.metadata_json.data(),
               static_cast<std::streamsize>(meta_len));

    if (!file_.good()) {
        throw std::runtime_error(
            "I/O error reading sample " + std::to_string(index)
            + " from " + path_
        );
    }

    return sample;
}

// ---------------------------------------------------------------------------
// Batch reading
// ---------------------------------------------------------------------------
Batch ShardReader::read_batch(const std::vector<uint32_t>& indices) {
    std::vector<Sample> samples;
    samples.reserve(indices.size());
    for (const uint32_t idx : indices) {
        samples.push_back(read_sample(idx));
    }
    return Batch::from_samples(
        samples,
        static_cast<int32_t>(header_.image_channels),
        static_cast<int32_t>(header_.image_height),
        static_cast<int32_t>(header_.image_width),
        static_cast<int32_t>(header_.token_length)
    );
}

std::vector<Sample> ShardReader::read_all() {
    std::vector<Sample> result;
    result.reserve(header_.sample_count);
    for (uint32_t i = 0; i < header_.sample_count; ++i) {
        result.push_back(read_sample(i));
    }
    return result;
}

}  // namespace vlm
