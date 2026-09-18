/**
 * @file shard_reader.cpp
 * @brief Implementation of the VLM binary shard reader (v1 and v2).
 *
 * Parses the shard file format documented in docs/shard_format.md:
 *   [HEADER 64 B] [SAMPLE RECORDS …] [OFFSET TABLE] [FOOTER 12 B]
 *
 * Phase 2 enhancements:
 *   - AVX2 FMA–optimised uint8→float32 normalization (compile-time flag)
 *   - LZ4 on-the-fly decompression via vendored lz4.c
 *   - Buffer-based sample parsing (single I/O per sample, cache-friendly)
 *
 * CRC32 verification uses the same ISO 3309 polynomial as Python's
 * zlib.crc32 (reflected polynomial 0xEDB88320).
 */

#include "shard_reader.h"
#include "detail/sample_parser.h"
#include "detail/crc32.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

// Vendored LZ4 (decompression only)
extern "C" {
#include "lz4.h"
}

namespace vlm {

// Import shared parsing functions from the detail header
using detail::parse_sample_from_buffer;
using detail::normalize_channel;

namespace {

void assert_little_endian() {
    const uint32_t probe = 1u;
    if (*reinterpret_cast<const uint8_t*>(&probe) != 1u)
        throw std::runtime_error(
            "vlm::ShardReader requires a little-endian host.");
}

}  // anonymous namespace


// Magic constants
static constexpr char MAGIC_START_STR[9] = "VLMSHARD";
static constexpr char MAGIC_END_STR[9]   = "SHARDEND";


// Constructor / Destructor
ShardReader::ShardReader(const std::string& path) : path_(path) {
    assert_little_endian();
    file_.open(path, std::ios::binary);
    if (!file_.is_open())
        throw std::runtime_error("Failed to open shard file: " + path);
    parse_header();
    parse_offset_table();
    verify_footer();
}

ShardReader::~ShardReader() {
    if (file_.is_open()) file_.close();
}


// Header parsing (64 bytes, v1 and v2)
void ShardReader::parse_header() {
    char raw[HEADER_SIZE];
    file_.seekg(0);
    file_.read(raw, HEADER_SIZE);
    if (static_cast<size_t>(file_.gcount()) < HEADER_SIZE)
        throw std::runtime_error("Shard file too small for header: " + path_);

    if (std::memcmp(raw, MAGIC_START_STR, 8) != 0)
        throw std::runtime_error("Invalid shard magic in: " + path_);

    std::memcpy(&header_.version,          raw +  8, 4);
    std::memcpy(&header_.sample_count,     raw + 12, 4);
    std::memcpy(&header_.offset_table_pos, raw + 16, 8);
    std::memcpy(&header_.image_channels,   raw + 24, 4);
    std::memcpy(&header_.image_height,     raw + 28, 4);
    std::memcpy(&header_.image_width,      raw + 32, 4);
    std::memcpy(&header_.token_length,     raw + 36, 4);

    if (header_.version == FORMAT_VERSION) {
        header_.flags = 0;
    } else if (header_.version == FORMAT_VERSION_V2) {
        std::memcpy(&header_.flags, raw + 40, 4);
    } else {
        throw std::runtime_error(
            "Unsupported shard version " + std::to_string(header_.version));
    }
}


// Offset table parsing
void ShardReader::parse_offset_table() {
    file_.seekg(static_cast<std::streamoff>(header_.offset_table_pos));
    offsets_.resize(header_.sample_count);
    for (uint32_t i = 0; i < header_.sample_count; ++i) {
        uint64_t offset = 0, length = 0;
        file_.read(reinterpret_cast<char*>(&offset), 8);
        file_.read(reinterpret_cast<char*>(&length), 8);
        if (!file_.good())
            throw std::runtime_error("Truncated offset table in: " + path_);
        offsets_[i] = {offset, length};
    }
}


// Footer verification (CRC32 + end magic)
void ShardReader::verify_footer() {
    const uint64_t footer_pos =
        header_.offset_table_pos
        + static_cast<uint64_t>(header_.sample_count) * 16u;

    file_.seekg(static_cast<std::streamoff>(footer_pos));
    char footer_buf[12];
    file_.read(footer_buf, 12);
    if (static_cast<size_t>(file_.gcount()) < 12)
        throw std::runtime_error("Truncated footer in: " + path_);

    uint32_t stored_crc = 0;
    std::memcpy(&stored_crc, footer_buf, 4);
    if (std::memcmp(footer_buf + 4, MAGIC_END_STR, 8) != 0)
        throw std::runtime_error("Invalid end magic in: " + path_);

    // Recompute CRC32 over all bytes before the footer
    file_.seekg(0);
    uint32_t running_crc = 0xFFFFFFFFu;
    constexpr size_t chunk_size = 65536;
    std::vector<uint8_t> buf(chunk_size);
    uint64_t remaining = footer_pos;
    while (remaining > 0) {
        const auto to_read = static_cast<std::streamsize>(
            std::min(remaining, static_cast<uint64_t>(chunk_size)));
        file_.read(reinterpret_cast<char*>(buf.data()), to_read);
        const auto got = static_cast<size_t>(file_.gcount());
        running_crc = detail::crc32_update(running_crc, buf.data(), got);
        remaining -= got;
    }
    if (stored_crc != (running_crc ^ 0xFFFFFFFFu))
        throw std::runtime_error("CRC32 mismatch in: " + path_);
}


// Public accessors
const ShardFileHeader& ShardReader::header() const { return header_; }
uint32_t ShardReader::sample_count() const { return header_.sample_count; }
void ShardReader::set_normalization(const NormalizationParams& p) { norm_ = p; }


Sample ShardReader::read_sample(uint32_t index) {
    if (index >= header_.sample_count)
        throw std::out_of_range(
            "Sample index " + std::to_string(index) + " out of range [0, "
            + std::to_string(header_.sample_count) + ")");

    const auto [offset, length] = offsets_[index];

    // Single I/O: read the entire sample record into a contiguous buffer
    std::vector<char> raw(length);
    file_.seekg(static_cast<std::streamoff>(offset));
    file_.read(raw.data(), static_cast<std::streamsize>(length));
    if (!file_.good())
        throw std::runtime_error(
            "I/O error reading sample " + std::to_string(index)
            + " from " + path_);

    if (header_.is_lz4_compressed()) {
        // ── LZ4 decompression path ─────────────────────────────
        if (length < 4)
            throw std::runtime_error(
                "Compressed sample too small for size prefix");

        // First 4 bytes: uncompressed size
        uint32_t uncompressed_size = 0;
        std::memcpy(&uncompressed_size, raw.data(), 4);

        const char*  comp_data = raw.data() + 4;
        const int    comp_size = static_cast<int>(length - 4);

        // Decompress
        std::vector<char> decompressed(uncompressed_size);
        const int result = LZ4_decompress_safe(
            comp_data, decompressed.data(),
            comp_size, static_cast<int>(uncompressed_size));

        if (result < 0)
            throw std::runtime_error(
                "LZ4 decompression failed (error " + std::to_string(result)
                + ") for sample " + std::to_string(index)
                + " in " + path_);

        return parse_sample_from_buffer(
            decompressed.data(),
            static_cast<size_t>(result),
            header_, norm_);
    }

    // ── Uncompressed path ──────────────────────────────────────
    return parse_sample_from_buffer(
        raw.data(), length, header_, norm_);
}


// Batch reading
Batch ShardReader::read_batch(const std::vector<uint32_t>& indices) {
    std::vector<Sample> samples;
    samples.reserve(indices.size());
    for (const uint32_t idx : indices)
        samples.push_back(read_sample(idx));

    if (header_.version >= FORMAT_VERSION_V2
        && header_.has_per_sample_dims())
    {
        return Batch::collate_padded(
            samples,
            static_cast<int32_t>(header_.image_channels),
            static_cast<int32_t>(header_.token_length));
    }

    return Batch::from_samples(
        samples,
        static_cast<int32_t>(header_.image_channels),
        static_cast<int32_t>(header_.image_height),
        static_cast<int32_t>(header_.image_width),
        static_cast<int32_t>(header_.token_length));
}

std::vector<Sample> ShardReader::read_all() {
    std::vector<Sample> result;
    result.reserve(header_.sample_count);
    for (uint32_t i = 0; i < header_.sample_count; ++i)
        result.push_back(read_sample(i));
    return result;
}

}  // namespace vlm
