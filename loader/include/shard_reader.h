#ifndef VLM_LOADER_SHARD_READER_H
#define VLM_LOADER_SHARD_READER_H

/**
 * @file shard_reader.h
 * @brief C++ shard reader for the VLM binary shard format (v1 and v2).
 *
 * Reads binary shard files produced by the Python ShardWriter, providing
 * random-access sample retrieval via the offset table.  Validates the
 * header magic, format version, and CRC32 footer checksum on open.
 *
 * Supports both Format v1 (float32, fixed dimensions) and Format v2
 * (uint8, per-sample dimensions, flags field).
 *
 * The binary format specification is documented in docs/shard_format.md.
 */

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "batch.h"

namespace vlm {

// ---------------------------------------------------------------------------
// Format constants (must match Python shard_writer.py exactly)
// ---------------------------------------------------------------------------
static constexpr uint32_t FORMAT_VERSION    = 1;
static constexpr uint32_t FORMAT_VERSION_V2 = 2;
static constexpr size_t   HEADER_SIZE       = 64;

// ---------------------------------------------------------------------------
// V2 header flags (bytes 40-43 of header, was reserved in v1)
// ---------------------------------------------------------------------------
static constexpr uint32_t FLAG_UINT8_STORAGE    = 0x01;  // bit 0
static constexpr uint32_t FLAG_PER_SAMPLE_DIMS  = 0x02;  // bit 1
static constexpr uint32_t FLAG_LZ4_COMPRESSED   = 0x04;  // bit 2
static constexpr uint32_t FLAG_ZSTD_COMPRESSED  = 0x08;  // bit 3

// ---------------------------------------------------------------------------
// ShardFileHeader -- parsed from the first 64 bytes of every shard file
// ---------------------------------------------------------------------------
struct ShardFileHeader {
    uint32_t version;
    uint32_t sample_count;
    uint64_t offset_table_pos;
    uint32_t image_channels;
    uint32_t image_height;    // v1: fixed H; v2: max_image_dim
    uint32_t image_width;     // v1: fixed W; v2: max_image_dim
    uint32_t token_length;
    uint32_t flags;           // v2 only; 0 for v1

    // Convenience accessors for flag bits
    bool is_uint8()           const { return (flags & FLAG_UINT8_STORAGE)   != 0; }
    bool has_per_sample_dims() const { return (flags & FLAG_PER_SAMPLE_DIMS) != 0; }
    bool is_lz4_compressed()  const { return (flags & FLAG_LZ4_COMPRESSED)  != 0; }
    bool is_zstd_compressed() const { return (flags & FLAG_ZSTD_COMPRESSED) != 0; }
};

// ---------------------------------------------------------------------------
// PerSampleDims -- 8-byte prefix in v2 sample records
// ---------------------------------------------------------------------------
struct PerSampleDims {
    uint16_t orig_h;
    uint16_t orig_w;
    uint16_t actual_h;
    uint16_t actual_w;
};

// ---------------------------------------------------------------------------
// Normalization parameters (passed at load time, not stored in file)
// ---------------------------------------------------------------------------
struct NormalizationParams {
    float mean[3] = {0.485f, 0.456f, 0.406f};  // ImageNet defaults
    float std[3]  = {0.229f, 0.224f, 0.225f};
};

// ---------------------------------------------------------------------------
// ShardReader -- random-access reader for a single binary shard
// ---------------------------------------------------------------------------
class ShardReader {
public:
    /**
     * Open a shard file and parse its header, offset table, and footer.
     *
     * @param path  Path to the .bin shard file.
     * @throws std::runtime_error on invalid magic, version, or CRC32.
     */
    explicit ShardReader(const std::string& path);

    ~ShardReader();

    // Non-copyable (owns a file handle)
    ShardReader(const ShardReader&) = delete;
    ShardReader& operator=(const ShardReader&) = delete;

    /** Return a const reference to the parsed file header. */
    const ShardFileHeader& header() const;

    /** Number of samples in this shard. */
    uint32_t sample_count() const;

    /**
     * Set normalization parameters for deferred uint8→float32 conversion.
     * Only used when reading v2 shards with uint8 storage.
     */
    void set_normalization(const NormalizationParams& params);

    /**
     * Read a single sample by zero-based index.
     *
     * Uses the offset table for O(1) seeking.
     * For v2 uint8 shards, applies deferred normalization.
     * @throws std::out_of_range if index >= sample_count().
     */
    Sample read_sample(uint32_t index);

    /**
     * Read multiple samples and pack them into a contiguous Batch.
     * For v2 shards with per-sample dims, uses collate_padded().
     *
     * @param indices  Vector of zero-based sample indices.
     */
    Batch read_batch(const std::vector<uint32_t>& indices);

    /** Read all samples in the shard. */
    std::vector<Sample> read_all();

private:
    void parse_header();
    void parse_offset_table();
    void verify_footer();

    std::string path_;
    std::ifstream file_;
    ShardFileHeader header_{};
    std::vector<std::pair<uint64_t, uint64_t>> offsets_;  // (offset, length)
    NormalizationParams norm_{};
};

}  // namespace vlm

#endif  // VLM_LOADER_SHARD_READER_H
