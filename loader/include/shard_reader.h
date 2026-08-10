#ifndef VLM_LOADER_SHARD_READER_H
#define VLM_LOADER_SHARD_READER_H

/**
 * @file shard_reader.h
 * @brief C++ shard reader for the VLM binary shard format.
 *
 * Reads binary shard files produced by the Python ShardWriter, providing
 * random-access sample retrieval via the offset table.  Validates the
 * header magic, format version, and CRC32 footer checksum on open.
 *
 * This reader is the production-grade C++ counterpart to the Python
 * shard_reader.py.  It is designed for:
 *   - Memory-efficient sample access (no full-file buffering)
 *   - Batch construction for GPU feeding
 *   - Integration with the distributed partitioning utilities
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
static constexpr uint32_t FORMAT_VERSION = 1;
static constexpr size_t   HEADER_SIZE    = 64;

// ---------------------------------------------------------------------------
// ShardFileHeader -- parsed from the first 64 bytes of every shard file
// ---------------------------------------------------------------------------
struct ShardFileHeader {
    uint32_t version;
    uint32_t sample_count;
    uint64_t offset_table_pos;
    uint32_t image_channels;
    uint32_t image_height;
    uint32_t image_width;
    uint32_t token_length;
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
     * Read a single sample by zero-based index.
     *
     * Uses the offset table for O(1) seeking.
     * @throws std::out_of_range if index >= sample_count().
     */
    Sample read_sample(uint32_t index);

    /**
     * Read multiple samples and pack them into a contiguous Batch.
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
};

}  // namespace vlm

#endif  // VLM_LOADER_SHARD_READER_H
