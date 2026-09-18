#ifndef VLM_LOADER_DETAIL_CRC32_H
#define VLM_LOADER_DETAIL_CRC32_H

#include <cstdint>
#include <cstddef>

namespace vlm {
namespace detail {

// CRC32 (ISO 3309 / ITU-T V.42, same as Python zlib.crc32)

struct CRC32Table {
    uint32_t entries[256];
    constexpr CRC32Table() : entries{} {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                crc = (crc & 1u) ? (crc >> 1u) ^ 0xEDB88320u : crc >> 1u;
            }
            entries[i] = crc;
        }
    }
};

inline constexpr CRC32Table crc32_table{};

inline uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        crc = crc32_table.entries[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
    }
    return crc;
}

}  // namespace detail
}  // namespace vlm

#endif  // VLM_LOADER_DETAIL_CRC32_H
