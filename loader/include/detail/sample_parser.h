#ifndef VLM_LOADER_DETAIL_SAMPLE_PARSER_H
#define VLM_LOADER_DETAIL_SAMPLE_PARSER_H

/**
 * @file detail/sample_parser.h
 * @brief Shared internal header: buffer-based sample parsing and SIMD
 *        normalization dispatch.
 *
 * This header is included by both shard_reader.cpp (sync path) and
 * async_loader.cpp (async/mmap path).  It contains the hot-path
 * functions that convert a raw byte buffer into a vlm::Sample.
 *
 * NOT part of the public API — lives under include/detail/.
 */

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "shard_reader.h"  // ShardFileHeader, PerSampleDims, NormalizationParams
#include "batch.h"          // Sample

// AVX2 + FMA intrinsics (guarded by compile-time flag from CMake)
#ifdef VLM_HAS_AVX2
#include <immintrin.h>
#endif

namespace vlm {
namespace detail {

// SIMD-accelerated uint8 → float32 normalization

#ifdef VLM_HAS_AVX2
/**
 * AVX2 + FMA vectorised normalisation.
 *
 * For each pixel: dst[i] = src[i] * scale + bias
 *   where  scale = 1 / (255 · σ)
 *          bias  = −μ / σ
 *
 * Processes 8 float values per iteration via a single FMA instruction.
 */
inline void normalize_channel_avx2(
    const uint8_t* __restrict src,
    float*         __restrict dst,
    size_t count,
    float mean,
    float std_val)
{
    const float scale = 1.0f / (255.0f * std_val);
    const float bias  = -mean / std_val;

    const __m256 vscale = _mm256_set1_ps(scale);
    const __m256 vbias  = _mm256_set1_ps(bias);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m128i raw8  = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(src + i));
        const __m256i raw32 = _mm256_cvtepu8_epi32(raw8);
        const __m256  fval  = _mm256_cvtepi32_ps(raw32);
        const __m256 result = _mm256_fmadd_ps(fval, vscale, vbias);
        _mm256_storeu_ps(dst + i, result);
    }
    // Scalar tail (0-7 remaining elements)
    for (; i < count; ++i)
        dst[i] = static_cast<float>(src[i]) * scale + bias;
}
#endif  // VLM_HAS_AVX2

/**
 * Scalar fallback for uint8→float32 normalization.
 * Uses the same fused scale+bias formulation for consistency.
 */
inline void normalize_channel_scalar(
    const uint8_t* __restrict src,
    float*         __restrict dst,
    size_t count,
    float mean,
    float std_val)
{
    const float scale = 1.0f / (255.0f * std_val);
    const float bias  = -mean / std_val;

    for (size_t i = 0; i < count; ++i)
        dst[i] = static_cast<float>(src[i]) * scale + bias;
}

/**
 * Dispatch to SIMD or scalar normalization based on compile-time flags.
 */
inline void normalize_channel(
    const uint8_t* src, float* dst, size_t count,
    float mean, float std_val)
{
#ifdef VLM_HAS_AVX2
    normalize_channel_avx2(src, dst, count, mean, std_val);
#else
    normalize_channel_scalar(src, dst, count, mean, std_val);
#endif
}


// Buffer-based sample parsing (unified for compressed / uncompressed)

/**
 * Parse a sample from a contiguous memory buffer.
 *
 * This is the inner hot-path: called for every sample regardless of
 * whether it was LZ4-compressed on disk.  All field reads are simple
 * pointer advances with bounds checking.
 *
 * @param buf      Pointer to the start of the sample record.
 * @param buf_len  Length of the sample record in bytes.
 * @param hdr      Parsed shard file header.
 * @param norm     Normalization parameters for uint8→float conversion.
 * @return A fully populated Sample.
 */
inline Sample parse_sample_from_buffer(
    const char* buf,
    size_t      buf_len,
    const ShardFileHeader& hdr,
    const NormalizationParams& norm)
{
    const char* ptr = buf;
    const char* end = buf + buf_len;

    auto advance = [&](size_t n) -> const char* {
        if (ptr + n > end)
            throw std::runtime_error("Sample buffer overrun");
        const char* p = ptr;
        ptr += n;
        return p;
    };

    Sample sample;

    const bool has_dims = hdr.has_per_sample_dims();
    const bool is_uint8 = hdr.is_uint8();
    const bool is_v2    = hdr.version >= FORMAT_VERSION_V2;

    // ── Per-sample dimensions (v2) ──────────────────────────────
    uint32_t img_h = hdr.image_height;
    uint32_t img_w = hdr.image_width;

    if (has_dims) {
        const char* dim_ptr = advance(sizeof(PerSampleDims));
        PerSampleDims dims{};
        std::memcpy(&dims, dim_ptr, sizeof(PerSampleDims));
        sample.orig_h   = dims.orig_h;
        sample.orig_w   = dims.orig_w;
        sample.actual_h = dims.actual_h;
        sample.actual_w = dims.actual_w;
        img_h = dims.actual_h;
        img_w = dims.actual_w;
    } else if (is_v2) {
        sample.actual_h = static_cast<uint16_t>(hdr.image_height);
        sample.actual_w = static_cast<uint16_t>(hdr.image_width);
    }

    // ── Image tensor ────────────────────────────────────────────
    const size_t pixels =
        static_cast<size_t>(hdr.image_channels) * img_h * img_w;

    if (is_uint8) {
        const auto* raw_u8 =
            reinterpret_cast<const uint8_t*>(advance(pixels));
        sample.image_tensor.resize(pixels);

        const uint32_t ch = hdr.image_channels;
        const size_t plane = static_cast<size_t>(img_h) * img_w;

        for (uint32_t c = 0; c < ch; ++c) {
            normalize_channel(
                raw_u8 + c * plane,
                sample.image_tensor.data() + c * plane,
                plane,
                norm.mean[c],
                norm.std[c]);
        }
    } else {
        const size_t nbytes = pixels * sizeof(float);
        const char* fp = advance(nbytes);
        sample.image_tensor.resize(pixels);
        std::memcpy(sample.image_tensor.data(), fp, nbytes);
    }

    // ── Token arrays (4 × T × int32) ───────────────────────────
    const size_t tok_len   = hdr.token_length;
    const size_t tok_bytes = tok_len * sizeof(int32_t);

    auto read_tokens = [&](std::vector<int32_t>& vec) {
        vec.resize(tok_len);
        const char* tp = advance(tok_bytes);
        std::memcpy(vec.data(), tp, tok_bytes);
    };

    read_tokens(sample.question_ids);
    read_tokens(sample.question_mask);
    read_tokens(sample.answer_ids);
    read_tokens(sample.answer_mask);

    // ── Metadata: uint32 length + UTF-8 JSON ────────────────────
    const char* len_ptr = advance(sizeof(uint32_t));
    uint32_t meta_len = 0;
    std::memcpy(&meta_len, len_ptr, sizeof(uint32_t));

    const char* json_ptr = advance(meta_len);
    sample.metadata_json.assign(json_ptr, meta_len);

    return sample;
}

}  // namespace detail
}  // namespace vlm

#endif  // VLM_LOADER_DETAIL_SAMPLE_PARSER_H
