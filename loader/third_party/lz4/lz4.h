/*
 * LZ4 - Fast LZ compression algorithm (decompression-only subset)
 * Vendored for the VLM data loader runtime.
 *
 * Original author : Yann Collet
 * Source           : https://github.com/lz4/lz4
 * License          : BSD-2-Clause (see LICENSE in this directory)
 *
 * This header exposes only the functions required by the VLM loader:
 *   - LZ4_decompress_safe()   : safe decompression with bounds checking
 *   - LZ4_compressBound()     : max compressed size (buffer sizing utility)
 *
 * Compression is handled on the Python side via the `lz4` PyPI package.
 */

#ifndef LZ4_H_VLM_VENDORED
#define LZ4_H_VLM_VENDORED

#ifdef __cplusplus
extern "C" {
#endif

/**
 * LZ4_decompress_safe() :
 *   Decompress an LZ4-compressed block.
 *
 * @param src             Compressed data (LZ4 block format, no frame header).
 * @param dst             Output buffer.
 * @param compressedSize  Exact size of the compressed data in bytes.
 * @param dstCapacity     Maximum size of the output buffer.
 *
 * @return  The number of bytes decompressed into dst (always <= dstCapacity),
 *          or a negative value on failure:
 *            -1  invalid arguments (NULL pointers, negative sizes)
 *            -2  output buffer overflow
 *            -3  malformed/corrupted compressed data
 */
int LZ4_decompress_safe(
    const char* src,
    char* dst,
    int compressedSize,
    int dstCapacity
);

/**
 * LZ4_compressBound() :
 *   Calculate the maximum possible compressed size for a given input size.
 *
 * @param inputSize  Uncompressed data size in bytes.
 * @return  Maximum compressed size, or 0 if inputSize is invalid.
 */
int LZ4_compressBound(int inputSize);

#ifdef __cplusplus
}
#endif

#endif /* LZ4_H_VLM_VENDORED */
