/*
 * LZ4 block decompression — clean-room implementation
 * Vendored for the VLM data loader runtime (decompression only).
 *
 * Implements the LZ4 block format as documented at:
 *   https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
 *
 * Original algorithm by Yann Collet.
 * Licensed under BSD-2-Clause (see LICENSE in this directory).
 *
 * ── Format overview ──────────────────────────────────────────────
 * An LZ4 block is a sequence of "sequences", each containing:
 *   1. Token byte: high nibble = literal length, low nibble = match length
 *   2. Optional extended literal length bytes (if high nibble == 15)
 *   3. Literal data (copied verbatim)
 *   4. Match offset (2 bytes LE) — points backward in the output
 *   5. Optional extended match length bytes (if low nibble == 15)
 *   6. Match data (copied from earlier output, may overlap)
 *
 * The last sequence in a block has NO match section (ends after literals).
 * ─────────────────────────────────────────────────────────────────
 */

#include "lz4.h"
#include <string.h>  /* memcpy */

/* LZ4 format constants */
#define LZ4_MINMATCH    4
#define LZ4_ML_BITS     4
#define LZ4_ML_MASK     ((1u << LZ4_ML_BITS) - 1)   /* 0x0F */
#define LZ4_RUN_BITS    (8 - LZ4_ML_BITS)
#define LZ4_RUN_MASK    ((1u << LZ4_RUN_BITS) - 1)  /* 0x0F */

int LZ4_decompress_safe(
    const char* src,
    char* dst,
    int compressedSize,
    int dstCapacity)
{
    /* ── Argument validation ─────────────────────────────────── */
    if (src == NULL || dst == NULL ||
        compressedSize < 0 || dstCapacity < 0)
    {
        return -1;
    }
    if (compressedSize == 0) {
        return 0;  /* empty input → empty output */
    }

    const unsigned char* ip     = (const unsigned char*)src;
    const unsigned char* ip_end = ip + compressedSize;
    unsigned char*       op     = (unsigned char*)dst;
    unsigned char*       op_end = op + dstCapacity;

    /* ── Main decompression loop ─────────────────────────────── */
    for (;;) {
        /* 1. Read token */
        if (ip >= ip_end) return -3;
        const unsigned char token = *ip++;

        /* 2. Decode literal length (high nibble) */
        unsigned int lit_len = (token >> LZ4_ML_BITS) & LZ4_RUN_MASK;
        if (lit_len == LZ4_RUN_MASK) {
            unsigned char extra;
            do {
                if (ip >= ip_end) return -3;
                extra = *ip++;
                lit_len += extra;
            } while (extra == 255);
        }

        /* 3. Copy literals */
        if (lit_len > 0) {
            if (ip + lit_len > ip_end)           return -3;  /* src overflow */
            if (op + lit_len > op_end)           return -2;  /* dst overflow */
            memcpy(op, ip, lit_len);
            ip += lit_len;
            op += lit_len;
        }

        /* 4. End-of-block check: last sequence has no match */
        if (ip >= ip_end) break;

        /* 5. Read match offset (2 bytes, little-endian) */
        if (ip + 2 > ip_end) return -3;
        const unsigned int offset = (unsigned int)ip[0]
                                  | ((unsigned int)ip[1] << 8);
        ip += 2;

        if (offset == 0)                                   return -3;
        if (offset > (unsigned int)(op - (unsigned char*)dst)) return -3;

        /* 6. Decode match length (low nibble + MINMATCH) */
        unsigned int match_len = (token & LZ4_ML_MASK) + LZ4_MINMATCH;
        if ((token & LZ4_ML_MASK) == LZ4_ML_MASK) {
            unsigned char extra;
            do {
                if (ip >= ip_end) return -3;
                extra = *ip++;
                match_len += extra;
            } while (extra == 255);
        }

        /* 7. Bounds-check the match copy */
        if (op + match_len > op_end) return -2;

        /* 8. Copy match (handles overlapping copies correctly) */
        const unsigned char* match_src = op - offset;
        if (offset >= match_len) {
            /* Non-overlapping: safe to use memcpy */
            memcpy(op, match_src, match_len);
            op += match_len;
        } else {
            /* Overlapping: byte-by-byte (the repeated-pattern case) */
            unsigned int i;
            for (i = 0; i < match_len; i++) {
                op[i] = match_src[i];
            }
            op += match_len;
        }
    }

    return (int)(op - (unsigned char*)dst);
}


int LZ4_compressBound(int inputSize)
{
    if (inputSize < 0) return 0;
    /* LZ4 worst case: inputSize + inputSize/255 + 16 */
    return inputSize + (inputSize / 255) + 16;
}
