#ifndef VLM_LOADER_BATCH_H
#define VLM_LOADER_BATCH_H

/**
 * @file batch.h
 * @brief Data structures for batched VQA samples loaded from binary shards.
 *
 * A Batch holds one or more samples in a contiguous, GPU-transfer-friendly
 * layout.  Each field (image tensor, question tokens, answer tokens) is
 * stored as a flat vector with known dimensions, so the caller can
 * interpret them as multi-dimensional tensors without any copying.
 *
 * Memory layout choices:
 * - Images are stored in NCHW order (batch, channel, height, width).
 * - Token arrays are stored in (batch, seq_len) order.
 * - All arrays are contiguous and can be wrapped in framework tensors
 *   (PyTorch, TensorFlow) via their from-pointer constructors.
 *
 * v2 additions:
 * - collate_padded(): dynamic padding to per-batch max(H) x max(W)
 * - padding_mask: per-sample binary mask (1 = real pixel, 0 = pad)
 * - Aspect ratio bucketing utilities for minimal padding waste
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace vlm {

/**
 * @brief A single VQA sample as loaded from a shard.
 *
 * Holds raw arrays for one sample.  The ShardReader returns these,
 * and the batcher collects them into a Batch.
 */
struct Sample {
    /// Normalized image tensor, float32, CHW layout, size = C*H*W.
    std::vector<float> image_tensor;

    /// Question token IDs, int32, length = token_length.
    std::vector<int32_t> question_ids;

    /// Question attention mask, int32, length = token_length.
    std::vector<int32_t> question_mask;

    /// Answer token IDs, int32, length = token_length.
    std::vector<int32_t> answer_ids;

    /// Answer attention mask, int32, length = token_length.
    std::vector<int32_t> answer_mask;

    /// JSON metadata string (deserialization is left to the caller).
    std::string metadata_json;

    // --- V2 per-sample dimensions (0 for v1 samples) ---
    uint16_t orig_h = 0;
    uint16_t orig_w = 0;
    uint16_t actual_h = 0;
    uint16_t actual_w = 0;
};


/**
 * @brief A batch of VQA samples with contiguous storage.
 *
 * Fields are stored as flat vectors.  The caller uses the dimension
 * fields to interpret the data as N-dimensional tensors.
 *
 * For example, image_data has total size
 *   batch_size * image_channels * image_height * image_width
 * and should be interpreted as a 4-D tensor in NCHW order.
 */
struct Batch {
    /// Number of samples in this batch.
    int32_t batch_size = 0;

    /// Image dimensions (same for all samples in a shard).
    int32_t image_channels = 0;
    int32_t image_height = 0;
    int32_t image_width = 0;

    /// Token sequence length (same for all samples).
    int32_t token_length = 0;

    /// Contiguous image data, NCHW layout, size = batch_size * C * H * W.
    std::vector<float> image_data;

    /// Contiguous question token IDs, size = batch_size * token_length.
    std::vector<int32_t> question_ids;

    /// Contiguous question attention masks, size = batch_size * token_length.
    std::vector<int32_t> question_mask;

    /// Contiguous answer token IDs, size = batch_size * token_length.
    std::vector<int32_t> answer_ids;

    /// Contiguous answer attention masks, size = batch_size * token_length.
    std::vector<int32_t> answer_mask;

    /// Per-sample metadata JSON strings.
    std::vector<std::string> metadata_json;

    // --- V2 dynamic padding fields ---

    /// Per-sample padding mask, NCHW layout [N, 1, max_H, max_W].
    /// 1.0f = real pixel, 0.0f = padding.  Empty for v1 batches.
    std::vector<float> padding_mask;

    /// Per-sample actual heights (before padding), length = batch_size.
    std::vector<int32_t> actual_heights;

    /// Per-sample actual widths (before padding), length = batch_size.
    std::vector<int32_t> actual_widths;

    /**
     * @brief Build a Batch from a vector of individual Samples (v1 path).
     *
     * Copies each sample's data into contiguous batch arrays.
     * All samples must have the same tensor dimensions.
     *
     * @param samples     Vector of Sample objects.
     * @param channels    Number of image channels.
     * @param height      Image height.
     * @param width       Image width.
     * @param tok_length  Token sequence length.
     * @return A populated Batch.
     */
    static Batch from_samples(
        const std::vector<Sample>& samples,
        int32_t channels, int32_t height, int32_t width,
        int32_t tok_length
    ) {
        Batch batch;
        batch.batch_size = static_cast<int32_t>(samples.size());
        batch.image_channels = channels;
        batch.image_height = height;
        batch.image_width = width;
        
        int32_t max_seq = 0;
        for (const auto& s : samples) {
            int32_t seq = 0;
            for (int32_t k = tok_length - 1; k >= 0; --k) {
                if (s.question_mask[k] != 0 || s.answer_mask[k] != 0) {
                    seq = k + 1;
                    break;
                }
            }
            max_seq = std::max(max_seq, seq);
        }
        max_seq = std::max(1, max_seq);
        batch.token_length = max_seq;

        const size_t img_size = static_cast<size_t>(channels) * height * width;
        const size_t tok_size = static_cast<size_t>(max_seq);

        batch.image_data.resize(samples.size() * img_size);
        batch.question_ids.resize(samples.size() * tok_size);
        batch.question_mask.resize(samples.size() * tok_size);
        batch.answer_ids.resize(samples.size() * tok_size);
        batch.answer_mask.resize(samples.size() * tok_size);
        batch.metadata_json.resize(samples.size());

        for (size_t i = 0; i < samples.size(); ++i) {
            std::copy(
                samples[i].image_tensor.begin(),
                samples[i].image_tensor.end(),
                batch.image_data.begin() + static_cast<ptrdiff_t>(i * img_size)
            );
            std::copy(
                samples[i].question_ids.begin(),
                samples[i].question_ids.begin() + max_seq,
                batch.question_ids.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                samples[i].question_mask.begin(),
                samples[i].question_mask.begin() + max_seq,
                batch.question_mask.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                samples[i].answer_ids.begin(),
                samples[i].answer_ids.begin() + max_seq,
                batch.answer_ids.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                samples[i].answer_mask.begin(),
                samples[i].answer_mask.begin() + max_seq,
                batch.answer_mask.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            batch.metadata_json[i] = samples[i].metadata_json;
        }

        return batch;
    }

    /**
     * @brief Build a dynamically-padded Batch from variable-size samples.
     *
     * Core thesis function.  Pads each batch on-the-fly to the
     * max(H) x max(W) of that specific batch, generating a binary
     * padding mask for ViT attention masking.
     *
     * Steps:
     *  1. Scan all samples for max_h = max(actual_h), max_w = max(actual_w).
     *  2. Allocate zero-initialized batch tensor [N, C, max_h, max_w].
     *  3. Copy each sample's [C, h_i, w_i] into top-left of padded slot.
     *  4. Generate padding_mask [N, 1, max_h, max_w] (1 = real, 0 = pad).
     *
     * @param samples     Vector of variable-size Sample objects (v2).
     * @param channels    Number of image channels.
     * @param tok_length  Token sequence length.
     * @return A Batch with dynamic padding and padding mask.
     */
    static Batch collate_padded(
        const std::vector<Sample>& samples,
        int32_t channels,
        int32_t tok_length
    ) {
        Batch batch;
        const auto n = static_cast<int32_t>(samples.size());
        batch.batch_size = n;
        batch.image_channels = channels;
        
        // Step 1: Find max dimensions across all samples in this batch
        int32_t max_h = 0;
        int32_t max_w = 0;
        int32_t max_seq = 0;
        for (const auto& s : samples) {
            max_h = std::max(max_h, static_cast<int32_t>(s.actual_h));
            max_w = std::max(max_w, static_cast<int32_t>(s.actual_w));
            
            int32_t seq = 0;
            for (int32_t k = tok_length - 1; k >= 0; --k) {
                if (s.question_mask[k] != 0 || s.answer_mask[k] != 0) {
                    seq = k + 1;
                    break;
                }
            }
            max_seq = std::max(max_seq, seq);
        }
        max_seq = std::max(1, max_seq);
        
        batch.image_height = max_h;
        batch.image_width = max_w;
        batch.token_length = max_seq;

        // Step 2: Allocate zero-initialized contiguous storage
        const size_t padded_img_size =
            static_cast<size_t>(channels) * max_h * max_w;
        const size_t mask_size =
            static_cast<size_t>(max_h) * max_w;  // single channel
        const size_t tok_size = static_cast<size_t>(max_seq);

        batch.image_data.assign(static_cast<size_t>(n) * padded_img_size, 0.0f);
        batch.padding_mask.assign(static_cast<size_t>(n) * mask_size, 0.0f);
        batch.actual_heights.resize(n);
        batch.actual_widths.resize(n);
        batch.question_ids.resize(static_cast<size_t>(n) * tok_size);
        batch.question_mask.resize(static_cast<size_t>(n) * tok_size);
        batch.answer_ids.resize(static_cast<size_t>(n) * tok_size);
        batch.answer_mask.resize(static_cast<size_t>(n) * tok_size);
        batch.metadata_json.resize(n);

        // Step 3 & 4: Copy each sample into its padded slot
        for (int32_t i = 0; i < n; ++i) {
            const auto& s = samples[i];
            const int32_t h_i = static_cast<int32_t>(s.actual_h);
            const int32_t w_i = static_cast<int32_t>(s.actual_w);
            batch.actual_heights[i] = h_i;
            batch.actual_widths[i] = w_i;

            // Copy image data: for each channel, copy h_i rows of w_i pixels
            // Source is dense [C, h_i, w_i]; dest is [C, max_h, max_w]
            for (int32_t c = 0; c < channels; ++c) {
                for (int32_t row = 0; row < h_i; ++row) {
                    const size_t src_offset =
                        static_cast<size_t>(c) * h_i * w_i
                        + static_cast<size_t>(row) * w_i;
                    const size_t dst_offset =
                        static_cast<size_t>(i) * padded_img_size
                        + static_cast<size_t>(c) * max_h * max_w
                        + static_cast<size_t>(row) * max_w;

                    std::memcpy(
                        batch.image_data.data() + dst_offset,
                        s.image_tensor.data() + src_offset,
                        static_cast<size_t>(w_i) * sizeof(float)
                    );
                }
            }

            // Generate padding mask [1, max_h, max_w] for this sample
            for (int32_t row = 0; row < h_i; ++row) {
                const size_t mask_offset =
                    static_cast<size_t>(i) * mask_size
                    + static_cast<size_t>(row) * max_w;
                std::fill_n(
                    batch.padding_mask.data() + mask_offset,
                    w_i,
                    1.0f
                );
            }

            // Copy token arrays
            std::copy(
                s.question_ids.begin(), s.question_ids.begin() + max_seq,
                batch.question_ids.begin()
                    + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                s.question_mask.begin(), s.question_mask.begin() + max_seq,
                batch.question_mask.begin()
                    + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                s.answer_ids.begin(), s.answer_ids.begin() + max_seq,
                batch.answer_ids.begin()
                    + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                s.answer_mask.begin(), s.answer_mask.begin() + max_seq,
                batch.answer_mask.begin()
                    + static_cast<ptrdiff_t>(i * tok_size)
            );
            batch.metadata_json[i] = s.metadata_json;
        }

        return batch;
    }
};


// ---------------------------------------------------------------------------
// Aspect Ratio Bucketing
// ---------------------------------------------------------------------------

/**
 * @brief Assigns sample indices to aspect-ratio buckets for minimal padding.
 *
 * Groups samples by quantized aspect ratio so that batches drawn from
 * a single bucket contain images of similar shape, dramatically reducing
 * padding waste.  This is critical for dynamic padding to be effective.
 *
 * Default bucket boundaries (aspect ratio = W/H):
 *   [0, 0.5) [0.5, 0.75) [0.75, 1.0) [1.0, 1.33) [1.33, 2.0) [2.0, inf)
 *
 * Usage:
 * @code
 *   auto buckets = vlm::bucket_by_aspect_ratio(samples, 6);
 *   for (auto& [bucket_id, indices] : buckets) {
 *       // Draw batch_size indices from this bucket
 *       auto batch_indices = indices.subspan(0, batch_size);
 *       auto batch = reader.read_batch(batch_indices);
 *   }
 * @endcode
 */
struct AspectRatioBucketer {
    /// Bucket boundaries (aspect ratio = W/H).  Length = num_buckets - 1.
    std::vector<float> boundaries;

    /// Per-bucket list of sample indices.
    std::vector<std::vector<uint32_t>> buckets;

    /**
     * Construct a bucketer with the given number of buckets.
     * Uses logarithmically-spaced boundaries centered around AR=1.0.
     *
     * @param num_buckets  Number of aspect ratio bins (default: 6).
     */
    explicit AspectRatioBucketer(int num_buckets = 6) {
        // Default boundaries: 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0
        // These are log-spaced around 1.0 for balanced bin sizes
        if (num_buckets <= 1) {
            // Single bucket: everything goes in bucket 0
            buckets.resize(1);
            return;
        }

        // Generate log-spaced boundaries
        const float log_min = std::log(0.5f);
        const float log_max = std::log(2.0f);
        const float step = (log_max - log_min) / static_cast<float>(num_buckets);

        boundaries.reserve(num_buckets - 1);
        for (int i = 1; i < num_buckets; ++i) {
            boundaries.push_back(std::exp(log_min + step * i));
        }
        buckets.resize(num_buckets);
    }

    /**
     * Assign a sample to its bucket based on aspect ratio.
     *
     * @param index     Sample index.
     * @param width     Image width.
     * @param height    Image height.
     */
    void add_sample(uint32_t index, uint16_t width, uint16_t height) {
        if (height == 0) return;
        const float ar = static_cast<float>(width) / static_cast<float>(height);
        int bucket_idx = static_cast<int>(boundaries.size());  // last bucket
        for (size_t b = 0; b < boundaries.size(); ++b) {
            if (ar < boundaries[b]) {
                bucket_idx = static_cast<int>(b);
                break;
            }
        }
        buckets[bucket_idx].push_back(index);
    }

    /**
     * Get batch-sized groups of indices from the same bucket.
     * Minimizes cross-aspect-ratio mixing within batches.
     *
     * @param batch_size  Desired batch size.
     * @return Vector of index vectors, each of length <= batch_size.
     */
    std::vector<std::vector<uint32_t>> get_batches(int batch_size) const {
        std::vector<std::vector<uint32_t>> result;
        for (const auto& bucket : buckets) {
            for (size_t start = 0; start < bucket.size();
                 start += static_cast<size_t>(batch_size))
            {
                size_t end = std::min(
                    start + static_cast<size_t>(batch_size),
                    bucket.size()
                );
                result.emplace_back(
                    bucket.begin() + static_cast<ptrdiff_t>(start),
                    bucket.begin() + static_cast<ptrdiff_t>(end)
                );
            }
        }
        return result;
    }
};

}  // namespace vlm

#endif  // VLM_LOADER_BATCH_H
