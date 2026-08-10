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
 */

#include <cstdint>
#include <string>
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

    /**
     * @brief Build a Batch from a vector of individual Samples.
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
        batch.token_length = tok_length;

        const size_t img_size = static_cast<size_t>(channels) * height * width;
        const size_t tok_size = static_cast<size_t>(tok_length);

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
                samples[i].question_ids.end(),
                batch.question_ids.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                samples[i].question_mask.begin(),
                samples[i].question_mask.end(),
                batch.question_mask.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                samples[i].answer_ids.begin(),
                samples[i].answer_ids.end(),
                batch.answer_ids.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            std::copy(
                samples[i].answer_mask.begin(),
                samples[i].answer_mask.end(),
                batch.answer_mask.begin() + static_cast<ptrdiff_t>(i * tok_size)
            );
            batch.metadata_json[i] = samples[i].metadata_json;
        }

        return batch;
    }
};

}  // namespace vlm

#endif  // VLM_LOADER_BATCH_H
