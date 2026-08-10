#ifndef VLM_LOADER_DISTRIBUTED_H
#define VLM_LOADER_DISTRIBUTED_H

/**
 * @file distributed.h
 * @brief Intrinsic data-partitioning utilities for distributed loading.
 *
 * These functions compute which sample indices each worker/GPU should
 * read, ensuring that the full dataset is covered with zero overlap.
 * The partitioning is performed *inside the loader* rather than by an
 * external orchestrator, making the loader intrinsically distributed.
 *
 * Supported strategies:
 *   - "contiguous"  -- each worker gets a contiguous slice of indices.
 *                       Best for sequential I/O locality.
 *   - "interleaved" -- round-robin assignment (worker r reads indices
 *                       r, r+W, r+2W, …).  Better load-balance when
 *                       sample sizes vary.
 *
 * Usage with ShardReader:
 * @code
 *   auto indices = vlm::get_worker_indices(reader.sample_count(),
 *                                          my_rank, world_size,
 *                                          "contiguous");
 *   auto batch = reader.read_batch(indices);
 * @endcode
 */

#include <cstdint>
#include <string>
#include <vector>

namespace vlm {

/**
 * Compute the sample indices assigned to a specific worker.
 *
 * @param total_samples  Total number of samples across the shard(s).
 * @param worker_id      This worker's zero-based rank.
 * @param num_workers    Total number of workers / GPUs.
 * @param strategy       "contiguous" or "interleaved".
 * @return Sorted vector of sample indices for this worker.
 * @throws std::invalid_argument on bad rank, world_size, or strategy.
 */
std::vector<uint32_t> get_worker_indices(
    uint32_t total_samples,
    uint32_t worker_id,
    uint32_t num_workers,
    const std::string& strategy = "contiguous"
);

/**
 * Verify that a partitioning strategy covers all indices exactly once.
 *
 * Checks that the union of all workers' partitions equals {0, …, N-1}
 * and that no index appears in more than one partition.
 *
 * @param total_samples  Total number of samples.
 * @param num_workers    Number of workers.
 * @param strategy       "contiguous" or "interleaved".
 * @return true if the partition is correct, false otherwise.
 */
bool verify_partition(
    uint32_t total_samples,
    uint32_t num_workers,
    const std::string& strategy = "contiguous"
);

}  // namespace vlm

#endif  // VLM_LOADER_DISTRIBUTED_H
