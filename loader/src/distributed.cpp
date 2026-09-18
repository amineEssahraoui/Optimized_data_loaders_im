/**
 * @file distributed.cpp
 * @brief Implementation of intrinsic data-partitioning for distributed loading.
 *
 * Provides two partitioning strategies that divide a global sample index
 * space across N workers with zero overlap and full coverage:
 *
 *   "contiguous"  -- Worker r gets indices [r*chunk, (r+1)*chunk).
 *                    The last worker may get fewer samples.
 *   "interleaved" -- Worker r gets indices r, r+N, r+2N, ...
 *
 * Both strategies guarantee:
 *   - Union of all partitions == {0, 1, ..., total-1}
 *   - Intersection of any two partitions == {} (empty)
 *
 * The partitioning is deterministic and requires no communication
 * between workers -- each worker computes its own partition locally.
 */

#include "distributed.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace vlm {

// get_worker_indices
std::vector<uint32_t> get_worker_indices(
    uint32_t total_samples,
    uint32_t worker_id,
    uint32_t num_workers,
    const std::string& strategy)
{
    if (num_workers == 0) {
        throw std::invalid_argument("num_workers must be >= 1");
    }
    if (worker_id >= num_workers) {
        throw std::invalid_argument(
            "worker_id (" + std::to_string(worker_id)
            + ") must be < num_workers (" + std::to_string(num_workers) + ")"
        );
    }

    std::vector<uint32_t> indices;

    if (strategy == "contiguous") {
        // Each worker gets a contiguous chunk.
        // chunk_size = ceil(total / num_workers)
        const uint32_t chunk_size =
            (total_samples + num_workers - 1) / num_workers;
        const uint32_t start = worker_id * chunk_size;
        const uint32_t end   = std::min(start + chunk_size, total_samples);

        indices.reserve(end > start ? end - start : 0);
        for (uint32_t i = start; i < end; ++i) {
            indices.push_back(i);
        }

    } else if (strategy == "interleaved") {
        // Round-robin: worker r reads indices r, r+W, r+2W, ...
        indices.reserve(
            (total_samples + num_workers - 1) / num_workers
        );
        for (uint32_t i = worker_id; i < total_samples; i += num_workers) {
            indices.push_back(i);
        }

    } else {
        throw std::invalid_argument(
            "Unknown partitioning strategy: '" + strategy
            + "'. Supported: 'contiguous', 'interleaved'"
        );
    }

    return indices;
}

// verify_partition
bool verify_partition(
    uint32_t total_samples,
    uint32_t num_workers,
    const std::string& strategy)
{
    if (num_workers == 0) {
        return total_samples == 0;
    }

    std::set<uint32_t> all_indices;
    uint32_t total_count = 0;

    for (uint32_t w = 0; w < num_workers; ++w) {
        const auto partition = get_worker_indices(
            total_samples, w, num_workers, strategy
        );

        for (const uint32_t idx : partition) {
            // Check for duplicates / overlap
            if (!all_indices.insert(idx).second) {
                return false;  // duplicate found
            }
        }
        total_count += static_cast<uint32_t>(partition.size());
    }

    // Must cover exactly {0, 1, ..., total_samples - 1}
    if (total_count != total_samples) {
        return false;
    }
    if (all_indices.size() != total_samples) {
        return false;
    }
    // Verify range: all indices in [0, total_samples)
    if (!all_indices.empty() && *all_indices.rbegin() >= total_samples) {
        return false;
    }

    return true;
}

}  // namespace vlm
