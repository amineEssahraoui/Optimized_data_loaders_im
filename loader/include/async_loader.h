#ifndef VLM_LOADER_ASYNC_LOADER_H
#define VLM_LOADER_ASYNC_LOADER_H

/**
 * @file async_loader.h
 * @brief High-performance asynchronous shard loader with thread pool,
 *        double buffering, and deterministic shuffle.
 *
 * AsyncShardLoader eliminates I/O bottlenecks by:
 *   1. Memory-mapping all shard files (zero user-space copy on read)
 *   2. Prefetching batches in a background thread pool
 *   3. Double buffering via a lock-free SPSC queue so batch i+1
 *      is ready while the GPU computes batch i
 *   4. Providing a deterministic shuffle buffer seeded by
 *      (seed + epoch) for reproducible training
 *
 * Thread safety:
 *   - next() / has_next() are called from a single consumer thread
 *     (the Python main thread or a DataLoader worker)
 *   - Internal worker threads are managed exclusively by this class
 *   - reset() must NOT be called concurrently with next()
 *
 * Usage (C++):
 * @code
 *   vlm::AsyncLoaderConfig cfg;
 *   cfg.shard_paths = {"shard_000.bin", "shard_001.bin"};
 *   cfg.batch_size  = 32;
 *   cfg.seed        = 42;
 *
 *   vlm::AsyncShardLoader loader(cfg);
 *   while (loader.has_next()) {
 *       vlm::Batch batch = loader.next();
 *       // ... process batch
 *   }
 *   loader.reset();  // next epoch
 * @endcode
 */

#include <cstdint>
#include <string>
#include <vector>

#include "batch.h"
#include "shard_reader.h"  // ShardFileHeader, NormalizationParams

namespace vlm {

// Configuration

/**
 * @brief Configuration for the async shard loader.
 */
struct AsyncLoaderConfig {
    /// Ordered list of shard file paths.
    std::vector<std::string> shard_paths;

    /// Number of samples per batch.
    int32_t batch_size = 32;

    /// Number of batches to prefetch ahead (double buffering depth).
    /// With prefetch_depth=2, batch i+1 is ready while GPU computes batch i.
    int32_t prefetch_depth = 2;

    /// Number of I/O worker threads in the thread pool.
    int32_t num_workers = 4;

    /// Base PRNG seed for deterministic shuffle.
    uint64_t seed = 42;

    /// Size of the shuffle buffer (0 = sequential, no shuffle).
    /// Larger values give better randomization at the cost of memory.
    int32_t shuffle_buffer_size = 1024;

    /// Normalization parameters for deferred uint8→float32 conversion.
    NormalizationParams norm;

    /// Distributed training: this worker's rank.
    uint32_t worker_rank = 0;

    /// Distributed training: total number of workers.
    uint32_t world_size = 1;

    /// Distributed partitioning strategy ("contiguous" or "interleaved").
    std::string partition_strategy = "contiguous";

    /// Safe Mode: explicitly compute and verify CRC32 of mapped memory.
    bool verify_crc = false;
};


// Checkpoint (lightweight: only epoch + seed)

/**
 * @brief Minimal checkpoint state for resuming training.
 *
 * Only stores the epoch number and seed.  The shuffle buffer state
 * is reconstructed deterministically from these two values.
 */
struct LoaderCheckpoint {
    uint64_t epoch = 0;
    uint64_t seed  = 0;
};


// AsyncShardLoader

// Forward-declare the implementation to hide threading details
struct AsyncShardLoaderImpl;

/**
 * @brief Async, multi-threaded, zero-copy shard loader with prefetching.
 *
 * Uses the PIMPL idiom to keep threading headers out of the public API.
 */
class AsyncShardLoader {
public:
    /**
     * Construct the loader, memory-map all shards, and start worker threads.
     *
     * @param config  Loader configuration.
     * @throws std::runtime_error if any shard cannot be opened or parsed.
     */
    explicit AsyncShardLoader(AsyncLoaderConfig config);

    /**
     * Destructor: signals workers to stop and joins all threads.
     */
    ~AsyncShardLoader();

    // Non-copyable, non-movable (owns threads and mapped files)
    AsyncShardLoader(const AsyncShardLoader&) = delete;
    AsyncShardLoader& operator=(const AsyncShardLoader&) = delete;
    AsyncShardLoader(AsyncShardLoader&&) = delete;
    AsyncShardLoader& operator=(AsyncShardLoader&&) = delete;

    // ── Iterator protocol ──────────────────────────────────────────

    /**
     * Get the next prefetched batch.
     *
     * Blocks if the prefetch queue is temporarily empty (workers are
     * still loading).  Returns immediately if a batch is already queued.
     *
     * @return The next Batch.
     * @throws std::runtime_error if called after the epoch is exhausted
     *         (has_next() returns false).
     */
    Batch next();

    /**
     * Check whether more batches remain in the current epoch.
     */
    bool has_next() const;

    // ── Epoch control ──────────────────────────────────────────────

    /**
     * Reset for a new epoch.
     *
     * Drains the prefetch queue, increments the epoch counter,
     * re-seeds the shuffle PRNG, and restarts prefetching from the
     * beginning of the dataset.
     *
     * Must NOT be called concurrently with next().
     */
    void reset();

    // ── Checkpointing ──────────────────────────────────────────────

    /**
     * Save a lightweight checkpoint (epoch + seed).
     *
     * Can be called at any point during iteration.
     */
    LoaderCheckpoint checkpoint() const;

    /**
     * Restore from a checkpoint and reset to that epoch's state.
     *
     * After restore(), calling next() yields the same sequence as
     * the original run at that checkpoint.
     */
    void restore(const LoaderCheckpoint& cp);

    // ── Stats ──────────────────────────────────────────────────────

    /** Total number of samples across all shards (for this worker). */
    uint64_t total_samples() const;

    /** Current epoch number (0-based). */
    uint64_t epoch() const;

private:
    std::unique_ptr<AsyncShardLoaderImpl> impl_;
};

}  // namespace vlm

#endif  // VLM_LOADER_ASYNC_LOADER_H
