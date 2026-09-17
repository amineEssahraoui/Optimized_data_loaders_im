/**
 * @file async_loader.cpp
 * @brief Implementation of the high-performance async shard loader.
 *
 * Architecture:
 *
 *   ┌────────────────────────────────────────────────────────┐
 *   │                  AsyncShardLoaderImpl                  │
 *   │                                                       │
 *   │  MappedFile[] ──┐                                     │
 *   │                 ├─▶ ShardInfo[] (header + offsets)     │
 *   │                 │                                     │
 *   │  ShuffleBuffer ─┤                                     │
 *   │                 │   ┌──────────────────────────────┐   │
 *   │                 ├──▶│  Worker threads (N)          │   │
 *   │                 │   │  • Pull batch indices        │   │
 *   │                 │   │  • Parse from mmap'd region  │   │
 *   │                 │   │  • Collate into Batch        │   │
 *   │                 │   │  • Push to SPSC queue        │   │
 *   │                 │   └──────────┬───────────────────┘   │
 *   │                                │                       │
 *   │                 SPSCQueue<Batch> (prefetch_depth)       │
 *   │                                │                       │
 *   │                 next() ◀───────┘                       │
 *   └────────────────────────────────────────────────────────┘
 *
 * Thread safety:
 *   - Workers only READ from mmap'd memory (no synchronization needed)
 *   - The work dispatch uses a mutex + condition variable (batches are
 *     coarse-grained, so contention is minimal)
 *   - The SPSC queue is lock-free between the single dispatcher thread
 *     and the consumer
 */

#include "async_loader.h"
#include "mmap_file.h"
#include "spsc_queue.h"
#include "distributed.h"
#include "detail/sample_parser.h"

// Vendored LZ4 (decompression only)
extern "C" {
#include "lz4.h"
}

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace vlm {

// ═══════════════════════════════════════════════════════════════════
// Per-shard metadata (header + offset table, parsed from mmap)
// ═══════════════════════════════════════════════════════════════════

namespace {

static constexpr char MAGIC_START[9] = "VLMSHARD";

/**
 * Parsed shard info: the header and offset table extracted from the
 * memory-mapped file.  No ifstream needed.
 */
struct ShardInfo {
    ShardFileHeader header{};
    std::vector<std::pair<uint64_t, uint64_t>> offsets;  // (offset, length)
};

/**
 * Parse the 64-byte header from a memory-mapped shard.
 */
ShardFileHeader parse_header_from_mmap(const uint8_t* data, size_t file_size) {
    if (file_size < HEADER_SIZE)
        throw std::runtime_error("Shard file too small for header");

    if (std::memcmp(data, MAGIC_START, 8) != 0)
        throw std::runtime_error("Invalid shard magic");

    ShardFileHeader hdr{};
    const auto* raw = reinterpret_cast<const char*>(data);
    std::memcpy(&hdr.version,          raw +  8, 4);
    std::memcpy(&hdr.sample_count,     raw + 12, 4);
    std::memcpy(&hdr.offset_table_pos, raw + 16, 8);
    std::memcpy(&hdr.image_channels,   raw + 24, 4);
    std::memcpy(&hdr.image_height,     raw + 28, 4);
    std::memcpy(&hdr.image_width,      raw + 32, 4);
    std::memcpy(&hdr.token_length,     raw + 36, 4);

    if (hdr.version == FORMAT_VERSION) {
        hdr.flags = 0;
    } else if (hdr.version == FORMAT_VERSION_V2) {
        std::memcpy(&hdr.flags, raw + 40, 4);
    } else {
        throw std::runtime_error(
            "Unsupported shard version " + std::to_string(hdr.version));
    }

    return hdr;
}

/**
 * Parse the offset table from a memory-mapped shard.
 */
std::vector<std::pair<uint64_t, uint64_t>> parse_offsets_from_mmap(
    const uint8_t* data,
    size_t file_size,
    const ShardFileHeader& hdr)
{
    const size_t table_size =
        static_cast<size_t>(hdr.sample_count) * 16u;
    if (hdr.offset_table_pos + table_size > file_size)
        throw std::runtime_error("Offset table exceeds file bounds");

    std::vector<std::pair<uint64_t, uint64_t>> offsets(hdr.sample_count);
    const auto* ptr =
        reinterpret_cast<const char*>(data + hdr.offset_table_pos);

    for (uint32_t i = 0; i < hdr.sample_count; ++i) {
        uint64_t offset = 0, length = 0;
        std::memcpy(&offset, ptr + i * 16,     8);
        std::memcpy(&length, ptr + i * 16 + 8, 8);
        offsets[i] = {offset, length};
    }

    return offsets;
}

/**
 * Read a single sample from a memory-mapped shard, handling LZ4
 * decompression if the shard is compressed.
 */
Sample read_sample_from_mmap(
    const uint8_t* shard_data,
    size_t shard_size,
    const ShardInfo& info,
    uint32_t local_index,
    const NormalizationParams& norm)
{
    const auto [offset, length] = info.offsets[local_index];

    if (offset + length > shard_size)
        throw std::runtime_error(
            "Sample record exceeds mapped file bounds");

    const auto* sample_ptr =
        reinterpret_cast<const char*>(shard_data + offset);

    if (info.header.is_lz4_compressed()) {
        if (length < 4)
            throw std::runtime_error(
                "Compressed sample too small for size prefix");

        uint32_t uncompressed_size = 0;
        std::memcpy(&uncompressed_size, sample_ptr, 4);

        const char* comp_data = sample_ptr + 4;
        const int   comp_size = static_cast<int>(length - 4);

        std::vector<char> decompressed(uncompressed_size);
        const int result = LZ4_decompress_safe(
            comp_data, decompressed.data(),
            comp_size, static_cast<int>(uncompressed_size));

        if (result < 0)
            throw std::runtime_error(
                "LZ4 decompression failed (error "
                + std::to_string(result) + ")");

        return detail::parse_sample_from_buffer(
            decompressed.data(),
            static_cast<size_t>(result),
            info.header, norm);
    }

    return detail::parse_sample_from_buffer(
        sample_ptr, length, info.header, norm);
}

}  // anonymous namespace


// ═══════════════════════════════════════════════════════════════════
// Global sample address: (shard_id, local_index)
// ═══════════════════════════════════════════════════════════════════

struct SampleAddress {
    uint32_t shard_id;
    uint32_t local_index;
};


// ═══════════════════════════════════════════════════════════════════
// AsyncShardLoaderImpl (PIMPL body)
// ═══════════════════════════════════════════════════════════════════

struct AsyncShardLoaderImpl {
    // ── Configuration ──────────────────────────────────────────────
    AsyncLoaderConfig config;

    // ── Mapped shards ──────────────────────────────────────────────
    std::vector<MappedFile>  mapped_files;
    std::vector<ShardInfo>   shard_infos;

    // ── Global sample index ────────────────────────────────────────
    //   all_addresses[i] = {shard_id, local_index}
    //   After distributed partitioning, this contains only this
    //   worker's share.
    std::vector<SampleAddress> all_addresses;

    // ── Epoch / shuffle state ──────────────────────────────────────
    uint64_t              epoch_   = 0;
    std::mt19937_64       rng_;
    std::vector<uint32_t> shuffled_order;  // indices into all_addresses
    size_t                cursor_  = 0;    // next index in shuffled_order

    // ── Thread pool ────────────────────────────────────────────────
    std::vector<std::thread>  workers;
    std::atomic<bool>         stop_flag{false};

    // Work items: each is a batch (vector of SampleAddresses)
    struct WorkItem {
        std::vector<SampleAddress> addresses;
    };

    std::mutex              work_mutex;
    std::condition_variable work_cv;
    std::vector<WorkItem>   work_queue;
    bool                    work_done = false;

    // ── Aggregator thread (SPSC single producer) ───────────────────
    std::vector<Batch>      pending_batches;
    std::mutex              pending_mutex;
    std::condition_variable pending_cv;
    std::thread             aggregator_thread;

    // ── Output queue (SPSC) ────────────────────────────────────────
    std::unique_ptr<SPSCQueue<Batch>> output_queue;
    std::mutex              output_mutex;
    std::condition_variable output_cv;

    // ── Dispatch thread ────────────────────────────────────────────
    std::thread dispatch_thread;

    // ── Completion tracking ────────────────────────────────────────
    std::atomic<uint64_t> batches_produced{0};
    uint64_t              total_batches = 0;
    std::atomic<bool>     epoch_finished{false};


    // ── Constructor ────────────────────────────────────────────────

    explicit AsyncShardLoaderImpl(AsyncLoaderConfig cfg)
        : config(std::move(cfg))
    {
        // 1. Memory-map all shards
        mapped_files.reserve(config.shard_paths.size());
        shard_infos.reserve(config.shard_paths.size());

        for (const auto& path : config.shard_paths) {
            mapped_files.emplace_back(path);
            const auto& mf = mapped_files.back();

            ShardInfo si;
            si.header  = parse_header_from_mmap(mf.data(), mf.size());
            si.offsets = parse_offsets_from_mmap(
                mf.data(), mf.size(), si.header);
            shard_infos.push_back(std::move(si));
        }

        // 2. Build global sample address table
        build_address_table();

        // 3. Create output queue (+1 for SPSC sentinel slot)
        output_queue = std::make_unique<SPSCQueue<Batch>>(
            static_cast<size_t>(config.prefetch_depth) + 1);

        // 4. Initialize epoch
        init_epoch();

        // 5. Start worker threads
        start_workers();
    }

    ~AsyncShardLoaderImpl() {
        shutdown();
    }


    // ── Address table construction ─────────────────────────────────

    void build_address_table() {
        // Compute total samples across all shards
        uint64_t total = 0;
        for (const auto& si : shard_infos)
            total += si.header.sample_count;

        // Build flat address table: global_index → (shard, local)
        std::vector<SampleAddress> global_addrs;
        global_addrs.reserve(static_cast<size_t>(total));

        for (uint32_t s = 0; s < shard_infos.size(); ++s) {
            for (uint32_t i = 0; i < shard_infos[s].header.sample_count; ++i) {
                global_addrs.push_back({s, i});
            }
        }

        // Apply distributed partitioning
        if (config.world_size > 1) {
            auto indices = get_worker_indices(
                static_cast<uint32_t>(global_addrs.size()),
                config.worker_rank,
                config.world_size,
                config.partition_strategy);

            all_addresses.clear();
            all_addresses.reserve(indices.size());
            for (const uint32_t idx : indices)
                all_addresses.push_back(global_addrs[idx]);
        } else {
            all_addresses = std::move(global_addrs);
        }
    }


    // ── Shuffle buffer ─────────────────────────────────────────────

    void init_epoch() {
        epoch_finished.store(false);
        batches_produced.store(0);
        cursor_ = 0;

        // Seed PRNG deterministically: seed + epoch
        rng_.seed(config.seed + epoch_);

        // Build shuffled order
        const size_t n = all_addresses.size();
        shuffled_order.resize(n);
        std::iota(shuffled_order.begin(), shuffled_order.end(), 0u);

        if (config.shuffle_buffer_size > 0 && n > 1) {
            // Reservoir-based approximate shuffle
            // For a true epoch-level shuffle when buffer >= n, this
            // degenerates to a full Fisher-Yates shuffle.
            apply_shuffle_buffer();
        }

        // Compute total batches
        total_batches = (n + static_cast<size_t>(config.batch_size) - 1)
                      / static_cast<size_t>(config.batch_size);
    }

    /**
     * Apply a streaming shuffle buffer.
     *
     * If shuffle_buffer_size >= n, this is a full Fisher-Yates shuffle.
     * Otherwise, it's a reservoir-based approximation that provides
     * good randomization with bounded memory.
     */
    void apply_shuffle_buffer() {
        const size_t n = shuffled_order.size();
        const size_t buf_size = std::min(
            static_cast<size_t>(config.shuffle_buffer_size), n);

        if (buf_size >= n) {
            // Full shuffle
            for (size_t i = n - 1; i > 0; --i) {
                std::uniform_int_distribution<size_t> dist(0, i);
                const size_t j = dist(rng_);
                std::swap(shuffled_order[i], shuffled_order[j]);
            }
            return;
        }

        // Streaming reservoir shuffle:
        // 1. Fill buffer with first buf_size elements
        // 2. For each subsequent element, pick random slot in buffer,
        //    emit the displaced element, insert new one
        std::vector<uint32_t> buffer(
            shuffled_order.begin(),
            shuffled_order.begin() + static_cast<ptrdiff_t>(buf_size));
        std::vector<uint32_t> result;
        result.reserve(n);

        std::uniform_int_distribution<size_t> dist(0, buf_size - 1);

        for (size_t i = buf_size; i < n; ++i) {
            const size_t j = dist(rng_);
            result.push_back(buffer[j]);
            buffer[j] = shuffled_order[i];
        }

        // Drain remaining buffer in random order
        for (size_t i = buffer.size(); i > 0; --i) {
            std::uniform_int_distribution<size_t> drain_dist(0, i - 1);
            const size_t j = drain_dist(rng_);
            result.push_back(buffer[j]);
            buffer[j] = buffer[i - 1];
        }

        shuffled_order = std::move(result);
    }


    // ── Worker threads ─────────────────────────────────────────────

    void start_workers() {
        stop_flag.store(false);
        work_done = false;

        // Start the dispatch thread (produces work items)
        dispatch_thread = std::thread([this] { dispatch_loop(); });

        // Start worker threads (consume work items, produce Batches)
        const int nw = config.num_workers;
        workers.reserve(nw);
        for (int i = 0; i < nw; ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }

        // Start aggregator thread
        aggregator_thread = std::thread([this] { aggregator_loop(); });
    }

    void shutdown() {
        stop_flag.store(true);

        // Wake up all waiting workers and aggregator
        work_cv.notify_all();
        pending_cv.notify_all();
        output_cv.notify_all();

        // Join dispatch thread
        if (dispatch_thread.joinable())
            dispatch_thread.join();

        // Join workers
        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }
        workers.clear();

        // Join aggregator thread
        if (aggregator_thread.joinable())
            aggregator_thread.join();
    }

    /**
     * Dispatch loop: runs in its own thread.
     *
     * Produces WorkItems (batch-sized groups of SampleAddresses)
     * from the shuffled order and posts them to the work queue.
     */
    void dispatch_loop() {
        const size_t n = shuffled_order.size();
        const size_t bs = static_cast<size_t>(config.batch_size);

        size_t pos = 0;
        while (pos < n && !stop_flag.load(std::memory_order_relaxed)) {
            const size_t end = std::min(pos + bs, n);

            WorkItem item;
            item.addresses.reserve(end - pos);
            for (size_t i = pos; i < end; ++i) {
                item.addresses.push_back(
                    all_addresses[shuffled_order[i]]);
            }

            // Post to work queue
            {
                std::lock_guard<std::mutex> lock(work_mutex);
                work_queue.push_back(std::move(item));
            }
            work_cv.notify_one();

            pos = end;
        }

        // Signal that all work has been dispatched
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            work_done = true;
        }
        work_cv.notify_all();
    }

    /**
     * Worker loop: runs in each worker thread.
     *
     * Pulls WorkItems from the work queue, parses samples from the
     * mmap'd shard data, collates into Batches, and pushes them
     * to the SPSC output queue.
     */
    void worker_loop() {
        while (!stop_flag.load(std::memory_order_relaxed)) {
            WorkItem item;

            // ── Pull next work item ────────────────────────────────
            {
                std::unique_lock<std::mutex> lock(work_mutex);
                work_cv.wait(lock, [this] {
                    return !work_queue.empty() || work_done
                        || stop_flag.load(std::memory_order_relaxed);
                });

                if (stop_flag.load(std::memory_order_relaxed))
                    return;

                if (work_queue.empty()) {
                    if (work_done) return;
                    continue;
                }

                item = std::move(work_queue.front());
                work_queue.erase(work_queue.begin());
            }

            // ── Parse samples ──────────────────────────────────────
            std::vector<Sample> samples;
            samples.reserve(item.addresses.size());

            for (const auto& addr : item.addresses) {
                if (stop_flag.load(std::memory_order_relaxed))
                    return;

                samples.push_back(read_sample_from_mmap(
                    mapped_files[addr.shard_id].data(),
                    mapped_files[addr.shard_id].size(),
                    shard_infos[addr.shard_id],
                    addr.local_index,
                    config.norm));
            }

            if (stop_flag.load(std::memory_order_relaxed))
                return;

            // ── Collate into Batch ─────────────────────────────────
            Batch batch;
            if (!samples.empty()) {
                const auto& hdr = shard_infos[item.addresses[0].shard_id].header;

                if (hdr.version >= FORMAT_VERSION_V2
                    && hdr.has_per_sample_dims())
                {
                    batch = Batch::collate_padded(
                        samples,
                        static_cast<int32_t>(hdr.image_channels),
                        static_cast<int32_t>(hdr.token_length));
                } else {
                    batch = Batch::from_samples(
                        samples,
                        static_cast<int32_t>(hdr.image_channels),
                        static_cast<int32_t>(hdr.image_height),
                        static_cast<int32_t>(hdr.image_width),
                        static_cast<int32_t>(hdr.token_length));
                }
            }

            // ── Push to pending batches ────────────────────────────
            {
                std::lock_guard<std::mutex> lock(pending_mutex);
                pending_batches.push_back(std::move(batch));
            }
            pending_cv.notify_one();
        }
    }

    /**
     * Aggregator loop: runs in its own thread.
     *
     * Drains pending_batches and pushes to the SPSC output queue.
     * This is the single producer for the SPSC queue, preventing races.
     */
    void aggregator_loop() {
        while (!stop_flag.load(std::memory_order_relaxed)) {
            std::vector<Batch> local_batches;
            {
                std::unique_lock<std::mutex> lock(pending_mutex);
                pending_cv.wait(lock, [this] {
                    return !pending_batches.empty() || stop_flag.load(std::memory_order_relaxed);
                });

                if (stop_flag.load(std::memory_order_relaxed) && pending_batches.empty())
                    return;

                local_batches = std::move(pending_batches);
                pending_batches.clear();
            }

            for (auto& batch : local_batches) {
                // Spin with yield until there's space (bounded by prefetch_depth)
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    if (output_queue->try_push(std::move(batch))) {
                        batches_produced.fetch_add(1, std::memory_order_release);
                        // Notify consumer
                        output_cv.notify_one();
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        }
    }


    // ── Consumer API ───────────────────────────────────────────────

    Batch next_batch() {
        if (cursor_ >= total_batches)
            throw std::runtime_error(
                "AsyncShardLoader: no more batches in this epoch");

        Batch batch;

        // Spin-wait with condition variable notification
        {
            std::unique_lock<std::mutex> lock(output_mutex);
            while (!output_queue->try_pop(batch)) {
                if (stop_flag.load(std::memory_order_relaxed))
                    throw std::runtime_error(
                        "AsyncShardLoader: shutdown during next()");

                // Wait with timeout to handle edge cases
                output_cv.wait_for(lock, std::chrono::milliseconds(1));
            }
        }

        ++cursor_;

        if (cursor_ >= total_batches)
            epoch_finished.store(true);

        return batch;
    }

    bool has_next() const {
        return cursor_ < total_batches;
    }

    void reset() {
        // 1. Shutdown existing workers
        shutdown();

        // 2. Drain any remaining items in the output queue
        Batch tmp;
        while (output_queue->try_pop(tmp)) {}

        // 3. Clear work queue and pending batches
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            work_queue.clear();
        }
        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending_batches.clear();
        }

        // 4. Advance epoch
        ++epoch_;

        // 5. Re-initialize shuffle and restart
        init_epoch();
        start_workers();
    }

    LoaderCheckpoint get_checkpoint() const {
        return {epoch_, config.seed};
    }

    void restore_checkpoint(const LoaderCheckpoint& cp) {
        // 1. Shutdown
        shutdown();

        // 2. Drain
        Batch tmp;
        while (output_queue->try_pop(tmp)) {}
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            work_queue.clear();
        }
        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending_batches.clear();
        }

        // 3. Restore state
        epoch_ = cp.epoch;
        config.seed = cp.seed;

        // 4. Re-initialize and start
        init_epoch();
        start_workers();
    }
};


// ═══════════════════════════════════════════════════════════════════
// AsyncShardLoader public methods (delegate to impl)
// ═══════════════════════════════════════════════════════════════════

AsyncShardLoader::AsyncShardLoader(AsyncLoaderConfig config)
    : impl_(std::make_unique<AsyncShardLoaderImpl>(std::move(config)))
{}

AsyncShardLoader::~AsyncShardLoader() = default;

Batch AsyncShardLoader::next() {
    return impl_->next_batch();
}

bool AsyncShardLoader::has_next() const {
    return impl_->has_next();
}

void AsyncShardLoader::reset() {
    impl_->reset();
}

LoaderCheckpoint AsyncShardLoader::checkpoint() const {
    return impl_->get_checkpoint();
}

void AsyncShardLoader::restore(const LoaderCheckpoint& cp) {
    impl_->restore_checkpoint(cp);
}

uint64_t AsyncShardLoader::total_samples() const {
    return static_cast<uint64_t>(impl_->all_addresses.size());
}

uint64_t AsyncShardLoader::epoch() const {
    return impl_->epoch_;
}

}  // namespace vlm
