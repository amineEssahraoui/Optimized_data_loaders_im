#ifndef VLM_LOADER_SPSC_QUEUE_H
#define VLM_LOADER_SPSC_QUEUE_H

/**
 * @file spsc_queue.h
 * @brief Bounded, lock-free, cache-line-aligned SPSC ring buffer.
 *
 * This queue is designed for a single-producer (thread pool dispatcher)
 * and single-consumer (the Python iteration thread calling `next()`).
 *
 * Key properties:
 *   - Wait-free try_push / try_pop (no locks, no CAS loops)
 *   - Cache-line padding between head and tail to prevent false sharing
 *   - Fixed capacity determined at construction (no heap allocations
 *     during push/pop)
 *   - Supports move-only types (e.g., Batch with large vectors)
 *
 * Usage:
 * @code
 *   vlm::SPSCQueue<vlm::Batch> q(4);  // capacity = 4 slots
 *   // Producer thread:
 *   while (!q.try_push(std::move(batch))) { // spin or yield }
 *   // Consumer thread:
 *   Batch b;
 *   while (!q.try_pop(b)) { // spin or yield }
 * @endcode
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

namespace vlm {

// Cache line size for alignment (64 bytes on x86/ARM)
#ifndef VLM_CACHE_LINE_SIZE
#define VLM_CACHE_LINE_SIZE 64
#endif

/**
 * @brief Bounded single-producer single-consumer lock-free queue.
 *
 * @tparam T  Element type (must be move-constructible).
 */
template <typename T>
class SPSCQueue {
public:
    /**
     * Construct with the given capacity.
     * @param capacity  Maximum number of elements.  Must be >= 1.
     */
    explicit SPSCQueue(size_t capacity)
        : capacity_(capacity)
        , buffer_(capacity)
    {}

    ~SPSCQueue() = default;

    // Non-copyable, non-movable (contains atomics)
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    /**
     * Try to enqueue an element (producer side).
     *
     * @param value  Value to move into the queue.
     * @return true if enqueued, false if the queue is full.
     */
    bool try_push(T&& value) noexcept {
        const size_t cur_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (cur_tail + 1) % capacity_;

        // Full check: if next_tail == head, queue is full
        if (next_tail == head_.load(std::memory_order_acquire))
            return false;

        buffer_[cur_tail] = std::move(value);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    /**
     * Try to dequeue an element (consumer side).
     *
     * @param[out] value  Receives the dequeued element via move.
     * @return true if dequeued, false if the queue is empty.
     */
    bool try_pop(T& value) noexcept {
        const size_t cur_head = head_.load(std::memory_order_relaxed);

        // Empty check: if head == tail, queue is empty
        if (cur_head == tail_.load(std::memory_order_acquire))
            return false;

        value = std::move(buffer_[cur_head]);
        head_.store((cur_head + 1) % capacity_, std::memory_order_release);
        return true;
    }

    /**
     * Check if the queue is empty (approximate, for monitoring only).
     */
    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire)
            == tail_.load(std::memory_order_acquire);
    }

    /**
     * Approximate number of elements in the queue.
     */
    size_t size_approx() const noexcept {
        const size_t h = head_.load(std::memory_order_acquire);
        const size_t t = tail_.load(std::memory_order_acquire);
        return (t >= h) ? (t - h) : (capacity_ - h + t);
    }

    /**
     * Maximum capacity.
     * Note: usable capacity is capacity_ - 1 (one slot is sentinel).
     */
    size_t capacity() const noexcept { return capacity_; }

private:
    const size_t capacity_;

    // Cache-line aligned head and tail to prevent false sharing.
    // head_ is only written by the consumer; tail_ only by the producer.
    alignas(VLM_CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(VLM_CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};

    // Slot storage
    std::vector<T> buffer_;
};

}  // namespace vlm

#endif  // VLM_LOADER_SPSC_QUEUE_H
