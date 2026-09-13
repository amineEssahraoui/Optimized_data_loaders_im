#ifndef VLM_LOADER_GPU_TRANSFER_H
#define VLM_LOADER_GPU_TRANSFER_H

/**
 * @file gpu_transfer.h
 * @brief Conditional GPU transfer module with pinned memory, async
 *        cudaMemcpy, and DLPack v0.8 export.
 *
 * Compiled only when VLM_HAS_CUDA is defined (set by CMake when a
 * CUDA toolkit is found).  When CUDA is not available, provides a
 * CPU-fallback struct that holds batch pointers without any transfer.
 *
 * Key design decisions:
 *   - Accepts user-provided CUDA streams (void* for ABI compatibility)
 *     to integrate seamlessly with PyTorch's stream management
 *   - Uses cudaHostAlloc for pinned staging buffers (optimal DMA)
 *   - Exports device tensors via DLPack v0.8 for zero-copy handoff
 *     to PyTorch (torch.from_dlpack)
 *
 * Usage (with CUDA):
 * @code
 *   vlm::gpu::PinnedBuffer staging(required_bytes);
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *
 *   auto gb = vlm::gpu::transfer_batch(batch, staging,
 *                                       static_cast<void*>(stream), 0);
 *   // ... use gb.device_image etc.
 *   vlm::gpu::free_gpu_batch(gb);
 * @endcode
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "batch.h"

// ═══════════════════════════════════════════════════════════════════
// DLPack v0.8 types (inlined to avoid external dependency)
// ═══════════════════════════════════════════════════════════════════

#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#ifdef __cplusplus
extern "C" {
#endif

/** DLPack version. */
#define DLPACK_MAJOR_VERSION 0
#define DLPACK_MINOR_VERSION 8

/** Device type codes. */
typedef enum {
    kDLCPU          = 1,
    kDLCUDA         = 2,
    kDLCUDAHost     = 3,
    kDLCUDAManaged  = 13,
} DLDeviceType;

/** Device descriptor. */
typedef struct {
    DLDeviceType device_type;
    int32_t      device_id;
} DLDevice;

/** Data type descriptor. */
typedef enum : uint8_t {
    kDLInt   = 0U,
    kDLUInt  = 1U,
    kDLFloat = 2U,
} DLDataTypeCode;

typedef struct {
    uint8_t  code;    // DLDataTypeCode
    uint8_t  bits;    // number of bits
    uint16_t lanes;   // number of lanes (1 for scalar)
} DLDataType;

/** N-dimensional tensor. */
typedef struct {
    void*      data;
    DLDevice   device;
    int32_t    ndim;
    DLDataType dtype;
    int64_t*   shape;
    int64_t*   strides;  // can be NULL for contiguous
    uint64_t   byte_offset;
} DLTensor;

/** Managed tensor with destructor callback (DLPack v0.8). */
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void*    manager_ctx;
    void     (*deleter)(struct DLManagedTensor*);
} DLManagedTensor;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DLPACK_DLPACK_H_


namespace vlm {
namespace gpu {

// ═══════════════════════════════════════════════════════════════════
// GPUBatch — holds device-side pointers for a transferred batch
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Container for GPU-resident batch data.
 *
 * When CUDA is available, holds device pointers allocated via
 * cudaMalloc.  When CUDA is not available, holds nullptr (no-op).
 */
struct GPUBatch {
    void* device_image  = nullptr;  ///< float* [N,C,H,W] on device
    void* device_q_ids  = nullptr;  ///< int32* [N,T] on device
    void* device_q_mask = nullptr;  ///< int32* [N,T] on device
    void* device_a_ids  = nullptr;  ///< int32* [N,T] on device
    void* device_a_mask = nullptr;  ///< int32* [N,T] on device

    int32_t batch_size = 0;
    int32_t C = 0, H = 0, W = 0;   ///< Image dimensions
    int32_t T = 0;                  ///< Token length

    int32_t device_id = 0;          ///< CUDA device ordinal

    bool on_device = false;         ///< true if pointers are device-side
};


// ═══════════════════════════════════════════════════════════════════
// CUDA path (compiled only with VLM_HAS_CUDA)
// ═══════════════════════════════════════════════════════════════════

#ifdef VLM_HAS_CUDA

#include <cuda_runtime.h>

/**
 * @brief RAII pinned (page-locked) host memory buffer.
 *
 * Uses cudaHostAlloc for optimal host→device DMA transfer bandwidth.
 * Non-copyable, movable.
 */
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(size_t bytes) : size_(bytes) {
        if (bytes == 0) return;
        cudaError_t err = cudaHostAlloc(&ptr_, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess)
            throw std::runtime_error(
                "cudaHostAlloc failed: "
                + std::string(cudaGetErrorString(err)));
    }

    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    // Non-copyable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Movable
    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_  = nullptr;
        o.size_ = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_  = o.ptr_;
            size_ = o.size_;
            o.ptr_  = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    void* data()         noexcept { return ptr_; }
    const void* data()   const noexcept { return ptr_; }
    size_t size()        const noexcept { return size_; }

    /**
     * Ensure the buffer is at least `required` bytes.
     * Reallocates (with cudaFreeHost + cudaHostAlloc) if too small.
     */
    void ensure(size_t required) {
        if (required <= size_) return;
        if (ptr_) cudaFreeHost(ptr_);
        ptr_ = nullptr;
        size_ = 0;
        cudaError_t err = cudaHostAlloc(&ptr_, required, cudaHostAllocDefault);
        if (err != cudaSuccess)
            throw std::runtime_error(
                "cudaHostAlloc failed on resize: "
                + std::string(cudaGetErrorString(err)));
        size_ = required;
    }

private:
    void*  ptr_  = nullptr;
    size_t size_ = 0;
};


// ── Helper: check CUDA errors ──────────────────────────────────────
namespace detail {
inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
}
}  // namespace detail


/**
 * Transfer a CPU Batch to GPU memory via a pinned staging buffer.
 *
 * @param batch      The CPU-side batch to transfer.
 * @param staging    Pinned buffer used as staging area (resized if needed).
 * @param stream     User-provided cudaStream_t (as void* for ABI compat).
 *                   Pass nullptr for the default stream.
 * @param device_id  CUDA device ordinal (default 0).
 * @return GPUBatch with device pointers (caller must free_gpu_batch).
 */
inline GPUBatch transfer_batch(
    const Batch& batch,
    PinnedBuffer& staging,
    void* stream = nullptr,
    int device_id = 0)
{
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    detail::cuda_check(
        cudaSetDevice(device_id), "cudaSetDevice");

    GPUBatch gb;
    gb.batch_size = batch.batch_size;
    gb.C = batch.image_channels;
    gb.H = batch.image_height;
    gb.W = batch.image_width;
    gb.T = batch.token_length;
    gb.device_id = device_id;
    gb.on_device = true;

    const size_t img_bytes =
        static_cast<size_t>(batch.batch_size)
        * batch.image_channels * batch.image_height * batch.image_width
        * sizeof(float);
    const size_t tok_bytes =
        static_cast<size_t>(batch.batch_size)
        * batch.token_length * sizeof(int32_t);

    // Allocate device memory for all tensors
    detail::cuda_check(
        cudaMalloc(&gb.device_image, img_bytes), "cudaMalloc image");
    detail::cuda_check(
        cudaMalloc(&gb.device_q_ids, tok_bytes), "cudaMalloc q_ids");
    detail::cuda_check(
        cudaMalloc(&gb.device_q_mask, tok_bytes), "cudaMalloc q_mask");
    detail::cuda_check(
        cudaMalloc(&gb.device_a_ids, tok_bytes), "cudaMalloc a_ids");
    detail::cuda_check(
        cudaMalloc(&gb.device_a_mask, tok_bytes), "cudaMalloc a_mask");

    // Ensure staging buffer is large enough for the largest tensor
    const size_t max_bytes = std::max(img_bytes, tok_bytes);
    staging.ensure(max_bytes);

    // ── Transfer image data ────────────────────────────────────────
    std::memcpy(staging.data(), batch.image_data.data(), img_bytes);
    detail::cuda_check(
        cudaMemcpyAsync(gb.device_image, staging.data(), img_bytes,
                        cudaMemcpyHostToDevice, cuda_stream),
        "cudaMemcpyAsync image");

    // Sync before reusing the staging buffer for the next tensor
    detail::cuda_check(
        cudaStreamSynchronize(cuda_stream), "stream sync after image");

    // ── Transfer token tensors ─────────────────────────────────────
    auto transfer_tokens = [&](const std::vector<int32_t>& src, void* dst) {
        std::memcpy(staging.data(), src.data(), tok_bytes);
        detail::cuda_check(
            cudaMemcpyAsync(dst, staging.data(), tok_bytes,
                            cudaMemcpyHostToDevice, cuda_stream),
            "cudaMemcpyAsync tokens");
        detail::cuda_check(
            cudaStreamSynchronize(cuda_stream), "stream sync after tokens");
    };

    transfer_tokens(batch.question_ids,  gb.device_q_ids);
    transfer_tokens(batch.question_mask, gb.device_q_mask);
    transfer_tokens(batch.answer_ids,    gb.device_a_ids);
    transfer_tokens(batch.answer_mask,   gb.device_a_mask);

    return gb;
}


/**
 * Free all device memory held by a GPUBatch.
 */
inline void free_gpu_batch(GPUBatch& gb) {
    if (!gb.on_device) return;
    cudaSetDevice(gb.device_id);
    if (gb.device_image)  cudaFree(gb.device_image);
    if (gb.device_q_ids)  cudaFree(gb.device_q_ids);
    if (gb.device_q_mask) cudaFree(gb.device_q_mask);
    if (gb.device_a_ids)  cudaFree(gb.device_a_ids);
    if (gb.device_a_mask) cudaFree(gb.device_a_mask);
    gb = GPUBatch{};
}


// ── DLPack export ──────────────────────────────────────────────────

namespace dlpack_detail {

/**
 * Context for DLPack managed tensor destruction.
 * Stores the shape/strides arrays and the device pointer ownership info.
 */
struct DLPackContext {
    int64_t shape[5];    // max 5D: [N, C, H, W] or [N, T]
    int64_t strides[5];
    int     device_id;
    bool    owns_data;   // if true, cudaFree on delete
};

inline void dl_deleter(DLManagedTensor* self) {
    if (self) {
        auto* ctx = static_cast<DLPackContext*>(self->manager_ctx);
        if (ctx && ctx->owns_data && self->dl_tensor.data) {
            cudaSetDevice(ctx->device_id);
            cudaFree(self->dl_tensor.data);
        }
        delete ctx;
        delete self;
    }
}

}  // namespace dlpack_detail

/**
 * Export a device float pointer as a DLPack managed tensor.
 *
 * @param data       Device pointer.
 * @param shape      Tensor shape (e.g., {N, C, H, W}).
 * @param ndim       Number of dimensions.
 * @param device_id  CUDA device ordinal.
 * @param owns       If true, the DLPack deleter will cudaFree the pointer.
 * @return Heap-allocated DLManagedTensor (caller transfers ownership).
 */
inline DLManagedTensor* make_dlpack_float(
    void* data,
    const int64_t* shape,
    int32_t ndim,
    int device_id,
    bool owns = false)
{
    auto* ctx = new dlpack_detail::DLPackContext{};
    for (int i = 0; i < ndim; ++i) ctx->shape[i] = shape[i];
    // Row-major strides
    ctx->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        ctx->strides[i] = ctx->strides[i + 1] * ctx->shape[i + 1];
    ctx->device_id = device_id;
    ctx->owns_data = owns;

    auto* mt = new DLManagedTensor{};
    mt->dl_tensor.data = data;
    mt->dl_tensor.device = {kDLCUDA, device_id};
    mt->dl_tensor.ndim = ndim;
    mt->dl_tensor.dtype = {kDLFloat, 32, 1};
    mt->dl_tensor.shape = ctx->shape;
    mt->dl_tensor.strides = ctx->strides;
    mt->dl_tensor.byte_offset = 0;
    mt->manager_ctx = ctx;
    mt->deleter = dlpack_detail::dl_deleter;

    return mt;
}

/**
 * Export a device int32 pointer as a DLPack managed tensor.
 */
inline DLManagedTensor* make_dlpack_int32(
    void* data,
    const int64_t* shape,
    int32_t ndim,
    int device_id,
    bool owns = false)
{
    auto* ctx = new dlpack_detail::DLPackContext{};
    for (int i = 0; i < ndim; ++i) ctx->shape[i] = shape[i];
    ctx->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        ctx->strides[i] = ctx->strides[i + 1] * ctx->shape[i + 1];
    ctx->device_id = device_id;
    ctx->owns_data = owns;

    auto* mt = new DLManagedTensor{};
    mt->dl_tensor.data = data;
    mt->dl_tensor.device = {kDLCUDA, device_id};
    mt->dl_tensor.ndim = ndim;
    mt->dl_tensor.dtype = {kDLInt, 32, 1};
    mt->dl_tensor.shape = ctx->shape;
    mt->dl_tensor.strides = ctx->strides;
    mt->dl_tensor.byte_offset = 0;
    mt->manager_ctx = ctx;
    mt->deleter = dlpack_detail::dl_deleter;

    return mt;
}

#else  // !VLM_HAS_CUDA

// ═══════════════════════════════════════════════════════════════════
// CPU-fallback stubs (no CUDA available)
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief CPU-fallback pinned buffer (just a regular heap allocation).
 */
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t bytes) : size_(bytes) {
        if (bytes > 0) ptr_ = new uint8_t[bytes];
    }
    ~PinnedBuffer() { delete[] static_cast<uint8_t*>(ptr_); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; o.size_ = 0; }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            delete[] static_cast<uint8_t*>(ptr_);
            ptr_ = o.ptr_; size_ = o.size_;
            o.ptr_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    void*  data()  noexcept { return ptr_; }
    size_t size()  const noexcept { return size_; }
    void ensure(size_t required) {
        if (required <= size_) return;
        delete[] static_cast<uint8_t*>(ptr_);
        ptr_ = new uint8_t[required];
        size_ = required;
    }

private:
    void*  ptr_  = nullptr;
    size_t size_ = 0;
};

/** CPU fallback: no actual GPU transfer. */
inline GPUBatch transfer_batch(
    const Batch& /*batch*/,
    PinnedBuffer& /*staging*/,
    void* /*stream*/ = nullptr,
    int /*device_id*/ = 0)
{
    GPUBatch gb;
    gb.on_device = false;
    return gb;
}

/** CPU fallback: no-op. */
inline void free_gpu_batch(GPUBatch& gb) {
    gb = GPUBatch{};
}

#endif  // VLM_HAS_CUDA

}  // namespace gpu
}  // namespace vlm

#endif  // VLM_LOADER_GPU_TRANSFER_H
