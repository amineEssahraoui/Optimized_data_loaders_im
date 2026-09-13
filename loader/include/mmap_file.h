#ifndef VLM_LOADER_MMAP_FILE_H
#define VLM_LOADER_MMAP_FILE_H

/**
 * @file mmap_file.h
 * @brief Cross-platform memory-mapped file wrapper (RAII, header-only).
 *
 * Provides zero-copy access to shard files by mapping them directly
 * into the process address space.  On POSIX systems uses mmap(2);
 * on Windows uses CreateFileMapping / MapViewOfFile.
 *
 * The mapped region is read-only and private (MAP_PRIVATE on POSIX,
 * FILE_MAP_READ on Windows).  MAP_POPULATE is used on Linux for
 * kernel readahead of the entire file.
 *
 * Usage:
 * @code
 *   vlm::MappedFile mf("/data/shard_000.bin");
 *   auto span = mf.view(offset, length);
 *   // span.data() points directly into the mapped page
 * @endcode
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

// ── Platform detection ─────────────────────────────────────────────
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace vlm {

/**
 * @brief Lightweight read-only span over a contiguous byte region.
 *
 * Does NOT own the memory.  Valid only while the parent MappedFile
 * is alive.  Intentionally minimal to avoid C++20 std::span dependency.
 */
struct ByteSpan {
    const uint8_t* data_ptr = nullptr;
    size_t         length   = 0;

    const uint8_t* data()  const noexcept { return data_ptr; }
    size_t         size()  const noexcept { return length; }
    bool           empty() const noexcept { return length == 0; }

    const uint8_t& operator[](size_t i) const noexcept {
        return data_ptr[i];
    }
};


/**
 * @brief RAII memory-mapped file.
 *
 * Non-copyable, movable.  Maps the entire file on construction
 * and unmaps on destruction.
 */
class MappedFile {
public:
    MappedFile() = default;

    /**
     * Open and memory-map a file.
     * @param path  Filesystem path to the file.
     * @throws std::runtime_error on open / mapping failure.
     */
    explicit MappedFile(const std::string& path) {
        open(path);
    }

    ~MappedFile() {
        close();
    }

    // Non-copyable
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // Movable
    MappedFile(MappedFile&& other) noexcept { swap(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            swap(other);
        }
        return *this;
    }

    /** Pointer to the start of the mapped region. */
    const uint8_t* data() const noexcept { return data_; }

    /** Total size of the mapped file in bytes. */
    size_t size() const noexcept { return size_; }

    /** Whether a file is currently mapped. */
    bool is_open() const noexcept { return data_ != nullptr; }

    /**
     * Return a ByteSpan over [offset, offset+length).
     * @throws std::out_of_range if the range exceeds the file size.
     */
    ByteSpan view(size_t offset, size_t length) const {
        if (offset + length > size_)
            throw std::out_of_range(
                "MappedFile::view: offset=" + std::to_string(offset)
                + " length=" + std::to_string(length)
                + " exceeds file size=" + std::to_string(size_));
        return {data_ + offset, length};
    }

    /** Return a span over the entire file. */
    ByteSpan view_all() const noexcept {
        return {data_, size_};
    }

private:
    void swap(MappedFile& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
#ifdef _WIN32
        std::swap(file_handle_, other.file_handle_);
        std::swap(mapping_handle_, other.mapping_handle_);
#else
        std::swap(fd_, other.fd_);
#endif
    }

    // ── Platform-specific implementation ───────────────────────────

#ifdef _WIN32
    // ── Windows implementation ─────────────────────────────────────
    void open(const std::string& path) {
        file_handle_ = ::CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);

        if (file_handle_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error(
                "MappedFile: failed to open file: " + path);

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(file_handle_, &file_size)) {
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error(
                "MappedFile: failed to get file size: " + path);
        }
        size_ = static_cast<size_t>(file_size.QuadPart);

        if (size_ == 0) {
            // Empty files can't be mapped; leave data_ null, size_ 0
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return;
        }

        mapping_handle_ = ::CreateFileMappingA(
            file_handle_,
            nullptr,
            PAGE_READONLY,
            static_cast<DWORD>(size_ >> 32),
            static_cast<DWORD>(size_ & 0xFFFFFFFF),
            nullptr);

        if (mapping_handle_ == nullptr) {
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error(
                "MappedFile: CreateFileMapping failed: " + path);
        }

        void* view = ::MapViewOfFile(
            mapping_handle_,
            FILE_MAP_READ,
            0, 0, 0);  // map entire file

        if (view == nullptr) {
            ::CloseHandle(mapping_handle_);
            ::CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error(
                "MappedFile: MapViewOfFile failed: " + path);
        }

        data_ = static_cast<const uint8_t*>(view);
    }

    void close() noexcept {
        if (data_) {
            ::UnmapViewOfFile(static_cast<LPCVOID>(data_));
            data_ = nullptr;
        }
        if (mapping_handle_) {
            ::CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
        size_ = 0;
    }

    const uint8_t* data_   = nullptr;
    size_t         size_   = 0;
    HANDLE file_handle_    = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;

#else
    // ── POSIX implementation ───────────────────────────────────────
    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error(
                "MappedFile: failed to open file: " + path);

        struct stat st {};
        if (::fstat(fd_, &st) < 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "MappedFile: fstat failed: " + path);
        }
        size_ = static_cast<size_t>(st.st_size);

        if (size_ == 0) {
            ::close(fd_);
            fd_ = -1;
            return;
        }

        int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
        flags |= MAP_POPULATE;  // Linux: trigger readahead
#endif

        void* addr = ::mmap(nullptr, size_, PROT_READ, flags, fd_, 0);
        if (addr == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "MappedFile: mmap failed: " + path);
        }

        data_ = static_cast<const uint8_t*>(addr);

        // Advise sequential access for optimal readahead
#ifdef MADV_SEQUENTIAL
        ::madvise(const_cast<void*>(static_cast<const void*>(data_)),
                  size_, MADV_SEQUENTIAL);
#endif
    }

    void close() noexcept {
        if (data_) {
            ::munmap(const_cast<void*>(
                static_cast<const void*>(data_)), size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
    }

    const uint8_t* data_ = nullptr;
    size_t         size_ = 0;
    int            fd_   = -1;
#endif  // _WIN32
};

}  // namespace vlm

#endif  // VLM_LOADER_MMAP_FILE_H
