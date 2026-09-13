cmake_minimum_required(VERSION 3.16)
project(vlm_loader VERSION 0.2.0 LANGUAGES C CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Threading (required for async loader) ─────────────────────────────────
find_package(Threads REQUIRED)

# ── Optional CUDA detection ───────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    set(VLM_HAS_CUDA ON)
    message(STATUS "CUDA found: enabling GPU transfer module")
else()
    set(VLM_HAS_CUDA OFF)
    message(STATUS "CUDA not found: GPU transfer disabled (CPU fallback)")
endif()

# ── Build the loader as a static library ──────────────────────────────────
add_library(vlm_loader STATIC
    src/shard_reader.cpp
    src/distributed.cpp
    src/async_loader.cpp
)

target_include_directories(vlm_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Vendored LZ4 (decompression only) ────────────────────────────────────
add_library(vlm_lz4 STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4/lz4.c
)
target_include_directories(vlm_lz4 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4
)
set_target_properties(vlm_lz4 PROPERTIES
    C_STANDARD 99
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(vlm_loader PRIVATE vlm_lz4)
target_include_directories(vlm_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4
)

# ── Link threading library ────────────────────────────────────────────────
target_link_libraries(vlm_loader PUBLIC Threads::Threads)

# ── CUDA configuration ───────────────────────────────────────────────────
if(VLM_HAS_CUDA)
    target_compile_definitions(vlm_loader PUBLIC VLM_HAS_CUDA=1)
    if(TARGET CUDA::cudart)
        target_link_libraries(vlm_loader PUBLIC CUDA::cudart)
    endif()
endif()

# Enable warnings for development builds
# Also enable SIMD intrinsics for deferred normalization (v2 path)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vlm_loader PRIVATE -Wall -Wextra -Wpedantic)
    # Enable AVX2 + FMA for vectorised uint8->float32 normalization
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    if(HAS_AVX2)
        target_compile_options(vlm_loader PRIVATE -mavx2 -mfma)
        target_compile_definitions(vlm_loader PRIVATE VLM_HAS_AVX2=1)
    endif()
elseif(MSVC)
    target_compile_options(vlm_loader PRIVATE /W4)
    # MSVC: AVX2 for vectorised normalization
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/arch:AVX2" MSVC_HAS_AVX2)
    if(MSVC_HAS_AVX2)
        target_compile_options(vlm_loader PRIVATE /arch:AVX2)
        target_compile_definitions(vlm_loader PRIVATE VLM_HAS_AVX2=1)
    endif()
endif()

# Optional: build pybind11 bindings (if pybind11 is found)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    # Try to find pybind11 via CMake (pip install pybind11 provides this)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found, building Python bindings")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../bindings
                         ${CMAKE_CURRENT_BINARY_DIR}/bindings)
    else()
        message(WARNING "pybind11 not found. Python bindings will not be built. "
                        "Install with: pip install pybind11")
    endif()
endif()
