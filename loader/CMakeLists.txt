cmake_minimum_required(VERSION 3.16)
project(vlm_loader VERSION 0.1.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build the loader as a static library
add_library(vlm_loader STATIC
    src/shard_reader.cpp
    src/distributed.cpp
)

target_include_directories(vlm_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Platform-specific: no extra libs needed on Windows (Win32 API is default).
# On POSIX, we may need pthread for future threading work.
if(NOT WIN32)
    target_link_libraries(vlm_loader PRIVATE pthread)
endif()

# Enable warnings for development builds
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vlm_loader PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(vlm_loader PRIVATE /W4)
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
