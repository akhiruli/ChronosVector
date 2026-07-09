cmake_minimum_required(VERSION 3.25)
project(chronosv
    VERSION 0.1.0
    DESCRIPTION "Embeddable C++23 sliding-window vector store for streaming time-series embeddings"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Options ---------------------------------------------------------------

option(CHRONOSV_BUILD_TESTS         "Build unit + integration tests"                    ON)
option(CHRONOSV_BUILD_BENCH         "Build benchmarks"                                  OFF)
option(CHRONOSV_BUILD_EXAMPLES      "Build examples"                                    OFF)
option(CHRONOSV_BUILD_SOAK          "Build long-running soak test"                      OFF)
option(CHRONOSV_BUILD_INT8_RECALL   "Build INT8 recall harness (needs external data)"  OFF)
option(CHRONOSV_ENABLE_INT8         "Enable INT8 quantized storage path"                ON)
option(CHRONOSV_ENABLE_ASAN    "AddressSanitizer"               OFF)
option(CHRONOSV_ENABLE_TSAN    "ThreadSanitizer"                OFF)
option(CHRONOSV_ENABLE_UBSAN   "UBSanitizer"                    OFF)

# Sanitizers are mutually exclusive at build time.
set(_san_count 0)
foreach(_san CHRONOSV_ENABLE_ASAN CHRONOSV_ENABLE_TSAN CHRONOSV_ENABLE_UBSAN)
    if(${_san})
        math(EXPR _san_count "${_san_count} + 1")
    endif()
endforeach()
if(_san_count GREATER 1)
    message(FATAL_ERROR "Enable at most one of ASAN / TSAN / UBSAN")
endif()

# ---- Dependencies ----------------------------------------------------------

# Eigen: vendored as a git submodule at third_party/eigen, pinned to a stable tag.
# Header-only, so we consume it via its include path directly (no add_subdirectory
# needed and no Eigen CMake config required).
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen"
    CACHE PATH "Path to vendored Eigen headers")
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR
        "Eigen not found at ${EIGEN_INCLUDE_DIR}. Did you `git submodule update --init`?")
endif()

# RocksDB: required system install (brew install rocksdb / apt install librocksdb-dev),
# linked dynamically. Not vendored. Used by src/storage_rocksdb.cpp for cold-tier
#
# Discovery: two-tier because Ubuntu's librocksdb-dev historically ships pkg-config
# .pc files but not RocksDBConfig.cmake, while Homebrew ships both. Try find_package
# first (works on macOS + newer Linux distros), fall back to PkgConfig if needed.
find_package(RocksDB QUIET CONFIG)
if(NOT RocksDB_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ROCKSDB REQUIRED IMPORTED_TARGET rocksdb)
    # ALIAS can't point at IMPORTED targets on older CMake; use INTERFACE wrapper.
    if(NOT TARGET RocksDB::rocksdb)
        add_library(RocksDB::rocksdb INTERFACE IMPORTED)
        target_link_libraries(RocksDB::rocksdb INTERFACE PkgConfig::ROCKSDB)
    endif()
    message(STATUS "RocksDB: using PkgConfig (${ROCKSDB_VERSION})")
else()
    message(STATUS "RocksDB: using find_package CONFIG mode")
endif()

# ---- Library ---------------------------------------------------------------

# Phase 1: header-heavy library. src/ will grow in Phase 2 (engine.cpp, eviction.cpp,
# storage_rocksdb.cpp). For now we compile only what exists.
add_library(chronosv
    src/kernels.cpp
    src/engine.cpp
    src/block_codec.cpp
    src/storage_rocksdb.cpp
)
add_library(chronosv::chronosv ALIAS chronosv)

target_include_directories(chronosv
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# Treat Eigen as a SYSTEM include so its C++23-deprecation warnings (Eigen 3.4.0
# still uses std::float_denorm_style / std::denorm_absent) don't trip -Werror.
target_include_directories(chronosv SYSTEM PRIVATE ${EIGEN_INCLUDE_DIR})

# RocksDB is a PRIVATE dependency — its headers appear only in src/storage_rocksdb.cpp,
# not in any public header. Callers of libchronosv don't need to know about it.
target_link_libraries(chronosv PRIVATE RocksDB::rocksdb)

target_compile_options(chronosv PRIVATE
    -Wall -Wextra -Wpedantic -Werror
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)

if(CHRONOSV_ENABLE_INT8)
    target_compile_definitions(chronosv PUBLIC CHRONOSV_ENABLE_INT8=1)
endif()

# Sanitizer flags (applied to library and any dependent target that inherits).
set(_san_flags "")
if(CHRONOSV_ENABLE_ASAN)
    list(APPEND _san_flags -fsanitize=address -fno-omit-frame-pointer)
elseif(CHRONOSV_ENABLE_TSAN)
    list(APPEND _san_flags -fsanitize=thread -fno-omit-frame-pointer)
elseif(CHRONOSV_ENABLE_UBSAN)
    list(APPEND _san_flags -fsanitize=undefined -fno-omit-frame-pointer)
endif()
if(_san_flags)
    target_compile_options(chronosv PUBLIC ${_san_flags})
    target_link_options(chronosv PUBLIC ${_san_flags})
endif()

# ---- Subdirectories --------------------------------------------------------

if(CHRONOSV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CHRONOSV_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(CHRONOSV_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(CHRONOSV_BUILD_SOAK)
    add_subdirectory(soak)
endif()

if(CHRONOSV_BUILD_INT8_RECALL)
    add_subdirectory(tests/int8_recall)
endif()

# ---- Install ---------------------------------------------------------------

include(GNUInstallDirs)
install(TARGETS chronosv
    EXPORT chronosv-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/chronosv
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(EXPORT chronosv-targets
    FILE chronosv-config.cmake
    NAMESPACE chronosv::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chronosv
)

message(STATUS "chronosv ${PROJECT_VERSION} configured "
               "(tests=${CHRONOSV_BUILD_TESTS}, "
               "int8=${CHRONOSV_ENABLE_INT8}, "
               "san=${_san_flags})")
