# Benchmarks — opt-in via -DCHRONOSV_BUILD_BENCH=ON.
# google/benchmark via FetchContent (test-only style; not part of shipping lib).
include(FetchContent)

set(FETCHCONTENT_QUIET ON)

# We want google/benchmark WITHOUT its own tests / gtest dependency.
set(BENCHMARK_ENABLE_TESTING          OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL          OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS      OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS            OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(benchmark)

# google/benchmark builds with -Werror by default in some configs; disable so
# a compiler upgrade doesn't break our benches.
if(TARGET benchmark)
    target_compile_options(benchmark PRIVATE -Wno-error)
endif()

# ---- Bench executables -----------------------------------------------------

add_executable(bench_ring bench_ring.cpp)
target_link_libraries(bench_ring PRIVATE chronosv benchmark::benchmark)
target_include_directories(bench_ring PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(bench_ring PRIVATE -Wno-error)

add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE chronosv benchmark::benchmark)
target_include_directories(bench_kernels PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(bench_kernels SYSTEM PRIVATE ${EIGEN_INCLUDE_DIR})
target_compile_options(bench_kernels PRIVATE -Wno-error)

add_executable(bench_engine bench_engine.cpp)
target_link_libraries(bench_engine PRIVATE chronosv benchmark::benchmark)
target_compile_options(bench_engine PRIVATE -Wno-error)

# -------- Competitor comparison bench --------
#
# hnswlib fetched at CMake configure time. Header-only, small (~5 files),
# so it doesn't inflate build time meaningfully. Kept entirely inside
# bench/ — nothing about the shipping library depends on it.
FetchContent_Declare(
    hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG        v0.8.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(hnswlib)
if(NOT hnswlib_POPULATED)
    FetchContent_MakeAvailable(hnswlib)
endif()

add_executable(bench_compare bench_compare.cpp)
target_link_libraries(bench_compare PRIVATE chronosv benchmark::benchmark)
target_include_directories(bench_compare SYSTEM PRIVATE ${hnswlib_SOURCE_DIR})
target_compile_options(bench_compare PRIVATE -Wno-error)
