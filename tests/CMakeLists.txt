# Catch2 via FetchContent — keeps test-only dep out of the shipping artifact
# and out of the repo. First configure downloads ~2 MB; subsequent configures
# reuse the cached tree under build/_deps/.
include(FetchContent)

set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# ---- Test executables ------------------------------------------------------

add_executable(test_ring_buffer test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer PRIVATE chronosv Catch2::Catch2WithMain)
target_include_directories(test_ring_buffer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE chronosv Catch2::Catch2WithMain)
target_include_directories(test_kernels PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_kernels SYSTEM PRIVATE ${EIGEN_INCLUDE_DIR})

add_executable(test_engine test_engine.cpp)
target_link_libraries(test_engine PRIVATE chronosv Catch2::Catch2WithMain)

add_executable(test_storage_backend test_storage_backend.cpp)
target_link_libraries(test_storage_backend PRIVATE chronosv Catch2::Catch2WithMain)

add_executable(test_block_codec test_block_codec.cpp)
target_link_libraries(test_block_codec PRIVATE chronosv Catch2::Catch2WithMain)
target_include_directories(test_block_codec PRIVATE ${CMAKE_SOURCE_DIR}/src)

# INT8 kernel tests — built only when CHRONOSV_ENABLE_INT8=ON (the default).
# The engine gates INT8 with `#ifndef CHRONOSV_ENABLE_INT8` and returns
# CHRONOSV_ERR_UNSUPPORTED at runtime if a downstream build explicitly
# turns the flag off. These tests exercise the ON path.
# NOTE: catch_discover_tests() call is deferred to the discovery block below
# (after include(Catch)).
if(CHRONOSV_ENABLE_INT8)
    add_executable(test_kernels_int8 test_kernels_int8.cpp)
    target_link_libraries(test_kernels_int8 PRIVATE chronosv Catch2::Catch2WithMain)
    target_include_directories(test_kernels_int8 PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_options(test_kernels_int8 PRIVATE -Wno-error)
endif()

# Test drives RocksDB directly for corruption injection, so it needs both
# our internal header and the RocksDB include path.
add_executable(test_storage_rocksdb test_storage_rocksdb.cpp)
target_link_libraries(test_storage_rocksdb PRIVATE chronosv Catch2::Catch2WithMain RocksDB::rocksdb)
target_include_directories(test_storage_rocksdb PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Integration tests use RocksDB directly for corruption injection and take
# a few seconds each — hence they live in their own executable rather than
# sharing test_engine's process.
add_executable(test_phase2_integration test_phase2_integration.cpp)
target_link_libraries(test_phase2_integration PRIVATE chronosv Catch2::Catch2WithMain RocksDB::rocksdb)

# Suppress -Werror inside Catch2 headers by allowing a few known-noisy warnings
# in the test targets only. Catch2 v3.x is generally clean but toolchain
# specifics can trip -Wpedantic.
foreach(_t test_ring_buffer test_kernels test_engine test_storage_backend test_block_codec test_storage_rocksdb test_phase2_integration)
    target_compile_options(${_t} PRIVATE -Wno-error)
endforeach()

# ---- CTest registration ----------------------------------------------------

include(CTest)
include(Catch)
catch_discover_tests(test_ring_buffer)
catch_discover_tests(test_kernels)
catch_discover_tests(test_engine)
catch_discover_tests(test_storage_backend)
catch_discover_tests(test_block_codec)
catch_discover_tests(test_storage_rocksdb)
if(CHRONOSV_ENABLE_INT8)
    catch_discover_tests(test_kernels_int8)
endif()
# The RocksDB integration tests trip ASan's container-overflow detector
# and TSan on std::vector / thread operations inside RocksDB's own code —
# well-known false positives when sanitizer-instrumented code links
# against a non-instrumented library (see google/sanitizers wiki:
# AddressSanitizerContainerOverflow). We add these as three individual
# ctest entries (one per Catch2 test case) with the sanitizer opt-out
# environment applied. All other sanitizer checks (heap-buffer-overflow,
# use-after-free, leaks, most races) remain active.
set(_integ_env
    "ASAN_OPTIONS=detect_container_overflow=0"
    "TSAN_OPTIONS=ignore_noninstrumented_modules=1")
foreach(_name
    "Integration: N producers + eviction + cold-tier writes (5 s)"
    "Integration: concurrent chronosv_flush is idempotent (no deadlock)"
    "Integration: chronosv_open succeeds when a persisted block is corrupt")
    add_test(NAME "${_name}"
             COMMAND test_phase2_integration "${_name}"
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties("${_name}" PROPERTIES ENVIRONMENT "${_integ_env}")
endforeach()
