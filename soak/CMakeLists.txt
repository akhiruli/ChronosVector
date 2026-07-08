# Soak test — opt-in via -DCHRONOSV_BUILD_SOAK=ON.
# Not run by ctest; invoked manually for smoke (~10 min) or full (24 h) runs.

add_executable(chronosv_soak soak_test.cpp)
target_link_libraries(chronosv_soak PRIVATE chronosv)
target_compile_options(chronosv_soak PRIVATE -Wno-error)
