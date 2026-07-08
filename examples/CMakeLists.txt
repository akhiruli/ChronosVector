# Examples — opt-in via -DCHRONOSV_BUILD_EXAMPLES=ON.

add_executable(chronosv_cli chronosv_cli.cpp)
target_link_libraries(chronosv_cli PRIVATE chronosv)
target_compile_options(chronosv_cli PRIVATE -Wno-error)

# Walkthrough: streaming anomaly detection + metrics sink. Runs standalone,
# no RocksDB required (cold_path is NULL). See the file header for what it
# demonstrates.
add_executable(anomaly_stream anomaly_stream.cpp)
target_link_libraries(anomaly_stream PRIVATE chronosv)
target_compile_options(anomaly_stream PRIVATE -Wno-error)
