# INT8 recall validation harness — opt-in.
#
# Enable with -DCHRONOSV_BUILD_INT8_RECALL=ON. Requires CHRONOSV_ENABLE_INT8=ON.
# Not wired into ctest — needs external data (see README.md); run manually
# after preparing a dataset via prepare_sift1m.sh or prepare_bert.py.

if(NOT CHRONOSV_BUILD_INT8_RECALL)
    return()
endif()

if(NOT CHRONOSV_ENABLE_INT8)
    message(FATAL_ERROR
        "CHRONOSV_BUILD_INT8_RECALL=ON requires CHRONOSV_ENABLE_INT8=ON")
endif()

add_executable(test_int8_recall test_int8_recall.cpp)
target_link_libraries(test_int8_recall PRIVATE chronosv)
# Not in ctest — this is a manual perf/recall harness, not a pass/fail unit test.
