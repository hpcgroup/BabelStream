# Identical build flags to the native `omp` model — only difference is the
# `.cpp` source we compile. Guarantees compiler/runtime parity for fair perf.
include(${CMAKE_CURRENT_SOURCE_DIR}/src/omp/model.cmake)

# clacc-emitted source uses the C `restrict` keyword which C++ rejects.
register_definitions(restrict=__restrict)
