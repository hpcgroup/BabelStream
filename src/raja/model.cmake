register_flag_required(RAJA_BACK_END "Specify whether we target CPU/CUDA/HIP/SYCL")

macro(setup)
    if (POLICY CMP0104)
        cmake_policy(SET CMP0104 OLD)
    endif ()

    set(CMAKE_CXX_STANDARD 14)

    find_package(RAJA REQUIRED)
    find_package(umpire REQUIRED)

    register_link_library(RAJA umpire)
    if (${RAJA_BACK_END} STREQUAL "CUDA")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -forward-unknown-to-host-compiler -arch=sm_80 -extended-lambda --expt-relaxed-constexpr -restrict -keep")
    endif ()
endmacro()
