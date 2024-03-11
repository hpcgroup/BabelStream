register_flag_optional(RAJA_BACK_END "Specify whether we target CPU/CUDA/HIP/SYCL" "CPU")

register_flag_optional(MANAGED_ALLOC "Use UM instead of device-only allocation" "OFF")

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
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -extended-lambda --expt-relaxed-constexpr --restrict --keep")

        register_definitions(RAJA_TARGET_GPU)
    elseif (${RAJA_BACK_END} STREQUAL "HIP")
        find_package(hip REQUIRED)

        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 14)
        set(CMAKE_HIP_SEPARABLE_COMPILATION ON)

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)

        register_definitions(RAJA_TARGET_GPU)
    elseif (${RAJA_BACK_END} STREQUAL "SYCL")
        register_definitions(RAJA_TARGET_GPU)
    else()
        register_definitions(RAJA_TARGET_CPU)
        message(STATUS "Falling Back to CPU")
    endif ()

    if (MANAGED_ALLOC)
        register_definitions(BABELSTREAM_MANAGED_ALLOC)
    endif ()
endmacro()
