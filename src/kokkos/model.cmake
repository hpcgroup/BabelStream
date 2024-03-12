macro(setup)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_EXTENSIONS OFF)

    find_package(Kokkos REQUIRED)

    register_link_library(Kokkos::kokkos)

    if (${KOKKOS_BACK_END} STREQUAL "CUDA")
        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
    elseif (${KOKKOS_BACK_END} STREQUAL "HIP")
        find_package(hip REQUIRED)

        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)
        set(CMAKE_HIP_SEPARABLE_COMPILATION ON)

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
    endif ()

endmacro()
