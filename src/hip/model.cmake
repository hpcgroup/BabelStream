
register_flag_required(CMAKE_CXX_COMPILER
        "Absolute path to the AMD HIP C++ compiler")

register_flag_optional(MEM "Device memory mode:
        DEFAULT   - allocate host and device memory pointers.
        MANAGED   - use HIP Managed Memory.
        PAGEFAULT - shared memory, only host pointers allocated."
        "DEFAULT")

macro(setup)
    # nothing to do here as hipcc does everything correctly, what a surprise!
    register_definitions(${MEM})

    enable_language(HIP)
    set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
endmacro()
