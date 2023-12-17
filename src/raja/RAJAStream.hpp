// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "RAJA/RAJA.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "RAJA"

#if defined(RAJA_TARGET_CPU)
#if defined(RAJA_ENABLE_OPENMP)
    using exec_policy = RAJA::omp_parallel_for_exec;
#else
    using exec_policy = RAJA::seq_exec;
#endif
#else
    const size_t block_size = 256;
#if defined(RAJA_ENABLE_CUDA)
    using exec_policy = RAJA::cuda_exec<block_size>;
#elif defined(RAJA_ENABLE_HIP)
    using exec_policy = RAJA::hip_exec<block_size>;
#elif defined(RAJA_ENABLE_SYCL)
    using exec_policy = RAJA::sycl_exec<block_size>;
#endif
#endif



template <class T>
class RAJAStream : public Stream<T> {
  protected:
    // Size of arrays
	const int array_size;
	const RAJA::TypedRangeSegment<int> range;

    // Umpire Allocators
    umpire::ResourceManager &rm = umpire::ResourceManager::getInstance();
#if defined(RAJA_ENABLE_CPU)
    umpire::Allocator alloc = rm.getAllocator("HOST");
#else
    umpire::Allocator alloc = rm.getAllocator("DEVICE");
#endif

    // Device side pointers to arrays
    T* d_a;
    T* d_b;
    T* d_c;

  public:
    RAJAStream(const int, const int);
    ~RAJAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
	virtual void nstream() override;
	virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(
            std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

