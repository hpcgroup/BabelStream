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
#include "umpire/strategy/AlignedAllocator.hpp"

#include "Stream.h"

#define TBSIZE 1024

#define IMPLEMENTATION_STRING "RAJA"
#define ALIGNMENT (2*1024*1024) // 2MB

#if defined(RAJA_TARGET_CPU)
#if defined(RAJA_ENABLE_OPENMP)
    using exec_policy = RAJA::omp_parallel_for_exec;
    using reduce_policy = RAJA::omp_reduce;
#else
    using exec_policy = RAJA::seq_exec;
    using reduce_policy = RAJA::seq_reduce;
#endif
#else
#if defined(RAJA_ENABLE_CUDA)
    using exec_policy = RAJA::cuda_exec<TBSIZE>;
    using reduce_policy = RAJA::cuda_reduce;
#elif defined(RAJA_ENABLE_HIP)
    using exec_policy = RAJA::hip_exec<TBSIZE>;
    using reduce_policy = RAJA::hip_reduce;
#elif defined(RAJA_ENABLE_SYCL)
    using exec_policy = RAJA::sycl_exec<TBSIZE>;
    using reduce_policy = RAJA::sycl_reduce;
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
#if defined(RAJA_TARGET_CPU)
  umpire::Allocator alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>("aligned_allocator", rm.getAllocator("HOST"), ALIGNMENT);

#else
#if defined(BABELSTREAM_MANAGED_ALLOC)
  umpire::Allocator alloc = rm.getAllocator("UM");
#else
  umpire::Allocator alloc = rm.getAllocator("DEVICE");
#endif
#endif

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
