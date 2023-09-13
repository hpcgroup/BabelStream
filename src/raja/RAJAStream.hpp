// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include "RAJA/RAJA.hpp"

#ifdef RAJA_USE_CHAI
#include <chai/ManagedArray.hpp>
#endif

#include "Stream.h"

#define IMPLEMENTATION_STRING "RAJA"

#ifdef RAJA_TARGET_CPU
// TODO verify old and new templates are semantically equal
//typedef RAJA::ExecPolicy<
//        RAJA::seq_segit,
//        RAJA::omp_parallel_for_exec> policy;

typedef RAJA::omp_parallel_for_exec policy;
typedef RAJA::omp_reduce reduce_policy;
#elif RAJA_TARGET_AMD
const size_t block_size = 128;
typedef RAJA::hip_exec<block_size> policy;
typedef RAJA::hip_reduce reduce_policy;
#else
const size_t block_size = 128;
// TODO verify old and new templates are semantically equal
//typedef RAJA::IndexSet::ExecPolicy<
//        RAJA::seq_segit,
//        RAJA::cuda_exec<block_size>> policy;
//typedef RAJA::cuda_reduce<block_size> reduce_policy;
typedef RAJA::cuda_exec<block_size> policy;
typedef RAJA::cuda_reduce reduce_policy;
#endif

using RAJA::RangeSegment;


template <class T>
class RAJAStream : public Stream<T>
{
  protected:
    // Size of arrays
  const int array_size;
  const RangeSegment range;

    // Device side pointers to arrays
#ifdef RAJA_USE_CHAI
    chai::ManagedArray<T>* d_a;
    chai::ManagedArray<T>* d_b;
    chai::ManagedArray<T>* d_c;
#else
    T* d_a;
    T* d_b;
    T* d_c;
#endif

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
