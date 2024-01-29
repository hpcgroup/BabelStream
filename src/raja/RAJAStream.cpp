
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include <stdexcept>
#include "RAJAStream.hpp"

template <class T>
RAJAStream<T>::RAJAStream(const int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE), range(0, ARRAY_SIZE)
{
    d_a = static_cast<T*>(alloc.allocate(sizeof(T) * array_size));
    d_b = static_cast<T*>(alloc.allocate(sizeof(T) * array_size));
    d_c = static_cast<T*>(alloc.allocate(sizeof(T) * array_size));
}

template <class T>
RAJAStream<T>::~RAJAStream()
{
    rm.getAllocator(d_a).deallocate(d_a);
    rm.getAllocator(d_b).deallocate(d_b);
    rm.getAllocator(d_c).deallocate(d_c);
}

template <class T>
void RAJAStream<T>::init_arrays(T initA, T initB, T initC)
{
  T* a = d_a;
  T* b = d_b;
  T* c = d_c;
  RAJA::forall<default_policy>(range, [=] RAJA_HOST_DEVICE (RAJA::Index_type index) {
    a[index] = initA;
    b[index] = initB;
    c[index] = initC;
  });
}

template <class T>
void RAJAStream<T>::read_arrays(
        std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  auto host_alloc = rm.getAllocator("HOST");
  auto strategy = host_alloc.getAllocationStrategy();

  umpire::util::AllocationRecord recordA{a.data(), sizeof(T) * array_size, strategy};
  umpire::util::AllocationRecord recordB{a.data(), sizeof(T) * array_size, strategy};
  umpire::util::AllocationRecord recordC{a.data(), sizeof(T) * array_size, strategy};

  rm.registerAllocation(a.data(), recordA);
  rm.registerAllocation(b.data(), recordB);
  rm.registerAllocation(c.data(), recordC);
  
  rm.copy(a.data(), d_a);
  rm.copy(b.data(), d_b);
  rm.copy(c.data(), d_c);
}

template <class T>
void RAJAStream<T>::copy()
{
  T* a = d_a;
  T* c = d_c;
  RAJA::forall<default_policy>(range, [=] RAJA_HOST_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
  T* b = d_b;
  T* c = d_c;
  const T scalar = startScalar;
  RAJA::forall<default_policy>(range, [=] RAJA_HOST_DEVICE (RAJA::Index_type index)
  {
    b[index] = scalar*c[index];
  });
}

template <class T>
void RAJAStream<T>::add()
{
  T* a = d_a;
  T* b = d_b;
  T* c = d_c;
  RAJA::forall<default_policy>(range, [=] RAJA_HOST_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index] + b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
  T* a = d_a;
  T* b = d_b;
  T* c = d_c;
  const T scalar = startScalar;
  RAJA::forall<default_policy>(range, [=] RAJA_HOST_DEVICE (RAJA::Index_type index)
  {
    a[index] = b[index] + scalar*c[index];
  });
}

template <class T>
void RAJAStream<T>::nstream()
{
    T* a = d_a;
    T* b = d_b;
    T* c = d_c;
    const T scalar = startScalar;

    RAJA::forall<default_policy>(range, [=] RAJA_HOST_DEVICE (RAJA::Index_type index) {
      a[index] += b[index] + scalar * c[index];
      }
    );
}

template <class T>
T RAJAStream<T>::dot()
{
  T* a = d_a;
  T* b = d_b;
  T sum{};

  RAJA::forall<reduce_policy>(range,
    RAJA::expt::Reduce<RAJA::operators::plus>(&sum),
    [=] RAJA_HOST_DEVICE (RAJA::Index_type index, T &_sum) {
      _sum += a[index] * b[index];
  });

  return sum;
}


void listDevices(void)
{
  std::cout << "This is not the device you are looking for.";
}


std::string getDeviceName(const int device)
{
  return "RAJA";
}


std::string getDeviceDriver(const int device)
{
  return "RAJA";
}

template class RAJAStream<float>;
template class RAJAStream<double>;
