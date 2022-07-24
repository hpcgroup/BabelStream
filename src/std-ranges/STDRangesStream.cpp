// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDRangesStream.hpp"

#include <algorithm>
#include <execution>
#include <ranges>

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

#ifdef USE_VECTOR
#define BEGIN(x) (x).begin()
#define END(x) (x).end()
#else
#define BEGIN(x) (x)
#define END(x) ((x) + array_size)
#endif

// There are three execution policies:
// auto exe_policy = std::execution::seq;
// auto exe_policy = std::execution::par;
constexpr auto exe_policy = std::execution::par_unseq;

template <class T>
STDRangesStream<T>::STDRangesStream(const int ARRAY_SIZE, int device)
 : array_size{ARRAY_SIZE},
#ifdef USE_VECTOR
  a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE)
#else
  a((T *) aligned_alloc(ALIGNMENT, sizeof(T) * ARRAY_SIZE)),
  b((T *) aligned_alloc(ALIGNMENT, sizeof(T) * ARRAY_SIZE)),
  c((T *) aligned_alloc(ALIGNMENT, sizeof(T) * ARRAY_SIZE))
#endif
{ std::cout <<"Backing storage typeid: " << typeid(a).name() << std::endl; }

template <class T>
void STDRangesStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::for_each_n(
    exe_policy,
    std::views::iota(0).begin(), array_size, // loop range
    [&] (int i) {
      a[i] = initA;
      b[i] = initB;
      c[i] = initC;
    }
  );
}

template <class T>
void STDRangesStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  // Element-wise copy.
    std::copy(BEGIN(a), END(a), h_a.begin());
    std::copy(BEGIN(b), END(b), h_b.begin());
    std::copy(BEGIN(c), END(c), h_c.begin());
}

template <class T>
void STDRangesStream<T>::copy()
{
  std::for_each_n(
    exe_policy,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      c[i] = a[i];
    }
  );
}

template <class T>
void STDRangesStream<T>::mul()
{
  const T scalar = startScalar;

  std::for_each_n(
    exe_policy,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      b[i] = scalar * c[i];
    }
  );
}

template <class T>
void STDRangesStream<T>::add()
{
  std::for_each_n(
    exe_policy,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      c[i] = a[i] + b[i];
    }
  );
}

template <class T>
void STDRangesStream<T>::triad()
{
  const T scalar = startScalar;

  std::for_each_n(
    exe_policy,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      a[i] = b[i] + scalar * c[i];
    }
  );
}

template <class T>
void STDRangesStream<T>::nstream()
{
  const T scalar = startScalar;

  std::for_each_n(
    exe_policy,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      a[i] += b[i] + scalar * c[i];
    }
  );
}

template <class T>
T STDRangesStream<T>::dot()
{
  // sum += a[i] * b[i];
  return
    std::transform_reduce(
      exe_policy,
      BEGIN(a), END(a), BEGIN(b), 0.0);
}

void listDevices(void)
{
  std::cout << "C++20 does not expose devices" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}

template class STDRangesStream<float>;
template class STDRangesStream<double>;

#undef BEGIN
#undef END
