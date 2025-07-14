#pragma once

#include <numa.h>

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#define ROUND_UP(a, b) ((a + b - 1) / b)

#define __forceinline __attribute__((always_inline)) inline

void set_data(double *matrix, uint64_t size, uint64_t seed, double min_value,
              double max_value)
{
#pragma omp parallel
  {
    uint64_t tid = omp_get_thread_num();
    uint64_t value = (tid * 1034871 + 10581) * seed;
    uint64_t mul = 192499;
    uint64_t add = 6837199;
    for (uint64_t i = 0; i < 50 + tid; ++i)
      value = value * mul + add;
#pragma omp for
    for (uint64_t i = 0; i < size; ++i)
    {
      value = value * mul + add;
      matrix[i] = (double)value / (double)(uint64_t)(-1) * (max_value - min_value) + min_value;
    }
  }
}
