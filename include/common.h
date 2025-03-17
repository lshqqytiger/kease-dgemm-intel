#pragma once

#include <numa.h>

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#define ALLOC(arr, size)              \
  do                                  \
  {                                   \
    arr = numa_alloc_onnode(size, 1); \
  } while (0)
#define FREE(arr, size)   \
  do                      \
  {                       \
    numa_free(arr, size); \
  } while (0)
