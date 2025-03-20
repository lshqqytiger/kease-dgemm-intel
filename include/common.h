#pragma once

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#ifdef USE_MCDRAM
#include <hbwmalloc.h>

__forceinline void *alloc_impl(size_t size)
{
  void *ptr = hbw_malloc(size);
  hbw_posix_memalign(ptr, 64, size);
  return ptr;
}

#define ALLOC(arr, size) (arr = alloc_impl(size))
#define FREE(arr, size) hbw_free(arr)
#else
#include <numa.h>

#define ALLOC(arr, size) (arr = numa_alloc_onnode(size, 1))
#define FREE(arr, size) numa_free(arr, size)
#endif
