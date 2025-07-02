#pragma once

#include <numa.h>

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#define ROUND_UP(a, b) ((a + b - 1) / b)

#ifndef NUMA_NODE_MCDRAM
#define NUMA_NODE_MCDRAM 1
#endif
#define numa_alloc(size) numa_alloc_onnode(size, NUMA_NODE_MCDRAM)

#define __forceinline __attribute__((always_inline)) inline
