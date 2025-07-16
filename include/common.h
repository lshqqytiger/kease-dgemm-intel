#pragma once

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#define ROUND_UP(a, b) ((a + b - 1) / b)

#define __forceinline __attribute__((always_inline)) inline
