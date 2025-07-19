#pragma once

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#define ROUND_UP(a, b) ((a + b - 1) / b)

#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define READ 0
#define WRITE 1

#define LOCALITY_NONE 0
#define LOCALITY_LOW 1
#define LOCALITY_MODERATE 2
#define LOCALITY_HIGH 3

#define __forceinline __attribute__((always_inline)) inline

#define STRING_(x) #x
#define STRING(x) STRING_(x)
#define CONCAT(a, b) a##b
#define ASM_PREFETCH(locality, operand) " " STRING(CONCAT(prefetch, locality)) " " operand " \t\n"
#define ASM_REPEAT(n, exp, post) exp CONCAT(ASM_REPEAT_, n)(post exp)

#define ASM_REPEAT_1(exp)
#define ASM_REPEAT_2(exp) exp
#define ASM_REPEAT_3(exp) exp exp
#define ASM_REPEAT_4(exp) exp exp exp
#define ASM_REPEAT_5(exp) exp exp exp exp
#define ASM_REPEAT_6(exp) exp exp exp exp exp
#define ASM_REPEAT_7(exp) exp exp exp exp exp exp
#define ASM_REPEAT_8(exp) exp exp exp exp exp exp exp
#define ASM_REPEAT_9(exp) exp exp exp exp exp exp exp exp
#define ASM_REPEAT_10(exp) exp exp exp exp exp exp exp exp exp
#define ASM_REPEAT_11(exp) exp exp exp exp exp exp exp exp exp exp
#define ASM_REPEAT_12(exp) exp exp exp exp exp exp exp exp exp exp exp
#define ASM_REPEAT_13(exp) exp exp exp exp exp exp exp exp exp exp exp exp
#define ASM_REPEAT_14(exp) exp exp exp exp exp exp exp exp exp exp exp exp exp
#define ASM_REPEAT_15(exp) exp exp exp exp exp exp exp exp exp exp exp exp exp exp
#define ASM_REPEAT_16(exp) exp exp exp exp exp exp exp exp exp exp exp exp exp exp exp
