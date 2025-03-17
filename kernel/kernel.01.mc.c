/**
 * @file kernel.01.mc.c
 * @author Enoch Jung
 * @brief dgemm for
 *        - core : 1 core
 *        - A     : ColMajor
 *        - B     : ColMajor
 *        - C     : ColMajor
 *        - k     : even number
 *        - alpha : -1.0
 *        - beta  : +1.0
 * @date 2024-04-17
 */

// #define DEBUG
// #define PLAY
// #define PARALLEL_WITH_OMP
// #define ORIGIN_MC

#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <omp.h>
#include <sched.h>
#include <assert.h>
#include <stdint.h>
#include <immintrin.h>

#include "cblas_format.h"
#include "common.h"

#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define TOTAL_CORE 68
#define NT1 17
#define NT2 (TOTAL_CORE / NT1)

#ifndef CM
#define CM 17
#endif

#define CN (TOTAL_CORE / CM)

#define MR 8
#define NR 24

#ifndef MB
// #define MB (MR * 48)
#define MB (MR * 48)
#endif

#ifndef NB
// #define NB (NR * 1)
#define NB (NR * 1)
#endif

#ifndef KB
// #define KB 64
#define KB 64
#endif

#define MA (MB * 1000)
#define NA (NB * 1000)
#define KA (KB * 1000)

// 17 8 24 75 5 64

#ifndef ORIGIN_MC

// #define LCAM
#define BDL

void micro_kernel_8x24_ppc_anbp(
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    uint64_t ldc,
    const double *restrict _A_next)
{
    register double *tmp_C = C;
#ifdef BDL
    register double ldb = 6000;
#endif

    asm volatile(
#ifndef LCAM
        " vmovapd (%[A]), %%zmm31         \t\n"
        " vmovupd (%[C]),  %%zmm0  \t\n"
        " vmovupd (%[C], %[ldc],1),  %%zmm1  \t\n"
        " vmovupd (%[C], %[ldc],2),  %%zmm2  \t\n"
        " vmovupd (%[C], %[ldc3],1),  %%zmm3  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd (%[C]),  %%zmm4  \t\n"
        " vmovupd (%[C], %[ldc],1),  %%zmm5  \t\n"
        " vmovupd (%[C], %[ldc],2),  %%zmm6  \t\n"
        " vmovupd (%[C], %[ldc3],1),  %%zmm7  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd (%[C]),  %%zmm8  \t\n"
        " vmovupd (%[C], %[ldc],1),  %%zmm9  \t\n"
        " vmovupd (%[C], %[ldc],2), %%zmm10  \t\n"
        " vmovupd (%[C], %[ldc3],1), %%zmm11  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd (%[C]), %%zmm12  \t\n"
        " vmovupd (%[C], %[ldc],1), %%zmm13  \t\n"
        " vmovupd (%[C], %[ldc],2), %%zmm14  \t\n"
        " vmovupd (%[C], %[ldc3],1), %%zmm15  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd (%[C]), %%zmm16  \t\n"
        " vmovupd (%[C], %[ldc],1), %%zmm17  \t\n"
        " vmovupd (%[C], %[ldc],2), %%zmm18  \t\n"
        " vmovupd (%[C], %[ldc3],1), %%zmm19  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd (%[C]), %%zmm20  \t\n"
        " vmovupd (%[C], %[ldc],1), %%zmm21  \t\n"
        " vmovupd (%[C], %[ldc],2), %%zmm22  \t\n"
        " vmovupd (%[C], %[ldc3],1), %%zmm23  \t\n"
    /*
    " lea (%[C], %[ldc],4), %[C] \t\n"
    " vmovupd (%[C]), %%zmm24  \t\n"
    " vmovupd (%[C], %[ldc],1), %%zmm25  \t\n"
    " vmovupd (%[C], %[ldc],2), %%zmm26  \t\n"
    " vmovupd (%[C], %[ldc3],1), %%zmm27  \t\n"
    " lea (%[C], %[ldc],4), %[C] \t\n"
    " vmovupd (%[C]), %%zmm28  \t\n"
    " vmovupd (%[C], %[ldc],1), %%zmm29  \t\n"
    */
#else
        " vmovapd (%[A]), %%zmm31         \t\n"
        " vpxord  %%zmm0,  %%zmm0, %%zmm0 \t\n"
        " vmovapd %%zmm0,  %%zmm1         \t\n"
        " vmovapd %%zmm0,  %%zmm2         \t\n"
        " vmovapd %%zmm0,  %%zmm3         \t\n"
        " vmovapd %%zmm0,  %%zmm4         \t\n"
        " vmovapd %%zmm0,  %%zmm5         \t\n"
        " vmovapd %%zmm0,  %%zmm6         \t\n"
        " vmovapd %%zmm0,  %%zmm7         \t\n"
        " vmovapd %%zmm0,  %%zmm8         \t\n"
        " vmovapd %%zmm0,  %%zmm9         \t\n"
        " vmovapd %%zmm0, %%zmm10         \t\n"
        " vmovapd %%zmm0, %%zmm11         \t\n"
        " vmovapd %%zmm0, %%zmm12         \t\n"
        " vmovapd %%zmm0, %%zmm13         \t\n"
        " vmovapd %%zmm0, %%zmm14         \t\n"
        " vmovapd %%zmm0, %%zmm15         \t\n"
        " vmovapd %%zmm0, %%zmm16         \t\n"
        " vmovapd %%zmm0, %%zmm17         \t\n"
        " vmovapd %%zmm0, %%zmm18         \t\n"
        " vmovapd %%zmm0, %%zmm19         \t\n"
        " vmovapd %%zmm0, %%zmm20         \t\n"
        " vmovapd %%zmm0, %%zmm21         \t\n"
        " vmovapd %%zmm0, %%zmm22         \t\n"
        " vmovapd %%zmm0, %%zmm23         \t\n"

        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#endif
        : [C] "+r"(tmp_C)
        : [ldc] "r"(ldc * 8), [ldc3] "r"(ldc * 8 * 3), [A] "r"(_A)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23", "zmm31");

    kk >>= 1;
#pragma unroll(4)
    for (uint64_t i = 0; LIKELY(i < kk); ++i)
    {
        asm volatile(

#ifdef BDL
            " vmovapd       0x40(%[A]),        %%zmm30          \t\n"
            " vfnmadd231pd      (%[B])%{1to8}, %%zmm31,  %%zmm0 \t\n"
            " vfnmadd231pd   0x8(%[B])%{1to8}, %%zmm31,  %%zmm1 \t\n"
            " vfnmadd231pd  0x10(%[B])%{1to8}, %%zmm31,  %%zmm2 \t\n"
            " vfnmadd231pd  0x18(%[B])%{1to8}, %%zmm31,  %%zmm3 \t\n"
            " vfnmadd231pd  0x20(%[B])%{1to8}, %%zmm31,  %%zmm4 \t\n"
            " vfnmadd231pd  0x28(%[B])%{1to8}, %%zmm31,  %%zmm5 \t\n"
            " vfnmadd231pd  0x30(%[B])%{1to8}, %%zmm31,  %%zmm6 \t\n"
            " vfnmadd231pd  0x38(%[B])%{1to8}, %%zmm31,  %%zmm7 \t\n"
            " vfnmadd231pd  0x40(%[B])%{1to8}, %%zmm31,  %%zmm8 \t\n"
            " vfnmadd231pd  0x48(%[B])%{1to8}, %%zmm31,  %%zmm9 \t\n"
            " vfnmadd231pd  0x50(%[B])%{1to8}, %%zmm31, %%zmm10 \t\n"
            " vfnmadd231pd  0x58(%[B])%{1to8}, %%zmm31, %%zmm11 \t\n"
            " vfnmadd231pd  0x60(%[B])%{1to8}, %%zmm31, %%zmm12 \t\n"
            " vfnmadd231pd  0x68(%[B])%{1to8}, %%zmm31, %%zmm13 \t\n"
            " vfnmadd231pd  0x70(%[B])%{1to8}, %%zmm31, %%zmm14 \t\n"
            " vfnmadd231pd  0x78(%[B])%{1to8}, %%zmm31, %%zmm15 \t\n"
            " vfnmadd231pd  0x80(%[B])%{1to8}, %%zmm31, %%zmm16 \t\n"
            " vfnmadd231pd  0x88(%[B])%{1to8}, %%zmm31, %%zmm17 \t\n"
            " vfnmadd231pd  0x90(%[B])%{1to8}, %%zmm31, %%zmm18 \t\n"
            " vfnmadd231pd  0x98(%[B])%{1to8}, %%zmm31, %%zmm19 \t\n"
            " vfnmadd231pd  0xa0(%[B])%{1to8}, %%zmm31, %%zmm20 \t\n"
            " vfnmadd231pd  0xa8(%[B])%{1to8}, %%zmm31, %%zmm21 \t\n"
            " vfnmadd231pd  0xb0(%[B])%{1to8}, %%zmm31, %%zmm22 \t\n"
            " vfnmadd231pd  0xb8(%[B])%{1to8}, %%zmm31, %%zmm23 \t\n"

            " vmovapd       0x80(%[A]),        %%zmm31          \t\n"
            " vfnmadd231pd  0xc0(%[B])%{1to8}, %%zmm30,  %%zmm0 \t\n"
            " vfnmadd231pd  0xc8(%[B])%{1to8}, %%zmm30,  %%zmm1 \t\n"
            " vfnmadd231pd  0xd0(%[B])%{1to8}, %%zmm30,  %%zmm2 \t\n"
            " vfnmadd231pd  0xd8(%[B])%{1to8}, %%zmm30,  %%zmm3 \t\n"
            " vfnmadd231pd  0xe0(%[B])%{1to8}, %%zmm30,  %%zmm4 \t\n"
            " vfnmadd231pd  0xe8(%[B])%{1to8}, %%zmm30,  %%zmm5 \t\n"
            " vfnmadd231pd  0xf0(%[B])%{1to8}, %%zmm30,  %%zmm6 \t\n"
            " vfnmadd231pd  0xf8(%[B])%{1to8}, %%zmm30,  %%zmm7 \t\n"
            " vfnmadd231pd 0x100(%[B])%{1to8}, %%zmm30,  %%zmm8 \t\n"
            " vfnmadd231pd 0x108(%[B])%{1to8}, %%zmm30,  %%zmm9 \t\n"
            " vfnmadd231pd 0x110(%[B])%{1to8}, %%zmm30, %%zmm10 \t\n"
            " vfnmadd231pd 0x118(%[B])%{1to8}, %%zmm30, %%zmm11 \t\n"
            " vfnmadd231pd 0x120(%[B])%{1to8}, %%zmm30, %%zmm12 \t\n"
            " vfnmadd231pd 0x128(%[B])%{1to8}, %%zmm30, %%zmm13 \t\n"
            " vfnmadd231pd 0x130(%[B])%{1to8}, %%zmm30, %%zmm14 \t\n"
            " vfnmadd231pd 0x138(%[B])%{1to8}, %%zmm30, %%zmm15 \t\n"
            " vfnmadd231pd 0x140(%[B])%{1to8}, %%zmm30, %%zmm16 \t\n"
            " vfnmadd231pd 0x148(%[B])%{1to8}, %%zmm30, %%zmm17 \t\n"
            " vfnmadd231pd 0x150(%[B])%{1to8}, %%zmm30, %%zmm18 \t\n"
            " vfnmadd231pd 0x158(%[B])%{1to8}, %%zmm30, %%zmm19 \t\n"
            " vfnmadd231pd 0x160(%[B])%{1to8}, %%zmm30, %%zmm20 \t\n"
            " vfnmadd231pd 0x168(%[B])%{1to8}, %%zmm30, %%zmm21 \t\n"
            " vfnmadd231pd 0x170(%[B])%{1to8}, %%zmm30, %%zmm22 \t\n"
            " vfnmadd231pd 0x178(%[B])%{1to8}, %%zmm30, %%zmm23 \t\n"

            " add  $0x80, %[A] \t\n"
            " add $0x180, %[B] \t\n"
#else
            " prefetcht0   0x480(%[A])                          \t\n"
            " vmovapd       0x40(%[A]),        %%zmm30          \t\n"
            " vfnmadd231pd      (%[B])%{1to8}, %%zmm31,  %%zmm0 \t\n"
            " vfnmadd231pd   0x8(%[B])%{1to8}, %%zmm31,  %%zmm1 \t\n"
            " vfnmadd231pd  0x10(%[B])%{1to8}, %%zmm31,  %%zmm2 \t\n"
            " vfnmadd231pd  0x18(%[B])%{1to8}, %%zmm31,  %%zmm3 \t\n"
            " vfnmadd231pd  0x20(%[B])%{1to8}, %%zmm31,  %%zmm4 \t\n"
            " vfnmadd231pd  0x28(%[B])%{1to8}, %%zmm31,  %%zmm5 \t\n"
            " vfnmadd231pd  0x30(%[B])%{1to8}, %%zmm31,  %%zmm6 \t\n"
            " vfnmadd231pd  0x38(%[B])%{1to8}, %%zmm31,  %%zmm7 \t\n"
            " vfnmadd231pd  0x40(%[B])%{1to8}, %%zmm31,  %%zmm8 \t\n"
            " vfnmadd231pd  0x48(%[B])%{1to8}, %%zmm31,  %%zmm9 \t\n"
            " vfnmadd231pd  0x50(%[B])%{1to8}, %%zmm31, %%zmm10 \t\n"
            " vfnmadd231pd  0x58(%[B])%{1to8}, %%zmm31, %%zmm11 \t\n"
            " vfnmadd231pd  0x60(%[B])%{1to8}, %%zmm31, %%zmm12 \t\n"
            " vfnmadd231pd  0x68(%[B])%{1to8}, %%zmm31, %%zmm13 \t\n"
            " vfnmadd231pd  0x70(%[B])%{1to8}, %%zmm31, %%zmm14 \t\n"
            " vfnmadd231pd  0x78(%[B])%{1to8}, %%zmm31, %%zmm15 \t\n"
            " vfnmadd231pd  0x80(%[B])%{1to8}, %%zmm31, %%zmm16 \t\n"
            " vfnmadd231pd  0x88(%[B])%{1to8}, %%zmm31, %%zmm17 \t\n"
            " vfnmadd231pd  0x90(%[B])%{1to8}, %%zmm31, %%zmm18 \t\n"
            " vfnmadd231pd  0x98(%[B])%{1to8}, %%zmm31, %%zmm19 \t\n"
            " vfnmadd231pd  0xa0(%[B])%{1to8}, %%zmm31, %%zmm20 \t\n"
            " vfnmadd231pd  0xa8(%[B])%{1to8}, %%zmm31, %%zmm21 \t\n"
            " vfnmadd231pd  0xb0(%[B])%{1to8}, %%zmm31, %%zmm22 \t\n"
            " vfnmadd231pd  0xb8(%[B])%{1to8}, %%zmm31, %%zmm23 \t\n"

            " prefetcht0   0x4c0(%[A])                          \t\n"
            " vmovapd       0x80(%[A]),        %%zmm31          \t\n"
            " vfnmadd231pd  0xc0(%[B])%{1to8}, %%zmm30,  %%zmm0 \t\n"
            " vfnmadd231pd  0xc8(%[B])%{1to8}, %%zmm30,  %%zmm1 \t\n"
            " vfnmadd231pd  0xd0(%[B])%{1to8}, %%zmm30,  %%zmm2 \t\n"
            " vfnmadd231pd  0xd8(%[B])%{1to8}, %%zmm30,  %%zmm3 \t\n"
            " vfnmadd231pd  0xe0(%[B])%{1to8}, %%zmm30,  %%zmm4 \t\n"
            " vfnmadd231pd  0xe8(%[B])%{1to8}, %%zmm30,  %%zmm5 \t\n"
            " vfnmadd231pd  0xf0(%[B])%{1to8}, %%zmm30,  %%zmm6 \t\n"
            " vfnmadd231pd  0xf8(%[B])%{1to8}, %%zmm30,  %%zmm7 \t\n"
            " vfnmadd231pd 0x100(%[B])%{1to8}, %%zmm30,  %%zmm8 \t\n"
            " vfnmadd231pd 0x108(%[B])%{1to8}, %%zmm30,  %%zmm9 \t\n"
            " vfnmadd231pd 0x110(%[B])%{1to8}, %%zmm30, %%zmm10 \t\n"
            " vfnmadd231pd 0x118(%[B])%{1to8}, %%zmm30, %%zmm11 \t\n"
            " vfnmadd231pd 0x120(%[B])%{1to8}, %%zmm30, %%zmm12 \t\n"
            " vfnmadd231pd 0x128(%[B])%{1to8}, %%zmm30, %%zmm13 \t\n"
            " vfnmadd231pd 0x130(%[B])%{1to8}, %%zmm30, %%zmm14 \t\n"
            " vfnmadd231pd 0x138(%[B])%{1to8}, %%zmm30, %%zmm15 \t\n"
            " vfnmadd231pd 0x140(%[B])%{1to8}, %%zmm30, %%zmm16 \t\n"
            " vfnmadd231pd 0x148(%[B])%{1to8}, %%zmm30, %%zmm17 \t\n"
            " vfnmadd231pd 0x150(%[B])%{1to8}, %%zmm30, %%zmm18 \t\n"
            " vfnmadd231pd 0x158(%[B])%{1to8}, %%zmm30, %%zmm19 \t\n"
            " vfnmadd231pd 0x160(%[B])%{1to8}, %%zmm30, %%zmm20 \t\n"
            " vfnmadd231pd 0x168(%[B])%{1to8}, %%zmm30, %%zmm21 \t\n"
            " vfnmadd231pd 0x170(%[B])%{1to8}, %%zmm30, %%zmm22 \t\n"
            " vfnmadd231pd 0x178(%[B])%{1to8}, %%zmm30, %%zmm23 \t\n"

            " add  $0x80, %[A] \t\n"
            " add $0x180, %[B] \t\n"
#endif
#ifdef BDL
            : [A] "+r"(_A), [B] "+r"(_B)
            : [ldb] "r"(ldb)
#else
            : [A] "+r"(_A), [B] "+r"(_B)
            :
#endif
            : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
              "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
              "zmm20", "zmm21", "zmm22", "zmm23", "zmm30", "zmm31");
    }

    asm volatile(
#ifndef LCAM
        " vmovupd  %%zmm0, (%[C])  \t\n"
        " vmovupd  %%zmm1, (%[C], %[ldc],1)  \t\n"
        " vmovupd  %%zmm2, (%[C], %[ldc],2)  \t\n"
        " vmovupd  %%zmm3, (%[C], %[ldc3],1)  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd  %%zmm4, (%[C])  \t\n"
        " vmovupd  %%zmm5, (%[C], %[ldc],1)  \t\n"
        " vmovupd  %%zmm6, (%[C], %[ldc],2)  \t\n"
        " vmovupd  %%zmm7, (%[C], %[ldc3],1)  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd  %%zmm8, (%[C])  \t\n"
        " vmovupd  %%zmm9, (%[C], %[ldc],1)  \t\n"
        " vmovupd %%zmm10, (%[C], %[ldc],2)  \t\n"
        " vmovupd %%zmm11, (%[C], %[ldc3],1)  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd %%zmm12, (%[C])  \t\n"
        " vmovupd %%zmm13, (%[C], %[ldc],1)  \t\n"
        " vmovupd %%zmm14, (%[C], %[ldc],2)  \t\n"
        " vmovupd %%zmm15, (%[C], %[ldc3],1)  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd %%zmm16, (%[C])  \t\n"
        " vmovupd %%zmm17, (%[C], %[ldc],1)  \t\n"
        " vmovupd %%zmm18, (%[C], %[ldc],2)  \t\n"
        " vmovupd %%zmm19, (%[C], %[ldc3],1)  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd %%zmm20, (%[C])  \t\n"
        " vmovupd %%zmm21, (%[C], %[ldc],1)  \t\n"
        " vmovupd %%zmm22, (%[C], %[ldc],2)  \t\n"
        " vmovupd %%zmm23, (%[C], %[ldc3],1)  \t\n"
    /*
    " lea (%[C], %[ldc],4), %[C] \t\n"
    " vmovupd %%zmm24, (%[C])  \t\n"
    " vmovupd %%zmm25, (%[C], %[ldc],1)  \t\n"
    " vmovupd %%zmm26, (%[C], %[ldc],2)  \t\n"
    " vmovupd %%zmm27, (%[C], %[ldc3],1)  \t\n"
    " lea (%[C], %[ldc],4), %[C] \t\n"
    " vmovupd %%zmm28, (%[C])  \t\n"
    " vmovupd %%zmm29, (%[C], %[ldc],1)  \t\n"
    */
#else
        " prefetcht0     (%[A_next])                 \t\n"
        " prefetcht0 0x40(%[A_next])                 \t\n"
        " prefetcht0 0x80(%[A_next])                 \t\n"
        " prefetcht0 0xc0(%[A_next])                 \t\n"

        " vaddpd  (%[C]),            %%zmm0,  %%zmm0 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm1,  %%zmm1 \t\n"
        " vaddpd  (%[C], %[ldc],2),  %%zmm2,  %%zmm2 \t\n"
        " vaddpd  (%[C],%[ldc3],1),  %%zmm3,  %%zmm3 \t\n"
        " vmovupd  %%zmm0, (%[C])                    \t\n"
        " vmovupd  %%zmm1, (%[C], %[ldc],1)          \t\n"
        " vmovupd  %%zmm2, (%[C], %[ldc],2)          \t\n"
        " vmovupd  %%zmm3, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),            %%zmm4,  %%zmm4 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm5,  %%zmm5 \t\n"
        " vaddpd  (%[C], %[ldc],2),  %%zmm6,  %%zmm6 \t\n"
        " vaddpd  (%[C],%[ldc3],1),  %%zmm7,  %%zmm7 \t\n"
        " vmovupd  %%zmm4, (%[C])                    \t\n"
        " vmovupd  %%zmm5, (%[C], %[ldc],1)          \t\n"
        " vmovupd  %%zmm6, (%[C], %[ldc],2)          \t\n"
        " vmovupd  %%zmm7, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),            %%zmm8,  %%zmm8 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm9,  %%zmm9 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm10, %%zmm10 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm11, %%zmm11 \t\n"
        " vmovupd  %%zmm8, (%[C])                    \t\n"
        " vmovupd  %%zmm9, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm10, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm11, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm12, %%zmm12 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm13, %%zmm13 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm14, %%zmm14 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm15, %%zmm15 \t\n"
        " vmovupd %%zmm12, (%[C])                    \t\n"
        " vmovupd %%zmm13, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm14, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm15, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm16, %%zmm16 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm17, %%zmm17 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm18, %%zmm18 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm19, %%zmm19 \t\n"
        " vmovupd %%zmm16, (%[C])                    \t\n"
        " vmovupd %%zmm17, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm18, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm19, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm20, %%zmm20 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm21, %%zmm21 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm22, %%zmm22 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm23, %%zmm23 \t\n"
        " vmovupd %%zmm20, (%[C])                    \t\n"
        " vmovupd %%zmm21, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm22, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm23, (%[C],%[ldc3],1)          \t\n"
#endif
        : [C] "+r"(C)
        : [ldc] "r"(ldc * 8), [ldc3] "r"(ldc * 8 * 3), [A_next] "r"(_A_next)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23");
}

#else

void micro_kernel_8x24_ppc_anbp(
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    uint64_t ldc,
    const double *restrict _A_next)
{
    register double *tmp_C = C;

    asm volatile(
        " vmovapd (%[A]), %%zmm31         \t\n"
        " vpxord  %%zmm0,  %%zmm0, %%zmm0 \t\n"
        " vmovapd %%zmm0,  %%zmm1         \t\n"
        " vmovapd %%zmm0,  %%zmm2         \t\n"
        " vmovapd %%zmm0,  %%zmm3         \t\n"
        " vmovapd %%zmm0,  %%zmm4         \t\n"
        " vmovapd %%zmm0,  %%zmm5         \t\n"
        " vmovapd %%zmm0,  %%zmm6         \t\n"
        " vmovapd %%zmm0,  %%zmm7         \t\n"
        " vmovapd %%zmm0,  %%zmm8         \t\n"
        " vmovapd %%zmm0,  %%zmm9         \t\n"
        " vmovapd %%zmm0, %%zmm10         \t\n"
        " vmovapd %%zmm0, %%zmm11         \t\n"
        " vmovapd %%zmm0, %%zmm12         \t\n"
        " vmovapd %%zmm0, %%zmm13         \t\n"
        " vmovapd %%zmm0, %%zmm14         \t\n"
        " vmovapd %%zmm0, %%zmm15         \t\n"
        " vmovapd %%zmm0, %%zmm16         \t\n"
        " vmovapd %%zmm0, %%zmm17         \t\n"
        " vmovapd %%zmm0, %%zmm18         \t\n"
        " vmovapd %%zmm0, %%zmm19         \t\n"
        " vmovapd %%zmm0, %%zmm20         \t\n"
        " vmovapd %%zmm0, %%zmm21         \t\n"
        " vmovapd %%zmm0, %%zmm22         \t\n"
        " vmovapd %%zmm0, %%zmm23         \t\n"

        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
        : [C] "+r"(tmp_C)
        : [ldc] "r"(ldc * 8), [ldc3] "r"(ldc * 8 * 3), [A] "r"(_A)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23", "zmm31");

    kk >>= 1;
#pragma unroll(4)
    for (uint64_t i = 0; LIKELY(i < kk); ++i)
    {
        asm volatile(

            " prefetcht0   0x480(%[A])                          \t\n"
            " vmovapd       0x40(%[A]),        %%zmm30          \t\n"
            " vfnmadd231pd      (%[B])%{1to8}, %%zmm31,  %%zmm0 \t\n"
            " vfnmadd231pd   0x8(%[B])%{1to8}, %%zmm31,  %%zmm1 \t\n"
            " vfnmadd231pd  0x10(%[B])%{1to8}, %%zmm31,  %%zmm2 \t\n"
            " vfnmadd231pd  0x18(%[B])%{1to8}, %%zmm31,  %%zmm3 \t\n"
            " vfnmadd231pd  0x20(%[B])%{1to8}, %%zmm31,  %%zmm4 \t\n"
            " vfnmadd231pd  0x28(%[B])%{1to8}, %%zmm31,  %%zmm5 \t\n"
            " vfnmadd231pd  0x30(%[B])%{1to8}, %%zmm31,  %%zmm6 \t\n"
            " vfnmadd231pd  0x38(%[B])%{1to8}, %%zmm31,  %%zmm7 \t\n"
            " vfnmadd231pd  0x40(%[B])%{1to8}, %%zmm31,  %%zmm8 \t\n"
            " vfnmadd231pd  0x48(%[B])%{1to8}, %%zmm31,  %%zmm9 \t\n"
            " vfnmadd231pd  0x50(%[B])%{1to8}, %%zmm31, %%zmm10 \t\n"
            " vfnmadd231pd  0x58(%[B])%{1to8}, %%zmm31, %%zmm11 \t\n"
            " vfnmadd231pd  0x60(%[B])%{1to8}, %%zmm31, %%zmm12 \t\n"
            " vfnmadd231pd  0x68(%[B])%{1to8}, %%zmm31, %%zmm13 \t\n"
            " vfnmadd231pd  0x70(%[B])%{1to8}, %%zmm31, %%zmm14 \t\n"
            " vfnmadd231pd  0x78(%[B])%{1to8}, %%zmm31, %%zmm15 \t\n"
            " vfnmadd231pd  0x80(%[B])%{1to8}, %%zmm31, %%zmm16 \t\n"
            " vfnmadd231pd  0x88(%[B])%{1to8}, %%zmm31, %%zmm17 \t\n"
            " vfnmadd231pd  0x90(%[B])%{1to8}, %%zmm31, %%zmm18 \t\n"
            " vfnmadd231pd  0x98(%[B])%{1to8}, %%zmm31, %%zmm19 \t\n"
            " vfnmadd231pd  0xa0(%[B])%{1to8}, %%zmm31, %%zmm20 \t\n"
            " vfnmadd231pd  0xa8(%[B])%{1to8}, %%zmm31, %%zmm21 \t\n"
            " vfnmadd231pd  0xb0(%[B])%{1to8}, %%zmm31, %%zmm22 \t\n"
            " vfnmadd231pd  0xb8(%[B])%{1to8}, %%zmm31, %%zmm23 \t\n"

            " prefetcht0   0x4c0(%[A])                          \t\n"
            " vmovapd       0x80(%[A]),        %%zmm31          \t\n"
            " vfnmadd231pd  0xc0(%[B])%{1to8}, %%zmm30,  %%zmm0 \t\n"
            " vfnmadd231pd  0xc8(%[B])%{1to8}, %%zmm30,  %%zmm1 \t\n"
            " vfnmadd231pd  0xd0(%[B])%{1to8}, %%zmm30,  %%zmm2 \t\n"
            " vfnmadd231pd  0xd8(%[B])%{1to8}, %%zmm30,  %%zmm3 \t\n"
            " vfnmadd231pd  0xe0(%[B])%{1to8}, %%zmm30,  %%zmm4 \t\n"
            " vfnmadd231pd  0xe8(%[B])%{1to8}, %%zmm30,  %%zmm5 \t\n"
            " vfnmadd231pd  0xf0(%[B])%{1to8}, %%zmm30,  %%zmm6 \t\n"
            " vfnmadd231pd  0xf8(%[B])%{1to8}, %%zmm30,  %%zmm7 \t\n"
            " vfnmadd231pd 0x100(%[B])%{1to8}, %%zmm30,  %%zmm8 \t\n"
            " vfnmadd231pd 0x108(%[B])%{1to8}, %%zmm30,  %%zmm9 \t\n"
            " vfnmadd231pd 0x110(%[B])%{1to8}, %%zmm30, %%zmm10 \t\n"
            " vfnmadd231pd 0x118(%[B])%{1to8}, %%zmm30, %%zmm11 \t\n"
            " vfnmadd231pd 0x120(%[B])%{1to8}, %%zmm30, %%zmm12 \t\n"
            " vfnmadd231pd 0x128(%[B])%{1to8}, %%zmm30, %%zmm13 \t\n"
            " vfnmadd231pd 0x130(%[B])%{1to8}, %%zmm30, %%zmm14 \t\n"
            " vfnmadd231pd 0x138(%[B])%{1to8}, %%zmm30, %%zmm15 \t\n"
            " vfnmadd231pd 0x140(%[B])%{1to8}, %%zmm30, %%zmm16 \t\n"
            " vfnmadd231pd 0x148(%[B])%{1to8}, %%zmm30, %%zmm17 \t\n"
            " vfnmadd231pd 0x150(%[B])%{1to8}, %%zmm30, %%zmm18 \t\n"
            " vfnmadd231pd 0x158(%[B])%{1to8}, %%zmm30, %%zmm19 \t\n"
            " vfnmadd231pd 0x160(%[B])%{1to8}, %%zmm30, %%zmm20 \t\n"
            " vfnmadd231pd 0x168(%[B])%{1to8}, %%zmm30, %%zmm21 \t\n"
            " vfnmadd231pd 0x170(%[B])%{1to8}, %%zmm30, %%zmm22 \t\n"
            " vfnmadd231pd 0x178(%[B])%{1to8}, %%zmm30, %%zmm23 \t\n"

            " add  $0x80, %[A] \t\n"
            " add $0x180, %[B] \t\n"
            : [A] "+r"(_A), [B] "+r"(_B)
            :
            : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
              "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
              "zmm20", "zmm21", "zmm22", "zmm23", "zmm30", "zmm31");
    }

    asm volatile(
        " prefetcht0     (%[A_next])                 \t\n"
        " prefetcht0 0x40(%[A_next])                 \t\n"
        " prefetcht0 0x80(%[A_next])                 \t\n"
        " prefetcht0 0xc0(%[A_next])                 \t\n"

        " vaddpd  (%[C]),            %%zmm0,  %%zmm0 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm1,  %%zmm1 \t\n"
        " vaddpd  (%[C], %[ldc],2),  %%zmm2,  %%zmm2 \t\n"
        " vaddpd  (%[C],%[ldc3],1),  %%zmm3,  %%zmm3 \t\n"
        " vmovupd  %%zmm0, (%[C])                    \t\n"
        " vmovupd  %%zmm1, (%[C], %[ldc],1)          \t\n"
        " vmovupd  %%zmm2, (%[C], %[ldc],2)          \t\n"
        " vmovupd  %%zmm3, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),            %%zmm4,  %%zmm4 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm5,  %%zmm5 \t\n"
        " vaddpd  (%[C], %[ldc],2),  %%zmm6,  %%zmm6 \t\n"
        " vaddpd  (%[C],%[ldc3],1),  %%zmm7,  %%zmm7 \t\n"
        " vmovupd  %%zmm4, (%[C])                    \t\n"
        " vmovupd  %%zmm5, (%[C], %[ldc],1)          \t\n"
        " vmovupd  %%zmm6, (%[C], %[ldc],2)          \t\n"
        " vmovupd  %%zmm7, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),            %%zmm8,  %%zmm8 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm9,  %%zmm9 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm10, %%zmm10 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm11, %%zmm11 \t\n"
        " vmovupd  %%zmm8, (%[C])                    \t\n"
        " vmovupd  %%zmm9, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm10, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm11, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm12, %%zmm12 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm13, %%zmm13 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm14, %%zmm14 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm15, %%zmm15 \t\n"
        " vmovupd %%zmm12, (%[C])                    \t\n"
        " vmovupd %%zmm13, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm14, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm15, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm16, %%zmm16 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm17, %%zmm17 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm18, %%zmm18 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm19, %%zmm19 \t\n"
        " vmovupd %%zmm16, (%[C])                    \t\n"
        " vmovupd %%zmm17, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm18, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm19, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm20, %%zmm20 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm21, %%zmm21 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm22, %%zmm22 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm23, %%zmm23 \t\n"
        " vmovupd %%zmm20, (%[C])                    \t\n"
        " vmovupd %%zmm21, (%[C], %[ldc],1)          \t\n"
        " vmovupd %%zmm22, (%[C], %[ldc],2)          \t\n"
        " vmovupd %%zmm23, (%[C],%[ldc3],1)          \t\n"
        : [C] "+r"(C)
        : [ldc] "r"(ldc * 8), [ldc3] "r"(ldc * 8 * 3), [A_next] "r"(_A_next)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23");
}
#endif

void micro_dxpy_cc(
    uint64_t m,
    uint64_t n,
    double *restrict C,
    uint64_t ldc,
    const double *restrict _C)
{
    for (uint64_t i = 0; i < n; ++i)
    {
        for (uint64_t j = 0; j < m; ++j)
        {
            C[i * ldc + j] += _C[i * MR + j];
        }
    }
}

// #define INNER_M_N
void inner_kernel_ppc_anbp(
    uint64_t mm,
    uint64_t nn,
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    uint64_t ldc)
{
    uint64_t mmc = (mm + MR - 1) / MR;
    uint64_t mmr = mm % MR;
    uint64_t nnc = (nn + NR - 1) / NR;
    uint64_t nnr = nn % NR;

    const double *A_now;
    const double *B_now;
    const double *A_next;
    const double *B_next;

    A_next = _A;
    B_next = _B;

#ifdef INNER_M_N
#ifdef PARALLEL_WITH_OMP
#pragma omp parallel for num_threads(NT2)
#endif
    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

        const double *A_now = _A + mmi * MR * kk;
        const double *A_next = mmi != mmc - 1 ? A_now + MR * kk : _A;

        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

            const double *B_now = _B + nni * NR * kk;
            const double *B_next = nni != nnc - 1 ? B_now + NR * kk : _B;
#else
    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

        B_now = B_next;
        B_next = nni != nnc - 1 ? B_next + NR * kk : _B;

        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

            A_now = A_next;
            A_next = mmi != mmc - 1 ? A_next + MR * kk : _A;
#endif

#ifdef BDL
            if (LIKELY(mmm == MR && nnn == NR))
            {
                micro_kernel_8x24_ppc_anbp(kk, A_now, _B + nni * 600, C + mmi * MR + nni * NR * ldc, ldc, A_next);
            }
            else
            {
                double _C[MR * NR] __attribute__((aligned(64))) = {};
                micro_kernel_8x24_ppc_anbp(kk, A_now, _B + nni * 600, _C, MR, A_next);
                micro_dxpy_cc(mmm, nnn, C + mmi * MR + nni * NR * ldc, ldc, _C);
            }
#else
            if (LIKELY(mmm == MR && nnn == NR))
            {
                micro_kernel_8x24_ppc_anbp(kk, A_now, B_now, C + mmi * MR + nni * NR * ldc, ldc, A_next);
            }
            else
            {
                double _C[MR * NR] __attribute__((aligned(64))) = {};
                micro_kernel_8x24_ppc_anbp(kk, A_now, B_now, _C, MR, A_next);
                micro_dxpy_cc(mmm, nnn, C + mmi * MR + nni * NR * ldc, ldc, _C);
            }
#endif
        }
    }
}

void packacc(
    uint64_t mm,
    uint64_t kk,
    const double *restrict A,
    uint64_t lda,
    double *restrict _A)
{
    uint64_t mmc = (mm + MR - 1) / MR;
    uint64_t mmr = mm % MR;
    uint64_t kkc = (kk + CACHE_ELEM - 1) / CACHE_ELEM;
    uint64_t kkr = kk % CACHE_ELEM;

    const double *A_now = A;
    const double *A_k_next = A;

#ifdef PARALLEL_WITH_OMP
#pragma omp parallel for num_threads(TOTAL_CORE)
#endif
    for (uint64_t kki = 0; kki < kkc; ++kki)
    {
        uint64_t kkk = (kki != kkc - 1 || kkr == 0) ? CACHE_ELEM : kkr;

        A_now = A_k_next;
        A_k_next += lda * CACHE_ELEM;

        asm volatile(
            " prefetchnta     (%[A])            \t\n"
            " prefetchnta 0x40(%[A])            \t\n"
            " prefetchnta 0x80(%[A])            \t\n"
            " prefetchnta 0xc0(%[A])            \t\n"
            " prefetchnta     (%[A], %[lda],1)  \t\n"
            " prefetchnta 0x40(%[A], %[lda],1)  \t\n"
            " prefetchnta 0x80(%[A], %[lda],1)  \t\n"
            " prefetchnta 0xc0(%[A], %[lda],1)  \t\n"
            " prefetchnta     (%[A], %[lda],2)  \t\n"
            " prefetchnta 0x40(%[A], %[lda],2)  \t\n"
            " prefetchnta 0x80(%[A], %[lda],2)  \t\n"
            " prefetchnta 0xc0(%[A], %[lda],2)  \t\n"
            " prefetchnta     (%[A],%[lda3],1)  \t\n"
            " prefetchnta 0x40(%[A],%[lda3],1)  \t\n"
            " prefetchnta 0x80(%[A],%[lda3],1)  \t\n"
            " prefetchnta 0xc0(%[A],%[lda3],1)  \t\n"
            " prefetchnta     (%[B])            \t\n"
            " prefetchnta 0x40(%[B])            \t\n"
            " prefetchnta 0x80(%[B])            \t\n"
            " prefetchnta 0xc0(%[B])            \t\n"
            " prefetchnta     (%[B], %[lda],1)  \t\n"
            " prefetchnta 0x40(%[B], %[lda],1)  \t\n"
            " prefetchnta 0x80(%[B], %[lda],1)  \t\n"
            " prefetchnta 0xc0(%[B], %[lda],1)  \t\n"
            " prefetchnta     (%[B], %[lda],2)  \t\n"
            " prefetchnta 0x40(%[B], %[lda],2)  \t\n"
            " prefetchnta 0x80(%[B], %[lda],2)  \t\n"
            " prefetchnta 0xc0(%[B], %[lda],2)  \t\n"
            " prefetchnta     (%[B],%[lda3],1)  \t\n"
            " prefetchnta 0x40(%[B],%[lda3],1)  \t\n"
            " prefetchnta 0x80(%[B],%[lda3],1)  \t\n"
            " prefetchnta 0xc0(%[B],%[lda3],1)  \t\n"
            :
            : [A] "r"(A_k_next), [B] "r"(A_k_next + lda * 4), [lda] "r"(lda * 8), [lda3] "r"(lda * 8 * 3));

        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

            register double *_A_now = _A + mmi * MR * kk + kki * MR * CACHE_ELEM;

            if (LIKELY(mmm == MR && kkk == CACHE_ELEM))
            {
                register const double *tmp_A = A_now;

                asm volatile(
                    " vmovupd (%[A]),           %%zmm0   \t\n"
                    " vmovupd (%[A], %[lda],1), %%zmm1   \t\n"
                    " vmovupd (%[A], %[lda],2), %%zmm2   \t\n"
                    " vmovupd (%[A],%[lda3],1), %%zmm3   \t\n"
                    " lea     (%[A], %[lda],4), %[A]     \t\n"

                    " vmovupd (%[A]),           %%zmm4   \t\n"
                    " vmovupd (%[A], %[lda],1), %%zmm5   \t\n"
                    " vmovupd (%[A], %[lda],2), %%zmm6   \t\n"
                    " vmovupd (%[A],%[lda3],1), %%zmm7   \t\n"

                    " vmovupd  %%zmm0,      (%[_A])      \t\n"
                    " vmovupd  %%zmm1,  0x40(%[_A])      \t\n"
                    " vmovupd  %%zmm2,  0x80(%[_A])      \t\n"
                    " vmovupd  %%zmm3,  0xc0(%[_A])      \t\n"

                    " vmovupd  %%zmm4, 0x100(%[_A])      \t\n"
                    " vmovupd  %%zmm5, 0x140(%[_A])      \t\n"
                    " vmovupd  %%zmm6, 0x180(%[_A])      \t\n"
                    " vmovupd  %%zmm7, 0x1c0(%[_A])      \t\n"
                    : [A] "+r"(tmp_A), [_A] "+r"(_A_now)
                    : [lda] "r"(lda * 8), [lda3] "r"(lda * 8 * 3)
                    : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7");
            }
            else
            {
                for (uint64_t kkki = 0; kkki < kkk; ++kkki)
                {
                    for (uint64_t mmmi = 0; mmmi < mmm; ++mmmi)
                    {
                        _A_now[mmmi + kkki * MR] = A_now[mmmi + kkki * lda];
                    }
                }
            }

            A_now += MR;
        }
    }
}

void transpose(double *dst, const double *src, int ld)
{
    __m512d r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r0a, r0b, r0c, r0d, r0e, r0f;

    r00 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 0 * ld)), _mm256_loadu_pd(src + 4 * ld), 1);
    r01 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 1 * ld)), _mm256_loadu_pd(src + 5 * ld), 1);
    r02 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 2 * ld)), _mm256_loadu_pd(src + 6 * ld), 1);
    r03 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 3 * ld)), _mm256_loadu_pd(src + 7 * ld), 1);
    r04 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 0 * ld + 4)), _mm256_loadu_pd(src + 4 * ld + 4), 1);
    r05 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 1 * ld + 4)), _mm256_loadu_pd(src + 5 * ld + 4), 1);
    r06 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 2 * ld + 4)), _mm256_loadu_pd(src + 6 * ld + 4), 1);
    r07 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(src + 3 * ld + 4)), _mm256_loadu_pd(src + 7 * ld + 4), 1);

    r08 = _mm512_mask_permutex_pd(r00, 0xcc, r02, 0x4e);
    r09 = _mm512_mask_permutex_pd(r01, 0xcc, r03, 0x4e);
    r0a = _mm512_mask_permutex_pd(r02, 0x33, r00, 0x4e);
    r0b = _mm512_mask_permutex_pd(r03, 0x33, r01, 0x4e);
    r0c = _mm512_mask_permutex_pd(r04, 0xcc, r06, 0x4e);
    r0d = _mm512_mask_permutex_pd(r05, 0xcc, r07, 0x4e);
    r0e = _mm512_mask_permutex_pd(r06, 0x33, r04, 0x4e);
    r0f = _mm512_mask_permutex_pd(r07, 0x33, r05, 0x4e);

    r00 = _mm512_mask_permute_pd(r08, 0xaa, r09, 0x55);
    r01 = _mm512_mask_permute_pd(r09, 0x55, r08, 0x55);
    r02 = _mm512_mask_permute_pd(r0a, 0xaa, r0b, 0x55);
    r03 = _mm512_mask_permute_pd(r0b, 0x55, r0a, 0x55);
    r04 = _mm512_mask_permute_pd(r0c, 0xaa, r0d, 0x55);
    r05 = _mm512_mask_permute_pd(r0d, 0x55, r0c, 0x55);
    r06 = _mm512_mask_permute_pd(r0e, 0xaa, r0f, 0x55);
    r07 = _mm512_mask_permute_pd(r0f, 0x55, r0e, 0x55);

    _mm512_store_pd(dst + 0 * NR, r00);
    _mm512_store_pd(dst + 1 * NR, r01);
    _mm512_store_pd(dst + 2 * NR, r02);
    _mm512_store_pd(dst + 3 * NR, r03);
    _mm512_store_pd(dst + 4 * NR, r04);
    _mm512_store_pd(dst + 5 * NR, r05);
    _mm512_store_pd(dst + 6 * NR, r06);
    _mm512_store_pd(dst + 7 * NR, r07);
}

void packbcr(
    uint64_t kk,
    uint64_t nn,
    const double *restrict B,
    uint64_t ldb,
    double *restrict _B)
{
    uint64_t nnc = (nn + NR - 1) / NR;
    uint64_t nnr = nn % NR;
    uint64_t kkc = (kk + 7) / 8;
    uint64_t kkr = kk % 8;
    for (uint64_t j = 0; j < kkc; ++j)
    {
        uint64_t kkk = (j != kkc - 1 || kkr == 0) ? 8 : kkr;

        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
            uint64_t nnnc = (nnn + 7) / 8;
            uint64_t nnnr = nnn % 8;

            for (uint64_t i = 0; i < nnnc; ++i)
            {
                uint64_t nnnn = (i != nnnc - 1 || nnnr == 0) ? 8 : nnnr;
                if (kkk == 8 && nnnn == 8)
                {
                    transpose(_B + nni * NR * kk + i * 8 + j * NR * 8, B + nni * NR * ldb + i * 8 * ldb + j * 8, ldb);
                }
                else
                {
                    for (uint64_t ii = 0; ii < nnnn; ++ii)
                    {
                        for (uint64_t jj = 0; jj < kkk; ++jj)
                        {
                            _B[j * NR * 8 + nni * NR * kk + i * 8 + ii + jj * NR] = B[j * 8 + nni * NR * ldb + i * 8 * ldb + ii * ldb + jj];
                        }
                    }
                }
            }
        }
    }
}

struct thread_info
{
    pthread_t thread_id;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    const double *A;
    uint64_t lda;
    const double *B;
    uint64_t ldb;
    double *restrict C;
    uint64_t ldc;
};

static void *middle_kernel(
    void *arg)
{
    struct thread_info *tinfo = arg;
    const uint64_t m = tinfo->m;
    const uint64_t n = tinfo->n;
    const uint64_t k = tinfo->k;
    const double *A = tinfo->A;
    const uint64_t lda = tinfo->lda;
    const double *B = tinfo->B;
    const uint64_t ldb = tinfo->ldb;
    double *C = tinfo->C;
    const uint64_t ldc = tinfo->ldc;

    uint64_t mc = (m + MB - 1) / MB;
    uint64_t mr = m % MB;
    uint64_t nc = (n + NB - 1) / NB;
    uint64_t nr = n % NB;
    uint64_t kc = (k + KB - 1) / KB;
    uint64_t kr = k % KB;

    double *_A;
    double *_B;

#ifdef DEBUG
    const double start_time = omp_get_wtime();
#endif

    ALLOC(_A, sizeof(double) * (MB + MR) * KB);
    ALLOC(_B, sizeof(double) * KB * (NB + NR));

#ifdef PLAY
    for (uint64_t i = 0; i < MB * KB; ++i)
        _A[i] = i;
    for (uint64_t i = 0; i < KB * NB; ++i)
        _B[i] = i;

    /*
        uint64_t cnt = ((m + MR - 1) / MR) * ((n + NR - 1) / NR) * ((k + KB - 1) / KB);
        for (uint64_t i = 0; i < cnt; ++i) {
            micro_kernel_8x24_ppc_anbp(KB, _A, _B, C, ldc, _A);
        }
        */

    for (uint64_t mi = 0; mi < mc; ++mi)
    {
        uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            // for (uint64_t i = 0; i < MB * KB; ++i) _A[i] = i;

            // packacc(mm, kk, A + mi * MB + ki * KB * lda, lda, _A);

            for (uint64_t ni = 0; ni < nc; ++ni)
            {
                uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

                // for (uint64_t i = 0; i < KB * NB; ++i) _B[i] = i;

                // packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);

                inner_kernel_ppc_anbp(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
            }
        }
    }
#else
    for (uint64_t mi = 0; mi < mc; ++mi)
    {
        uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            packacc(mm, kk, A + mi * MB + ki * KB * lda, lda, _A);

            for (uint64_t ni = 0; ni < nc; ++ni)
            {
                uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

#ifdef BDL
                inner_kernel_ppc_anbp(mm, nn, kk, _A, B + ki * KB + ni * NB * ldb, C + mi * MB + ni * NB * ldc, ldc);
#else
                packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);

                inner_kernel_ppc_anbp(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
#endif
            }
        }
    }
#endif

    FREE(_A, sizeof(double) * (MB + MR) * KB);
    FREE(_B, sizeof(double) * KB * (NB + NR));

#ifdef DEBUG
    const double end_time = omp_get_wtime();
    double *mem;
    mem = malloc(sizeof(double));
    *mem = (end_time - start_time);

    return (void *)mem;
#else
    return (void *)0;
#endif
}

uint64_t min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

void call_dgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    int64_t m,
    int64_t n,
    int64_t k,
    double alpha,
    const double *A,
    int64_t lda,
    const double *B,
    int64_t ldb,
    double beta,
    double *C,
    int64_t ldc)
{
    uint64_t is_C_row = (layout == CblasRowMajor ? 1 : 0);
    uint64_t is_A_row = (TransA == CblasTrans ? !is_C_row : is_C_row);
    uint64_t is_B_row = (TransB == CblasTrans ? !is_C_row : is_C_row);

    assert(is_A_row == 0);
    assert(is_B_row == 0);
    assert(is_C_row == 0);
    assert(k % 2 == 0);
    assert(alpha == -1.0);
    assert(beta == 1.0);

#ifdef PARALLEL_WITH_OMP
    uint64_t mc = (m + MB - 1) / MB;
    uint64_t mr = m % MB;
    uint64_t nc = (n + NB - 1) / NB;
    uint64_t nr = n % NB;
    uint64_t kc = (k + KB - 1) / KB;
    uint64_t kr = k % KB;

    double *_A;

    ALLOC(_A, sizeof(double) * (MB + MR) * KB);

    for (uint64_t mi = 0; mi < mc; ++mi)
    {
        uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            packacc(mm, kk, A + mi * MB + ki * KB * lda, lda, _A);

#pragma omp parallel num_threads(NT1)
            {
                double *_B;
                ALLOC(_B, sizeof(double) * KB * (NB + NR));

#pragma omp for
                for (uint64_t ni = 0; ni < nc; ++ni)
                {
                    uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

                    packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);

                    inner_kernel_ppc_anbp(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
                }

                FREE(_B, sizeof(double) * KB * (NB + NR));
            }
        }
    }

    FREE(_A, sizeof(double) * (MB + MR) * KB);
#else
    uint64_t mc = (m + MA - 1) / MA;
    uint64_t mr = m % MA;
    uint64_t nc = (n + NA - 1) / NA;
    uint64_t nr = n % NA;
    uint64_t kc = (k + KA - 1) / KA;
    uint64_t kr = k % KA;

    for (uint64_t mi = 0; mi < mc; ++mi)
    {
        uint64_t mm = (mi != mc - 1 || mr == 0) ? MA : mr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            uint64_t kk = (ki != kc - 1 || kr == 0) ? KA : kr;

            for (uint64_t ni = 0; ni < nc; ++ni)
            {
                uint64_t nn = (ni != nc - 1 || nr == 0) ? NA : nr;

                const uint64_t total_m_jobs = (mm + MR - 1) / MR;
                const uint64_t min_each_m_jobs = total_m_jobs / CM;
                const uint64_t rest_m_jobs = total_m_jobs % CM;

                const uint64_t total_n_jobs = (nn + NR - 1) / NR;
                const uint64_t min_each_n_jobs = total_n_jobs / CN;
                const uint64_t rest_n_jobs = total_n_jobs % CN;

                struct thread_info tinfo[TOTAL_CORE];
                cpu_set_t mask;

                pthread_attr_t attr;
                pthread_attr_init(&attr);

                for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid)
                {
                    const uint64_t m_pid = pid % CM;
                    const uint64_t n_pid = pid / CM;

                    const uint64_t my_m_idx_start = (m_pid)*min_each_m_jobs + min(m_pid, rest_m_jobs);
                    const uint64_t my_m_idx_end = (m_pid + 1) * min_each_m_jobs + min(m_pid + 1, rest_m_jobs);
                    const uint64_t my_m_start = min(my_m_idx_start * MR, mm);
                    const uint64_t my_m_end = min(my_m_idx_end * MR, mm);
                    const uint64_t my_m_size = my_m_end - my_m_start;

                    const uint64_t my_n_idx_start = (n_pid)*min_each_n_jobs + min(n_pid, rest_n_jobs);
                    const uint64_t my_n_idx_end = (n_pid + 1) * min_each_n_jobs + min(n_pid + 1, rest_n_jobs);
                    const uint64_t my_n_start = min(my_n_idx_start * NR, nn);
                    const uint64_t my_n_end = min(my_n_idx_end * NR, nn);
                    const uint64_t my_n_size = my_n_end - my_n_start;

                    const double *A_start = A + my_m_start;
                    const double *B_start = B + my_n_start * ldb;
                    double *C_start = C + my_m_start * 1 + my_n_start * ldc;

                    tinfo[pid].m = my_m_size;
                    tinfo[pid].n = my_n_size;
                    tinfo[pid].k = kk;
                    tinfo[pid].A = A_start;
                    tinfo[pid].lda = lda;
                    tinfo[pid].B = B_start;
                    tinfo[pid].ldb = ldb;
                    tinfo[pid].C = C_start;
                    tinfo[pid].ldc = ldc;

                    CPU_ZERO(&mask);
                    CPU_SET(pid, &mask);

                    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask);
                    pthread_create(&tinfo[pid].thread_id, &attr, &middle_kernel, &tinfo[pid]);
                }

                for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid)
                {
                    void *res;
                    pthread_join(tinfo[pid].thread_id, &res);
                }

                pthread_attr_destroy(&attr);
            }
        }
    }
#endif
}
