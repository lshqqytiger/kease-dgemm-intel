/**
 * @file kernel.01.oc.c
 * @author Enoch Jung
 * @brief dgemm for
 *        - core : 1 core
 *        - A     : ColMajor
 *        - B     : ColMajor
 *        - C     : ColMajor
 *        - k     : even number
 *        - alpha : -1.0
 *        - beta  : +1.0
 * @date 2023-10-24
 */

#include <assert.h>
#include <stdint.h>
#include <immintrin.h>

#include "cblas_format.h"
#include "common.h"

#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define MR 8
#define NR 24

#ifndef MB
#define MB (MR * 250) // [150] => [1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150]
#endif
#ifndef NB
#define NB (NR * 1) // [ 50] => [1, 2, 5, 10, 25, 50]
#endif
#ifndef KB
#define KB 64 // 64
#endif

#define LCAM

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
#ifndef LCAM
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
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd (%[C]), %%zmm24  \t\n"
        " vmovupd (%[C], %[ldc],1), %%zmm25  \t\n"
        " vmovupd (%[C], %[ldc],2), %%zmm26  \t\n"
        " vmovupd (%[C], %[ldc3],1), %%zmm27  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd (%[C]), %%zmm28  \t\n"
        " vmovupd (%[C], %[ldc],1), %%zmm29  \t\n"
#else
        " vpxord  %%zmm0,  %%zmm0, %%zmm0 \t\n"
        " vmovapd %%zmm0,  %%zmm1         \t\n"
        " vmovapd %%zmm0,  %%zmm2         \t\n"
        " vmovapd %%zmm0,  %%zmm3         \t\n"
#if NR > 4
        " vmovapd %%zmm0,  %%zmm4         \t\n"
#endif
#if NR > 5
        " vmovapd %%zmm0,  %%zmm5         \t\n"
#endif
#if NR > 6
        " vmovapd %%zmm0,  %%zmm6         \t\n"
#endif
#if NR > 7
        " vmovapd %%zmm0,  %%zmm7         \t\n"
#endif
#if NR > 8
        " vmovapd %%zmm0,  %%zmm8         \t\n"
#endif
#if NR > 9
        " vmovapd %%zmm0,  %%zmm9         \t\n"
#endif
#if NR > 10
        " vmovapd %%zmm0, %%zmm10         \t\n"
#endif
#if NR > 11
        " vmovapd %%zmm0, %%zmm11         \t\n"
#endif
#if NR > 12
        " vmovapd %%zmm0, %%zmm12         \t\n"
#endif
#if NR > 13
        " vmovapd %%zmm0, %%zmm13         \t\n"
#endif
#if NR > 14
        " vmovapd %%zmm0, %%zmm14         \t\n"
#endif
#if NR > 15
        " vmovapd %%zmm0, %%zmm15         \t\n"
#endif
#if NR > 16
        " vmovapd %%zmm0, %%zmm16         \t\n"
#endif
#if NR > 17
        " vmovapd %%zmm0, %%zmm17         \t\n"
#endif
#if NR > 18
        " vmovapd %%zmm0, %%zmm18         \t\n"
#endif
#if NR > 19
        " vmovapd %%zmm0, %%zmm19         \t\n"
#endif
#if NR > 20
        " vmovapd %%zmm0, %%zmm20         \t\n"
#endif
#if NR > 21
        " vmovapd %%zmm0, %%zmm21         \t\n"
#endif
#if NR > 22
        " vmovapd %%zmm0, %%zmm22         \t\n"
#endif
#if NR > 23
        " vmovapd %%zmm0, %%zmm23         \t\n"
#endif
#if NR > 24
        " vmovapd %%zmm0, %%zmm24         \t\n"
#endif
#if NR > 25
        " vmovapd %%zmm0, %%zmm25         \t\n"
#endif
#if NR > 26
        " vmovapd %%zmm0, %%zmm26         \t\n"
#endif
#if NR > 27
        " vmovapd %%zmm0, %%zmm27         \t\n"
#endif
#if NR > 28
        " vmovapd %%zmm0, %%zmm28         \t\n"
#endif
#if NR > 29
        " vmovapd %%zmm0, %%zmm29         \t\n"
#endif

        " prefetcht0 (%[C])                 \t\n"
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#if NR > 4
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
#endif
#if NR > 5
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
#endif
#if NR > 6
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
#endif
#if NR > 7
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#endif
#if NR > 8
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
#endif
#if NR > 9
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
#endif
#if NR > 10
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
#endif
#if NR > 11
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#endif
#if NR > 12
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
#endif
#if NR > 13
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
#endif
#if NR > 14
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
#endif
#if NR > 15
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#endif
#if NR > 16
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
#endif
#if NR > 17
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
#endif
#if NR > 18
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
#endif
#if NR > 19
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#endif
#if NR > 20
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
#endif
#if NR > 21
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
#endif
#if NR > 22
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
#endif
#if NR > 23
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#endif
#if NR > 24
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
#endif
#if NR > 25
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
#endif
#if NR > 26
        " prefetcht0 (%[C], %[ldc],2)       \t\n"
#endif
#if NR > 27
        " prefetcht0 (%[C],%[ldc3],1)       \t\n"
#endif
#if NR > 28
        " lea        (%[C], %[ldc],4), %[C] \t\n"
        " prefetcht0 (%[C])                 \t\n"
#endif
#if NR > 29
        " prefetcht0 (%[C], %[ldc],1)       \t\n"
#endif
#endif
        : [C] "+r"(tmp_C)
        : [ldc] "r"(ldc * 8), [ldc3] "r"(ldc * 8 * 3), [A] "r"(_A)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31");

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
#if NR > 4
            " vfnmadd231pd  0x20(%[B])%{1to8}, %%zmm31,  %%zmm4 \t\n"
#endif
#if NR > 5
            " vfnmadd231pd  0x28(%[B])%{1to8}, %%zmm31,  %%zmm5 \t\n"
#endif
#if NR > 6
            " vfnmadd231pd  0x30(%[B])%{1to8}, %%zmm31,  %%zmm6 \t\n"
#endif
#if NR > 7
            " vfnmadd231pd  0x38(%[B])%{1to8}, %%zmm31,  %%zmm7 \t\n"
#endif
#if NR > 8
            " vfnmadd231pd  0x40(%[B])%{1to8}, %%zmm31,  %%zmm8 \t\n"
#endif
#if NR > 9
            " vfnmadd231pd  0x48(%[B])%{1to8}, %%zmm31,  %%zmm9 \t\n"
#endif
#if NR > 10
            " vfnmadd231pd  0x50(%[B])%{1to8}, %%zmm31, %%zmm10 \t\n"
#endif
#if NR > 11
            " vfnmadd231pd  0x58(%[B])%{1to8}, %%zmm31, %%zmm11 \t\n"
#endif
#if NR > 12
            " vfnmadd231pd  0x60(%[B])%{1to8}, %%zmm31, %%zmm12 \t\n"
#endif
#if NR > 13
            " vfnmadd231pd  0x68(%[B])%{1to8}, %%zmm31, %%zmm13 \t\n"
#endif
#if NR > 14
            " vfnmadd231pd  0x70(%[B])%{1to8}, %%zmm31, %%zmm14 \t\n"
#endif
#if NR > 15
            " vfnmadd231pd  0x78(%[B])%{1to8}, %%zmm31, %%zmm15 \t\n"
#endif
#if NR > 16
            " vfnmadd231pd  0x80(%[B])%{1to8}, %%zmm31, %%zmm16 \t\n"
#endif
#if NR > 17
            " vfnmadd231pd  0x88(%[B])%{1to8}, %%zmm31, %%zmm17 \t\n"
#endif
#if NR > 18
            " vfnmadd231pd  0x90(%[B])%{1to8}, %%zmm31, %%zmm18 \t\n"
#endif
#if NR > 19
            " vfnmadd231pd  0x98(%[B])%{1to8}, %%zmm31, %%zmm19 \t\n"
#endif
#if NR > 20
            " vfnmadd231pd  0xa0(%[B])%{1to8}, %%zmm31, %%zmm20 \t\n"
#endif
#if NR > 21
            " vfnmadd231pd  0xa8(%[B])%{1to8}, %%zmm31, %%zmm21 \t\n"
#endif
#if NR > 22
            " vfnmadd231pd  0xb0(%[B])%{1to8}, %%zmm31, %%zmm22 \t\n"
#endif
#if NR > 23
            " vfnmadd231pd  0xb8(%[B])%{1to8}, %%zmm31, %%zmm23 \t\n"
#endif
#if NR > 24
            " vfnmadd231pd  0xc0(%[B])%{1to8}, %%zmm31, %%zmm24 \t\n"
#endif
#if NR > 25
            " vfnmadd231pd  0xc8(%[B])%{1to8}, %%zmm31, %%zmm25 \t\n"
#endif
#if NR > 26
            " vfnmadd231pd  0xd0(%[B])%{1to8}, %%zmm31, %%zmm26 \t\n"
#endif
#if NR > 27
            " vfnmadd231pd  0xd8(%[B])%{1to8}, %%zmm31, %%zmm27 \t\n"
#endif
#if NR > 28
            " vfnmadd231pd  0xe0(%[B])%{1to8}, %%zmm31, %%zmm28 \t\n"
#endif
#if NR > 29
            " vfnmadd231pd  0xe8(%[B])%{1to8}, %%zmm31, %%zmm29 \t\n"
#endif

            " prefetcht0   0x4c0(%[A])                          \t\n"
            " vmovapd       0x80(%[A]),        %%zmm31          \t\n"
            " vfnmadd231pd  0xc0(%[B])%{1to8}, %%zmm30,  %%zmm0 \t\n"
            " vfnmadd231pd  0xc8(%[B])%{1to8}, %%zmm30,  %%zmm1 \t\n"
            " vfnmadd231pd  0xd0(%[B])%{1to8}, %%zmm30,  %%zmm2 \t\n"
            " vfnmadd231pd  0xd8(%[B])%{1to8}, %%zmm30,  %%zmm3 \t\n"
#if NR > 4
            " vfnmadd231pd  0xe0(%[B])%{1to8}, %%zmm30,  %%zmm4 \t\n"
#endif
#if NR > 5
            " vfnmadd231pd  0xe8(%[B])%{1to8}, %%zmm30,  %%zmm5 \t\n"
#endif
#if NR > 6
            " vfnmadd231pd  0xf0(%[B])%{1to8}, %%zmm30,  %%zmm6 \t\n"
#endif
#if NR > 7
            " vfnmadd231pd  0xf8(%[B])%{1to8}, %%zmm30,  %%zmm7 \t\n"
#endif
#if NR > 8
            " vfnmadd231pd 0x100(%[B])%{1to8}, %%zmm30,  %%zmm8 \t\n"
#endif
#if NR > 9
            " vfnmadd231pd 0x108(%[B])%{1to8}, %%zmm30,  %%zmm9 \t\n"
#endif
#if NR > 10
            " vfnmadd231pd 0x110(%[B])%{1to8}, %%zmm30, %%zmm10 \t\n"
#endif
#if NR > 11
            " vfnmadd231pd 0x118(%[B])%{1to8}, %%zmm30, %%zmm11 \t\n"
#endif
#if NR > 12
            " vfnmadd231pd 0x120(%[B])%{1to8}, %%zmm30, %%zmm12 \t\n"
#endif
#if NR > 13
            " vfnmadd231pd 0x128(%[B])%{1to8}, %%zmm30, %%zmm13 \t\n"
#endif
#if NR > 14
            " vfnmadd231pd 0x130(%[B])%{1to8}, %%zmm30, %%zmm14 \t\n"
#endif
#if NR > 15
            " vfnmadd231pd 0x138(%[B])%{1to8}, %%zmm30, %%zmm15 \t\n"
#endif
#if NR > 16
            " vfnmadd231pd 0x140(%[B])%{1to8}, %%zmm30, %%zmm16 \t\n"
#endif
#if NR > 17
            " vfnmadd231pd 0x148(%[B])%{1to8}, %%zmm30, %%zmm17 \t\n"
#endif
#if NR > 18
            " vfnmadd231pd 0x150(%[B])%{1to8}, %%zmm30, %%zmm18 \t\n"
#endif
#if NR > 19
            " vfnmadd231pd 0x158(%[B])%{1to8}, %%zmm30, %%zmm19 \t\n"
#endif
#if NR > 20
            " vfnmadd231pd 0x160(%[B])%{1to8}, %%zmm30, %%zmm20 \t\n"
#endif
#if NR > 21
            " vfnmadd231pd 0x168(%[B])%{1to8}, %%zmm30, %%zmm21 \t\n"
#endif
#if NR > 22
            " vfnmadd231pd 0x170(%[B])%{1to8}, %%zmm30, %%zmm22 \t\n"
#endif
#if NR > 23
            " vfnmadd231pd 0x178(%[B])%{1to8}, %%zmm30, %%zmm23 \t\n"
#endif
#if NR > 24
            " vfnmadd231pd 0x180(%[B])%{1to8}, %%zmm30, %%zmm24 \t\n"
#endif
#if NR > 25
            " vfnmadd231pd 0x188(%[B])%{1to8}, %%zmm30, %%zmm25 \t\n"
#endif
#if NR > 26
            " vfnmadd231pd 0x190(%[B])%{1to8}, %%zmm30, %%zmm26 \t\n"
#endif
#if NR > 27
            " vfnmadd231pd 0x198(%[B])%{1to8}, %%zmm30, %%zmm27 \t\n"
#endif
#if NR > 28
            " vfnmadd231pd 0x1a0(%[B])%{1to8}, %%zmm30, %%zmm28 \t\n"
#endif
#if NR > 29
            " vfnmadd231pd 0x1a8(%[B])%{1to8}, %%zmm30, %%zmm29 \t\n"
#endif

            " add  $0x80, %[A] \t\n"
#if NR > 29
            " add $0x1e0, %[B] \t\n"
#elif NR > 28
            " add $0x1d0, %[B] \t\n"
#elif NR > 27
            " add $0x1c0, %[B] \t\n"
#elif NR > 26
            " add $0x1b0, %[B] \t\n"
#elif NR > 25
            " add $0x1a0, %[B] \t\n"
#elif NR > 24
            " add $0x190, %[B] \t\n"
#elif NR > 23
            " add $0x180, %[B] \t\n"
#elif NR > 22
            " add $0x170, %[B] \t\n"
#elif NR > 21
            " add $0x160, %[B] \t\n"
#elif NR > 20
            " add $0x150, %[B] \t\n"
#elif NR > 19
            " add $0x140, %[B] \t\n"
#elif NR > 18
            " add $0x130, %[B] \t\n"
#elif NR > 17
            " add $0x120, %[B] \t\n"
#elif NR > 16
            " add $0x110, %[B] \t\n"
#elif NR > 15
            " add $0x100, %[B] \t\n"
#elif NR > 14
            " add  $0xf0, %[B] \t\n"
#elif NR > 13
            " add  $0xe0, %[B] \t\n"
#elif NR > 12
            " add  $0xd0, %[B] \t\n"
#elif NR > 11
            " add  $0xc0, %[B] \t\n"
#elif NR > 10
            " add  $0xb0, %[B] \t\n"
#elif NR > 9
            " add  $0xa0, %[B] \t\n"
#elif NR > 8
            " add  $0x90, %[B] \t\n"
#elif NR > 7
            " add  $0x80, %[B] \t\n"
#elif NR > 6
            " add  $0x70, %[B] \t\n"
#elif NR > 5
            " add  $0x60, %[B] \t\n"
#elif NR > 4
            " add  $0x50, %[B] \t\n"
#else
            " add  $0x40, %[B] \t\n"
#endif
            : [A] "+r"(_A), [B] "+r"(_B)
            :
            : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
              "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
              "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
              "zmm30", "zmm31");
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
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd %%zmm24, (%[C])  \t\n"
        " vmovupd %%zmm25, (%[C], %[ldc],1)  \t\n"
        " vmovupd %%zmm26, (%[C], %[ldc],2)  \t\n"
        " vmovupd %%zmm27, (%[C], %[ldc3],1)  \t\n"
        " lea (%[C], %[ldc],4), %[C] \t\n"
        " vmovupd %%zmm28, (%[C])  \t\n"
        " vmovupd %%zmm29, (%[C], %[ldc],1)  \t\n"
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

#if NR > 4
        " lea     (%[C], %[ldc],4), %[C]             \t\n"
        " vaddpd  (%[C]),            %%zmm4,  %%zmm4 \t\n"
#endif
#if NR > 5
        " vaddpd  (%[C], %[ldc],1),  %%zmm5,  %%zmm5 \t\n"
#endif
#if NR > 6
        " vaddpd  (%[C], %[ldc],2),  %%zmm6,  %%zmm6 \t\n"
#endif
#if NR > 7
        " vaddpd  (%[C],%[ldc3],1),  %%zmm7,  %%zmm7 \t\n"
#endif
#if NR > 4
        " vmovupd  %%zmm4, (%[C])                    \t\n"
#endif
#if NR > 5
        " vmovupd  %%zmm5, (%[C], %[ldc],1)          \t\n"
#endif
#if NR > 6
        " vmovupd  %%zmm6, (%[C], %[ldc],2)          \t\n"
#endif
#if NR > 7
        " vmovupd  %%zmm7, (%[C],%[ldc3],1)          \t\n"
#endif

#if NR > 8
        " lea     (%[C], %[ldc],4), %[C]             \t\n"
        " vaddpd  (%[C]),            %%zmm8,  %%zmm8 \t\n"
#endif
#if NR > 9
        " vaddpd  (%[C], %[ldc],1),  %%zmm9,  %%zmm9 \t\n"
#endif
#if NR > 10
        " vaddpd  (%[C], %[ldc],2), %%zmm10, %%zmm10 \t\n"
#endif
#if NR > 11
        " vaddpd  (%[C],%[ldc3],1), %%zmm11, %%zmm11 \t\n"
#endif
#if NR > 8
        " vmovupd  %%zmm8, (%[C])                    \t\n"
#endif
#if NR > 9
        " vmovupd  %%zmm9, (%[C], %[ldc],1)          \t\n"
#endif
#if NR > 10
        " vmovupd %%zmm10, (%[C], %[ldc],2)          \t\n"
#endif
#if NR > 11
        " vmovupd %%zmm11, (%[C],%[ldc3],1)          \t\n"
#endif

#if NR > 12
        " lea     (%[C], %[ldc],4), %[C]             \t\n"
        " vaddpd  (%[C]),           %%zmm12, %%zmm12 \t\n"
#endif
#if NR > 13
        " vaddpd  (%[C], %[ldc],1), %%zmm13, %%zmm13 \t\n"
#endif
#if NR > 14
        " vaddpd  (%[C], %[ldc],2), %%zmm14, %%zmm14 \t\n"
#endif
#if NR > 15
        " vaddpd  (%[C],%[ldc3],1), %%zmm15, %%zmm15 \t\n"
#endif
#if NR > 12
        " vmovupd %%zmm12, (%[C])                    \t\n"
#endif
#if NR > 13
        " vmovupd %%zmm13, (%[C], %[ldc],1)          \t\n"
#endif
#if NR > 14
        " vmovupd %%zmm14, (%[C], %[ldc],2)          \t\n"
#endif
#if NR > 15
        " vmovupd %%zmm15, (%[C],%[ldc3],1)          \t\n"
#endif

#if NR > 16
        " lea     (%[C], %[ldc],4), %[C]             \t\n"
        " vaddpd  (%[C]),           %%zmm16, %%zmm16 \t\n"
#endif
#if NR > 17
        " vaddpd  (%[C], %[ldc],1), %%zmm17, %%zmm17 \t\n"
#endif
#if NR > 18
        " vaddpd  (%[C], %[ldc],2), %%zmm18, %%zmm18 \t\n"
#endif
#if NR > 19
        " vaddpd  (%[C],%[ldc3],1), %%zmm19, %%zmm19 \t\n"
#endif
#if NR > 16
        " vmovupd %%zmm16, (%[C])                    \t\n"
#endif
#if NR > 17
        " vmovupd %%zmm17, (%[C], %[ldc],1)          \t\n"
#endif
#if NR > 18
        " vmovupd %%zmm18, (%[C], %[ldc],2)          \t\n"
#endif
#if NR > 19
        " vmovupd %%zmm19, (%[C],%[ldc3],1)          \t\n"
#endif

#if NR > 20
        " lea     (%[C], %[ldc],4), %[C]             \t\n"
        " vaddpd  (%[C]),           %%zmm20, %%zmm20 \t\n"
#endif
#if NR > 21
        " vaddpd  (%[C], %[ldc],1), %%zmm21, %%zmm21 \t\n"
#endif
#if NR > 22
        " vaddpd  (%[C], %[ldc],2), %%zmm22, %%zmm22 \t\n"
#endif
#if NR > 23
        " vaddpd  (%[C],%[ldc3],1), %%zmm23, %%zmm23 \t\n"
#endif
#if NR > 20
        " vmovupd %%zmm20, (%[C])                    \t\n"
#endif
#if NR > 21
        " vmovupd %%zmm21, (%[C], %[ldc],1)          \t\n"
#endif
#if NR > 22
        " vmovupd %%zmm22, (%[C], %[ldc],2)          \t\n"
#endif
#if NR > 23
        " vmovupd %%zmm23, (%[C],%[ldc3],1)          \t\n"
#endif

#if NR > 24
        " lea     (%[C], %[ldc],4), %[C]             \t\n"
        " vaddpd  (%[C]),           %%zmm24, %%zmm24 \t\n"
#endif
#if NR > 25
        " vaddpd  (%[C], %[ldc],1), %%zmm25, %%zmm25 \t\n"
#endif
#if NR > 26
        " vaddpd  (%[C], %[ldc],2), %%zmm26, %%zmm26 \t\n"
#endif
#if NR > 27
        " vaddpd  (%[C],%[ldc3],1), %%zmm27, %%zmm27 \t\n"
#endif
#if NR > 24
        " vmovupd %%zmm24, (%[C])                    \t\n"
#endif
#if NR > 25
        " vmovupd %%zmm25, (%[C], %[ldc],1)          \t\n"
#endif
#if NR > 26
        " vmovupd %%zmm26, (%[C], %[ldc],2)          \t\n"
#endif
#if NR > 27
        " vmovupd %%zmm27, (%[C],%[ldc3],1)          \t\n"
#endif

#if NR > 28
        " lea     (%[C], %[ldc],4), %[C]             \t\n"
        " vaddpd  (%[C]),           %%zmm28, %%zmm28 \t\n"
#endif
#if NR > 29
        " vaddpd  (%[C], %[ldc],1), %%zmm29, %%zmm29 \t\n"
#endif
#if NR > 28
        " vmovupd %%zmm28, (%[C])                    \t\n"
#endif
#if NR > 29
        " vmovupd %%zmm29, (%[C], %[ldc],1)          \t\n"
#endif
#endif
        : [C] "+r"(C)
        : [ldc] "r"(ldc * 8), [ldc3] "r"(ldc * 8 * 3), [A_next] "r"(_A_next)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31");
}

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

// #define INNER_MN

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

#ifdef INNER_MN
    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

        A_now = A_next;
        A_next = mmi != mmc - 1 ? A_next + MR * kk : _A;

        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

            B_now = B_next;
            B_next = nni != nnc - 1 ? B_next + NR * kk : _B;
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
        }
    }
}

// #define PACKACC_M_FIRST

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

#ifdef PACKACC_M_FIRST
    const double *A_now = A;
    const double *A_m_next = A;
    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

        A_now = A_m_next;
        A_m_next += MR;

        /*
                register const double* tmp_A = A_m_next;

                asm volatile(
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                " lea        (%[A], %[lda],4), %[A] \t\n"
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                " lea        (%[A], %[lda],4), %[A] \t\n"
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                " lea        (%[A], %[lda],4), %[A] \t\n"
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                " lea        (%[A], %[lda],4), %[A] \t\n"
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                " lea        (%[A], %[lda],4), %[A] \t\n"
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                " lea        (%[A], %[lda],4), %[A] \t\n"
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                " lea        (%[A], %[lda],4), %[A] \t\n"
                " prefetcht0 (%[A])                 \t\n"
                " prefetcht0 (%[A], %[lda],1)       \t\n"
                " prefetcht0 (%[A], %[lda],2)       \t\n"
                " prefetcht0 (%[A],%[lda3],1)       \t\n"
                : [A]"+r"(tmp_A)
                : [lda]"r"(lda*8), [lda3]"r"(lda*8*3)
                );
                */

        for (uint64_t kki = 0; kki < kkc; ++kki)
        {
            uint64_t kkk = (kki != kkc - 1 || kkr == 0) ? CACHE_ELEM : kkr;
#else
    const double *A_now = A;
    const double *A_k_next = A;

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

        /*
                register const double* _A_k_next = _A + kki * MR * CACHE_ELEM;
                asm volatile(
                " prefetcht0      (%[A0])            \t\n"
                " prefetcht0  0x40(%[A0])            \t\n"
                " prefetcht0  0x80(%[A0])            \t\n"
                " prefetcht0  0xc0(%[A0])            \t\n"
                " prefetcht0 0x100(%[A0])            \t\n"
                " prefetcht0 0x140(%[A0])            \t\n"
                " prefetcht0 0x180(%[A0])            \t\n"
                " prefetcht0 0x1c0(%[A0])            \t\n"
                " prefetcht0      (%[A1])            \t\n"
                " prefetcht0  0x40(%[A1])            \t\n"
                " prefetcht0  0x80(%[A1])            \t\n"
                " prefetcht0  0xc0(%[A1])            \t\n"
                " prefetcht0 0x100(%[A1])            \t\n"
                " prefetcht0 0x140(%[A1])            \t\n"
                " prefetcht0 0x180(%[A1])            \t\n"
                " prefetcht0 0x1c0(%[A1])            \t\n"
                " prefetcht0      (%[A2])            \t\n"
                " prefetcht0  0x40(%[A2])            \t\n"
                " prefetcht0  0x80(%[A2])            \t\n"
                " prefetcht0  0xc0(%[A2])            \t\n"
                " prefetcht0 0x100(%[A2])            \t\n"
                " prefetcht0 0x140(%[A2])            \t\n"
                " prefetcht0 0x180(%[A2])            \t\n"
                " prefetcht0 0x1c0(%[A2])            \t\n"
                " prefetcht0      (%[A3])            \t\n"
                " prefetcht0  0x40(%[A3])            \t\n"
                " prefetcht0  0x80(%[A3])            \t\n"
                " prefetcht0  0xc0(%[A3])            \t\n"
                " prefetcht0 0x100(%[A3])            \t\n"
                " prefetcht0 0x140(%[A3])            \t\n"
                " prefetcht0 0x180(%[A3])            \t\n"
                " prefetcht0 0x1c0(%[A3])            \t\n"
                :
                : [A0]"r"(_A_k_next), [A1]"r"(_A_k_next+MR*kk), [A2]"r"(_A_k_next+MR*kk*2), [A3]"r"(_A_k_next+MR*kk*3)
                );
                */

        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;
#endif

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

#ifdef PACKACC_M_FIRST
            A_now += CACHE_ELEM * lda;
#else
            A_now += MR;
#endif
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

// #define ORIGIN_PACKBCR

void packbcr(
    uint64_t kk,
    uint64_t nn,
    const double *restrict B,
    uint64_t ldb,
    double *restrict _B)
{
    uint64_t nnc = (nn + NR - 1) / NR;
    uint64_t nnr = nn % NR;

#ifdef ORIGIN_PACKBCR
    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
        for (uint64_t i = 0; i < nnn; ++i)
        {
            for (uint64_t j = 0; j < kk; ++j)
            {
                _B[nni * NR * kk + i + j * NR] = B[nni * NR * ldb + i * ldb + j];
            }
        }
    }
#else
    uint64_t kkc = (kk + 7) / 8;
    uint64_t kkr = kk % 8;
    for (uint64_t j = 0; j < kkc; ++j)
    {
        uint64_t kkk = (j != kkc - 1 || kkr == 0) ? 8 : kkr;
        /*
        const double* B_k_next = B + (j + 6) * 8;
        asm volatile(
        " prefetchnta (%[B0])            \t\n"
        " prefetchnta (%[B0], %[ldb],1)  \t\n"
        " prefetchnta (%[B0], %[ldb],2)  \t\n"
        " prefetchnta (%[B0],%[ldb3],1)  \t\n"
        " prefetchnta (%[B4])            \t\n"
        " prefetchnta (%[B4], %[ldb],1)  \t\n"
        " prefetchnta (%[B4], %[ldb],2)  \t\n"
        " prefetchnta (%[B4],%[ldb3],1)  \t\n"
        " prefetchnta (%[B8])            \t\n"
        " prefetchnta (%[B8], %[ldb],1)  \t\n"
        " prefetchnta (%[B8], %[ldb],2)  \t\n"
        " prefetchnta (%[B8],%[ldb3],1)  \t\n"
        " prefetchnta (%[B12])            \t\n"
        " prefetchnta (%[B12], %[ldb],1)  \t\n"
        " prefetchnta (%[B12], %[ldb],2)  \t\n"
        " prefetchnta (%[B12],%[ldb3],1)  \t\n"
        " prefetchnta (%[B16])            \t\n"
        " prefetchnta (%[B16], %[ldb],1)  \t\n"
        " prefetchnta (%[B16], %[ldb],2)  \t\n"
        " prefetchnta (%[B16],%[ldb3],1)  \t\n"
        " prefetchnta (%[B20])            \t\n"
        " prefetchnta (%[B20], %[ldb],1)  \t\n"
        " prefetchnta (%[B20], %[ldb],2)  \t\n"
        " prefetchnta (%[B20],%[ldb3],1)  \t\n"
        :
        : [B0]"r"(B_k_next), [B4]"r"(B_k_next+ldb*4), [B8]"r"(B_k_next+ldb*8), [B12]"r"(B_k_next+ldb*12), [B16]"r"(B_k_next+ldb*16), [B20]"r"(B_k_next+ldb*20), [ldb]"r"(ldb), [ldb3]"r"(ldb*3)
        );
        */

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
#endif
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

    uint64_t mc = (m + MB - 1) / MB;
    uint64_t mr = m % MB;
    uint64_t nc = (n + NB - 1) / NB;
    uint64_t nr = n % NB;
    uint64_t kc = (k + KB - 1) / KB;
    uint64_t kr = k % KB;

    double *_A = numa_alloc(sizeof(double) * (MB + MR) * KB);
    double *_B = numa_alloc(sizeof(double) * KB * NB);

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

                packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);

                inner_kernel_ppc_anbp(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
            }
        }
    }

    numa_free(_A, sizeof(double) * (MB + MR) * KB);
    numa_free(_B, sizeof(double) * KB * NB);
}
