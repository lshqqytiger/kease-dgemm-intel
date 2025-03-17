#include <stdint.h>

#include "cblas_format.h"
#include "common.h"

#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define MR 8
#define NR 24
#define MB (MR * 150) // [150] => [1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150]
#define NB (NR * 1)   // [ 50] => [1, 2, 5, 10, 25, 50]
#define KB 1200

/*
static void micro_kernel_8x24_ppc_an(uint64_t k) {
#pragma unroll(16)
    for (uint64_t i = 0; LIKELY(i < k); ++i) {
        asm volatile(
        " vfnmadd231pd   %%zmm0, %%zmm31,  %%zmm0 \t\n"
        " vfnmadd231pd   %%zmm1, %%zmm31,  %%zmm1 \t\n"
        " vfnmadd231pd   %%zmm2, %%zmm31,  %%zmm2 \t\n"
        " vfnmadd231pd   %%zmm3, %%zmm31,  %%zmm3 \t\n"
        " vfnmadd231pd   %%zmm4, %%zmm31,  %%zmm4 \t\n"
        " vfnmadd231pd   %%zmm5, %%zmm31,  %%zmm5 \t\n"
        " vfnmadd231pd   %%zmm6, %%zmm31,  %%zmm6 \t\n"
        " vfnmadd231pd   %%zmm7, %%zmm31,  %%zmm7 \t\n"
        " vfnmadd231pd   %%zmm8, %%zmm31,  %%zmm8 \t\n"
        " vfnmadd231pd   %%zmm9, %%zmm31,  %%zmm9 \t\n"
        " vfnmadd231pd  %%zmm10, %%zmm31, %%zmm10 \t\n"
        " vfnmadd231pd  %%zmm11, %%zmm31, %%zmm11 \t\n"
        " vfnmadd231pd  %%zmm12, %%zmm31, %%zmm12 \t\n"
        " vfnmadd231pd  %%zmm13, %%zmm31, %%zmm13 \t\n"
        " vfnmadd231pd  %%zmm14, %%zmm31, %%zmm14 \t\n"
        " vfnmadd231pd  %%zmm15, %%zmm31, %%zmm15 \t\n"
        " vfnmadd231pd  %%zmm16, %%zmm31, %%zmm16 \t\n"
        " vfnmadd231pd  %%zmm17, %%zmm31, %%zmm17 \t\n"
        " vfnmadd231pd  %%zmm18, %%zmm31, %%zmm18 \t\n"
        " vfnmadd231pd  %%zmm19, %%zmm31, %%zmm19 \t\n"
        " vfnmadd231pd  %%zmm20, %%zmm31, %%zmm20 \t\n"
        " vfnmadd231pd  %%zmm21, %%zmm31, %%zmm21 \t\n"
        " vfnmadd231pd  %%zmm22, %%zmm31, %%zmm22 \t\n"
        " vfnmadd231pd  %%zmm23, %%zmm31, %%zmm23 \t\n"
        :
        :
        :  "zmm0",  "zmm1",  "zmm2",  "zmm3",  "zmm4",  "zmm5",  "zmm6",  "zmm7",  "zmm8",  "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23"
        );
    }
}
*/

void micro_kernel_8x24_ppc_an(
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    uint64_t ldc)
{
    register double *tmp_C = C;

    asm volatile(
        " vmovaps (%[A]), %%zmm31         \t\n"
        " vpxord  %%zmm0,  %%zmm0, %%zmm0 \t\n"
        " vmovaps %%zmm0,  %%zmm1         \t\n"
        " vmovaps %%zmm0,  %%zmm2         \t\n"
        " vmovaps %%zmm0,  %%zmm3         \t\n"
        " vmovaps %%zmm0,  %%zmm4         \t\n"
        " vmovaps %%zmm0,  %%zmm5         \t\n"
        " vmovaps %%zmm0,  %%zmm6         \t\n"
        " vmovaps %%zmm0,  %%zmm7         \t\n"
        " vmovaps %%zmm0,  %%zmm8         \t\n"
        " vmovaps %%zmm0,  %%zmm9         \t\n"
        " vmovaps %%zmm0, %%zmm10         \t\n"
        " vmovaps %%zmm0, %%zmm11         \t\n"
        " vmovaps %%zmm0, %%zmm12         \t\n"
        " vmovaps %%zmm0, %%zmm13         \t\n"
        " vmovaps %%zmm0, %%zmm14         \t\n"
        " vmovaps %%zmm0, %%zmm15         \t\n"
        " vmovaps %%zmm0, %%zmm16         \t\n"
        " vmovaps %%zmm0, %%zmm17         \t\n"
        " vmovaps %%zmm0, %%zmm18         \t\n"
        " vmovaps %%zmm0, %%zmm19         \t\n"
        " vmovaps %%zmm0, %%zmm20         \t\n"
        " vmovaps %%zmm0, %%zmm21         \t\n"
        " vmovaps %%zmm0, %%zmm22         \t\n"
        " vmovaps %%zmm0, %%zmm23         \t\n"

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
#pragma unroll(2)
    for (uint64_t i = 0; LIKELY(i < kk); ++i)
    {
        asm volatile(

            " vmovaps       0x40(%[A]),        %%zmm30          \t\n"
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

            " vmovaps       0x80(%[A]),        %%zmm31          \t\n"
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
        " vaddpd  (%[C]),            %%zmm0,  %%zmm0 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm1,  %%zmm1 \t\n"
        " vaddpd  (%[C], %[ldc],2),  %%zmm2,  %%zmm2 \t\n"
        " vaddpd  (%[C],%[ldc3],1),  %%zmm3,  %%zmm3 \t\n"
        " vmovups  %%zmm0, (%[C])                    \t\n"
        " vmovups  %%zmm1, (%[C], %[ldc],1)          \t\n"
        " vmovups  %%zmm2, (%[C], %[ldc],2)          \t\n"
        " vmovups  %%zmm3, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),            %%zmm4,  %%zmm4 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm5,  %%zmm5 \t\n"
        " vaddpd  (%[C], %[ldc],2),  %%zmm6,  %%zmm6 \t\n"
        " vaddpd  (%[C],%[ldc3],1),  %%zmm7,  %%zmm7 \t\n"
        " vmovups  %%zmm4, (%[C])                    \t\n"
        " vmovups  %%zmm5, (%[C], %[ldc],1)          \t\n"
        " vmovups  %%zmm6, (%[C], %[ldc],2)          \t\n"
        " vmovups  %%zmm7, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),            %%zmm8,  %%zmm8 \t\n"
        " vaddpd  (%[C], %[ldc],1),  %%zmm9,  %%zmm9 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm10, %%zmm10 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm11, %%zmm11 \t\n"
        " vmovups  %%zmm8, (%[C])                    \t\n"
        " vmovups  %%zmm9, (%[C], %[ldc],1)          \t\n"
        " vmovups %%zmm10, (%[C], %[ldc],2)          \t\n"
        " vmovups %%zmm11, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm12, %%zmm12 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm13, %%zmm13 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm14, %%zmm14 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm15, %%zmm15 \t\n"
        " vmovups %%zmm12, (%[C])                    \t\n"
        " vmovups %%zmm13, (%[C], %[ldc],1)          \t\n"
        " vmovups %%zmm14, (%[C], %[ldc],2)          \t\n"
        " vmovups %%zmm15, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm16, %%zmm16 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm17, %%zmm17 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm18, %%zmm18 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm19, %%zmm19 \t\n"
        " vmovups %%zmm16, (%[C])                    \t\n"
        " vmovups %%zmm17, (%[C], %[ldc],1)          \t\n"
        " vmovups %%zmm18, (%[C], %[ldc],2)          \t\n"
        " vmovups %%zmm19, (%[C],%[ldc3],1)          \t\n"
        " lea     (%[C], %[ldc],4), %[C]             \t\n"

        " vaddpd  (%[C]),           %%zmm20, %%zmm20 \t\n"
        " vaddpd  (%[C], %[ldc],1), %%zmm21, %%zmm21 \t\n"
        " vaddpd  (%[C], %[ldc],2), %%zmm22, %%zmm22 \t\n"
        " vaddpd  (%[C],%[ldc3],1), %%zmm23, %%zmm23 \t\n"
        " vmovups %%zmm20, (%[C])                    \t\n"
        " vmovups %%zmm21, (%[C], %[ldc],1)          \t\n"
        " vmovups %%zmm22, (%[C], %[ldc],2)          \t\n"
        " vmovups %%zmm23, (%[C],%[ldc3],1)          \t\n"
        : [C] "+r"(C)
        : [ldc] "r"(ldc * 8), [ldc3] "r"(ldc * 8 * 3)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9",
          "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23");
}

void call_dgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    const int64_t m_,
    const int64_t n_,
    const int64_t k_,
    const double alpha,
    const double *A,
    const int64_t lda_,
    const double *B,
    const int64_t ldb_,
    const double beta,
    double *C,
    const int64_t ldc_)
{
    const uint64_t cnt = (m_ / MR) * (n_ / NR) * (k_ / KB);

    double *_A = NULL;
    double *_B = NULL;

    ALLOC(_A, sizeof(double) * (MB + MR) * KB);
    ALLOC(_B, sizeof(double) * KB * NB);

    for (uint64_t i = 0; LIKELY(i < cnt); ++i)
    {
        micro_kernel_8x24_ppc_an(KB, A, _B, C, MR);
    }

    FREE(_A, sizeof(double) * (MB + MR) * KB);
    FREE(_B, sizeof(double) * KB * NB);
}
