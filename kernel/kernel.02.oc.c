/**
 * @file kernel.02.oc.c
 * @author Enoch Jung
 * @brief dgemm for
 *        - core : 1 core
 *        - A     : ColMajor
 *        - B     : ColMajor
 *        - C     : ColMajor
 *        - k     : divisible by 2
 *        - alpha : -1.0
 *        - beta  : +1.0
 * @date 2023-10-25
 */

#include <assert.h>
#include <stdint.h>

#include "cblas_format.h"
#include "common.h"

#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define MR 8
#define NR 30
#define MB (MR * 90)
#define NB (NR * 1)
#define KB 60

void micro_kernel_8x30_ppc_anbp(
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict _C,
    const double *restrict _A_next)
{
    asm volatile(
        " prefetchnta (%[A_next])           \t\n"
        " vmovapd          (%[A]), %%zmm30  \t\n"
        " vpxord  %%zmm0,  %%zmm0,  %%zmm0  \t\n"
        " vmovapd %%zmm0,  %%zmm1           \t\n"
        " vmovapd %%zmm0,  %%zmm2           \t\n"
        " vmovapd %%zmm0,  %%zmm3           \t\n"
        " vmovapd %%zmm0,  %%zmm4           \t\n"
        " vmovapd %%zmm0,  %%zmm5           \t\n"
        " vmovapd %%zmm0,  %%zmm6           \t\n"
        " vmovapd %%zmm0,  %%zmm7           \t\n"
        " vmovapd %%zmm0,  %%zmm8           \t\n"
        " vmovapd %%zmm0,  %%zmm9           \t\n"
        " vmovapd %%zmm0, %%zmm10           \t\n"
        " vmovapd %%zmm0, %%zmm11           \t\n"
        " vmovapd %%zmm0, %%zmm12           \t\n"
        " vmovapd %%zmm0, %%zmm13           \t\n"
        " vmovapd %%zmm0, %%zmm14           \t\n"
        " vmovapd %%zmm0, %%zmm15           \t\n"
        " vmovapd %%zmm0, %%zmm16           \t\n"
        " vmovapd %%zmm0, %%zmm17           \t\n"
        " vmovapd %%zmm0, %%zmm18           \t\n"
        " vmovapd %%zmm0, %%zmm19           \t\n"
        " vmovapd %%zmm0, %%zmm20           \t\n"
        " vmovapd %%zmm0, %%zmm21           \t\n"
        " vmovapd %%zmm0, %%zmm22           \t\n"
        " vmovapd %%zmm0, %%zmm23           \t\n"
        " vmovapd %%zmm0, %%zmm24           \t\n"
        " vmovapd %%zmm0, %%zmm25           \t\n"
        " vmovapd %%zmm0, %%zmm26           \t\n"
        " vmovapd %%zmm0, %%zmm27           \t\n"
        " vmovapd %%zmm0, %%zmm28           \t\n"
        " vmovapd %%zmm0, %%zmm29           \t\n"

        " prefetcht0 0x00(%[C])           \t\n"
        " prefetcht0 0x08(%[C])           \t\n"
        " prefetcht0 0x10(%[C])           \t\n"
        " prefetcht0 0x18(%[C])           \t\n"
        " prefetcht0 0x20(%[C])           \t\n"
        " prefetcht0 0x28(%[C])           \t\n"
        " prefetcht0 0x30(%[C])           \t\n"
        " prefetcht0 0x38(%[C])           \t\n"
        " prefetcht0 0x40(%[C])           \t\n"
        " prefetcht0 0x48(%[C])           \t\n"
        " prefetcht0 0x50(%[C])           \t\n"
        " prefetcht0 0x58(%[C])           \t\n"
        " prefetcht0 0x60(%[C])           \t\n"
        " prefetcht0 0x68(%[C])           \t\n"
        " prefetcht0 0x70(%[C])           \t\n"
        " prefetcht0 0x78(%[C])           \t\n"
        " prefetcht0 0x80(%[C])           \t\n"
        " prefetcht0 0x88(%[C])           \t\n"
        " prefetcht0 0x90(%[C])           \t\n"
        " prefetcht0 0x98(%[C])           \t\n"
        " prefetcht0 0xa0(%[C])           \t\n"
        " prefetcht0 0xa8(%[C])           \t\n"
        " prefetcht0 0xb0(%[C])           \t\n"
        " prefetcht0 0xb8(%[C])           \t\n"
        " prefetcht0 0xc0(%[C])           \t\n"
        " prefetcht0 0xc8(%[C])           \t\n"
        " prefetcht0 0xd0(%[C])           \t\n"
        " prefetcht0 0xd8(%[C])           \t\n"
        " prefetcht0 0xe0(%[C])           \t\n"
        " prefetcht0 0xe8(%[C])           \t\n"
        :
        : [A] "r"(_A), [A_next] "r"(_A_next), [C] "r"(_C)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30");

    kk >>= 1;
#pragma unroll(8)
    for (uint64_t i = 0; LIKELY(i < kk); ++i)
    {
        asm volatile(
            " prefetchnta     0x00(%[A_next])                     \t\n"
            " vmovapd        0x040(%[A]),        %%zmm31          \t\n"
            " vfnmadd231pd   0x000(%[B])%{1to8}, %%zmm30,  %%zmm0 \t\n"
            " vfnmadd231pd   0x008(%[B])%{1to8}, %%zmm30,  %%zmm1 \t\n"
            " vfnmadd231pd   0x010(%[B])%{1to8}, %%zmm30,  %%zmm2 \t\n"
            " vfnmadd231pd   0x018(%[B])%{1to8}, %%zmm30,  %%zmm3 \t\n"
            " vfnmadd231pd   0x020(%[B])%{1to8}, %%zmm30,  %%zmm4 \t\n"
            " vfnmadd231pd   0x028(%[B])%{1to8}, %%zmm30,  %%zmm5 \t\n"
            " vfnmadd231pd   0x030(%[B])%{1to8}, %%zmm30,  %%zmm6 \t\n"
            " vfnmadd231pd   0x038(%[B])%{1to8}, %%zmm30,  %%zmm7 \t\n"
            " vfnmadd231pd   0x040(%[B])%{1to8}, %%zmm30,  %%zmm8 \t\n"
            " vfnmadd231pd   0x048(%[B])%{1to8}, %%zmm30,  %%zmm9 \t\n"
            " vfnmadd231pd   0x050(%[B])%{1to8}, %%zmm30, %%zmm10 \t\n"
            " vfnmadd231pd   0x058(%[B])%{1to8}, %%zmm30, %%zmm11 \t\n"
            " vfnmadd231pd   0x060(%[B])%{1to8}, %%zmm30, %%zmm12 \t\n"
            " vfnmadd231pd   0x068(%[B])%{1to8}, %%zmm30, %%zmm13 \t\n"
            " vfnmadd231pd   0x070(%[B])%{1to8}, %%zmm30, %%zmm14 \t\n"
            " vfnmadd231pd   0x078(%[B])%{1to8}, %%zmm30, %%zmm15 \t\n"
            " vfnmadd231pd   0x080(%[B])%{1to8}, %%zmm30, %%zmm16 \t\n"
            " vfnmadd231pd   0x088(%[B])%{1to8}, %%zmm30, %%zmm17 \t\n"
            " vfnmadd231pd   0x090(%[B])%{1to8}, %%zmm30, %%zmm18 \t\n"
            " vfnmadd231pd   0x098(%[B])%{1to8}, %%zmm30, %%zmm19 \t\n"
            " vfnmadd231pd   0x0a0(%[B])%{1to8}, %%zmm30, %%zmm20 \t\n"
            " vfnmadd231pd   0x0a8(%[B])%{1to8}, %%zmm30, %%zmm21 \t\n"
            " vfnmadd231pd   0x0b0(%[B])%{1to8}, %%zmm30, %%zmm22 \t\n"
            " vfnmadd231pd   0x0b8(%[B])%{1to8}, %%zmm30, %%zmm23 \t\n"
            " vfnmadd231pd   0x0c0(%[B])%{1to8}, %%zmm30, %%zmm24 \t\n"
            " vfnmadd231pd   0x0c8(%[B])%{1to8}, %%zmm30, %%zmm25 \t\n"
            " vfnmadd231pd   0x0d0(%[B])%{1to8}, %%zmm30, %%zmm26 \t\n"
            " vfnmadd231pd   0x0d8(%[B])%{1to8}, %%zmm30, %%zmm27 \t\n"
            " vfnmadd231pd   0x0e0(%[B])%{1to8}, %%zmm30, %%zmm28 \t\n"
            " vfnmadd231pd   0x0e8(%[B])%{1to8}, %%zmm30, %%zmm29 \t\n"

            " prefetchnta     0x40(%[A_next])                     \t\n"
            " add $0x080, %[A_next] \t\n"
            " vmovapd        0x080(%[A]),        %%zmm30          \t\n"
            " add $0x080, %[A] \t\n"
            " vfnmadd231pd   0x0f0(%[B])%{1to8}, %%zmm31,  %%zmm0 \t\n"
            " vfnmadd231pd   0x0f8(%[B])%{1to8}, %%zmm31,  %%zmm1 \t\n"
            " vfnmadd231pd   0x100(%[B])%{1to8}, %%zmm31,  %%zmm2 \t\n"
            " vfnmadd231pd   0x108(%[B])%{1to8}, %%zmm31,  %%zmm3 \t\n"
            " vfnmadd231pd   0x110(%[B])%{1to8}, %%zmm31,  %%zmm4 \t\n"
            " vfnmadd231pd   0x118(%[B])%{1to8}, %%zmm31,  %%zmm5 \t\n"
            " vfnmadd231pd   0x120(%[B])%{1to8}, %%zmm31,  %%zmm6 \t\n"
            " vfnmadd231pd   0x128(%[B])%{1to8}, %%zmm31,  %%zmm7 \t\n"
            " vfnmadd231pd   0x130(%[B])%{1to8}, %%zmm31,  %%zmm8 \t\n"
            " vfnmadd231pd   0x138(%[B])%{1to8}, %%zmm31,  %%zmm9 \t\n"
            " vfnmadd231pd   0x140(%[B])%{1to8}, %%zmm31, %%zmm10 \t\n"
            " vfnmadd231pd   0x148(%[B])%{1to8}, %%zmm31, %%zmm11 \t\n"
            " vfnmadd231pd   0x150(%[B])%{1to8}, %%zmm31, %%zmm12 \t\n"
            " vfnmadd231pd   0x158(%[B])%{1to8}, %%zmm31, %%zmm13 \t\n"
            " vfnmadd231pd   0x160(%[B])%{1to8}, %%zmm31, %%zmm14 \t\n"
            " vfnmadd231pd   0x168(%[B])%{1to8}, %%zmm31, %%zmm15 \t\n"
            " vfnmadd231pd   0x170(%[B])%{1to8}, %%zmm31, %%zmm16 \t\n"
            " vfnmadd231pd   0x178(%[B])%{1to8}, %%zmm31, %%zmm17 \t\n"
            " vfnmadd231pd   0x180(%[B])%{1to8}, %%zmm31, %%zmm18 \t\n"
            " vfnmadd231pd   0x188(%[B])%{1to8}, %%zmm31, %%zmm19 \t\n"
            " vfnmadd231pd   0x190(%[B])%{1to8}, %%zmm31, %%zmm20 \t\n"
            " vfnmadd231pd   0x198(%[B])%{1to8}, %%zmm31, %%zmm21 \t\n"
            " vfnmadd231pd   0x1a0(%[B])%{1to8}, %%zmm31, %%zmm22 \t\n"
            " vfnmadd231pd   0x1a8(%[B])%{1to8}, %%zmm31, %%zmm23 \t\n"
            " vfnmadd231pd   0x1b0(%[B])%{1to8}, %%zmm31, %%zmm24 \t\n"
            " vfnmadd231pd   0x1b8(%[B])%{1to8}, %%zmm31, %%zmm25 \t\n"
            " vfnmadd231pd   0x1c0(%[B])%{1to8}, %%zmm31, %%zmm26 \t\n"
            " vfnmadd231pd   0x1c8(%[B])%{1to8}, %%zmm31, %%zmm27 \t\n"
            " vfnmadd231pd   0x1d0(%[B])%{1to8}, %%zmm31, %%zmm28 \t\n"
            " vfnmadd231pd   0x1d8(%[B])%{1to8}, %%zmm31, %%zmm29 \t\n"

            " add $0x1e0, %[B] \t\n"
            : [A] "+r"(_A), [B] "+r"(_B), [A_next] "+r"(_A_next)
            :
            : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
              "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
              "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
              "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31");
    }

    asm volatile(
        " vaddpd  0x00(%[C]),  %%zmm0,  %%zmm0  \t\n"
        " vaddpd  0x08(%[C]),  %%zmm1,  %%zmm1  \t\n"
        " vaddpd  0x10(%[C]),  %%zmm2,  %%zmm2  \t\n"
        " vaddpd  0x18(%[C]),  %%zmm3,  %%zmm3  \t\n"
        " vaddpd  0x20(%[C]),  %%zmm4,  %%zmm4  \t\n"
        " vaddpd  0x28(%[C]),  %%zmm5,  %%zmm5  \t\n"
        " vaddpd  0x30(%[C]),  %%zmm6,  %%zmm6  \t\n"
        " vaddpd  0x38(%[C]),  %%zmm7,  %%zmm7  \t\n"
        " vaddpd  0x40(%[C]),  %%zmm8,  %%zmm8  \t\n"
        " vaddpd  0x48(%[C]),  %%zmm9,  %%zmm9  \t\n"
        " vaddpd  0x50(%[C]), %%zmm10, %%zmm10  \t\n"
        " vaddpd  0x58(%[C]), %%zmm11, %%zmm11  \t\n"

        " vaddpd  0x60(%[C]), %%zmm12, %%zmm12  \t\n"
        " vmovupd  %%zmm0, 0x00(%[C])  \t\n"
        " vaddpd  0x68(%[C]), %%zmm13, %%zmm13  \t\n"
        " vmovupd  %%zmm1, 0x08(%[C])  \t\n"
        " vaddpd  0x70(%[C]), %%zmm14, %%zmm14  \t\n"
        " vmovupd  %%zmm2, 0x10(%[C])  \t\n"
        " vaddpd  0x78(%[C]), %%zmm15, %%zmm15  \t\n"
        " vmovupd  %%zmm3, 0x18(%[C])  \t\n"
        " vaddpd  0x80(%[C]), %%zmm16, %%zmm16  \t\n"
        " vmovupd  %%zmm4, 0x20(%[C])  \t\n"
        " vaddpd  0x88(%[C]), %%zmm17, %%zmm17  \t\n"
        " vmovupd  %%zmm5, 0x28(%[C])  \t\n"
        " vaddpd  0x90(%[C]), %%zmm18, %%zmm18  \t\n"
        " vmovupd  %%zmm6, 0x30(%[C])  \t\n"
        " vaddpd  0x98(%[C]), %%zmm19, %%zmm19  \t\n"
        " vmovupd  %%zmm7, 0x38(%[C])  \t\n"
        " vaddpd  0xa0(%[C]), %%zmm20, %%zmm20  \t\n"
        " vmovupd  %%zmm8, 0x40(%[C])  \t\n"
        " vaddpd  0xa8(%[C]), %%zmm21, %%zmm21  \t\n"
        " vmovupd  %%zmm9, 0x48(%[C])  \t\n"
        " vaddpd  0xb0(%[C]), %%zmm22, %%zmm22  \t\n"
        " vmovupd %%zmm10, 0x50(%[C])  \t\n"
        " vaddpd  0xb8(%[C]), %%zmm23, %%zmm23  \t\n"
        " vmovupd %%zmm11, 0x58(%[C])  \t\n"
        " vaddpd  0xc0(%[C]), %%zmm24, %%zmm24  \t\n"
        " vmovupd %%zmm12, 0x60(%[C])  \t\n"
        " vaddpd  0xc8(%[C]), %%zmm25, %%zmm25  \t\n"
        " vmovupd %%zmm13, 0x68(%[C])  \t\n"
        " vaddpd  0xd0(%[C]), %%zmm26, %%zmm26  \t\n"
        " vmovupd %%zmm14, 0x70(%[C])  \t\n"
        " vaddpd  0xd8(%[C]), %%zmm27, %%zmm27  \t\n"
        " vmovupd %%zmm15, 0x78(%[C])  \t\n"
        " vaddpd  0xe0(%[C]), %%zmm28, %%zmm28  \t\n"
        " vmovupd %%zmm16, 0x80(%[C])  \t\n"
        " vaddpd  0xe8(%[C]), %%zmm29, %%zmm29  \t\n"
        " vmovupd %%zmm17, 0x88(%[C])  \t\n"

        " vmovupd %%zmm18, 0x90(%[C])  \t\n"
        " vmovupd %%zmm19, 0x98(%[C])  \t\n"
        " vmovupd %%zmm20, 0xa0(%[C])  \t\n"
        " vmovupd %%zmm21, 0xa8(%[C])  \t\n"
        " vmovupd %%zmm22, 0xb0(%[C])  \t\n"
        " vmovupd %%zmm23, 0xb8(%[C])  \t\n"
        " vmovupd %%zmm24, 0xc0(%[C])  \t\n"
        " vmovupd %%zmm25, 0xc8(%[C])  \t\n"
        " vmovupd %%zmm26, 0xd0(%[C])  \t\n"
        " vmovupd %%zmm27, 0xd8(%[C])  \t\n"
        " vmovupd %%zmm28, 0xe0(%[C])  \t\n"
        " vmovupd %%zmm29, 0xe8(%[C])  \t\n"
        :
        : [C] "r"(_C)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
          "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29");
}

void inner_kernel_ppc_anbp(
    uint64_t mm,
    uint64_t nn,
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict _C)
{
    uint64_t mmc = (mm + MR - 1) / MR;
    uint64_t mmr = mm % MR;
    uint64_t nnc = (nn + NR - 1) / NR;
    uint64_t nnr = nn % NR;

    const double *A_now = _A;
    const double *A_next = (mmc > 1) ? _A + MR * kk : _A;

    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        // micro_kernel_8x30_ppc_anbp(kk, _A + mmi * MR * kk, _B, _C + mmi * MR * NR);
        // micro_kernel_8x30_ppc_anbp(kk, A_now, _B, _C, A_next);
        // micro_kernel_8x30_ppc_anbp(kk, A_now, _B, _C, A_now + 120);
        micro_kernel_8x30_ppc_anbp(kk, A_now, _B, _C + mmi * MR * NR, A_now + 120);

        A_now = A_next;
        A_next = (mmi < mmc - 1) ? A_next + MR * kk : _A;
    }
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

    static double *_A = NULL;
    static double *_B = NULL;
    static double *_C = NULL;

    if (_A == NULL)
    {
        ALLOC(_A, sizeof(double) * (MB * KB + MR));
        ALLOC(_B, sizeof(double) * KB * NB);
        ALLOC(_C, sizeof(double) * MB * NB);
    }

    for (uint64_t i = 0; i < (MB * KB + MR); ++i)
    {
        _A[i] = i * 1.0;
    }
    for (uint64_t i = 0; i < KB * NB; ++i)
    {
        _B[i] = i * 1.0;
    }
    for (uint64_t i = 0; i < MB * NB; ++i)
    {
        _C[i] = i * 1.0;
    }

    uint64_t cnt = m / MR * n / NR * k / KB;
    for (uint64_t i = 0; i < cnt; ++i)
    {
        micro_kernel_8x30_ppc_anbp(KB, _A, _B, _C, _A);
    }
    /*
        uint64_t cnt = ((m+MB-1)/MB) * ((n+NB-1)/NB) * ((k+KB-1)/KB);
        for (uint64_t i = 0; i < cnt; ++i) {
            inner_kernel_ppc_anbp(MB, NB, KB, _A, _B, C, ldc);
        }
        */

    /*
        for (uint64_t ki = 0; ki < kc; ++ki) {
            uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            for (uint64_t ni = 0; ni < nc; ++ni) {
                uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;
                //packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);

                for (uint64_t mi = 0; mi < mc; ++mi) {
                    uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;
                    //packacc(mm, kk, A + mi * MB + ki * KB * lda, lda, _A);

                    inner_kernel_ppc_anbp(mm, nn, kk, _A, _B, _C);
                }
            }
        }
        */
}
