// n     = 40
// alpha = 1.0
// beta  = 0.0

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#include "cblas_format.h"
#include "common.h"

#define CACHE_LINE 64
#define CACHE_ELEM 8

#ifndef USE_CILKPLUS
THIS_FILE_IS_ONLY_FOR_ICC;
#endif

#define TOTAL_CORE 1
#define CM 1
#define CN (TOTAL_CORE / CM)
#define MR 8
#define NR 16
#define MB (MR * 8)
#define NB (NR * 3)
#define KB 500

/*
#define MR 8
#define NR 20
#define MB 8
#define NB 40
#define KB 876
*/

#define NU 3

#define L1_DIST_A (MR * 10)
#define L1_DIST_B (NR * 10)

static void micro_kernel_ap(
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    uint64_t ldc)
{
#ifdef ORIGIN_KERNEL
    __m512d _A0;
    __m512d _C00, _C01, _C02, _C03, _C04, _C05, _C06, _C07, _C08, _C09,
        _C10, _C11, _C12, _C13, _C14, _C15, _C16, _C17, _C18, _C19;
    _C00 = _mm512_loadu_pd(C + ldc * 0);
    _C01 = _mm512_loadu_pd(C + ldc * 1);
    _C02 = _mm512_loadu_pd(C + ldc * 2);
    _C03 = _mm512_loadu_pd(C + ldc * 3);
    _C04 = _mm512_loadu_pd(C + ldc * 4);
    _C05 = _mm512_loadu_pd(C + ldc * 5);
    _C06 = _mm512_loadu_pd(C + ldc * 6);
    _C07 = _mm512_loadu_pd(C + ldc * 7);
    _C08 = _mm512_loadu_pd(C + ldc * 8);
    _C09 = _mm512_loadu_pd(C + ldc * 9);
    _C10 = _mm512_loadu_pd(C + ldc * 10);
    _C11 = _mm512_loadu_pd(C + ldc * 11);
    _C12 = _mm512_loadu_pd(C + ldc * 12);
    _C13 = _mm512_loadu_pd(C + ldc * 13);
    _C14 = _mm512_loadu_pd(C + ldc * 14);
    _C15 = _mm512_loadu_pd(C + ldc * 15);
    _C16 = _mm512_loadu_pd(C + ldc * 16);
    _C17 = _mm512_loadu_pd(C + ldc * 17);
    _C18 = _mm512_loadu_pd(C + ldc * 18);
    _C19 = _mm512_loadu_pd(C + ldc * 19);

#pragma unroll(NU)
    for (uint64_t i = 0; i < kk; ++i)
    {
        _mm_prefetch(_A + L1_DIST_A + 0, _MM_HINT_T0);
        _mm_prefetch(_B + L1_DIST_B + 0, _MM_HINT_T0);
        _mm_prefetch(_B + L1_DIST_B + 8, _MM_HINT_T0);
        _mm_prefetch(_B + L1_DIST_B + 16, _MM_HINT_T0);

        _A0 = _mm512_load_pd(_A);

        _C00 = _mm512_fmadd_pd(_mm512_set1_pd(_B[0]), _A0, _C00);
        _C01 = _mm512_fmadd_pd(_mm512_set1_pd(_B[1]), _A0, _C01);
        _C02 = _mm512_fmadd_pd(_mm512_set1_pd(_B[2]), _A0, _C02);
        _C03 = _mm512_fmadd_pd(_mm512_set1_pd(_B[3]), _A0, _C03);
        _C04 = _mm512_fmadd_pd(_mm512_set1_pd(_B[4]), _A0, _C04);
        _C05 = _mm512_fmadd_pd(_mm512_set1_pd(_B[5]), _A0, _C05);
        _C06 = _mm512_fmadd_pd(_mm512_set1_pd(_B[6]), _A0, _C06);
        _C07 = _mm512_fmadd_pd(_mm512_set1_pd(_B[7]), _A0, _C07);
        _C08 = _mm512_fmadd_pd(_mm512_set1_pd(_B[8]), _A0, _C08);
        _C09 = _mm512_fmadd_pd(_mm512_set1_pd(_B[9]), _A0, _C09);
        _C10 = _mm512_fmadd_pd(_mm512_set1_pd(_B[10]), _A0, _C10);
        _C11 = _mm512_fmadd_pd(_mm512_set1_pd(_B[11]), _A0, _C11);
        _C12 = _mm512_fmadd_pd(_mm512_set1_pd(_B[12]), _A0, _C12);
        _C13 = _mm512_fmadd_pd(_mm512_set1_pd(_B[13]), _A0, _C13);
        _C14 = _mm512_fmadd_pd(_mm512_set1_pd(_B[14]), _A0, _C14);
        _C15 = _mm512_fmadd_pd(_mm512_set1_pd(_B[15]), _A0, _C15);
        _C16 = _mm512_fmadd_pd(_mm512_set1_pd(_B[16]), _A0, _C16);
        _C17 = _mm512_fmadd_pd(_mm512_set1_pd(_B[17]), _A0, _C17);
        _C18 = _mm512_fmadd_pd(_mm512_set1_pd(_B[18]), _A0, _C18);
        _C19 = _mm512_fmadd_pd(_mm512_set1_pd(_B[19]), _A0, _C19);

        _A += MR;
        _B += NR;
    }

    //_mm_prefetch(A_next, _MM_HINT_T0);
    //_mm_prefetch(B_next, _MM_HINT_T0);

    _mm512_storeu_pd(C + ldc * 0, _C00);
    _mm512_storeu_pd(C + ldc * 1, _C01);
    _mm512_storeu_pd(C + ldc * 2, _C02);
    _mm512_storeu_pd(C + ldc * 3, _C03);
    _mm512_storeu_pd(C + ldc * 4, _C04);
    _mm512_storeu_pd(C + ldc * 5, _C05);
    _mm512_storeu_pd(C + ldc * 6, _C06);
    _mm512_storeu_pd(C + ldc * 7, _C07);
    _mm512_storeu_pd(C + ldc * 8, _C08);
    _mm512_storeu_pd(C + ldc * 9, _C09);
    _mm512_storeu_pd(C + ldc * 10, _C10);
    _mm512_storeu_pd(C + ldc * 11, _C11);
    _mm512_storeu_pd(C + ldc * 12, _C12);
    _mm512_storeu_pd(C + ldc * 13, _C13);
    _mm512_storeu_pd(C + ldc * 14, _C14);
    _mm512_storeu_pd(C + ldc * 15, _C15);
    _mm512_storeu_pd(C + ldc * 16, _C16);
    _mm512_storeu_pd(C + ldc * 17, _C17);
    _mm512_storeu_pd(C + ldc * 18, _C18);
    _mm512_storeu_pd(C + ldc * 19, _C19);
#else
    __m512d _A0, _A1;
    __m512d _C00, _C01, _C02, _C03, _C04, _C05, _C06, _C07, _C08, _C09,
        _C10, _C11, _C12, _C13, _C14, _C15;

    _A0 = _mm512_load_pd(_A);

    _C00 = _mm512_loadu_pd(C + ldc * 0);
    _C01 = _mm512_loadu_pd(C + ldc * 1);
    _C02 = _mm512_loadu_pd(C + ldc * 2);
    _C03 = _mm512_loadu_pd(C + ldc * 3);
    _C04 = _mm512_loadu_pd(C + ldc * 4);
    _C05 = _mm512_loadu_pd(C + ldc * 5);
    _C06 = _mm512_loadu_pd(C + ldc * 6);
    _C07 = _mm512_loadu_pd(C + ldc * 7);
    _C08 = _mm512_loadu_pd(C + ldc * 8);
    _C09 = _mm512_loadu_pd(C + ldc * 9);
    _C10 = _mm512_loadu_pd(C + ldc * 10);
    _C11 = _mm512_loadu_pd(C + ldc * 11);
    _C12 = _mm512_loadu_pd(C + ldc * 12);
    _C13 = _mm512_loadu_pd(C + ldc * 13);
    _C14 = _mm512_loadu_pd(C + ldc * 14);
    _C15 = _mm512_loadu_pd(C + ldc * 15);

    uint64_t kk_2 = kk >> 1;
    // #pragma unroll(NU)
    for (uint64_t i = 0; i < kk_2; ++i)
    {
        _mm_prefetch(_A + L1_DIST_A + 0, _MM_HINT_T0);
        _mm_prefetch(_A + L1_DIST_A + 8, _MM_HINT_T0);
        _mm_prefetch(_B + L1_DIST_B + 0, _MM_HINT_T0);
        _mm_prefetch(_B + L1_DIST_B + 8, _MM_HINT_T0);
        _mm_prefetch(_B + L1_DIST_B + 16, _MM_HINT_T0);
        _mm_prefetch(_B + L1_DIST_B + 24, _MM_HINT_T0);

        _A1 = _mm512_load_pd(_A + MR * 1);

        _C00 = _mm512_fmadd_pd(_mm512_set1_pd(_B[0]), _A0, _C00);
        _C01 = _mm512_fmadd_pd(_mm512_set1_pd(_B[1]), _A0, _C01);
        _C02 = _mm512_fmadd_pd(_mm512_set1_pd(_B[2]), _A0, _C02);
        _C03 = _mm512_fmadd_pd(_mm512_set1_pd(_B[3]), _A0, _C03);
        _C04 = _mm512_fmadd_pd(_mm512_set1_pd(_B[4]), _A0, _C04);
        _C05 = _mm512_fmadd_pd(_mm512_set1_pd(_B[5]), _A0, _C05);
        _C06 = _mm512_fmadd_pd(_mm512_set1_pd(_B[6]), _A0, _C06);
        _C07 = _mm512_fmadd_pd(_mm512_set1_pd(_B[7]), _A0, _C07);
        _C08 = _mm512_fmadd_pd(_mm512_set1_pd(_B[8]), _A0, _C08);
        _C09 = _mm512_fmadd_pd(_mm512_set1_pd(_B[9]), _A0, _C09);
        _C10 = _mm512_fmadd_pd(_mm512_set1_pd(_B[10]), _A0, _C10);
        _C11 = _mm512_fmadd_pd(_mm512_set1_pd(_B[11]), _A0, _C11);
        _C12 = _mm512_fmadd_pd(_mm512_set1_pd(_B[12]), _A0, _C12);
        _C13 = _mm512_fmadd_pd(_mm512_set1_pd(_B[13]), _A0, _C13);
        _C14 = _mm512_fmadd_pd(_mm512_set1_pd(_B[14]), _A0, _C14);
        _C15 = _mm512_fmadd_pd(_mm512_set1_pd(_B[15]), _A0, _C15);

        _A0 = _mm512_load_pd(_A + MR * 2);

        _C00 = _mm512_fmadd_pd(_mm512_set1_pd(_B[16]), _A1, _C00);
        _C01 = _mm512_fmadd_pd(_mm512_set1_pd(_B[17]), _A1, _C01);
        _C02 = _mm512_fmadd_pd(_mm512_set1_pd(_B[18]), _A1, _C02);
        _C03 = _mm512_fmadd_pd(_mm512_set1_pd(_B[19]), _A1, _C03);
        _C04 = _mm512_fmadd_pd(_mm512_set1_pd(_B[20]), _A1, _C04);
        _C05 = _mm512_fmadd_pd(_mm512_set1_pd(_B[21]), _A1, _C05);
        _C06 = _mm512_fmadd_pd(_mm512_set1_pd(_B[22]), _A1, _C06);
        _C07 = _mm512_fmadd_pd(_mm512_set1_pd(_B[23]), _A1, _C07);
        _C08 = _mm512_fmadd_pd(_mm512_set1_pd(_B[24]), _A1, _C08);
        _C09 = _mm512_fmadd_pd(_mm512_set1_pd(_B[25]), _A1, _C09);
        _C10 = _mm512_fmadd_pd(_mm512_set1_pd(_B[26]), _A1, _C10);
        _C11 = _mm512_fmadd_pd(_mm512_set1_pd(_B[27]), _A1, _C11);
        _C12 = _mm512_fmadd_pd(_mm512_set1_pd(_B[28]), _A1, _C12);
        _C13 = _mm512_fmadd_pd(_mm512_set1_pd(_B[29]), _A1, _C13);
        _C14 = _mm512_fmadd_pd(_mm512_set1_pd(_B[30]), _A1, _C14);
        _C15 = _mm512_fmadd_pd(_mm512_set1_pd(_B[31]), _A1, _C15);

        _A += MR * 2;
        _B += NR * 2;
    }

    //_mm_prefetch(A_next, _MM_HINT_T0);
    //_mm_prefetch(B_next, _MM_HINT_T0);

    _mm512_storeu_pd(C + ldc * 0, _C00);
    _mm512_storeu_pd(C + ldc * 1, _C01);
    _mm512_storeu_pd(C + ldc * 2, _C02);
    _mm512_storeu_pd(C + ldc * 3, _C03);
    _mm512_storeu_pd(C + ldc * 4, _C04);
    _mm512_storeu_pd(C + ldc * 5, _C05);
    _mm512_storeu_pd(C + ldc * 6, _C06);
    _mm512_storeu_pd(C + ldc * 7, _C07);
    _mm512_storeu_pd(C + ldc * 8, _C08);
    _mm512_storeu_pd(C + ldc * 9, _C09);
    _mm512_storeu_pd(C + ldc * 10, _C10);
    _mm512_storeu_pd(C + ldc * 11, _C11);
    _mm512_storeu_pd(C + ldc * 12, _C12);
    _mm512_storeu_pd(C + ldc * 13, _C13);
    _mm512_storeu_pd(C + ldc * 14, _C14);
    _mm512_storeu_pd(C + ldc * 15, _C15);
#endif
}

static void micro_dxpy(uint64_t mmm, uint64_t nnn, double *restrict C,
                       uint64_t ldc, const double *restrict _C)
{
    for (uint64_t i = 0; i < nnn; ++i)
    {
        C [0:mmm] += _C [0:mmm];
        C += ldc;
        _C += MR;
    }
}

static void inner_kernel_ppc_ap(
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

            if (LIKELY(mmm == MR && nnn == NR))
            {
                micro_kernel_ap(kk, A_now, B_now, C + mmi * MR + nni * NR * ldc, ldc);
            }
            else
            {
                double _C[MR * NR] __attribute__((aligned(CACHE_LINE))) = {};
                micro_kernel_ap(kk, A_now, B_now, _C, MR);
                micro_dxpy(mmm, nnn, C + mmi * MR + nni * NR * ldc, ldc, _C);
            }
        }
    }
}

static void store_transposed_8x8(double *restrict dst, const double *restrict src, uint64_t src_ld)
{
    __m512d a0, a1, a2, a3, a4, a5, a6, a7;
    __m512d b0, b1, b2, b3, b4, b5, b6, b7;

    a0 = _mm512_loadu_pd(src + src_ld * 0);
    a1 = _mm512_loadu_pd(src + src_ld * 1);
    a2 = _mm512_loadu_pd(src + src_ld * 2);
    a3 = _mm512_loadu_pd(src + src_ld * 3);
    a4 = _mm512_loadu_pd(src + src_ld * 4);
    a5 = _mm512_loadu_pd(src + src_ld * 5);
    a6 = _mm512_loadu_pd(src + src_ld * 6);
    a7 = _mm512_loadu_pd(src + src_ld * 7);

    b0 = _mm512_unpacklo_pd(a0, a1);
    b1 = _mm512_unpackhi_pd(a0, a1);
    b2 = _mm512_unpacklo_pd(a2, a3);
    b3 = _mm512_unpackhi_pd(a2, a3);
    b4 = _mm512_unpacklo_pd(a4, a5);
    b5 = _mm512_unpackhi_pd(a4, a5);
    b6 = _mm512_unpacklo_pd(a6, a7);
    b7 = _mm512_unpackhi_pd(a6, a7);

    a0 = _mm512_shuffle_f64x2(b0, b2, 0x44);
    a1 = _mm512_shuffle_f64x2(b1, b3, 0x44);
    a2 = _mm512_shuffle_f64x2(b0, b2, 0xEE);
    a3 = _mm512_shuffle_f64x2(b1, b3, 0xEE);
    a4 = _mm512_shuffle_f64x2(b4, b6, 0x44);
    a5 = _mm512_shuffle_f64x2(b5, b7, 0x44);
    a6 = _mm512_shuffle_f64x2(b4, b6, 0xEE);
    a7 = _mm512_shuffle_f64x2(b5, b7, 0xEE);

    b0 = _mm512_shuffle_f64x2(a0, a4, 0x88);
    b1 = _mm512_shuffle_f64x2(a1, a5, 0x88);
    b2 = _mm512_shuffle_f64x2(a0, a4, 0xDD);
    b3 = _mm512_shuffle_f64x2(a1, a5, 0xDD);
    b4 = _mm512_shuffle_f64x2(a2, a6, 0x88);
    b5 = _mm512_shuffle_f64x2(a3, a7, 0x88);
    b6 = _mm512_shuffle_f64x2(a2, a6, 0xDD);
    b7 = _mm512_shuffle_f64x2(a3, a7, 0xDD);

    _mm512_store_pd(dst + 0x00, b0);
    _mm512_store_pd(dst + 0x08, b1);
    _mm512_store_pd(dst + 0x10, b2);
    _mm512_store_pd(dst + 0x18, b3);
    _mm512_store_pd(dst + 0x20, b4);
    _mm512_store_pd(dst + 0x28, b5);
    _mm512_store_pd(dst + 0x30, b6);
    _mm512_store_pd(dst + 0x38, b7);
}

static void packacc(uint64_t mm, uint64_t kk, const double *restrict A, uint64_t lda,
                    double *restrict _A)
{
    const uint64_t mmc = (mm + MR - 1) / MR;
    const uint64_t mmr = mm % MR;

    for (uint64_t j = 0; j < kk; ++j)
    {
        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            const uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;
            _A [mmi * MR * kk + j * MR:mmm] = A [mmi * MR + j * lda:mmm];
        }
    }
}

static void packarc(uint64_t mm, uint64_t kk, const double *restrict A, uint64_t lda,
                    double *restrict _A)
{
#if MR == 8 && 0 == 1
    uint64_t mmc = mm / MR;
    uint64_t mmr = mm % MR;
    uint64_t kkc = kk / CACHE_ELEM;
    uint64_t kkr = kk % CACHE_ELEM;

    for (uint64_t i = 0; i < mmc; ++i)
    {
        for (uint64_t j = 0; j < kkc; ++j)
        {
            store_transposed_8x8(_A + i * MR * kk + j * CACHE_ELEM * MR, A + i * MR * lda + j * CACHE_ELEM, lda);
        }
    }
    if (kkr > 0)
    {
        for (uint64_t i = 0; i < mmc; ++i)
        {
            for (uint64_t j = kkc * CACHE_ELEM; j < kk; ++j)
            {
                _A [i * MR * kk + j * MR:MR] = A [i * MR * lda + j:MR:lda];
            }
        }
    }
    _A += mmc * kk * MR;
    A += mmc * lda * MR;
    if (mmr > 0)
    {
        for (uint64_t j = 0; j < kk; ++j)
        {
            _A [0:mmr] = A [0:mmr:lda];
            _A += MR;
            A += 1;
        }
    }
#else
    const uint64_t q = mm / MR;
    const uint64_t r = mm % MR;
    for (uint64_t i = 0; i < q; ++i)
    {
        for (uint64_t j = 0; j < kk; ++j)
        {
            _A [j * MR + i * kk * MR:MR] = A [j + i * MR * lda:MR:lda];
        }
    }
    _A += q * kk * MR;
    A += q * lda * MR;
    if (r > 0)
    {
        for (uint64_t j = 0; j < kk; ++j)
        {
            _A [0:r] = A [0:r:lda];
            _A += MR;
            A += 1;
        }
    }
#endif
}

static void packbcr(uint64_t kk, uint64_t nn, const double *restrict B, uint64_t ldb,
                    double *restrict _B)
{
    const uint64_t nnc = nn / NR;
    const uint64_t nnr = nn % NR;
    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        for (uint64_t kki = 0; kki < kk; ++kki)
        {
            _B [nni * NR * kk + kki * NR:NR] = B [nni * NR * ldb + kki:NR:ldb];
        }
    }
    if (nnr > 0)
    {
        _B += nnc * NR * kk;
        B += nnc * NR * ldb;
        for (uint64_t kki = 0; kki < kk; ++kki)
        {
            _B [0:nnr] = B [0:nnr:ldb];
            _B += NR;
            B += 1;
        }
    }
}

static void packbrr(uint64_t kk, uint64_t nn, const double *restrict B, uint64_t ldb,
                    double *restrict _B)
{
    const uint64_t nnc = nn / NR;
    const uint64_t nnr = nn % NR;
    for (uint64_t kki = 0; kki < kk; ++kki)
    {
        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            _B [kki * NR + nni * kk * NR:NR] = B [kki * ldb + nni * NR:NR];
        }
    }
    if (nnr > 0)
    {
        _B += nnc * NR * kk;
        B += nnc * NR;
        for (uint64_t kki = 0; kki < kk; ++kki)
        {
            _B [0:nnr] = B [0:nnr];
            _B += NR;
            B += ldb;
        }
    }
}

static void middle_kernel_xxc_ap(
    uint64_t nota,
    uint64_t notb,
    uint64_t m,
    uint64_t n,
    uint64_t k,
    const double *restrict A,
    uint64_t lda,
    const double *restrict B,
    uint64_t ldb,
    double *restrict C,
    uint64_t ldc)
{
    double *_A;
    double *_B;

    ALLOC(_A, sizeof(double) * (MB + MR) * KB);
    ALLOC(_B, sizeof(double) * KB * (NB + NR));

    const uint64_t mc = (m + MB - 1) / MB;
    const uint64_t mr = m % MB;
    const uint64_t nc = (n + NB - 1) / NB;
    const uint64_t nr = n % NB;
    const uint64_t kc = (k + KB - 1) / KB;
    const uint64_t kr = k % KB;

    for (uint64_t ni = 0; ni < nc; ++ni)
    {
        const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            if (notb)
                packbcr(kk, nn, B + ki * KB * 1 + ni * NB * ldb, ldb, _B);
            else
                packbrr(kk, nn, B + ki * KB * ldb + ni * NB * 1, ldb, _B);

            for (uint64_t mi = 0; mi < mc; ++mi)
            {
                const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

                if (nota)
                    packacc(mm, kk, A + mi * MB * 1 + ki * KB * lda, lda, _A);
                else
                    packarc(mm, kk, A + mi * MB * lda + ki * KB * 1, lda, _A);

                inner_kernel_ppc_ap(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
            }
        }
    }

    FREE(_A);
    FREE(_B);
}

static uint64_t min(
    uint64_t a,
    uint64_t b)
{
    return a < b ? a : b;
}

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, const int64_t m_, const int64_t n_,
                const int64_t k_, const double alpha, const double *A,
                const int64_t lda_, const double *B, const int64_t ldb_,
                const double beta, double *C, const int64_t ldc_)
{
    const uint64_t nota = (int)(transa[0] == 'N' || transa[0] == 'n');
    const uint64_t notb = (int)(transb[0] == 'N' || transb[0] == 'n');
    const uint64_t m = (int)(*_m);
    const uint64_t n = (int)(*_n);
    const uint64_t k = (int)(*_k);
    const uint64_t lda = (int)(*_lda);
    const uint64_t ldb = (int)(*_ldb);
    const uint64_t ldc = (int)(*_ldc);

    // beta = 0.0
#pragma omp parallel for
    for (uint64_t i = 0; i < n; ++i)
    {
        memset(C + i * ldc, 0, sizeof(double) * m);
    }

    const uint64_t total_m_jobs = (m + MR - 1) / MR;
    const uint64_t min_each_m_jobs = total_m_jobs / CM;
    const uint64_t rest_m_jobs = total_m_jobs % CM;

    const uint64_t total_n_jobs = (n + NR - 1) / NR;
    const uint64_t min_each_n_jobs = total_n_jobs / CN;
    const uint64_t rest_n_jobs = total_n_jobs % CN;

#pragma omp parallel
    {
        const uint64_t tid = omp_get_thread_num();
        const uint64_t m_tid = tid % CM;
        const uint64_t n_tid = tid / CM;

        const uint64_t my_m_idx_start = (m_tid)*min_each_m_jobs + min(m_tid, rest_m_jobs);
        const uint64_t my_m_idx_end = (m_tid + 1) * min_each_m_jobs + min(m_tid + 1, rest_m_jobs);
        const uint64_t my_m_start = min(my_m_idx_start * MR, m);
        const uint64_t my_m_end = min(my_m_idx_end * MR, m);
        const uint64_t my_m_size = my_m_end - my_m_start;

        const uint64_t my_n_idx_start = (n_tid)*min_each_n_jobs + min(n_tid, rest_n_jobs);
        const uint64_t my_n_idx_end = (n_tid + 1) * min_each_n_jobs + min(n_tid + 1, rest_n_jobs);
        const uint64_t my_n_start = min(my_n_idx_start * NR, n);
        const uint64_t my_n_end = min(my_n_idx_end * NR, n);
        const uint64_t my_n_size = my_n_end - my_n_start;

        const double *A_start;
        const double *B_start;
        double *C_start;

        if (nota)
            A_start = A + my_m_start * 1;
        else
            A_start = A + my_m_start * lda;
        if (notb)
            B_start = B + my_n_start * ldb;
        else
            B_start = B + my_n_start * 1;
        if (1)
            C_start = C + my_m_start * 1 + my_n_start * ldc;
        else
            C_start = C + my_m_start * ldc + my_n_start * 1;

        middle_kernel_xxc_ap(nota, notb, my_m_size, my_n_size, k, A_start, lda, B_start, ldb, C_start, ldc);
    }
}
