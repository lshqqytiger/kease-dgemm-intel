// n     = 40
// alpha = 1.0
// beta  = 0.0

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#include "common.h"

#define CACHE_ELEM 8

#define MR 8
#define NR 20
#define MB 8
#define NB 40
#define KB 876

#define NU 3

#define L1_DIST_A 192
#define L1_DIST_B 340

static void micro_kernel0_ap(uint64_t kk, const double *restrict _A, const double *restrict _B,
                             double *restrict C, uint64_t ldc)
{
    __m512d _A0;
    __m512d _C0, _C1, _C2, _C3, _C4, _C5, _C6, _C7, _C8, _C9, _C10, _C11, _C12, _C13, _C14, _C15, _C16, _C17, _C18, _C19;
    _C0 = _mm512_loadu_pd(&C[0 * ldc]);
    _C1 = _mm512_loadu_pd(&C[1 * ldc]);
    _C2 = _mm512_loadu_pd(&C[2 * ldc]);
    _C3 = _mm512_loadu_pd(&C[3 * ldc]);
    _C4 = _mm512_loadu_pd(&C[4 * ldc]);
    _C5 = _mm512_loadu_pd(&C[5 * ldc]);
    _C6 = _mm512_loadu_pd(&C[6 * ldc]);
    _C7 = _mm512_loadu_pd(&C[7 * ldc]);
    _C8 = _mm512_loadu_pd(&C[8 * ldc]);
    _C9 = _mm512_loadu_pd(&C[9 * ldc]);
    _C10 = _mm512_loadu_pd(&C[10 * ldc]);
    _C11 = _mm512_loadu_pd(&C[11 * ldc]);
    _C12 = _mm512_loadu_pd(&C[12 * ldc]);
    _C13 = _mm512_loadu_pd(&C[13 * ldc]);
    _C14 = _mm512_loadu_pd(&C[14 * ldc]);
    _C15 = _mm512_loadu_pd(&C[15 * ldc]);
    _C16 = _mm512_loadu_pd(&C[16 * ldc]);
    _C17 = _mm512_loadu_pd(&C[17 * ldc]);
    _C18 = _mm512_loadu_pd(&C[18 * ldc]);
    _C19 = _mm512_loadu_pd(&C[19 * ldc]);

#pragma unroll(NU)
    for (uint64_t i = 0; i < kk; ++i)
    {
        _mm_prefetch((const void *)&_A[L1_DIST_A + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 8], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 16], _MM_HINT_T0);
        _A0 = _mm512_loadu_pd(&_A[0]);
        _C0 = _mm512_fmadd_pd(_mm512_set1_pd(_B[0]), _A0, _C0);
        _C1 = _mm512_fmadd_pd(_mm512_set1_pd(_B[1]), _A0, _C1);
        _C2 = _mm512_fmadd_pd(_mm512_set1_pd(_B[2]), _A0, _C2);
        _C3 = _mm512_fmadd_pd(_mm512_set1_pd(_B[3]), _A0, _C3);
        _C4 = _mm512_fmadd_pd(_mm512_set1_pd(_B[4]), _A0, _C4);
        _C5 = _mm512_fmadd_pd(_mm512_set1_pd(_B[5]), _A0, _C5);
        _C6 = _mm512_fmadd_pd(_mm512_set1_pd(_B[6]), _A0, _C6);
        _C7 = _mm512_fmadd_pd(_mm512_set1_pd(_B[7]), _A0, _C7);
        _C8 = _mm512_fmadd_pd(_mm512_set1_pd(_B[8]), _A0, _C8);
        _C9 = _mm512_fmadd_pd(_mm512_set1_pd(_B[9]), _A0, _C9);
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
    _mm512_storeu_pd(&C[0 * ldc], _C0);
    _mm512_storeu_pd(&C[1 * ldc], _C1);
    _mm512_storeu_pd(&C[2 * ldc], _C2);
    _mm512_storeu_pd(&C[3 * ldc], _C3);
    _mm512_storeu_pd(&C[4 * ldc], _C4);
    _mm512_storeu_pd(&C[5 * ldc], _C5);
    _mm512_storeu_pd(&C[6 * ldc], _C6);
    _mm512_storeu_pd(&C[7 * ldc], _C7);
    _mm512_storeu_pd(&C[8 * ldc], _C8);
    _mm512_storeu_pd(&C[9 * ldc], _C9);
    _mm512_storeu_pd(&C[10 * ldc], _C10);
    _mm512_storeu_pd(&C[11 * ldc], _C11);
    _mm512_storeu_pd(&C[12 * ldc], _C12);
    _mm512_storeu_pd(&C[13 * ldc], _C13);
    _mm512_storeu_pd(&C[14 * ldc], _C14);
    _mm512_storeu_pd(&C[15 * ldc], _C15);
    _mm512_storeu_pd(&C[16 * ldc], _C16);
    _mm512_storeu_pd(&C[17 * ldc], _C17);
    _mm512_storeu_pd(&C[18 * ldc], _C18);
    _mm512_storeu_pd(&C[19 * ldc], _C19);
}

static void micro_kernel1_ap(uint64_t kk, const double *restrict _A, const double *restrict _B, double *restrict _C)
{
    __m512d _A0;
    __m512d _C0, _C1, _C2, _C3, _C4, _C5, _C6, _C7, _C8, _C9, _C10, _C11, _C12, _C13, _C14, _C15, _C16, _C17, _C18, _C19;
    _C0 = _mm512_setzero_pd();
    _C1 = _mm512_setzero_pd();
    _C2 = _mm512_setzero_pd();
    _C3 = _mm512_setzero_pd();
    _C4 = _mm512_setzero_pd();
    _C5 = _mm512_setzero_pd();
    _C6 = _mm512_setzero_pd();
    _C7 = _mm512_setzero_pd();
    _C8 = _mm512_setzero_pd();
    _C9 = _mm512_setzero_pd();
    _C10 = _mm512_setzero_pd();
    _C11 = _mm512_setzero_pd();
    _C12 = _mm512_setzero_pd();
    _C13 = _mm512_setzero_pd();
    _C14 = _mm512_setzero_pd();
    _C15 = _mm512_setzero_pd();
    _C16 = _mm512_setzero_pd();
    _C17 = _mm512_setzero_pd();
    _C18 = _mm512_setzero_pd();
    _C19 = _mm512_setzero_pd();

#pragma unroll(NU)
    for (uint64_t i = 0; i < kk; ++i)
    {
        _mm_prefetch((const void *)&_A[L1_DIST_A + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 8], _MM_HINT_T0);
        _mm_prefetch((const void *)&_B[L1_DIST_B + 16], _MM_HINT_T0);
        _A0 = _mm512_loadu_pd(&_A[0]);
        _C0 = _mm512_fmadd_pd(_mm512_set1_pd(_B[0]), _A0, _C0);
        _C1 = _mm512_fmadd_pd(_mm512_set1_pd(_B[1]), _A0, _C1);
        _C2 = _mm512_fmadd_pd(_mm512_set1_pd(_B[2]), _A0, _C2);
        _C3 = _mm512_fmadd_pd(_mm512_set1_pd(_B[3]), _A0, _C3);
        _C4 = _mm512_fmadd_pd(_mm512_set1_pd(_B[4]), _A0, _C4);
        _C5 = _mm512_fmadd_pd(_mm512_set1_pd(_B[5]), _A0, _C5);
        _C6 = _mm512_fmadd_pd(_mm512_set1_pd(_B[6]), _A0, _C6);
        _C7 = _mm512_fmadd_pd(_mm512_set1_pd(_B[7]), _A0, _C7);
        _C8 = _mm512_fmadd_pd(_mm512_set1_pd(_B[8]), _A0, _C8);
        _C9 = _mm512_fmadd_pd(_mm512_set1_pd(_B[9]), _A0, _C9);
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
    _mm512_storeu_pd(&_C[0 * MR], _C0);
    _mm512_storeu_pd(&_C[1 * MR], _C1);
    _mm512_storeu_pd(&_C[2 * MR], _C2);
    _mm512_storeu_pd(&_C[3 * MR], _C3);
    _mm512_storeu_pd(&_C[4 * MR], _C4);
    _mm512_storeu_pd(&_C[5 * MR], _C5);
    _mm512_storeu_pd(&_C[6 * MR], _C6);
    _mm512_storeu_pd(&_C[7 * MR], _C7);
    _mm512_storeu_pd(&_C[8 * MR], _C8);
    _mm512_storeu_pd(&_C[9 * MR], _C9);
    _mm512_storeu_pd(&_C[10 * MR], _C10);
    _mm512_storeu_pd(&_C[11 * MR], _C11);
    _mm512_storeu_pd(&_C[12 * MR], _C12);
    _mm512_storeu_pd(&_C[13 * MR], _C13);
    _mm512_storeu_pd(&_C[14 * MR], _C14);
    _mm512_storeu_pd(&_C[15 * MR], _C15);
    _mm512_storeu_pd(&_C[16 * MR], _C16);
    _mm512_storeu_pd(&_C[17 * MR], _C17);
    _mm512_storeu_pd(&_C[18 * MR], _C18);
    _mm512_storeu_pd(&_C[19 * MR], _C19);
}

static void micro_dxpy(uint64_t mmm, uint64_t nnn, double *restrict C,
                       uint64_t ldc, const double *restrict _C)
{
    for (uint64_t i = 0; i < nnn; ++i)
    {
#ifdef USE_CILKPLUS
        C [0:mmm] += _C [0:mmm];
#else
        for (uint64_t j = 0; j < mmm; ++j)
        {
            C[j] += _C[j];
        }
#endif
        C += ldc;
        _C += MR;
    }
}

static void inner_kernel_n40_ap(
    uint64_t mm,
    uint64_t nn,
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    uint64_t ldc,
    double *restrict _C)
{
    uint64_t mmc = (mm + MR - 1) / MR;
    uint64_t mmr = mm % MR;
    uint64_t nnc = (nn + NR - 1) / NR;
    uint64_t nnr = nn % NR;

    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            uint64_t mmm = (mmi + 1 < mmc || mmr == 0) ? MR : mmr;

            if (mmm == MR && nnn == NR)
            {
                micro_kernel0_ap(kk, _A + mmi * MR * kk, _B + nni * NR * kk, C + mmi * MR + nni * NR * ldc, ldc);
            }
            else
            {
                micro_kernel1_ap(kk, _A + mmi * MR * kk, _B + nni * NR * kk, _C);
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
#ifdef USE_CILKPLUS
            _A [mmi * MR * kk + j * MR:mmm] = A [mmi * MR + j * lda:mmm];
#else
            memcpy(_A + mmi * MR * kk + j * MR, A + mmi * MR + j * lda, sizeof(double) * mmm);
#endif
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
#ifdef USE_CILKPLUS
        for (uint64_t i = 0; i < mmc; ++i)
        {
            for (uint64_t j = kkc * CACHE_ELEM; j < kk; ++j)
            {
                _A [i * MR * kk + j * MR:MR] = A [i * MR * lda + j:MR:lda];
            }
        }
#else
        asdf;
#endif
    }
    _A += mmc * kk * MR;
    A += mmc * lda * MR;
    if (mmr > 0)
    {
        for (uint64_t j = 0; j < kk; ++j)
        {
#ifdef USE_CILKPLUS
            _A [0:mmr] = A [0:mmr:lda];
#else
            for (uint64_t k = 0; k < r; ++k)
            {
                _A[k] = A[k * lda];
            }
#endif
            _A += MR;
            A += 1;
        }
    }
#else
#ifdef USE_CILKPLUS
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
#else
    const uint64_t q = mm / MR;
    const uint64_t r = mm % MR;
    for (uint64_t i = 0; i < q; ++i)
    {
        for (uint64_t j = 0; j < kk; ++j)
        {
            for (uint64_t k = 0; k < MR; ++k)
            {
                _A[j * MR + i * kk * MR + k] = A[j + i * MR * lda + k * lda];
            }
        }
    }
    _A += q * kk * MR;
    A += q * lda * MR;
    if (r > 0)
    {
        for (uint64_t j = 0; j < kk; ++j)
        {
            for (uint64_t k = 0; k < r; ++k)
            {
                _A[k] = A[k * lda];
            }
            _A += MR;
            A += 1;
        }
    }
#endif
#endif
}

static void packbcr(uint64_t kk, uint64_t nn, const double *restrict B, uint64_t ldb,
                    double *restrict _B)
{
#ifdef USE_CILKPLUS
    const uint64_t nnc = nn / NR;
    const uint64_t nnr = nn % NR;
#pragma omp parallel for
    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        for (uint64_t kki = 0; kki < kk; ++kki)
        {
            _B [nni * NR * kk + kki * NR:NR] = B [nni * NR * ldb + kki:NR:ldb];
        }
    }
    if (nnr > 0)
    {
        for (uint64_t kki = 0; kki < kk; ++kki)
        {
            _B [0:nnr] = B [0:nnr:ldb];
            _B += NR;
            B += 1;
        }
    }
#else
    const uint64_t nnc = (nn + NR - 1) / NR;
    const uint64_t nnr = nn % NR;
#pragma omp parallel for
    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
        for (uint64_t i = 0; i < nnn; ++i)
        {
            for (uint64_t j = 0; j < kk; ++j)
            {
                _B[nni * NR * kk + i + j * NR] = B[nni * NR * ldb + i * ldb + j];
            }
        }
    }
#endif
}

static void packbrr(uint64_t kk, uint64_t nn, const double *restrict B, uint64_t ldb,
                    double *restrict _B)
{
#ifdef USE_CILKPLUS
    const uint64_t nnc = nn / NR;
    const uint64_t nnr = nn % NR;
#pragma omp parallel for
    for (uint64_t kki = 0; kki < kk; ++kki)
    {
        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            _B [kki * NR + nni * kk * NR:NR] = B [kki * ldb + nni * NR:NR];
        }
    }
    _B += nnc * NR * kk;
    B += nnc * NR;
    if (nnr > 0)
    {
        for (uint64_t kki = 0; kki < kk; ++kki)
        {
            _B [0:nnr] = B [0:nnr];
            _B += NR;
            B += ldb;
        }
    }
#else
    const uint64_t nnc = (nn + NR - 1) / NR;
    const uint64_t nnr = nn % NR;
#pragma omp parallel for
    for (uint64_t j = 0; j < kk; ++j)
    {
        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
            memcpy(_B + j * NR + nni * kk * NR, B + j * ldb + nni * NR, sizeof(double) * nnn);
        }
    }
#endif
}

void userdgemm_n40_apbz(const char *transa, const char *transb, const int *_m,
                        const int *_n, const int *_k, const double *_alpha,
                        const double *restrict A, const int *_lda,
                        const double *restrict B, const int *_ldb,
                        const double *_beta, double *restrict C, const int *_ldc)
{
    const uint64_t nota = (int)(transa[0] == 'N' || transa[0] == 'n');
    const uint64_t notb = (int)(transb[0] == 'N' || transb[0] == 'n');
    const uint64_t m = (int)(*_m);
    // const uint64_t n = (int)(*_n);
    const uint64_t k = (int)(*_k);
    const uint64_t lda = (int)(*_lda);
    const uint64_t ldb = (int)(*_ldb);
    const uint64_t ldc = (int)(*_ldc);

    const uint64_t mc = (m + MB - 1) / MB;
    const uint64_t mr = m % MB;
    const uint64_t kc = (k + KB - 1) / KB;
    const uint64_t kr = k % KB;
    // const uint64_t nc = (n + NB - 1) / NB;
    // const uint64_t nr = n % NB;

    // beta = 0.0
#pragma omp parallel for
    for (uint64_t i = 0; i < NB; ++i)
    {
        memset(C + i * ldc, 0, sizeof(double) * m);
    }

    double *_B;
    ALLOC(_B, sizeof(double) * KB * NB);

    for (uint64_t ki = 0; ki < kc; ++ki)
    {
        const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

        if (notb)
            packbcr(kk, NB, B + ki * KB, ldb, _B);
        else
            packbrr(kk, NB, B + ki * KB * ldb, ldb, _B);

#pragma omp parallel
        {
            double *_A;
            double *_C;

            ALLOC(_A, sizeof(double) * MB * KB);
            ALLOC(_C, sizeof(double) * MR * NR);

#pragma omp for
            for (uint64_t mi = 0; mi < mc; ++mi)
            {
                const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

                if (nota)
                    packacc(mm, kk, A + mi * MB + ki * KB * lda, lda, _A);
                else
                    packarc(mm, kk, A + mi * MB * lda + ki * KB, lda, _A);

                inner_kernel_n40_ap(mm, NB, kk, _A, _B, C + mi * MB, ldc, _C);
            }

            FREE(_A);
            FREE(_C);
        }
    }

    FREE(_B);
}
