// alpha = -1.0
// beta  = +1.0

// #define ONE_CORE
// #define TUYEN

#include <stdint.h>
#include <immintrin.h>
#include <string.h>

#include "cblas_format.h"
#include "common.h"

#define MR 8
#define NR 31
#define MB 2400 // 2400
#define NB 31   // 31
#define KB 1620 // 1620

#define L1_DIST_A (MR * 20)
#define L1_DIST_B (NR * 20)

static void micro_kernel0(const uint64_t kk, const double *restrict A,
                          const double *restrict B, double *restrict C,
                          const uint64_t ldc)
{
    __m512d a0;
    __m512d c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14,
        c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28,
        c29, c30;

    c0 = _mm512_loadu_pd(C + 0 * ldc);
    c1 = _mm512_loadu_pd(C + 1 * ldc);
    c2 = _mm512_loadu_pd(C + 2 * ldc);
    c3 = _mm512_loadu_pd(C + 3 * ldc);
    c4 = _mm512_loadu_pd(C + 4 * ldc);
    c5 = _mm512_loadu_pd(C + 5 * ldc);
    c6 = _mm512_loadu_pd(C + 6 * ldc);
    c7 = _mm512_loadu_pd(C + 7 * ldc);
    c8 = _mm512_loadu_pd(C + 8 * ldc);
    c9 = _mm512_loadu_pd(C + 9 * ldc);
    c10 = _mm512_loadu_pd(C + 10 * ldc);
    c11 = _mm512_loadu_pd(C + 11 * ldc);
    c12 = _mm512_loadu_pd(C + 12 * ldc);
    c13 = _mm512_loadu_pd(C + 13 * ldc);
    c14 = _mm512_loadu_pd(C + 14 * ldc);
    c15 = _mm512_loadu_pd(C + 15 * ldc);
    c16 = _mm512_loadu_pd(C + 16 * ldc);
    c17 = _mm512_loadu_pd(C + 17 * ldc);
    c18 = _mm512_loadu_pd(C + 18 * ldc);
    c19 = _mm512_loadu_pd(C + 19 * ldc);
    c20 = _mm512_loadu_pd(C + 20 * ldc);
    c21 = _mm512_loadu_pd(C + 21 * ldc);
    c22 = _mm512_loadu_pd(C + 22 * ldc);
    c23 = _mm512_loadu_pd(C + 23 * ldc);
    c24 = _mm512_loadu_pd(C + 24 * ldc);
    c25 = _mm512_loadu_pd(C + 25 * ldc);
    c26 = _mm512_loadu_pd(C + 26 * ldc);
    c27 = _mm512_loadu_pd(C + 27 * ldc);
    c28 = _mm512_loadu_pd(C + 28 * ldc);
    c29 = _mm512_loadu_pd(C + 29 * ldc);
    c30 = _mm512_loadu_pd(C + 30 * ldc);

#pragma unroll(3)
    for (uint64_t i = 0; i < kk; ++i)
    {
        _mm_prefetch(A + L1_DIST_A + 0x0, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x0, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x8, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x10, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x18, _MM_HINT_T0);

        a0 = _mm512_load_pd(A);
        c0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), a0, c0);
        c1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), a0, c1);
        c2 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), a0, c2);
        c3 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), a0, c3);
        c4 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), a0, c4);
        c5 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), a0, c5);
        c6 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), a0, c6);
        c7 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), a0, c7);
        c8 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), a0, c8);
        c9 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), a0, c9);
        c10 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), a0, c10);
        c11 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), a0, c11);
        c12 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), a0, c12);
        c13 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), a0, c13);
        c14 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), a0, c14);
        c15 = _mm512_fnmadd_pd(_mm512_set1_pd(B[15]), a0, c15);
        c16 = _mm512_fnmadd_pd(_mm512_set1_pd(B[16]), a0, c16);
        c17 = _mm512_fnmadd_pd(_mm512_set1_pd(B[17]), a0, c17);
        c18 = _mm512_fnmadd_pd(_mm512_set1_pd(B[18]), a0, c18);
        c19 = _mm512_fnmadd_pd(_mm512_set1_pd(B[19]), a0, c19);
        c20 = _mm512_fnmadd_pd(_mm512_set1_pd(B[20]), a0, c20);
        c21 = _mm512_fnmadd_pd(_mm512_set1_pd(B[21]), a0, c21);
        c22 = _mm512_fnmadd_pd(_mm512_set1_pd(B[22]), a0, c22);
        c23 = _mm512_fnmadd_pd(_mm512_set1_pd(B[23]), a0, c23);
        c24 = _mm512_fnmadd_pd(_mm512_set1_pd(B[24]), a0, c24);
        c25 = _mm512_fnmadd_pd(_mm512_set1_pd(B[25]), a0, c25);
        c26 = _mm512_fnmadd_pd(_mm512_set1_pd(B[26]), a0, c26);
        c27 = _mm512_fnmadd_pd(_mm512_set1_pd(B[27]), a0, c27);
        c28 = _mm512_fnmadd_pd(_mm512_set1_pd(B[28]), a0, c28);
        c29 = _mm512_fnmadd_pd(_mm512_set1_pd(B[29]), a0, c29);
        c30 = _mm512_fnmadd_pd(_mm512_set1_pd(B[30]), a0, c30);

        A += MR;
        B += NR;
    }

    _mm512_storeu_pd(C + 0 * ldc, c0);
    _mm512_storeu_pd(C + 1 * ldc, c1);
    _mm512_storeu_pd(C + 2 * ldc, c2);
    _mm512_storeu_pd(C + 3 * ldc, c3);
    _mm512_storeu_pd(C + 4 * ldc, c4);
    _mm512_storeu_pd(C + 5 * ldc, c5);
    _mm512_storeu_pd(C + 6 * ldc, c6);
    _mm512_storeu_pd(C + 7 * ldc, c7);
    _mm512_storeu_pd(C + 8 * ldc, c8);
    _mm512_storeu_pd(C + 9 * ldc, c9);
    _mm512_storeu_pd(C + 10 * ldc, c10);
    _mm512_storeu_pd(C + 11 * ldc, c11);
    _mm512_storeu_pd(C + 12 * ldc, c12);
    _mm512_storeu_pd(C + 13 * ldc, c13);
    _mm512_storeu_pd(C + 14 * ldc, c14);
    _mm512_storeu_pd(C + 15 * ldc, c15);
    _mm512_storeu_pd(C + 16 * ldc, c16);
    _mm512_storeu_pd(C + 17 * ldc, c17);
    _mm512_storeu_pd(C + 18 * ldc, c18);
    _mm512_storeu_pd(C + 19 * ldc, c19);
    _mm512_storeu_pd(C + 20 * ldc, c20);
    _mm512_storeu_pd(C + 21 * ldc, c21);
    _mm512_storeu_pd(C + 22 * ldc, c22);
    _mm512_storeu_pd(C + 23 * ldc, c23);
    _mm512_storeu_pd(C + 24 * ldc, c24);
    _mm512_storeu_pd(C + 25 * ldc, c25);
    _mm512_storeu_pd(C + 26 * ldc, c26);
    _mm512_storeu_pd(C + 27 * ldc, c27);
    _mm512_storeu_pd(C + 28 * ldc, c28);
    _mm512_storeu_pd(C + 29 * ldc, c29);
    _mm512_storeu_pd(C + 30 * ldc, c30);
}

static void micro_kernel1(const uint64_t kk, const double *restrict A,
                          const double *restrict B, double *restrict C)
{
    __m512d a0;
    __m512d c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14,
        c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28,
        c29, c30;

    c0 = _mm512_setzero_pd();
    c1 = _mm512_setzero_pd();
    c2 = _mm512_setzero_pd();
    c3 = _mm512_setzero_pd();
    c4 = _mm512_setzero_pd();
    c5 = _mm512_setzero_pd();
    c6 = _mm512_setzero_pd();
    c7 = _mm512_setzero_pd();
    c8 = _mm512_setzero_pd();
    c9 = _mm512_setzero_pd();
    c10 = _mm512_setzero_pd();
    c11 = _mm512_setzero_pd();
    c12 = _mm512_setzero_pd();
    c13 = _mm512_setzero_pd();
    c14 = _mm512_setzero_pd();
    c15 = _mm512_setzero_pd();
    c16 = _mm512_setzero_pd();
    c17 = _mm512_setzero_pd();
    c18 = _mm512_setzero_pd();
    c19 = _mm512_setzero_pd();
    c20 = _mm512_setzero_pd();
    c21 = _mm512_setzero_pd();
    c22 = _mm512_setzero_pd();
    c23 = _mm512_setzero_pd();
    c24 = _mm512_setzero_pd();
    c25 = _mm512_setzero_pd();
    c26 = _mm512_setzero_pd();
    c27 = _mm512_setzero_pd();
    c28 = _mm512_setzero_pd();
    c29 = _mm512_setzero_pd();
    c30 = _mm512_setzero_pd();

#pragma unroll(3)
    for (uint64_t i = 0; i < kk; ++i)
    {
        _mm_prefetch(A + L1_DIST_A + 0x0, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x0, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x8, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x10, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0x18, _MM_HINT_T0);

        a0 = _mm512_load_pd(A);
        c0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), a0, c0);
        c1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), a0, c1);
        c2 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), a0, c2);
        c3 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), a0, c3);
        c4 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), a0, c4);
        c5 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), a0, c5);
        c6 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), a0, c6);
        c7 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), a0, c7);
        c8 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), a0, c8);
        c9 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), a0, c9);
        c10 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), a0, c10);
        c11 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), a0, c11);
        c12 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), a0, c12);
        c13 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), a0, c13);
        c14 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), a0, c14);
        c15 = _mm512_fnmadd_pd(_mm512_set1_pd(B[15]), a0, c15);
        c16 = _mm512_fnmadd_pd(_mm512_set1_pd(B[16]), a0, c16);
        c17 = _mm512_fnmadd_pd(_mm512_set1_pd(B[17]), a0, c17);
        c18 = _mm512_fnmadd_pd(_mm512_set1_pd(B[18]), a0, c18);
        c19 = _mm512_fnmadd_pd(_mm512_set1_pd(B[19]), a0, c19);
        c20 = _mm512_fnmadd_pd(_mm512_set1_pd(B[20]), a0, c20);
        c21 = _mm512_fnmadd_pd(_mm512_set1_pd(B[21]), a0, c21);
        c22 = _mm512_fnmadd_pd(_mm512_set1_pd(B[22]), a0, c22);
        c23 = _mm512_fnmadd_pd(_mm512_set1_pd(B[23]), a0, c23);
        c24 = _mm512_fnmadd_pd(_mm512_set1_pd(B[24]), a0, c24);
        c25 = _mm512_fnmadd_pd(_mm512_set1_pd(B[25]), a0, c25);
        c26 = _mm512_fnmadd_pd(_mm512_set1_pd(B[26]), a0, c26);
        c27 = _mm512_fnmadd_pd(_mm512_set1_pd(B[27]), a0, c27);
        c28 = _mm512_fnmadd_pd(_mm512_set1_pd(B[28]), a0, c28);
        c29 = _mm512_fnmadd_pd(_mm512_set1_pd(B[29]), a0, c29);
        c30 = _mm512_fnmadd_pd(_mm512_set1_pd(B[30]), a0, c30);

        A += MR;
        B += NR;
    }

    _mm512_store_pd(C + 0 * MR, c0);
    _mm512_store_pd(C + 1 * MR, c1);
    _mm512_store_pd(C + 2 * MR, c2);
    _mm512_store_pd(C + 3 * MR, c3);
    _mm512_store_pd(C + 4 * MR, c4);
    _mm512_store_pd(C + 5 * MR, c5);
    _mm512_store_pd(C + 6 * MR, c6);
    _mm512_store_pd(C + 7 * MR, c7);
    _mm512_store_pd(C + 8 * MR, c8);
    _mm512_store_pd(C + 9 * MR, c9);
    _mm512_store_pd(C + 10 * MR, c10);
    _mm512_store_pd(C + 11 * MR, c11);
    _mm512_store_pd(C + 12 * MR, c12);
    _mm512_store_pd(C + 13 * MR, c13);
    _mm512_store_pd(C + 14 * MR, c14);
    _mm512_store_pd(C + 15 * MR, c15);
    _mm512_store_pd(C + 16 * MR, c16);
    _mm512_store_pd(C + 17 * MR, c17);
    _mm512_store_pd(C + 18 * MR, c18);
    _mm512_store_pd(C + 19 * MR, c19);
    _mm512_store_pd(C + 20 * MR, c20);
    _mm512_store_pd(C + 21 * MR, c21);
    _mm512_store_pd(C + 22 * MR, c22);
    _mm512_store_pd(C + 23 * MR, c23);
    _mm512_store_pd(C + 24 * MR, c24);
    _mm512_store_pd(C + 25 * MR, c25);
    _mm512_store_pd(C + 26 * MR, c26);
    _mm512_store_pd(C + 27 * MR, c27);
    _mm512_store_pd(C + 28 * MR, c28);
    _mm512_store_pd(C + 29 * MR, c29);
    _mm512_store_pd(C + 30 * MR, c30);
}

static void micro_dxpy(const uint64_t mmm, const uint64_t nnn, double *restrict C,
                       const uint64_t ldc, const double *restrict _C)
{
    for (uint64_t i = 0; i < nnn; ++i)
    {
        for (uint64_t j = 0; j < mmm; ++j)
        {
            C[i * ldc + j] += _C[i * MR + j];
        }
    }
}

static void inner_kernel(const uint64_t mm, const uint64_t nn, const uint64_t kk,
                         const double *restrict _A, const double *restrict _B,
                         double *restrict C, const uint64_t ldc)
{
    const uint64_t mmc = (mm + MR - 1) / MR;
    const uint64_t mmr = mm % MR;
    const uint64_t nnc = (nn + NR - 1) / NR;
    const uint64_t nnr = nn % NR;

    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        const uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

            if (mmm == MR && nnn == NR)
            {
                micro_kernel0(kk, _A + mmi * MR * kk, _B + nni * NR * kk,
                              C + mmi * MR + nni * NR * ldc, ldc);
            }
            else
            {
                double _C[MR * NR] __attribute__((aligned(64)));
                micro_kernel1(kk, _A + mmi * MR * kk, _B + nni * NR * kk, _C);
                micro_dxpy(mmm, nnn, C + mmi * MR + nni * NR * ldc, ldc, _C);
            }
        }
    }
}

static void packacc(const uint64_t mm, const uint64_t kk, const double *restrict A,
                    const uint64_t lda, double *restrict _A)
{
    const uint64_t mmc = (mm + MR - 1) / MR;
    const uint64_t mmr = mm % MR;
#ifndef ONE_CORE
#pragma omp parallel for
#endif
    for (uint64_t j = 0; j < kk; ++j)
    {
        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            const uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;
            memcpy(_A + mmi * MR * kk + j * MR, A + mmi * MR + j * lda,
                   sizeof(double) * mmm);
        }
    }
}

static void packarc(const uint64_t mm, const uint64_t kk, const double *restrict A,
                    const uint64_t lda, double *restrict _A)
{
    const uint64_t mmc = (mm + MR - 1) / MR;
    const uint64_t mmr = mm % MR;
#ifndef ONE_CORE
#pragma omp parallel for
#endif
    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        const uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;
        for (uint64_t i = 0; i < mmm; ++i)
        {
            for (uint64_t j = 0; j < kk; ++j)
            {
                _A[mmi * MR * kk + i + j * MR] =
                    A[mmi * MR * lda + i * lda + j];
            }
        }
    }
}

static void packbcr(const uint64_t kk, const uint64_t nn, const double *restrict B,
                    const uint64_t ldb, double *restrict _B)
{
    const uint64_t nnc = (nn + NR - 1) / NR;
    const uint64_t nnr = nn % NR;
    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
        for (uint64_t i = 0; i < nnn; ++i)
        {
            for (uint64_t j = 0; j < kk; ++j)
            {
                _B[nni * NR * kk + i + j * NR] =
                    B[nni * NR * ldb + i * ldb + j];
            }
        }
    }
}

static void packbrr(const uint64_t kk, const uint64_t nn, const double *restrict B,
                    const uint64_t ldb, double *restrict _B)
{
    const uint64_t nnc = (nn + NR - 1) / NR;
    const uint64_t nnr = nn % NR;
    for (uint64_t j = 0; j < kk; ++j)
    {
        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
            memcpy(_B + j * NR + nni * kk * NR, B + j * ldb + nni * NR,
                   sizeof(double) * nnn);
        }
    }
}

void userdgemm_general(const char *transa, const char *transb, const int *_m,
                       const int *_n, const int *_k, const double *_alpha,
                       const double *restrict A, const int *_lda,
                       const double *restrict B, const int *_ldb,
                       const double *_beta, double *restrict C, const int *_ldc)
/*void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, const int64_t m, const int64_t n,
                const int64_t k, const double alpha, const double *A,
                const int64_t lda, const double *B, const int64_t ldb,
                const double beta, double *C, const int64_t ldc)*/
{
    // const char *transa = TransA == CblasNoTrans ? "N" : "T";
    // const char *transb = TransB == CblasNoTrans ? "N" : "T";

    const uint64_t nota = (int)(transa[0] == 'N' || transa[0] == 'n');
    const uint64_t notb = (int)(transb[0] == 'N' || transb[0] == 'n');

    const uint64_t m = *_m;
    const uint64_t n = *_n;
    const uint64_t k = *_k;
    const uint64_t lda = *_lda;
    const uint64_t ldb = *_ldb;
    const uint64_t ldc = *_ldc;

    const uint64_t mc = (m + MB - 1) / MB;
    const uint64_t mr = m % MB;
    const uint64_t nc = (n + NB - 1) / NB;
    const uint64_t nr = n % NB;
    const uint64_t kc = (k + KB - 1) / KB;
    const uint64_t kr = k % KB;

#ifndef TUYEN
    double *_A;
    ALLOC(_A, sizeof(double) * MB * KB);
#ifdef ONE_CORE
    double *_B;
    ALLOC(_B, sizeof(double) * KB * NB);
#endif

    for (uint64_t mi = 0; mi < mc; ++mi)
    {
        const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            if (nota)
                packacc(mm, kk, A + mi * MB + ki * KB * lda, lda, _A);
            else
                packarc(mm, kk, A + mi * MB * lda + ki * KB, lda, _A);

#ifndef ONE_CORE
#pragma omp parallel
            {
                double *_B;
                ALLOC(_B, sizeof(double) * KB * NB);

#pragma omp for
                for (uint64_t ni = 0; ni < nc; ++ni)
                {
                    const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

                    if (notb)
                        packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);
                    else
                        packbrr(kk, nn, B + ki * KB * ldb + ni * NB, ldb, _B);

                    inner_kernel(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
                }

                FREE(_B, sizeof(double) * KB * NB);
            }
#else
            for (uint64_t ni = 0; ni < nc; ++ni)
            {
                const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

                if (notb)
                    packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);
                else
                    packbrr(kk, nn, B + ki * KB * ldb + ni * NB, ldb, _B);

                inner_kernel(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
            }

#endif
        }
    }

    FREE(_A, sizeof(double) * MB * KB);
#ifdef ONE_CORE
    FREE(_B, sizeof(double) * KB * NB);
#endif
#else
    double *_A;
    double *_B;
    ALLOC(_A, sizeof(double) * MB * KB);
    ALLOC(_B, sizeof(double) * KB * NB);

    for (uint64_t mi = 0; mi < mc; ++mi)
    {
        const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            if (nota)
                packacc(mm, kk, A + mi * MB + ki * KB * lda, lda, _A);
            else
                packarc(mm, kk, A + mi * MB * lda + ki * KB, lda, _A);

            for (uint64_t ni = 0; ni < nc; ++ni)
            {
                const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

                if (notb)
                    packbcr(kk, nn, B + ki * KB + ni * NB * ldb, ldb, _B);
                else
                    packbrr(kk, nn, B + ki * KB * ldb + ni * NB, ldb, _B);

                inner_kernel(mm, nn, kk, _A, _B, C + mi * MB + ni * NB * ldc, ldc);
            }
        }
    }

    FREE(_A, sizeof(double) * MB * KB);
    FREE(_B, sizeof(double) * KB * NB);
#endif
}
