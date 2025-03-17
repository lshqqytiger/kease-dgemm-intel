// alpha = -1.0
// beta  = +1.0

#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

#include "common.h"

#define MR 16
#define NR 15
#define MB 25400
#define NB 45
#define KB 40
#define NU 1

#define L1_DIST_A (MR * 20)
#define L1_DIST_B (NR * 20)

static void micro_kernel0_k40_an(const double *restrict A, const double *restrict B,
                                 double *restrict C, uint64_t ldc)
{
    register __m512d _A0, _A1;
    register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0, _C14_0;
    register __m512d _C0_1, _C1_1, _C2_1, _C3_1, _C4_1, _C5_1, _C6_1, _C7_1, _C8_1, _C9_1, _C10_1, _C11_1, _C12_1, _C13_1, _C14_1;
    _C0_0 = _mm512_loadu_pd(&C[0 * ldc + 0]);
    _C0_1 = _mm512_loadu_pd(&C[0 * ldc + 8]);
    _C1_0 = _mm512_loadu_pd(&C[1 * ldc + 0]);
    _C1_1 = _mm512_loadu_pd(&C[1 * ldc + 8]);
    _C2_0 = _mm512_loadu_pd(&C[2 * ldc + 0]);
    _C2_1 = _mm512_loadu_pd(&C[2 * ldc + 8]);
    _C3_0 = _mm512_loadu_pd(&C[3 * ldc + 0]);
    _C3_1 = _mm512_loadu_pd(&C[3 * ldc + 8]);
    _C4_0 = _mm512_loadu_pd(&C[4 * ldc + 0]);
    _C4_1 = _mm512_loadu_pd(&C[4 * ldc + 8]);
    _C5_0 = _mm512_loadu_pd(&C[5 * ldc + 0]);
    _C5_1 = _mm512_loadu_pd(&C[5 * ldc + 8]);
    _C6_0 = _mm512_loadu_pd(&C[6 * ldc + 0]);
    _C6_1 = _mm512_loadu_pd(&C[6 * ldc + 8]);
    _C7_0 = _mm512_loadu_pd(&C[7 * ldc + 0]);
    _C7_1 = _mm512_loadu_pd(&C[7 * ldc + 8]);
    _C8_0 = _mm512_loadu_pd(&C[8 * ldc + 0]);
    _C8_1 = _mm512_loadu_pd(&C[8 * ldc + 8]);
    _C9_0 = _mm512_loadu_pd(&C[9 * ldc + 0]);
    _C9_1 = _mm512_loadu_pd(&C[9 * ldc + 8]);
    _C10_0 = _mm512_loadu_pd(&C[10 * ldc + 0]);
    _C10_1 = _mm512_loadu_pd(&C[10 * ldc + 8]);
    _C11_0 = _mm512_loadu_pd(&C[11 * ldc + 0]);
    _C11_1 = _mm512_loadu_pd(&C[11 * ldc + 8]);
    _C12_0 = _mm512_loadu_pd(&C[12 * ldc + 0]);
    _C12_1 = _mm512_loadu_pd(&C[12 * ldc + 8]);
    _C13_0 = _mm512_loadu_pd(&C[13 * ldc + 0]);
    _C13_1 = _mm512_loadu_pd(&C[13 * ldc + 8]);
    _C14_0 = _mm512_loadu_pd(&C[14 * ldc + 0]);
    _C14_1 = _mm512_loadu_pd(&C[14 * ldc + 8]);

#pragma unroll(NU)
    for (uint64_t i = 0; i < KB; ++i)
    {
        _mm_prefetch(A + L1_DIST_A + 0, _MM_HINT_T0);
        _mm_prefetch(A + L1_DIST_A + 8, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 0, _MM_HINT_T0);
        _mm_prefetch(B + L1_DIST_B + 8, _MM_HINT_T0);

        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A1, _C0_1);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A1, _C1_1);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A1, _C2_1);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A1, _C3_1);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A1, _C4_1);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A1, _C5_1);
        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A1, _C6_1);
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A1, _C7_1);
        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A1, _C8_1);
        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
        _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A1, _C9_1);
        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
        _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A1, _C10_1);
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
        _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A1, _C11_1);
        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
        _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A1, _C12_1);
        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
        _C13_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A1, _C13_1);
        _C14_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), _A0, _C14_0);
        _C14_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), _A1, _C14_1);

        A += MR;
        B += NR;
    }

    _mm512_storeu_pd(&C[0 * ldc + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * ldc + 8], _C0_1);
    _mm512_storeu_pd(&C[1 * ldc + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * ldc + 8], _C1_1);
    _mm512_storeu_pd(&C[2 * ldc + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * ldc + 8], _C2_1);
    _mm512_storeu_pd(&C[3 * ldc + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * ldc + 8], _C3_1);
    _mm512_storeu_pd(&C[4 * ldc + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * ldc + 8], _C4_1);
    _mm512_storeu_pd(&C[5 * ldc + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * ldc + 8], _C5_1);
    _mm512_storeu_pd(&C[6 * ldc + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * ldc + 8], _C6_1);
    _mm512_storeu_pd(&C[7 * ldc + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * ldc + 8], _C7_1);
    _mm512_storeu_pd(&C[8 * ldc + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * ldc + 8], _C8_1);
    _mm512_storeu_pd(&C[9 * ldc + 0], _C9_0);
    _mm512_storeu_pd(&C[9 * ldc + 8], _C9_1);
    _mm512_storeu_pd(&C[10 * ldc + 0], _C10_0);
    _mm512_storeu_pd(&C[10 * ldc + 8], _C10_1);
    _mm512_storeu_pd(&C[11 * ldc + 0], _C11_0);
    _mm512_storeu_pd(&C[11 * ldc + 8], _C11_1);
    _mm512_storeu_pd(&C[12 * ldc + 0], _C12_0);
    _mm512_storeu_pd(&C[12 * ldc + 8], _C12_1);
    _mm512_storeu_pd(&C[13 * ldc + 0], _C13_0);
    _mm512_storeu_pd(&C[13 * ldc + 8], _C13_1);
    _mm512_storeu_pd(&C[14 * ldc + 0], _C14_0);
    _mm512_storeu_pd(&C[14 * ldc + 8], _C14_1);
}

static void micro_kernel1_k40_an(const double *restrict A, const double *restrict B,
                                 double *restrict C)
{
    register __m512d _A0, _A1;
    register __m512d _C0_0, _C1_0, _C2_0, _C3_0, _C4_0, _C5_0, _C6_0, _C7_0, _C8_0, _C9_0, _C10_0, _C11_0, _C12_0, _C13_0, _C14_0;
    register __m512d _C0_1, _C1_1, _C2_1, _C3_1, _C4_1, _C5_1, _C6_1, _C7_1, _C8_1, _C9_1, _C10_1, _C11_1, _C12_1, _C13_1, _C14_1;
    _C0_0 = _mm512_setzero_pd();
    _C0_1 = _mm512_setzero_pd();
    _C1_0 = _mm512_setzero_pd();
    _C1_1 = _mm512_setzero_pd();
    _C2_0 = _mm512_setzero_pd();
    _C2_1 = _mm512_setzero_pd();
    _C3_0 = _mm512_setzero_pd();
    _C3_1 = _mm512_setzero_pd();
    _C4_0 = _mm512_setzero_pd();
    _C4_1 = _mm512_setzero_pd();
    _C5_0 = _mm512_setzero_pd();
    _C5_1 = _mm512_setzero_pd();
    _C6_0 = _mm512_setzero_pd();
    _C6_1 = _mm512_setzero_pd();
    _C7_0 = _mm512_setzero_pd();
    _C7_1 = _mm512_setzero_pd();
    _C8_0 = _mm512_setzero_pd();
    _C8_1 = _mm512_setzero_pd();
    _C9_0 = _mm512_setzero_pd();
    _C9_1 = _mm512_setzero_pd();
    _C10_0 = _mm512_setzero_pd();
    _C10_1 = _mm512_setzero_pd();
    _C11_0 = _mm512_setzero_pd();
    _C11_1 = _mm512_setzero_pd();
    _C12_0 = _mm512_setzero_pd();
    _C12_1 = _mm512_setzero_pd();
    _C13_0 = _mm512_setzero_pd();
    _C13_1 = _mm512_setzero_pd();
    _C14_0 = _mm512_setzero_pd();
    _C14_1 = _mm512_setzero_pd();

#pragma unroll(NU)
    for (uint64_t i = 0; i < KB; ++i)
    {
        _mm_prefetch((const void *)&A[L1_DIST_A + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&A[L1_DIST_A + 8], _MM_HINT_T0);
        _mm_prefetch((const void *)&B[L1_DIST_B + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&B[L1_DIST_B + 8], _MM_HINT_T0);

        _A0 = _mm512_loadu_pd(&A[0]);
        _A1 = _mm512_loadu_pd(&A[8]);

        _C0_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A0, _C0_0);
        _C0_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[0]), _A1, _C0_1);
        _C1_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A0, _C1_0);
        _C1_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[1]), _A1, _C1_1);
        _C2_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A0, _C2_0);
        _C2_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[2]), _A1, _C2_1);
        _C3_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A0, _C3_0);
        _C3_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[3]), _A1, _C3_1);
        _C4_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A0, _C4_0);
        _C4_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[4]), _A1, _C4_1);
        _C5_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A0, _C5_0);
        _C5_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[5]), _A1, _C5_1);
        _C6_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A0, _C6_0);
        _C6_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[6]), _A1, _C6_1);
        _C7_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A0, _C7_0);
        _C7_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[7]), _A1, _C7_1);
        _C8_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A0, _C8_0);
        _C8_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[8]), _A1, _C8_1);
        _C9_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A0, _C9_0);
        _C9_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[9]), _A1, _C9_1);
        _C10_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A0, _C10_0);
        _C10_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[10]), _A1, _C10_1);
        _C11_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A0, _C11_0);
        _C11_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[11]), _A1, _C11_1);
        _C12_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A0, _C12_0);
        _C12_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[12]), _A1, _C12_1);
        _C13_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A0, _C13_0);
        _C13_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[13]), _A1, _C13_1);
        _C14_0 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), _A0, _C14_0);
        _C14_1 = _mm512_fnmadd_pd(_mm512_set1_pd(B[14]), _A1, _C14_1);

        A += MR;
        B += NR;
    }

    _mm512_storeu_pd(&C[0 * MR + 0], _C0_0);
    _mm512_storeu_pd(&C[0 * MR + 8], _C0_1);
    _mm512_storeu_pd(&C[1 * MR + 0], _C1_0);
    _mm512_storeu_pd(&C[1 * MR + 8], _C1_1);
    _mm512_storeu_pd(&C[2 * MR + 0], _C2_0);
    _mm512_storeu_pd(&C[2 * MR + 8], _C2_1);
    _mm512_storeu_pd(&C[3 * MR + 0], _C3_0);
    _mm512_storeu_pd(&C[3 * MR + 8], _C3_1);
    _mm512_storeu_pd(&C[4 * MR + 0], _C4_0);
    _mm512_storeu_pd(&C[4 * MR + 8], _C4_1);
    _mm512_storeu_pd(&C[5 * MR + 0], _C5_0);
    _mm512_storeu_pd(&C[5 * MR + 8], _C5_1);
    _mm512_storeu_pd(&C[6 * MR + 0], _C6_0);
    _mm512_storeu_pd(&C[6 * MR + 8], _C6_1);
    _mm512_storeu_pd(&C[7 * MR + 0], _C7_0);
    _mm512_storeu_pd(&C[7 * MR + 8], _C7_1);
    _mm512_storeu_pd(&C[8 * MR + 0], _C8_0);
    _mm512_storeu_pd(&C[8 * MR + 8], _C8_1);
    _mm512_storeu_pd(&C[9 * MR + 0], _C9_0);
    _mm512_storeu_pd(&C[9 * MR + 8], _C9_1);
    _mm512_storeu_pd(&C[10 * MR + 0], _C10_0);
    _mm512_storeu_pd(&C[10 * MR + 8], _C10_1);
    _mm512_storeu_pd(&C[11 * MR + 0], _C11_0);
    _mm512_storeu_pd(&C[11 * MR + 8], _C11_1);
    _mm512_storeu_pd(&C[12 * MR + 0], _C12_0);
    _mm512_storeu_pd(&C[12 * MR + 8], _C12_1);
    _mm512_storeu_pd(&C[13 * MR + 0], _C13_0);
    _mm512_storeu_pd(&C[13 * MR + 8], _C13_1);
    _mm512_storeu_pd(&C[14 * MR + 0], _C14_0);
    _mm512_storeu_pd(&C[14 * MR + 8], _C14_1);
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

static void inner_loop_k40_an(const uint64_t mm, const uint64_t nn, const double *restrict _A,
                              const double *restrict _B, double *restrict C, const uint64_t ldc,
                              double *restrict _C)
{
    uint64_t mmc = (mm + MR - 1) / MR;
    uint64_t mmr = mm % MR;
    uint64_t nnc = (nn + NR - 1) / NR;
    uint64_t nnr = nn % NR;

    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        uint64_t mmm = (mmi + 1 < mmc || mmr == 0) ? MR : mmr;

        for (uint64_t nni = 0; nni < nnc; ++nni)
        {
            uint64_t nnn = (nni + 1 < nnc || nnr == 0) ? NR : nnr;

            if (mmm == MR && nnn == NR)
            {
                micro_kernel0_k40_an(_A + mmi * MR * KB, _B + nni * NR * KB, C + mmi * MR + nni * NR * ldc, ldc);
            }
            else
            {
                micro_kernel1_k40_an(_A + mmi * MR * KB, _B + nni * NR * KB, _C);
                micro_dxpy(mmm, nnn, C + mmi * MR + nni * NR * ldc, ldc, _C);
            }
        }
    }
}

static void packacc_k40(uint64_t mm, const double *restrict A, uint64_t lda,
                        double *restrict _A)
{
    uint64_t mmc = mm / MR;
    uint64_t mmr = mm % MR;

#pragma omp parallel for schedule(dynamic)
    for (uint64_t mmi = 0; mmi < mmc; ++mmi)
    {
        for (uint64_t ki = 0; ki < KB; ++ki)
        {
            _A [mmi * MR * KB + ki * MR:MR] = A [mmi * MR + ki * lda:MR];
        }
    }

    if (mmr > 0)
    {
        A += mmc * MR;
        _A += mmc * MR * KB;
        for (uint64_t ki = 0; ki < KB; ++ki)
        {
            _A [ki * MR:mmr] = A [ki * lda:mmr];
        }
    }
}

static void packbrr_k40(uint64_t nn, const double *restrict B, uint64_t ldb,
                        double *restrict _B)
{
    uint64_t nnc = nn / NR;
    uint64_t nnr = nn % NR;

    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        for (uint64_t ki = 0; ki < KB; ++ki)
        {
            _B [nni * NR * KB + ki * NR:NR] = B [nni * NR + ki * ldb:NR];
        }
    }

    if (nnr > 0)
    {
        B += nnc * NR;
        _B += nnc * NR * KB;
        for (uint64_t ki = 0; ki < KB; ++ki)
        {
            _B [ki * NR:nnr] = B [ki * ldb:nnr];
        }
    }
}

void userdgemm_k40_nt_anbp(const char *transa, const char *transb, const int *_m,
                           const int *_n, const int *_k, const double *_alpha,
                           const double *restrict A, const int *_lda,
                           const double *restrict B, const int *_ldb,
                           const double *_beta, double *restrict C, const int *_ldc)
{
    const uint64_t nota = (int)(transa[0] == 'N' || transa[0] == 'n');
    const uint64_t notb = (int)(transb[0] == 'N' || transb[0] == 'n');
    const uint64_t m = (int)(*_m);
    const uint64_t n = (int)(*_n);
    const uint64_t k = (int)(*_k);
    const uint64_t lda = (int)(*_lda);
    const uint64_t ldb = (int)(*_ldb);
    const uint64_t ldc = (int)(*_ldc);
    const uint64_t mc = (m + MB - 1) / MB;
    const uint64_t mr = m % MB;
    const uint64_t nc = (n + NB - 1) / NB;
    const uint64_t nr = n % NB;

    double *_A;
    ALLOC(_A, sizeof(double) * MB * KB);

    for (uint64_t mi = 0; mi < mc; ++mi)
    {
        uint64_t mm = (mi + 1 < mc || mr == 0) ? MB : mr;

        packacc_k40(mm, A + mi * MB, lda, _A);

#pragma omp parallel
        {
            double *_B;
            double *_C;
            ALLOC(_B, sizeof(double) * KB * NB);
            ALLOC(_C, sizeof(double) * MR * NR);

#pragma omp for
            for (uint64_t ni = 0; ni < nc; ++ni)
            {
                uint64_t nn = (ni + 1 < nc || nr == 0) ? NB : nr;

                packbrr_k40(nn, B + ni * NB, ldb, _B);
                inner_loop_k40_an(mm, nn, _A, _B, C + mi * MB + ni * NB * ldc, ldc, _C);
            }

            FREE(_B, sizeof(double) * KB * NB);
            FREE(_C, sizeof(double) * MR * NR);
        }
    }

    FREE(_A, sizeof(double) * MB * KB);
}
