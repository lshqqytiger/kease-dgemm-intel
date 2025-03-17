#define ONE_CORE

#ifndef FORTRAN_FORMAT
#include "cblas_format.h"
#endif

#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#ifndef ONE_CORE
#include <omp.h>
#endif

#include "common.h"

#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define MR 31
#define NR 8
#define MB (MR * 3)
#define NB (NR * 2500)
#define KB 536

#ifndef ONE_CORE
#define NT1 17
#define NT2 4
#endif

#define L1_DIST_A 620
#define L1_DIST_B 160

void micro_kernel0(int k, const double *A, const double *B, double *C, int ncol)
{
    int i;
    register __m512d _B0;
    register __m512d _C0_0, _C0_1, _C0_2, _C0_3, _C0_4, _C0_5, _C0_6, _C0_7, _C0_8, _C0_9, _C0_10, _C0_11, _C0_12, _C0_13, _C0_14, _C0_15, _C0_16, _C0_17, _C0_18, _C0_19, _C0_20, _C0_21, _C0_22, _C0_23, _C0_24, _C0_25, _C0_26, _C0_27, _C0_28, _C0_29, _C0_30;
    _C0_0 = _mm512_loadu_pd(&C[0 * ncol + 0]);
    _C0_1 = _mm512_loadu_pd(&C[1 * ncol + 0]);
    _C0_2 = _mm512_loadu_pd(&C[2 * ncol + 0]);
    _C0_3 = _mm512_loadu_pd(&C[3 * ncol + 0]);
    _C0_4 = _mm512_loadu_pd(&C[4 * ncol + 0]);
    _C0_5 = _mm512_loadu_pd(&C[5 * ncol + 0]);
    _C0_6 = _mm512_loadu_pd(&C[6 * ncol + 0]);
    _C0_7 = _mm512_loadu_pd(&C[7 * ncol + 0]);
    _C0_8 = _mm512_loadu_pd(&C[8 * ncol + 0]);
    _C0_9 = _mm512_loadu_pd(&C[9 * ncol + 0]);
    _C0_10 = _mm512_loadu_pd(&C[10 * ncol + 0]);
    _C0_11 = _mm512_loadu_pd(&C[11 * ncol + 0]);
    _C0_12 = _mm512_loadu_pd(&C[12 * ncol + 0]);
    _C0_13 = _mm512_loadu_pd(&C[13 * ncol + 0]);
    _C0_14 = _mm512_loadu_pd(&C[14 * ncol + 0]);
    _C0_15 = _mm512_loadu_pd(&C[15 * ncol + 0]);
    _C0_16 = _mm512_loadu_pd(&C[16 * ncol + 0]);
    _C0_17 = _mm512_loadu_pd(&C[17 * ncol + 0]);
    _C0_18 = _mm512_loadu_pd(&C[18 * ncol + 0]);
    _C0_19 = _mm512_loadu_pd(&C[19 * ncol + 0]);
    _C0_20 = _mm512_loadu_pd(&C[20 * ncol + 0]);
    _C0_21 = _mm512_loadu_pd(&C[21 * ncol + 0]);
    _C0_22 = _mm512_loadu_pd(&C[22 * ncol + 0]);
    _C0_23 = _mm512_loadu_pd(&C[23 * ncol + 0]);
    _C0_24 = _mm512_loadu_pd(&C[24 * ncol + 0]);
    _C0_25 = _mm512_loadu_pd(&C[25 * ncol + 0]);
    _C0_26 = _mm512_loadu_pd(&C[26 * ncol + 0]);
    _C0_27 = _mm512_loadu_pd(&C[27 * ncol + 0]);
    _C0_28 = _mm512_loadu_pd(&C[28 * ncol + 0]);
    _C0_29 = _mm512_loadu_pd(&C[29 * ncol + 0]);
    _C0_30 = _mm512_loadu_pd(&C[30 * ncol + 0]);
#pragma unroll(3)
    for (i = 0; i < k; i++)
    {
        _mm_prefetch((const void *)&A[L1_DIST_A + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&A[L1_DIST_A + 8], _MM_HINT_T0);
        _mm_prefetch((const void *)&A[L1_DIST_A + 16], _MM_HINT_T0);
        _mm_prefetch((const void *)&A[L1_DIST_A + 24], _MM_HINT_T0);
        _mm_prefetch((const void *)&B[L1_DIST_B + 0], _MM_HINT_T0);
        _B0 = _mm512_loadu_pd(&B[0]);
        _C0_0 = _mm512_fmadd_pd(_mm512_set1_pd(A[0]), _B0, _C0_0);
        _C0_1 = _mm512_fmadd_pd(_mm512_set1_pd(A[1]), _B0, _C0_1);
        _C0_2 = _mm512_fmadd_pd(_mm512_set1_pd(A[2]), _B0, _C0_2);
        _C0_3 = _mm512_fmadd_pd(_mm512_set1_pd(A[3]), _B0, _C0_3);
        _C0_4 = _mm512_fmadd_pd(_mm512_set1_pd(A[4]), _B0, _C0_4);
        _C0_5 = _mm512_fmadd_pd(_mm512_set1_pd(A[5]), _B0, _C0_5);
        _C0_6 = _mm512_fmadd_pd(_mm512_set1_pd(A[6]), _B0, _C0_6);
        _C0_7 = _mm512_fmadd_pd(_mm512_set1_pd(A[7]), _B0, _C0_7);
        _C0_8 = _mm512_fmadd_pd(_mm512_set1_pd(A[8]), _B0, _C0_8);
        _C0_9 = _mm512_fmadd_pd(_mm512_set1_pd(A[9]), _B0, _C0_9);
        _C0_10 = _mm512_fmadd_pd(_mm512_set1_pd(A[10]), _B0, _C0_10);
        _C0_11 = _mm512_fmadd_pd(_mm512_set1_pd(A[11]), _B0, _C0_11);
        _C0_12 = _mm512_fmadd_pd(_mm512_set1_pd(A[12]), _B0, _C0_12);
        _C0_13 = _mm512_fmadd_pd(_mm512_set1_pd(A[13]), _B0, _C0_13);
        _C0_14 = _mm512_fmadd_pd(_mm512_set1_pd(A[14]), _B0, _C0_14);
        _C0_15 = _mm512_fmadd_pd(_mm512_set1_pd(A[15]), _B0, _C0_15);
        _C0_16 = _mm512_fmadd_pd(_mm512_set1_pd(A[16]), _B0, _C0_16);
        _C0_17 = _mm512_fmadd_pd(_mm512_set1_pd(A[17]), _B0, _C0_17);
        _C0_18 = _mm512_fmadd_pd(_mm512_set1_pd(A[18]), _B0, _C0_18);
        _C0_19 = _mm512_fmadd_pd(_mm512_set1_pd(A[19]), _B0, _C0_19);
        _C0_20 = _mm512_fmadd_pd(_mm512_set1_pd(A[20]), _B0, _C0_20);
        _C0_21 = _mm512_fmadd_pd(_mm512_set1_pd(A[21]), _B0, _C0_21);
        _C0_22 = _mm512_fmadd_pd(_mm512_set1_pd(A[22]), _B0, _C0_22);
        _C0_23 = _mm512_fmadd_pd(_mm512_set1_pd(A[23]), _B0, _C0_23);
        _C0_24 = _mm512_fmadd_pd(_mm512_set1_pd(A[24]), _B0, _C0_24);
        _C0_25 = _mm512_fmadd_pd(_mm512_set1_pd(A[25]), _B0, _C0_25);
        _C0_26 = _mm512_fmadd_pd(_mm512_set1_pd(A[26]), _B0, _C0_26);
        _C0_27 = _mm512_fmadd_pd(_mm512_set1_pd(A[27]), _B0, _C0_27);
        _C0_28 = _mm512_fmadd_pd(_mm512_set1_pd(A[28]), _B0, _C0_28);
        _C0_29 = _mm512_fmadd_pd(_mm512_set1_pd(A[29]), _B0, _C0_29);
        _C0_30 = _mm512_fmadd_pd(_mm512_set1_pd(A[30]), _B0, _C0_30);
        A += MR;
        B += NR;
    }
    _mm512_storeu_pd(&C[0 * ncol + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ncol + 0], _C0_1);
    _mm512_storeu_pd(&C[2 * ncol + 0], _C0_2);
    _mm512_storeu_pd(&C[3 * ncol + 0], _C0_3);
    _mm512_storeu_pd(&C[4 * ncol + 0], _C0_4);
    _mm512_storeu_pd(&C[5 * ncol + 0], _C0_5);
    _mm512_storeu_pd(&C[6 * ncol + 0], _C0_6);
    _mm512_storeu_pd(&C[7 * ncol + 0], _C0_7);
    _mm512_storeu_pd(&C[8 * ncol + 0], _C0_8);
    _mm512_storeu_pd(&C[9 * ncol + 0], _C0_9);
    _mm512_storeu_pd(&C[10 * ncol + 0], _C0_10);
    _mm512_storeu_pd(&C[11 * ncol + 0], _C0_11);
    _mm512_storeu_pd(&C[12 * ncol + 0], _C0_12);
    _mm512_storeu_pd(&C[13 * ncol + 0], _C0_13);
    _mm512_storeu_pd(&C[14 * ncol + 0], _C0_14);
    _mm512_storeu_pd(&C[15 * ncol + 0], _C0_15);
    _mm512_storeu_pd(&C[16 * ncol + 0], _C0_16);
    _mm512_storeu_pd(&C[17 * ncol + 0], _C0_17);
    _mm512_storeu_pd(&C[18 * ncol + 0], _C0_18);
    _mm512_storeu_pd(&C[19 * ncol + 0], _C0_19);
    _mm512_storeu_pd(&C[20 * ncol + 0], _C0_20);
    _mm512_storeu_pd(&C[21 * ncol + 0], _C0_21);
    _mm512_storeu_pd(&C[22 * ncol + 0], _C0_22);
    _mm512_storeu_pd(&C[23 * ncol + 0], _C0_23);
    _mm512_storeu_pd(&C[24 * ncol + 0], _C0_24);
    _mm512_storeu_pd(&C[25 * ncol + 0], _C0_25);
    _mm512_storeu_pd(&C[26 * ncol + 0], _C0_26);
    _mm512_storeu_pd(&C[27 * ncol + 0], _C0_27);
    _mm512_storeu_pd(&C[28 * ncol + 0], _C0_28);
    _mm512_storeu_pd(&C[29 * ncol + 0], _C0_29);
    _mm512_storeu_pd(&C[30 * ncol + 0], _C0_30);
}

void micro_kernel1(int k, const double *A, const double *B, double *C, int ncol)
{
    int i;
    register __m512d _B0;
    register __m512d _C0_0, _C0_1, _C0_2, _C0_3, _C0_4, _C0_5, _C0_6, _C0_7, _C0_8, _C0_9, _C0_10, _C0_11, _C0_12, _C0_13, _C0_14, _C0_15, _C0_16, _C0_17, _C0_18, _C0_19, _C0_20, _C0_21, _C0_22, _C0_23, _C0_24, _C0_25, _C0_26, _C0_27, _C0_28, _C0_29, _C0_30;
    _C0_0 = _mm512_setzero_pd();
    _C0_1 = _mm512_setzero_pd();
    _C0_2 = _mm512_setzero_pd();
    _C0_3 = _mm512_setzero_pd();
    _C0_4 = _mm512_setzero_pd();
    _C0_5 = _mm512_setzero_pd();
    _C0_6 = _mm512_setzero_pd();
    _C0_7 = _mm512_setzero_pd();
    _C0_8 = _mm512_setzero_pd();
    _C0_9 = _mm512_setzero_pd();
    _C0_10 = _mm512_setzero_pd();
    _C0_11 = _mm512_setzero_pd();
    _C0_12 = _mm512_setzero_pd();
    _C0_13 = _mm512_setzero_pd();
    _C0_14 = _mm512_setzero_pd();
    _C0_15 = _mm512_setzero_pd();
    _C0_16 = _mm512_setzero_pd();
    _C0_17 = _mm512_setzero_pd();
    _C0_18 = _mm512_setzero_pd();
    _C0_19 = _mm512_setzero_pd();
    _C0_20 = _mm512_setzero_pd();
    _C0_21 = _mm512_setzero_pd();
    _C0_22 = _mm512_setzero_pd();
    _C0_23 = _mm512_setzero_pd();
    _C0_24 = _mm512_setzero_pd();
    _C0_25 = _mm512_setzero_pd();
    _C0_26 = _mm512_setzero_pd();
    _C0_27 = _mm512_setzero_pd();
    _C0_28 = _mm512_setzero_pd();
    _C0_29 = _mm512_setzero_pd();
    _C0_30 = _mm512_setzero_pd();
#pragma unroll(3)
    for (i = 0; i < k; i++)
    {
        _mm_prefetch((const void *)&A[L1_DIST_A + 0], _MM_HINT_T0);
        _mm_prefetch((const void *)&A[L1_DIST_A + 8], _MM_HINT_T0);
        _mm_prefetch((const void *)&A[L1_DIST_A + 16], _MM_HINT_T0);
        _mm_prefetch((const void *)&A[L1_DIST_A + 24], _MM_HINT_T0);
        _mm_prefetch((const void *)&B[L1_DIST_B + 0], _MM_HINT_T0);
        _B0 = _mm512_loadu_pd(&B[0]);
        _C0_0 = _mm512_fmadd_pd(_mm512_set1_pd(A[0]), _B0, _C0_0);
        _C0_1 = _mm512_fmadd_pd(_mm512_set1_pd(A[1]), _B0, _C0_1);
        _C0_2 = _mm512_fmadd_pd(_mm512_set1_pd(A[2]), _B0, _C0_2);
        _C0_3 = _mm512_fmadd_pd(_mm512_set1_pd(A[3]), _B0, _C0_3);
        _C0_4 = _mm512_fmadd_pd(_mm512_set1_pd(A[4]), _B0, _C0_4);
        _C0_5 = _mm512_fmadd_pd(_mm512_set1_pd(A[5]), _B0, _C0_5);
        _C0_6 = _mm512_fmadd_pd(_mm512_set1_pd(A[6]), _B0, _C0_6);
        _C0_7 = _mm512_fmadd_pd(_mm512_set1_pd(A[7]), _B0, _C0_7);
        _C0_8 = _mm512_fmadd_pd(_mm512_set1_pd(A[8]), _B0, _C0_8);
        _C0_9 = _mm512_fmadd_pd(_mm512_set1_pd(A[9]), _B0, _C0_9);
        _C0_10 = _mm512_fmadd_pd(_mm512_set1_pd(A[10]), _B0, _C0_10);
        _C0_11 = _mm512_fmadd_pd(_mm512_set1_pd(A[11]), _B0, _C0_11);
        _C0_12 = _mm512_fmadd_pd(_mm512_set1_pd(A[12]), _B0, _C0_12);
        _C0_13 = _mm512_fmadd_pd(_mm512_set1_pd(A[13]), _B0, _C0_13);
        _C0_14 = _mm512_fmadd_pd(_mm512_set1_pd(A[14]), _B0, _C0_14);
        _C0_15 = _mm512_fmadd_pd(_mm512_set1_pd(A[15]), _B0, _C0_15);
        _C0_16 = _mm512_fmadd_pd(_mm512_set1_pd(A[16]), _B0, _C0_16);
        _C0_17 = _mm512_fmadd_pd(_mm512_set1_pd(A[17]), _B0, _C0_17);
        _C0_18 = _mm512_fmadd_pd(_mm512_set1_pd(A[18]), _B0, _C0_18);
        _C0_19 = _mm512_fmadd_pd(_mm512_set1_pd(A[19]), _B0, _C0_19);
        _C0_20 = _mm512_fmadd_pd(_mm512_set1_pd(A[20]), _B0, _C0_20);
        _C0_21 = _mm512_fmadd_pd(_mm512_set1_pd(A[21]), _B0, _C0_21);
        _C0_22 = _mm512_fmadd_pd(_mm512_set1_pd(A[22]), _B0, _C0_22);
        _C0_23 = _mm512_fmadd_pd(_mm512_set1_pd(A[23]), _B0, _C0_23);
        _C0_24 = _mm512_fmadd_pd(_mm512_set1_pd(A[24]), _B0, _C0_24);
        _C0_25 = _mm512_fmadd_pd(_mm512_set1_pd(A[25]), _B0, _C0_25);
        _C0_26 = _mm512_fmadd_pd(_mm512_set1_pd(A[26]), _B0, _C0_26);
        _C0_27 = _mm512_fmadd_pd(_mm512_set1_pd(A[27]), _B0, _C0_27);
        _C0_28 = _mm512_fmadd_pd(_mm512_set1_pd(A[28]), _B0, _C0_28);
        _C0_29 = _mm512_fmadd_pd(_mm512_set1_pd(A[29]), _B0, _C0_29);
        _C0_30 = _mm512_fmadd_pd(_mm512_set1_pd(A[30]), _B0, _C0_30);
        A += MR;
        B += NR;
    }
    _mm512_storeu_pd(&C[0 * ncol + 0], _C0_0);
    _mm512_storeu_pd(&C[1 * ncol + 0], _C0_1);
    _mm512_storeu_pd(&C[2 * ncol + 0], _C0_2);
    _mm512_storeu_pd(&C[3 * ncol + 0], _C0_3);
    _mm512_storeu_pd(&C[4 * ncol + 0], _C0_4);
    _mm512_storeu_pd(&C[5 * ncol + 0], _C0_5);
    _mm512_storeu_pd(&C[6 * ncol + 0], _C0_6);
    _mm512_storeu_pd(&C[7 * ncol + 0], _C0_7);
    _mm512_storeu_pd(&C[8 * ncol + 0], _C0_8);
    _mm512_storeu_pd(&C[9 * ncol + 0], _C0_9);
    _mm512_storeu_pd(&C[10 * ncol + 0], _C0_10);
    _mm512_storeu_pd(&C[11 * ncol + 0], _C0_11);
    _mm512_storeu_pd(&C[12 * ncol + 0], _C0_12);
    _mm512_storeu_pd(&C[13 * ncol + 0], _C0_13);
    _mm512_storeu_pd(&C[14 * ncol + 0], _C0_14);
    _mm512_storeu_pd(&C[15 * ncol + 0], _C0_15);
    _mm512_storeu_pd(&C[16 * ncol + 0], _C0_16);
    _mm512_storeu_pd(&C[17 * ncol + 0], _C0_17);
    _mm512_storeu_pd(&C[18 * ncol + 0], _C0_18);
    _mm512_storeu_pd(&C[19 * ncol + 0], _C0_19);
    _mm512_storeu_pd(&C[20 * ncol + 0], _C0_20);
    _mm512_storeu_pd(&C[21 * ncol + 0], _C0_21);
    _mm512_storeu_pd(&C[22 * ncol + 0], _C0_22);
    _mm512_storeu_pd(&C[23 * ncol + 0], _C0_23);
    _mm512_storeu_pd(&C[24 * ncol + 0], _C0_24);
    _mm512_storeu_pd(&C[25 * ncol + 0], _C0_25);
    _mm512_storeu_pd(&C[26 * ncol + 0], _C0_26);
    _mm512_storeu_pd(&C[27 * ncol + 0], _C0_27);
    _mm512_storeu_pd(&C[28 * ncol + 0], _C0_28);
    _mm512_storeu_pd(&C[29 * ncol + 0], _C0_29);
    _mm512_storeu_pd(&C[30 * ncol + 0], _C0_30);
}

void micro_dxpy(int m, int n, double *restrict C, const double *restrict D, int ncol)
{
    int i;
    for (i = 0; i < m; ++i)
    {
        C [0:n] += D [i * NR:n];
        C += ncol;
    }
}

void packacc(int row, int col, const double *mt, int inc, double *bk)
{
    int q = row / MR;
    int r = row % MR;

    for (int i = 0; i < q; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            bk [i * MR * col + j * MR:MR] = mt [i * MR + j * inc:MR];
        }
    }
    bk += q * MR * col;
    mt += q * MR;
    if (r > 0)
    {
        for (int i = 0; i < col; ++i)
        {
            bk [0:r] = mt [0:r];
            bk [r:MR - r] = 0.0;
            bk += MR;
            mt += inc;
        }
    }
}

void packarc(int row, int col, const double *mt, int inc, double *bk)
{
    int q = row / MR;
    int r = row % MR;
    for (int i = 0; i < q; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            bk [0:MR] = mt [0:MR:inc];
            bk += MR;
            mt += 1;
        }
        mt += (inc * MR - col);
    }
    if (r > 0)
    {
        for (int j = 0; j < col; ++j)
        {
            bk [0:r] = mt [0:r:inc];
            bk [r:MR - r] = 0.0;
            bk += MR;
            mt += 1;
        }
    }
}

void packbcr(int row, int col, const double *mt, int inc, double *bk)
{
    int q = col / NR;
    int r = col % NR;
#ifndef ONE_CORE
#pragma omp parallel for num_threads(NT1)
#endif
    for (int i = 0; i < q; ++i)
    {
        for (int j = 0; j < row; ++j)
        {
            bk [0:NR] = mt [0:NR:inc];
            bk += NR;
            mt += 1;
        }
        mt += (inc * NR - row);
    }
    if (r > 0)
    {
        for (int j = 0; j < row; ++j)
        {
            bk [0:r] = mt [0:r:inc];
            bk [r:NR - r] = 0.0;
            bk += NR;
            mt += 1;
        }
    }
}

void packbrr(int row, int col, const double *mt, int inc, double *bk)
{
    int q = col / NR;
    int r = col % NR;

#ifndef ONE_CORE
#pragma omp parallel for num_threads(NT1)
#endif
    for (int j = 0; j < q; ++j)
    {
        for (int i = 0; i < row; ++i)
        {
            bk [i * NR + j * row * NR:NR] = mt [i * inc + j * NR:NR];
        }
    }
    bk += q * row * NR;
    mt += q * NR;
    if (r > 0)
    {
        for (int i = 0; i < row; ++i)
        {
            bk [0:r] = mt [0:r];
            bk [r:NR - r] = 0.0;
            bk += NR;
            mt += inc;
        }
    }
}

void ijrloop(int nota, const int m, const int n, const int ni, const int k, const int ki, const double *restrict A,
             const int la, const double *restrict B, double *restrict C, const int lc)
{
    int mq = (m + MB - 1) / MB;
    int md = m % MB;
    int nq = (n + NR - 1) / NR;
    int nd = n % NR;

#ifndef ONE_CORE
#pragma omp parallel num_threads(NT1)
#endif
    {
        double _A[MB * KB] __attribute__((aligned(64)));
#ifndef ONE_CORE
#pragma omp for schedule(dynamic)
#endif
        for (int i = 0; i < mq; ++i)
        {
            int mc = (i != mq - 1 || md == 0) ? MB : md;

            if (nota)
                packarc(mc, k, &A[ki * KB + i * MB * la], la, _A);
            else
                packacc(mc, k, &A[ki * KB * la + i * MB], la, _A);

#ifndef ONE_CORE
#pragma omp parallel num_threads(NT2)
#endif
            {
                double _C[MR * NR] __attribute__((aligned(64)));
#ifndef ONE_CORE
#pragma omp for schedule(dynamic)
#endif
                for (int j = 0; j < nq; ++j)
                {
                    int nc = (j != nq - 1 || nd == 0) ? NR : nd;
                    int pq = (mc + MR - 1) / MR;
                    int pd = mc % MR;
                    for (int p = 0; p < pq; ++p)
                    {
                        int pc = (p != pq - 1 || pd == 0) ? MR : pd;
                        if (pc == MR && nc == NR)
                        {
                            micro_kernel0(k, &_A[p * (MR)*k], &B[j * NR * k], &C[i * MB * lc + p * MR * lc + ni * NB + j * NR], lc);
                        }
                        else
                        {
                            micro_kernel1(k, &_A[p * (MR)*k], &B[j * NR * k], _C, NR);
                            micro_dxpy(pc, nc, &C[i * MB * lc + p * MR * lc + ni * NB + j * NR], _C, lc);
                        }
                    }
                }
            }
        }
    }
}

#ifdef FORTRAN_FORMAT
void userdgemm_(const char *transa, const char *transb, const int *_m,
                const int *_n, const int *_k, const double *_alpha,
                const double *restrict A, const int *_lda,
                const double *restrict B, const int *_ldb,
                const double *_beta, double *restrict C, const int *_ldc)
{
#else
void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, const int64_t m_, const int64_t n_,
                const int64_t k_, const double alpha, const double *A,
                const int64_t lda_, const double *B, const int64_t ldb_,
                const double beta, double *C, const int64_t ldc_)
{
    const int m = (int)m_;
    const int n = (int)n_;
    const int k = (int)k_;
    const int lda = (int)lda_;
    const int ldb = (int)ldb_;
    const int ldc = (int)ldc_;
    const char *transa = TransA == CblasNoTrans ? "N" : "T";
    const char *transb = TransB == CblasNoTrans ? "N" : "T";
    const int *_m = &m;
    const int *_n = &n;
    const int *_k = &k;
    const double *_alpha = &alpha;
    const int *_lda = &lda;
    const int *_ldb = &ldb;
    const double *_beta = &beta;
    const int *_ldc = &ldc;
#endif
    const int nota = (int)(transa[0] == 'N' || transa[0] == 'n');
    const int notb = (int)(transb[0] == 'N' || transb[0] == 'n');

    int nq = (n + NB - 1) / NB;
    int nd = n % NB;
    int kq = (k + KB - 1) / KB;
    int kd = k % KB;

#ifndef ONE_CORE
    omp_set_max_active_levels(8);
#endif

    double *_B;
    ALLOC(_B, sizeof(double) * KB * NB);

    for (int j = 0; j < nq; ++j)
    {
        int nc = (j != nq - 1 || nd == 0) ? NB : nd;

        for (int l = 0; l < kq; ++l)
        {
            int kc = (l != kq - 1 || kd == 0) ? KB : kd;

            if (notb)
                packbrr(kc, nc, &B[j * NB + l * KB * ldb], ldb, _B);
            else
                packbcr(kc, nc, &B[j * NB * ldb + l * KB], ldb, _B);

            ijrloop(nota, m, nc, j, kc, l, A, lda, _B, C, ldc);
        }
    }
    FREE(_B);
}
