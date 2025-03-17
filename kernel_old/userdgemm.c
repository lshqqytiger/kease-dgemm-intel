#ifndef FORTRAN_FORMAT
#include "cblas_format.h"
#endif

#include <stdint.h>
#include <stdlib.h>

void userdgemm_k40_nt_anbp(const char *transa, const char *transb, const int *_m,
                           const int *_n, const int *_k, const double *_alpha,
                           const double *restrict A, const int *_lda,
                           const double *restrict B, const int *_ldb,
                           const double *_beta, double *restrict C, const int *_ldc);

void userdgemm_n40_apbz(const char *transa, const char *transb, const int *_m,
                        const int *_n, const int *_k, const double *_alpha,
                        const double *restrict A, const int *_lda,
                        const double *restrict B, const int *_ldb,
                        const double *_beta, double *restrict C, const int *_ldc);

void userdgemm_general(const char *transa, const char *transb, const int *_m,
                       const int *_n, const int *_k, const double *_alpha,
                       const double *restrict A, const int *_lda,
                       const double *restrict B, const int *_ldb,
                       const double *_beta, double *restrict C, const int *_ldc);

#ifdef FORTRAN_FORMAT
void userdgemm_(const char *transa, const char *transb, const int *_m,
                const int *_n, const int *_k, const double *_alpha,
                const double *restrict A, const int *_lda,
                const double *restrict B, const int *_ldb,
                const double *_beta, double *restrict C, const int *_ldc) {
#else
void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, const int64_t m_, const int64_t n_,
                const int64_t k_, const double alpha, const double *A,
                const int64_t lda_, const double *B, const int64_t ldb_,
                const double beta, double *C, const int64_t ldc_) {
    const int     m      = (int)m_;
    const int     n      = (int)n_;
    const int     k      = (int)k_;
    const int     lda    = (int)lda_;
    const int     ldb    = (int)ldb_;
    const int     ldc    = (int)ldc_;
    const char*   transa = TransA == CblasNoTrans ? "N" : "T";
    const char*   transb = TransB == CblasNoTrans ? "N" : "T";
    const int*    _m     = &m;
    const int*    _n     = &n;
    const int*    _k     = &k;
    const double* _alpha = &alpha;
    const int*    _lda   = &lda;
    const int*    _ldb   = &ldb;
    const double* _beta  = &beta;
    const int*    _ldc   = &ldc;
#endif

    uint64_t nota = (int)(transa[0] == 'N' || transa[0] == 'n');
    uint64_t notb = (int)(transb[0] == 'N' || transb[0] == 'n');

    if (*_k == 40 && nota && !notb && *_alpha == -1.0 && *_beta == 1.0) {
        userdgemm_k40_nt_anbp(transa, transb, _m, _n, _k, _alpha, A, _lda, B, _ldb, _beta, C, _ldc);
    }
    else if (*_n == 40 && *_alpha == 1.0 && *_beta == 0.0) {
        userdgemm_n40_apbz(transa, transb, _m, _n, _k, _alpha, A, _lda, B, _ldb, _beta, C, _ldc);
    }
    else {
        userdgemm_general(transa, transb, _m, _n, _k, _alpha, A, _lda, B, _ldb, _beta, C, _ldc);
    }
}
