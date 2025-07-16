#include "helper.h"

#define M 600
#define N 600
#define K 600

#define lda M
#define ldb K
#define ldc M

void initialize(void **arg_in, void **arg_out, void **arg_val)
{
  void **arr = malloc(sizeof(void *) * 3);
  *arg_in = arr;

  arr[0] = numa_alloc(M * K * sizeof(double));
  arr[1] = numa_alloc(K * N * sizeof(double));
  arr[2] = numa_alloc(M * N * sizeof(double));

  *arg_out = numa_alloc(M * N * sizeof(double));
  *arg_val = numa_alloc(M * N * sizeof(double));

  set_data(arr[0], M * K, 100, -1.0, 1.0);
  set_data(arr[1], K * N, 200, -1.0, 1.0);
  set_data(arr[2], M * N, 300, -1.0, 1.0);
  memcpy(*arg_val, arr[2], sizeof(double) * M * N);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, -1.0, arr[0], lda, arr[1], ldb, 1.0, *arg_val, ldc);
}

void finalize(void *arg_in, void *arg_out, void *arg_val)
{
  void **arr = (void **)arg_in;

  numa_free(arr[0], M * K * sizeof(double));
  numa_free(arr[1], K * N * sizeof(double));
  numa_free(arr[2], M * N * sizeof(double));

  free(arr);

  numa_free(arg_out, M * N * sizeof(double));
  numa_free(arg_val, M * N * sizeof(double));
}

double evaluate(void *arg_in, void *arg_out)
{
  void **arr = (void **)arg_in;

  double *A = (double *)arr[0];
  double *B = (double *)arr[1];
  double *C = (double *)arr[2];

  memcpy(arg_out, C, sizeof(double) * M * N);

  const double start_time = omp_get_wtime();
  call_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, -1.0, A, lda, B, ldb, 1.0, arg_out, ldc);
  const double end_time = omp_get_wtime();

  return end_time - start_time;
}

bool validate(const void *arg_val, const void *arg_out)
{
  double *temp = numa_alloc(sizeof(double) * M * N);
  memcpy(temp, arg_val, sizeof(double) * M * N);

  cblas_daxpy(M * N, -1.0, arg_out, 1, temp, 1);
  double difference = cblas_dnrm2(M * N, temp, 1);
  numa_free(temp, sizeof(double) * M * N);

  return difference < 0.0001;
}
