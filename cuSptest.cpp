#include <stdio.h>
#include <stdlib.h>

#include <cusolverSp.h>
#include <cuda_runtime_api.h>

int main(int argc, char*argv[])
{
    cusolverSpHandle_t cusolverH = NULL:
    cusparseMatDescr_t descrA = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;

    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_b = NULL;
    int *rankA = NULL:
    double *x = NULL:
    double *p = NULL;
    double *min_norm = NULL;

    const int m = 4;
    constm int n = 2;
    const int nnzA = 4;
    double tol = 1.e-12;
    const int csrRowPtrA[m+1] = {0, 1, 2, 3, 4};
    const int csrColIndA[nnzA] = {0, 0, 1, 1};
    const int csrValA[nnzA] = {1.0, 2.0, 3.0, 4.0};
    const double b[m] = {1.0, 1.0, 1.0, 1.0};
    double h_x[n];
    //step2: create cusolver handle
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    //step3:copy data
    cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA * batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
    cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (m+1));
    cudaStat4 = cudaMalloc ((void**)&d_b         , sizeof(double) * m);
    cudaStat5 = cudaMalloc ((void**)&x, sizeof(double)*n);
    cudaStat6 = cudaMalloc ((void**)&p, sizeof(double)*n);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);
    assert(cudaStat6 == cudaSuccess);

    cudaStat1 = cudaMemcpy(d_csrValA   , csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1), cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(d_b, b, sizeof(double) * m, cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    cudaStat1 
    cusolverSpDcsrlsqvqr[Host](cusolverSpHandle_t handle,
                  int m,
                  int n,
                  int nnzA,
                  const cusparseMatDescr_t descrA,
                  const double *d_csrValA,
                  const int *d_csrRowPtrA,
                  const int *d_csrColIndA,
                  const double *d_b,
                  double tol,
                  int *rankA,
                  double *x,
                  int *p,
                  double *min_norm);
    assert(cudaStat1 == cudaSuccess);
    
    //trans the result to host
    cudaStat1 = cudaMemcpy(h_x, x, sizeof(double) * n, cudaMemcpyDeviceToHost);
    assert(cudaStat1 == cudaSuccess);

    //print the result
    for (int i=0; i<n; i++)
        printf("x[%d] = %E", i, x[i]);
    cudaFree(d_csrColIndA);
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(x);
    cudaFree(p);
    cudaFree(d_b);

    return(0)
}
