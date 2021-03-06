#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cusparse_v2.h>

#include <cusolverSp.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

int main(int argc, char*argv[])
{
    cusolverSpHandle_t cusolverH = NULL;
    cusparseMatDescr_t descrA = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;

    const int m = 4;
    const int n = 2;
    const int nnzA = 4;
    double tol = 1.e-12;
    /*int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_b = NULL;*/
    int *rankA = NULL;
    int *p = (int *)malloc(n * sizeof(int));
    double *min_norm = NULL;

    const int csrRowPtrA[m+1] = {0, 1, 2, 3, 4};
    const int csrColIndA[nnzA] = {0, 0, 1, 1};
    const double csrValA[nnzA] = {1.0, 2.0, 3.0, 4.0};
    const double b[m] = {1.0, 1.0, 1.0, 1.0};
    double *h_x = (double *)malloc(n * sizeof(double));
    //step2: create cusolver handle
    cusolver_status = cusolverSpCreate(&cusolverH);
    //assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    //step3:copy data
    /*cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA );
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
    assert(cudaStat4 == cudaSuccess);*/

    cusolver_status = cusolverSpDcsrlsqvqrHost(cusolverH,
                  m,
                  n,
                  nnzA,
                  descrA,
                  csrValA,
                  csrRowPtrA,
                  csrColIndA,
                  b,
                  tol,
                  rankA,
                  h_x,
                  p,
                  min_norm);
    assert(cusolver_status == cudaSuccess);
    
    //trans the result to host
    //cudaStat1 = cudaMemcpy(h_x, x, sizeof(double) * n, cudaMemcpyDeviceToHost);
    //assert(cudaStat1 == cudaSuccess);

    //print the result
    for (int i=0; i<n; i++)
        printf("x[%d] = %E", i, h_x[i]);
    /*cudaFree(d_csrColIndA);
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(x);
    cudaFree(d_b);*/

    return(0);
}
