#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

int main()
{
    // --- Initialize cuSPARSE
    cusparseHandle_t handle;    cusparseCreate(&handle);

    const int Nrows = 4;                        // --- Number of rows
    const int Ncols = 2;                        // --- Number of columns
    const int N     = Nrows;

    // --- Host side dense matrix
    /*double *h_A_dense = (double*)malloc(Nrows*Ncols*sizeof(*h_A_dense));

    // --- Column-major ordering
    h_A_dense[0] = 1.0f; h_A_dense[4] = 4.0f; h_A_dense[8]  = 0.0f; h_A_dense[12] = 0.0f; 
    h_A_dense[1] = 0.0f; h_A_dense[5] = 2.0f; h_A_dense[9]  = 3.0f; h_A_dense[13] = 0.0f; 
    h_A_dense[2] = 5.0f; h_A_dense[6] = 0.0f; h_A_dense[10] = 0.0f; h_A_dense[14] = 7.0f; 
    h_A_dense[3] = 0.0f; h_A_dense[7] = 0.0f; h_A_dense[11] = 9.0f; h_A_dense[15] = 0.0f; 

    //create device array and copy host to it
    double *d_A_dense;  cudaMalloc(&d_A_dense, Nrows * Ncols * sizeof(*d_A_dense));
    cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice);

    // --- Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;      cusparseCreateMatDescr(&descrA);
    cusparseSetMatType      (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase (descrA, CUSPARSE_INDEX_BASE_ZERO);  

    int nnz = 0;                                // --- Number of nonzero elements in dense matrix
    const int lda = Nrows;                      // --- Leading dimension of dense matrix
    // --- Device side number of nonzero elements per row
    int *d_nnzPerVector;    cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector));
    cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz);
    // --- Host side number of nonzero elements per row
    int *h_nnzPerVector = (int *)malloc(Nrows * sizeof(*h_nnzPerVector));
    cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost);

    printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
    for (int i = 0; i < Nrows; ++i) printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
    printf("\n");

    // --- Device side dense matrix
    double *d_A;            cudaMalloc(&d_A, nnz * sizeof(*d_A));
    int *d_A_RowIndices;    cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices));
    int *d_A_ColIndices;    cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices));

    cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices);

    // --- Host side dense matrix*/
    cusparseMatDescr_t descrA;      cusparseCreateMatDescr(&descrA);
    cusparseSetMatType      (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase (descrA, CUSPARSE_INDEX_BASE_ZERO);  
    int nnz = 4;
    double h_A[4] = {1.0, 2.0, 3.0, 4.0};
    int h_A_RowIndices[5] = {0, 1, 2, 3, 4};
    int h_A_ColIndices[4] = {0, 0, 1, 1};
    /*cudaMemcpy(h_A, d_A, nnz*sizeof(*h_A), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost);*/

    for (int i = 0; i < nnz; ++i) printf("A[%i] = %.0f ", i, h_A[i]); printf("\n");

    for (int i = 0; i < (Nrows + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

    for (int i = 0; i < nnz; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);   

    // --- Allocating and defining dense host and device data vectors
    double *h_y = (double *)malloc(Nrows * sizeof(double)); 
    h_y[0] = 1.0;  h_y[1] = 2.0; h_y[2] = 4.0; h_y[3] = 5.0;

    //double *d_y;        cudaMalloc(&d_y, Nrows * sizeof(double));  
    //cudaMemcpy(d_y, h_y, Nrows * sizeof(double), cudaMemcpyHostToDevice);

    // --- Allocating the host and device side result vector
    double *h_x = (double *)malloc(Ncols * sizeof(double)); 
    //double *d_x;        cudaMalloc(&d_x, Ncols * sizeof(double));

    // --- CUDA solver initialization
    cusolverSpHandle_t solver_handle;
    cusolverSpCreate(&solver_handle);

    //int singularity;
    // --- Using LU factorization
    //cusolveSafeCall(cusolverSpDcsrlsvluHost(solver_handle, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, 0, h_x, &singularity));
    // --- Using QR factorization
    //cusolveSafeCall(cusolverSpDcsrlsvqrHost(solver_handle, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, 0, h_x, &singularity));

    int rankA;
    int *p = (int *)malloc(N * sizeof(int));
    double min_norm;
    cusolverSpDcsrlsqvqrHost(solver_handle, N, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, &rankA, h_x, p, &min_norm);

    printf("Showing the results...\n");
    for (int i = 0; i < N; i++) printf("%f\n", h_x[i]);
    
    return 0;
}
