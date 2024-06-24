// CUDA 12.3.0
// https://docs.nvidia.com/cuda/archive/12.3.0/cublas/index.html
// https://docs.nvidia.com/cuda/archive/12.3.0/cusparse/index.html
// https://docs.nvidia.com/cuda/archive/12.3.0/cusolver/index.html

#include "structures.h"
// #define deterministic


// -------------- LEVEL 1 ROUTINES --------------

void dv_scale_D(dv_t* v, const double alpha) {
    AB(cublasDscal_v2(*(v->handle), v->n, &alpha, v->d_x_D, 1));
    AD(cudaDeviceSynchronize());
}

void dv_scale_S(dv_t* v, const float alpha) {
    AB(cublasSscal_v2(*(v->handle), v->n, &alpha, v->d_x_S, 1));
    AD(cudaDeviceSynchronize());
}

double dv_norm_D(dv_t* v) {
    double result;
    AB(cublasDnrm2_v2(*(v->handle), v->n, v->d_x_D, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}

float dv_norm_S(dv_t* v) {
    float result;
    AB(cublasSnrm2_v2(*(v->handle), v->n, v->d_x_S, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}

double dm_norm_D(dm_t* A) {
    double result;
    AB(cublasDnrm2_v2(*(A->handle), A->n * A->m, A->d_val_D, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}

float dm_norm_S(dm_t* A) {
    float result;
    AB(cublasSnrm2_v2(*(A->handle), A->n * A->m, A->d_val_S, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}

double spsm_norm_D(spsm_t* A, cublasHandle_t* handle) {
    double result;
    AB(cublasDnrm2_v2(*(handle), A->nnz, A->d_val_D, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}

float spsm_norm_S(spsm_t* A, cublasHandle_t* handle) {
    float result;
    AB(cublasSnrm2_v2(*(handle), A->nnz, A->d_val_S, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}

void dv_axpy_D(dv_t* v, dv_t* w, const double alpha) {
    AB(cublasDaxpy_v2(*(v->handle), v->n, &alpha, v->d_x_D, 1, w->d_x_D, 1));
    AD(cudaDeviceSynchronize());
}

void dv_axpy_S(dv_t* v, dv_t* w, const float alpha) {
    AB(cublasSaxpy_v2(*(v->handle), v->n, &alpha, v->d_x_S, 1, w->d_x_S, 1));
    AD(cudaDeviceSynchronize());
}

void dm_axpy_D(dm_t* A, dm_t* B, const double alpha) {
    AB(cublasDaxpy_v2(*(A->handle), A->n * A->m, &alpha, B->d_val_D, 1, A->d_val_D, 1));
    AD(cudaDeviceSynchronize());
}

void dm_axpy_S(dm_t* A, dm_t* B, const float alpha) {
    AB(cublasSaxpy_v2(*(A->handle), A->n * A->m, &alpha, B->d_val_S, 1, A->d_val_S, 1));
    AD(cudaDeviceSynchronize());
}

double dv_dv_D(dv_t* v, dv_t* w) {
    double result;
    AB(cublasDdot_v2(*(v->handle), v->n, v->d_x_D, 1, w->d_x_D, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}

float dv_dv_S(dv_t* v, dv_t* w) {
    float result;
    AB(cublasSdot_v2(*(v->handle), v->n, v->d_x_S, 1, w->d_x_S, 1, &result));
    AD(cudaDeviceSynchronize());
    return result;
}


// -------------- LEVEL 2 ROUTINES --------------

void spsm_dv_D(spsm_t* A, dv_t* v, dv_t* w) {
    AS(cusparseSpMV(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_D, A->descr_D, v->descr_D, &ZERO_D, w->descr_D, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, A->spsm_dv_buffer_D));
    AD(cudaDeviceSynchronize());
}

void spsm_dv_S(spsm_t* A, dv_t* v, dv_t* w) {
    AS(cusparseSpMV(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_S, v->descr_S, &ZERO_S, w->descr_S, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, A->spsm_dv_buffer_S));
    AD(cudaDeviceSynchronize());
}

void spsm_dv_H(spsm_t* A, dv_t* v, dv_t* w) {
    AS(cusparseSpMV(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, v->descr_H, &ZERO_S, w->descr_H, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, A->spsm_dv_buffer_H));
    AD(cudaDeviceSynchronize());
}

void spsm_dv_HS(spsm_t* A, dv_t* v, dv_t* w) {
    AS(cusparseSpMV(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, v->descr_H, &ZERO_S, w->descr_S, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, A->spsm_dv_buffer_HS));
    AD(cudaDeviceSynchronize());
}


// -------------- LEVEL 3 ROUTINES --------------

void dm_dm_D(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasDgemm_v2(*(A->handle), CUBLAS_OP_N, CUBLAS_OP_N, A->n, B->m, B->n, &ONE_D, A->d_val_D, A->n, B->d_val_D, B->n, &ZERO_D, C->d_val_D, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_transA_D(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasDgemm_v2(*(A->handle), CUBLAS_OP_T, CUBLAS_OP_N, A->m, B->m, B->n, &ONE_D, A->d_val_D, A->n, B->d_val_D, B->n, &ZERO_D, C->d_val_D, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_transB_D(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasDgemm_v2(*(A->handle), CUBLAS_OP_N, CUBLAS_OP_T, A->n, B->n, B->m, &ONE_D, A->d_val_D, A->n, B->d_val_D, B->n, &ZERO_D, C->d_val_D, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_S(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasSgemm_v2(*(A->handle), CUBLAS_OP_N, CUBLAS_OP_N, A->n, B->m, B->n, &ONE_S, A->d_val_S, A->n, B->d_val_S, B->n, &ZERO_S, C->d_val_S, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_transA_S(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasSgemm_v2(*(A->handle), CUBLAS_OP_T, CUBLAS_OP_N, A->m, B->m, B->n, &ONE_S, A->d_val_S, A->n, B->d_val_S, B->n, &ZERO_S, C->d_val_S, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_transB_S(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasSgemm_v2(*(A->handle), CUBLAS_OP_N, CUBLAS_OP_T, A->n, B->n, B->m, &ONE_S, A->d_val_S, A->n, B->d_val_S, B->n, &ZERO_S, C->d_val_S, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_H(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasHgemm(*(A->handle), CUBLAS_OP_N, CUBLAS_OP_N, A->n, B->m, B->n, &ONE_H, A->d_val_H, A->n, B->d_val_H, B->n, &ZERO_H, C->d_val_H, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_transA_H(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasHgemm(*(A->handle), CUBLAS_OP_T, CUBLAS_OP_N, A->m, B->m, B->n, &ONE_H, A->d_val_H, A->n, B->d_val_H, B->n, &ZERO_H, C->d_val_H, C->n));
    AD(cudaDeviceSynchronize());
}

void dm_dm_transB_H(dm_t* A, dm_t* B, dm_t* C) {
    AB(cublasHgemm(*(A->handle), CUBLAS_OP_N, CUBLAS_OP_T, A->n, B->n, B->m, &ONE_H, A->d_val_H, A->n, B->d_val_H, B->n, &ZERO_H, C->d_val_H, C->n));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_D(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_D, A->descr_D, B->descr_D, &ZERO_D, C->descr_D, CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_D));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_S(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_S, B->descr_S, &ZERO_S, C->descr_S, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_S));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_H(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, B->descr_H, &ZERO_S, C->descr_H, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_H));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_HS(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, B->descr_H, &ZERO_S, C->descr_S, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_HS));
    AD(cudaDeviceSynchronize());
}

double* spsm_m(spsm_t* A, const double* V, const int n, const int m) {
    double* AV = new double[n * m];
    double sum;
    int i, j, k;
    for (j = 0; j < m; j++) {
        for (i = 0; i < n; i++) {
            sum = 0.0;
            for (k = A->h_row[i]; k < A->h_row[i + 1]; k++) {
                sum += V[A->h_col[k] * m + j] * A->h_val[k];
            }
            AV[i * m + j] = sum;
        }
    }
    return AV;
}

double* tdm_m(tdm_t* T, const double* V, const int n, const int m) {
    int i, j;
    double c_ij;

    double* VT = new double[n * m];
    // first and last col with only 1 beta each
    for (i = 0; i < n; i++) {
        VT[i * m] = V[i * m] * T->alpha[0] + V[i * m + 1] * T->beta[1];
        VT[i * m + m - 1] = V[i * m + m - 1] * T->alpha[m - 1] + V[i * m + m - 2] * T->beta[m - 1];
    }
    // full values, only need to do triplets
    for (i = 0; i < n; i++) {
        for (j = 1; j < m - 1; j++) {
            c_ij = 0;
            c_ij += V[i * m + j - 1] * T->beta[j];
            c_ij += V[i * m + j    ] * T->alpha[j];
            c_ij += V[i * m + j + 1] * T->beta[j + 1];
            VT[i * m + j] = c_ij;
        }
    }
    return VT;
}


// -------------- QR DECOMPOSITION --------------

void dm_geqrf_D(dm_t* A, dm_t* R) {
    AD(cudaMemcpy(A->d_val_qr_D, A->d_val_D, A->n * A->m * sizeof(double), cudaMemcpyDeviceToDevice));
    AD(cudaDeviceSynchronize());
    AO(cusolverDnDgeqrf(A->solver_handle, A->n, A->m, A->d_val_qr_D, A->lda, A->d_tau_D, A->d_geqrf_buffer_D, A->geqrf_buffer_size, A->d_info));
    AD(cudaDeviceSynchronize());
    dm_to_upper_triangle_D(A, R);
    cudaDeviceSynchronize();
}

void dm_geqrf_S(dm_t* A, dm_t* R) {
    AD(cudaMemcpy(A->d_val_qr_S, A->d_val_S, A->n * A->m * sizeof(float), cudaMemcpyDeviceToDevice));
    AD(cudaDeviceSynchronize());
    AO(cusolverDnSgeqrf(A->solver_handle, A->n, A->m, A->d_val_qr_S, A->lda, A->d_tau_S, A->d_geqrf_buffer_S, A->geqrf_buffer_size, A->d_info));
    AD(cudaDeviceSynchronize());
    dm_to_upper_triangle_S(A, R);
    cudaDeviceSynchronize();
}

void dm_orgqr_D(dm_t* A, dm_t* Q) {
    AO(cusolverDnDorgqr(A->solver_handle, A->n, A->m, A->k, A->d_val_qr_D, A->lda, A->d_tau_D, A->d_geqrf_buffer_D, A->orgqr_buffer_size, A->d_info));
    AD(cudaDeviceSynchronize());
    AD(cudaMemcpy(Q->d_val_D, A->d_val_qr_D, A->n * A->m * sizeof(double), cudaMemcpyDeviceToDevice));
    AD(cudaDeviceSynchronize());
}

void dm_orgqr_S(dm_t* A, dm_t* Q) {
    AO(cusolverDnSorgqr(A->solver_handle, A->n, A->m, A->k, A->d_val_qr_S, A->lda, A->d_tau_S, A->d_geqrf_buffer_S, A->orgqr_buffer_size, A->d_info));
    AD(cudaDeviceSynchronize());
    AD(cudaMemcpy(Q->d_val_S, A->d_val_qr_S, A->n * A->m * sizeof(float), cudaMemcpyDeviceToDevice));
    AD(cudaDeviceSynchronize());
}


// -------------- ENABLE MIXED PRECISION --------------

void dv_use_S(dv_t* v) {
    AD(cudaMalloc(&(v->d_x_S), v->n * sizeof(float)));
    D2S(v->d_x_D, v->d_x_S, v->n);
    AS(cusparseCreateDnVec(&(v->descr_S), v->n, v->d_x_S, CUDA_R_32F));
    AD(cudaDeviceSynchronize());
    v->use_S = true;
}

void dv_use_H(dv_t* v) {
    AD(cudaMalloc(&(v->d_x_H), v->n * sizeof(half)));
    D2H(v->d_x_D, v->d_x_H, v->n);
    AS(cusparseCreateDnVec(&(v->descr_H), v->n, v->d_x_H, CUDA_R_16F));
    AD(cudaDeviceSynchronize());
    v->use_H = true;
}

void dm_use_S(dm_t* A) {
    AD(cudaMalloc(&(A->d_val_S), A->n * A->m * sizeof(float)));
    D2S(A->d_val_D, A->d_val_S, A->n * A->m);
    AS(cusparseCreateDnMat(&A->descr_S, A->n, A->m, A->n, A->d_val_S, CUDA_R_32F, CUSPARSE_ORDER_COL));
    AD(cudaDeviceSynchronize());
    A->use_S = true;
}

void dm_use_H(dm_t* A) {
    AD(cudaMalloc(&(A->d_val_H), A->n * A->m * sizeof(half)));
    D2H(A->d_val_D, A->d_val_H, A->n * A->m);
    AS(cusparseCreateDnMat(&A->descr_H, A->n, A->m, A->n, A->d_val_H, CUDA_R_16F, CUSPARSE_ORDER_COL));
    AD(cudaDeviceSynchronize());
    A->use_H = true;
}

void dm_use_qr_D(dm_t* A) {
    AO(cusolverDnCreate(&(A->solver_handle)));
    AD(cudaMalloc(&(A->d_info), sizeof(int)));
    A->tau_size = A->n > A->m ? A->m : A->n;
    A->k = A->m;
    AD(cudaMalloc(&(A->d_tau_D), A->tau_size * sizeof(double)));
    AD(cudaMalloc(&(A->d_val_qr_D), A->n * A->m * sizeof(double)));

    AO(cusolverDnDgeqrf_bufferSize(A->solver_handle, A->n, A->m, A->d_val_qr_D, A->lda, &(A->geqrf_buffer_size)));
    AO(cusolverDnDorgqr_bufferSize(A->solver_handle, A->n, A->m, A->k, A->d_val_qr_D, A->lda, A->d_tau_D, &(A->orgqr_buffer_size)));
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->d_geqrf_buffer_D), A->geqrf_buffer_size * sizeof(double)));
    AD(cudaMalloc(&(A->d_orgqr_buffer_D), A->orgqr_buffer_size * sizeof(double)));

    A->use_qr_D = true;
}

void dm_use_qr_S(dm_t* A) {
    AO(cusolverDnCreate(&(A->solver_handle)));
    AD(cudaMalloc(&(A->d_info), sizeof(int)));
    A->tau_size = A->n > A->m ? A->m : A->n;
    A->k = A->m;
    AD(cudaMalloc(&(A->d_tau_S), A->tau_size * sizeof(float)));
    AD(cudaMalloc(&(A->d_val_qr_S), A->n * A->m * sizeof(float)));

    AO(cusolverDnSgeqrf_bufferSize(A->solver_handle, A->n, A->m, A->d_val_qr_S, A->lda, &(A->geqrf_buffer_size)));
    AO(cusolverDnSorgqr_bufferSize(A->solver_handle, A->n, A->m, A->k, A->d_val_qr_S, A->lda, A->d_tau_S, &(A->orgqr_buffer_size)));
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->d_geqrf_buffer_S), A->geqrf_buffer_size * sizeof(float)));
    AD(cudaMalloc(&(A->d_orgqr_buffer_S), A->orgqr_buffer_size * sizeof(float)));

    A->use_qr_S = true;
}

void spsm_use_S(spsm_t* A) {
    AD(cudaMalloc(&(A->d_val_S), A->nnz * sizeof(float)));
    D2S(A->d_val_D, A->d_val_S, A->nnz);
    AS(cusparseCreateCsr(&(A->descr_S), A->n, A->n, A->nnz, A->d_row, A->d_col, A->d_val_S, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    AD(cudaDeviceSynchronize());
    A->use_S = true;
}

void spsm_use_H(spsm_t* A) {
    AD(cudaMalloc(&(A->d_val_H), A->nnz * sizeof(half)));
    D2H(A->d_val_D, A->d_val_H, A->nnz);
    AS(cusparseCreateCsr(&(A->descr_H), A->n, A->n, A->nnz, A->d_row, A->d_col, A->d_val_H, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    AD(cudaDeviceSynchronize());
    A->use_H = true;
}

void spsm_use_HS(spsm_t* A) {
    A->use_HS = true;
}


// -------------- CASTING & MEMORY TRANSFER --------------

/**
 * Device memory only, host memory is always double precision
 * 
 * h - half   (16 bit)
 * s - float  (32 bit)
 * d - double (64 bit)
 */

__global__ void _s2h(const float* xs, half* ys, const int n) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) ys[i] = __float2half(xs[i]);
}

__global__ void _d2h(const double* xs, half* ys, const int n) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) ys[i] = __double2half(xs[i]);
}

__global__ void _d2s(const double* xs, float* ys, const int n) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) ys[i] = (float) xs[i];
}

__global__ void _h2s(const half* xs, float* ys, const int n) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) ys[i] = __half2float(xs[i]);
}

__global__ void _h2d(const half* xs, double* ys, const int n) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) ys[i] = (double) __half2float(xs[i]);
}

__global__ void _s2d(const float* xs, double* ys, const int n) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) ys[i] = (double) xs[i];
}

void S2H(const float* xs, half* ys, const int n) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    _s2h<<<num_blocks, block_size>>>(xs, ys, n);
}

void D2H(const double* xs, half* ys, const int n) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    _d2h<<<num_blocks, block_size>>>(xs, ys, n);
}

void D2S(const double* xs, float* ys, const int n) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    _d2s<<<num_blocks, block_size>>>(xs, ys, n);
}

void H2S(const half* xs, float* ys, const int n) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    _h2s<<<num_blocks, block_size>>>(xs, ys, n);
}

void H2D(const half* xs, double* ys, const int n) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    _h2d<<<num_blocks, block_size>>>(xs, ys, n);
}

void S2D(const float* xs, double* ys, const int n) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    _s2d<<<num_blocks, block_size>>>(xs, ys, n);
}

void dv_host_to_device(dv_t* v) {
    AD(cudaMemcpy(v->d_x_D, v->h_x, v->n * sizeof(double), cudaMemcpyHostToDevice));
}

void dv_device_to_host(dv_t* v) {
    AD(cudaMemcpy(v->h_x, v->d_x_D, v->n * sizeof(double), cudaMemcpyDeviceToHost));
}

void dv_device_to_device_D(dv_t* v, dv_t* w) {
    AD(cudaMemcpy(w->d_x_D, v->d_x_D, v->n * sizeof(double), cudaMemcpyDeviceToDevice));
}

void dv_device_to_device_S(dv_t* v, dv_t* w) {
    AD(cudaMemcpy(w->d_x_S, v->d_x_S, v->n * sizeof(float), cudaMemcpyDeviceToDevice));
}

void dv_device_to_device_H(dv_t* v, dv_t* w) {
    AD(cudaMemcpy(w->d_x_H, v->d_x_H, v->n * sizeof(__half), cudaMemcpyDeviceToDevice));
}

void dm_host_to_device(dm_t* A) {
    AD(cudaMemcpy(A->d_val_D, A->h_val, A->n * A->m * sizeof(double), cudaMemcpyHostToDevice));
}

void dm_device_to_host(dm_t* A) {
    AD(cudaMemcpy(A->h_val, A->d_val_D, A->n * A->m * sizeof(double), cudaMemcpyDeviceToHost));
}

void dm_device_to_device_D(dm_t* A, dm_t* B) {
    AD(cudaMemcpy(B->d_val_D, A->d_val_D, A->n * A->m * sizeof(double), cudaMemcpyDeviceToDevice));
}

void dm_device_to_device_S(dm_t* A, dm_t* B) {
    AD(cudaMemcpy(B->d_val_S, A->d_val_S, A->n * A->m * sizeof(float), cudaMemcpyDeviceToDevice));
}

void dm_device_to_device_H(dm_t* A, dm_t* B) {
    AD(cudaMemcpy(B->d_val_H, A->d_val_H, A->n * A->m * sizeof(half), cudaMemcpyDeviceToDevice));
}

__global__ void _upper_triangle_D(double* A, double* B, const int n, const int m) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int col = i / n;
    const int row = i % n;
    const int j = col * m + row;
    if (row <= col) {
        B[j] = A[i];
    } else if (row < m) {
        B[j] = 0;
    }
}

__global__ void _upper_triangle_S(float* A, float* B, const int n, const int m) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int col = i / n;
    const int row = i % n;
    const int j = col * m + row;
    if (row <= col) {
        B[j] = A[i];
    } else if (row < m) {
        B[j] = 0;
    }
}

void dm_to_upper_triangle_D(dm_t* A, dm_t* B) {
    const int block_size = 256;
    const int num_blocks = (A->n * A->m + block_size - 1) / block_size;
    _upper_triangle_D<<<num_blocks, block_size>>>(A->d_val_qr_D, B->d_val_D, A->n, A->m);
}

void dm_to_upper_triangle_S(dm_t* A, dm_t* B) {
    const int block_size = 256;
    const int num_blocks = (A->n * A->m + block_size - 1) / block_size;
    _upper_triangle_S<<<num_blocks, block_size>>>(A->d_val_qr_S, B->d_val_S, A->n, A->m);
}


// -------------- OTHER UTILITY --------------

bool ends_with(const std::string &str, const std::string &suffix) {
    if (str.size() < suffix.size()) return false;
    return str.substr(str.size() - suffix.size()) == suffix;
}

void spsm_load_csr(spsm_t* A, const std::string &filepath) {
    std::ifstream file(filepath);
    int n, nnz;
    file >> n >> nnz;
    double* val = new double[nnz];
    int* col = new int[nnz];
    int* row = new int[n + 1];
    for (int i = 0; i < nnz; i++) { file >> val[i]; }
    for (int i = 0; i < nnz; i++) { file >> col[i]; }
    for (int i = 0; i <= n; ++i) { file >> row[i]; }
    file.close();
    A->n = n; A->nnz = nnz; A->h_val = val; A->h_col = col; A->h_row = row;
}


/**
 * Load .mtx file from matrix market. Must be symmetric.
*/
void spsm_load_mtx(spsm_t* A, const std::string &filepath) {
    int i = 0, j = 0, k = 0, mtx_nnz, n, nnz;

    std::ifstream file(filepath);
    std::string line;
    while (std::getline(file, line)) {
        // check if the line starts with '%'
        if (line.find("%") == 0) {
            continue;
        } else {
            // read cols, rows, nnz in half
            std::istringstream iss(line);
            iss >> n >> n >> mtx_nnz;
            break;
        }
    }

    double* mtx_val = new double[mtx_nnz];
    int* mtx_row = new int[mtx_nnz];
    int* mtx_col = new int[mtx_nnz];

    // col/row or row/col does not matter since it's symmetric
    // define first value as col, second as row
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> mtx_col[i] >> mtx_row[i] >> mtx_val[i];
        i++;
    }
    file.close();

    // nnz full = nnz half * 2 - nnz on diagonal
    // decrement row/col indices to start at 0
    nnz = mtx_nnz * 2;
    for (i = 0; i < mtx_nnz; i++) {
        mtx_row[i]--;
        mtx_col[i]--;
        if (mtx_col[i] == mtx_row[i]) {
            nnz--;
        }
    }

    double* val = new double[nnz];
    int* col = new int[nnz];
    int* row = new int[n + 1];
    row[0] = 0;

    // assume values are sorted by row then col
    // do multiple passes to maintain order
    k = 0;
    for(i = 0; i < n; i++) {
        // 1. pass: transpose columns to first half of rows
        for (j = 0; j < mtx_nnz; j++) {
            if (mtx_col[j] == i) {
                val[k] = mtx_val[j];
                col[k] = mtx_row[j];
                k++;
            }
        }
        // 2. pass: append rows to second half of rows
        for (j = 0; j < mtx_nnz; j++) {
            if (mtx_row[j] == i && mtx_row[j] != mtx_col[j]) {
                val[k] = mtx_val[j];
                col[k] = mtx_col[j];
                k++;
            }
        }
        row[i + 1] = k;
    }

    delete[] mtx_val;
    delete[] mtx_col;
    delete[] mtx_row;

    A->n = n; A->nnz = nnz; A->h_val = val; A->h_col = col; A->h_row = row;
}

void dv_swap(dv_t* &v, dv_t* &w) {
    dv_t* tmp = v;
    v = w;
    w = tmp;
}

void dm_swap(dm_t* &A, dm_t* &B) {
    dm_t* tmp = A;
    A = B;
    B = tmp;
}

void dm_assemble_blocks(dm_t** alpha, dm_t** beta, const int b, const int m, dm_t* Tm) {

    for (int i = 0; i < m; i++) {
        dm_device_to_host(alpha[i]);
        dm_device_to_host(beta[i]);
    }
    cudaDeviceSynchronize();
    
    for (int z = 0; z < m * b * m * b; z++) {
        Tm->h_val[z] = 0.0;
    }

    // i = 0, a b
    for (int j = 0; j < b; j++) {
        for (int i = 0; i < b; i++) {
            Tm->h_val[j * b * m + i    ] = alpha[0]->h_val[j * b + i];
            Tm->h_val[j * b * m + i + b] = beta[1]->h_val[j * b + i];
        }
    }

    // 0 < i < m, b-1' a b
    int s = 0; // shift
    for (int k = 1; k < m - 1; k++) {
        s = k * m * b * b + (k - 1) * b;
        for (int j = 0; j < b; j++) {
            for (int i = 0; i < b; i++) {
                Tm->h_val[j * b * m + i +         s] = beta[k]->h_val[i * b + j];
                Tm->h_val[j * b * m + i + b +     s] = alpha[k]->h_val[j * b + i];
                Tm->h_val[j * b * m + i + 2 * b + s] = beta[k + 1]->h_val[j * b + i];
            }
        }
    }

    // i = m - 1, b' a
    s = (m - 1) * m * b * b + (m - 2) * b;
    for (int j = 0; j < b; j++) {
        for (int i = 0; i < b; i++) {
            Tm->h_val[j * b * m + i +     s] = beta[m - 1]->h_val[i * b + j];
            Tm->h_val[j * b * m + i + b + s] = alpha[m - 1]->h_val[j * b + i];
        }
    }

    dm_host_to_device(Tm);
    cudaDeviceSynchronize();
}


// -------------- (DE)ALLOCATION --------------

void spsm_dv_allocate_buffer_D(spsm_t* A, dv_t* v, dv_t* w) {
    size_t buffer_size;
    AS(cusparseSpMV_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_D, A->descr_D, v->descr_D, &ZERO_D, w->descr_D, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dv_buffer_D), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dv_allocate_buffer_S(spsm_t* A, dv_t* v, dv_t* w) {
    size_t buffer_size;
    AS(cusparseSpMV_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_S, v->descr_S, &ZERO_S, w->descr_S, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dv_buffer_S), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dv_allocate_buffer_H(spsm_t* A, dv_t* v, dv_t* w) {
    size_t buffer_size;
    AS(cusparseSpMV_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, v->descr_H, &ZERO_S, w->descr_H, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dv_buffer_H), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dv_allocate_buffer_HS(spsm_t* A, dv_t* v, dv_t* w) {
    size_t buffer_size;
    AS(cusparseSpMV_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, v->descr_H, &ZERO_S, w->descr_S, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dv_buffer_HS), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_allocate_buffer_D(spsm_t* A, dm_t* B, dm_t* C) {
    size_t buffer_size;
    AS(cusparseSpMM_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_D, A->descr_D, B->descr_D, &ZERO_D, C->descr_D, CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2, &buffer_size))
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dm_buffer_D), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_allocate_buffer_S(spsm_t* A, dm_t* B, dm_t* C) {
    size_t buffer_size;
    AS(cusparseSpMM_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_S, B->descr_S, &ZERO_S, C->descr_S, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &buffer_size))
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dm_buffer_S), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_allocate_buffer_H(spsm_t* A, dm_t* B, dm_t* C) {
    size_t buffer_size;
    AS(cusparseSpMM_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, B->descr_H, &ZERO_S, C->descr_H, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &buffer_size))
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dm_buffer_H), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_allocate_buffer_HS(spsm_t* A, dm_t* B, dm_t* C) {
    size_t buffer_size;
    AS(cusparseSpMM_bufferSize(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, B->descr_H, &ZERO_S, C->descr_S, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &buffer_size));
    AD(cudaDeviceSynchronize());
    AD(cudaMalloc(&(A->spsm_dm_buffer_HS), buffer_size));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_preprocess_HS(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM_preprocess(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, B->descr_H, &ZERO_S, C->descr_S, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_HS));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_preprocess_D(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM_preprocess(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_D, A->descr_D, B->descr_D, &ZERO_D, C->descr_D, CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_D));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_preprocess_S(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM_preprocess(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_S, B->descr_S, &ZERO_S, C->descr_S, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_S));
    AD(cudaDeviceSynchronize());
}

void spsm_dm_preprocess_H(spsm_t* A, dm_t* B, dm_t* C) {
    AS(cusparseSpMM_preprocess(A->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE_S, A->descr_H, B->descr_H, &ZERO_S, C->descr_H, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, A->spsm_dm_buffer_H));
    AD(cudaDeviceSynchronize());
}

dv_t* dv_init(cublasHandle_t* handle, const int n) {
    dv_t* v = (dv_t*) malloc(sizeof(dv_t));
    v->use_S = false;
    v->use_H = false;
    v->handle = handle;
    double* h_x = (double*) malloc(n * sizeof(double));
    double* d_x_D;
    AD(cudaMalloc(&(d_x_D), n * sizeof(double)));
    AS(cusparseCreateDnVec(&(v->descr_D), n, d_x_D, CUDA_R_64F));
    AD(cudaDeviceSynchronize());
    v->n = n; v->h_x = h_x; v->d_x_D = d_x_D;
    return v;
}

dv_t* dv_init_rand(cublasHandle_t* handle, const int n) {
    dv_t* v = dv_init(handle, n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        #ifdef deterministic
            v->h_x[i] = (i % 9) * 0.1;
        #else
            v->h_x[i] = distribution(gen);
        #endif
        norm += v->h_x[i] * v->h_x[i];
    }
    dv_host_to_device(v);
    AD(cudaDeviceSynchronize());

    norm = std::sqrt(norm);    
    const double a = 1.0 / norm;
    AB(cublasDscal_v2(*handle, n, &a, v->d_x_D, 1));
    AD(cudaDeviceSynchronize());

    return v;
}

dm_t* dm_init(cublasHandle_t* handle, const int n, const int m) {
    dm_t* A = (dm_t*) malloc(sizeof(dm_t));
    A->handle = handle;
    A->use_S = false;
    A->use_H = false;
    A->use_qr_S = false;
    A->use_qr_D = false;
    double* h_val = (double*) malloc(n * m * sizeof(double));
    double* d_val_D;
    AD(cudaMalloc(&(d_val_D), n * m * sizeof(double)));
    AS(cusparseCreateDnMat(&A->descr_D, n, m, n, d_val_D, CUDA_R_64F, CUSPARSE_ORDER_COL));
    AD(cudaDeviceSynchronize());
    A->n = n; A->m = m; A->lda = n; A->h_val = h_val; A->d_val_D = d_val_D;
    return A;
}

dm_t* dm_init_rand(cublasHandle_t* handle, const int n, const int m) {
    dm_t* A = dm_init(handle, n, m);
    int i, j;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    #ifdef deterministic
        const int inc = m % 9 == 0 ? 8 : 9;
    #endif
    double* norm = (double*) malloc(m * sizeof(double));
    for (j = 0; j < m; j++) {
        norm[j] = 0.0;
    }

    for (i = 0; i < n * m; i++) {
        #ifdef deterministic
            A->h_val[i] = (i % inc) * 0.1;
        #else
            A->h_val[i] = distribution(gen);
        #endif
    }

    for (j = 0; j < m; j++) {
        for (i = 0; i < n; i++) {
            // idx = col * cols + row
            norm[j] += A->h_val[j * m + i] * A->h_val[j * m + i]; 
        }
    }

    for (j = 0; j < m; j++) {
        norm[j] = 1.0 / std::sqrt(norm[j]);
    }

    for (j = 0; j < m; j++) {
        for (i = 0; i < n; i++) {
            A->h_val[j * m + i] = A->h_val[j * m + i] * norm[j]; 
        }
    }

    dm_host_to_device(A);
    AD(cudaDeviceSynchronize());
    free(norm);

    return A;
}

spsm_t* spsm_init(const std::string &filepath) {

    spsm_t* A = (spsm_t*) malloc(sizeof(spsm_t));
    A->use_S = false;
    A->use_H = false;
    A->use_HS = false;
    A->use_dv = false;
    A->use_dm = false;
    AS(cusparseCreate(&(A->handle)));

    // mallocs h_val, h_col, h_row
    // initializes n, nnz
    if (ends_with(filepath, ".csr")) {
        spsm_load_csr(A, filepath);
    } else if (ends_with(filepath, ".mtx")) {
        spsm_load_mtx(A, filepath);
    }

    AD(cudaMalloc(&(A->d_val_D), A->nnz * sizeof(double)));
    AD(cudaMalloc(&(A->d_col), A->nnz * sizeof(int)));
    AD(cudaMalloc(&(A->d_row), (A->n + 1) * sizeof(int)));

    cudaMemcpy(A->d_val_D, A->h_val, A->nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A->d_col, A->h_col, A->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A->d_row, A->h_row, (A->n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    AS(cusparseCreateCsr(&(A->descr_D), A->n, A->n, A->nnz, A->d_row, A->d_col, A->d_val_D, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    AD(cudaDeviceSynchronize());

    return A;
}

tdm_t* tdm_init(const int m) {
    tdm_t* T = (tdm_t*) malloc(sizeof(tdm_t));
    double* alpha = new double[m];
    double* beta = new double[m];
    for(int i = 0; i < m; i++) {
        alpha[i] = beta[i] = 0;
    }
    T->alpha = alpha;
    T->beta = beta;
    return T;
}

void dv_free(dv_t* v) {
    // cuda handle is shared, don't destroy!
    free(v->h_x);
    AS(cusparseDestroyDnVec(v->descr_D));
    AD(cudaFree(v->d_x_D));
    if (v->use_S) {
        AS(cusparseDestroyDnVec(v->descr_S));
        AD(cudaFree(v->d_x_S));
    }
    if (v->use_H) {
        AS(cusparseDestroyDnVec(v->descr_H));
        AD(cudaFree(v->d_x_H));
    }
    free(v);
}

void dm_free(dm_t* A) {
    free(A->h_val);
    AS(cusparseDestroyDnMat(A->descr_D));
    AD(cudaFree(A->d_val_D));
    if (A->use_S) {
        AS(cusparseDestroyDnMat(A->descr_S));
        AD(cudaFree(A->d_val_S));
    }
    if (A->use_H) {
        AS(cusparseDestroyDnMat(A->descr_H));
        AD(cudaFree(A->d_val_H));
    }
    if (A->use_qr_D) {
        AO(cusolverDnDestroy(A->solver_handle));
        AD(cudaFree(A->d_info));
        AD(cudaFree(A->d_tau_D));
        AD(cudaFree(A->d_geqrf_buffer_D));
        AD(cudaFree(A->d_orgqr_buffer_D));
        AD(cudaFree(A->d_val_qr_D));
    }
    if (A->use_qr_S) {
        AO(cusolverDnDestroy(A->solver_handle));
        AD(cudaFree(A->d_info));
        AD(cudaFree(A->d_tau_S));
        AD(cudaFree(A->d_geqrf_buffer_S));
        AD(cudaFree(A->d_orgqr_buffer_S));
        AD(cudaFree(A->d_val_qr_S));
    }
    free(A);
}

void spsm_free(spsm_t* A) {
    AS(cusparseDestroy(A->handle));
    free(A->h_val);
    free(A->h_col);
    free(A->h_row);
    AS(cusparseDestroySpMat(A->descr_D));
    AD(cudaFree(A->d_val_D));
    if (A->use_S) {
        AS(cusparseDestroySpMat(A->descr_S));
        AD(cudaFree(A->d_val_S));
        if (A->use_dv) {
            AD(cudaFree(A->spsm_dv_buffer_S));
        }
        if (A->use_dm) {
            AD(cudaFree(A->spsm_dm_buffer_S));
        }
    }
    if (A->use_H) {
        AS(cusparseDestroySpMat(A->descr_H));
        AD(cudaFree(A->d_val_H));
    }
    if (A->use_H && !A->use_HS) { 
        if (A->use_dv) {
            AD(cudaFree(A->spsm_dv_buffer_H));
        }
        if (A->use_dm) {
            AD(cudaFree(A->spsm_dm_buffer_H));
        }
    }
    if (A->use_HS) {
        if (A->use_dv) {
            AD(cudaFree(A->spsm_dv_buffer_HS));
        }
        if (A->use_dm) {
            AD(cudaFree(A->spsm_dm_buffer_HS));
        }
    }
    if (!A->use_S && !A->use_H && !A->use_HS) {
        if (A->use_dv) {
            AD(cudaFree(A->spsm_dv_buffer_D));
        }
        if (A->use_dm) {
            AD(cudaFree(A->spsm_dm_buffer_D));
        }
    }
    AD(cudaFree(A->d_col));
    AD(cudaFree(A->d_row));
    free(A);
}

void tdm_free(tdm_t* T) {
    free(T->alpha);
    free(T->beta);
    free(T);
}

// ----------------- DEBUGGING -----------------

void print_matrix(const double* A, const int n, const int m) {
    std::cout << std::endl;
    if (n * m > 512) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                std::cout << (A[i * m + j] == 0.0 ? "0" : "1") << " ";
            }
            std::cout << std::endl;
        }
    } else {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                std::cout << std::setw(5) << std::fixed << std::setprecision(2) << A[i * m + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void print_col_major(double* val, int n, int m) {
    for (int i = 0; i < n ; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", val[j * n + i]);
        }
        printf("\n");
    }
    printf("\n");
}


__global__ void _print_H(half* xs, const int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", __half2float(xs[i]));
    }
    printf("\n");
}

__global__ void _print_S(float* xs, const int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", xs[i]);
    }
    printf("\n");
}

__global__ void _print_D(double* xs, const int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", xs[i]);
    }
    printf("\n");
}

void print_H(half* xs, const int n) {
    _print_H<<<1, 1>>>(xs, n);
}

void print_S(float* xs, const int n) {
    _print_S<<<1, 1>>>(xs, n);
}

void print_D(double* xs, const int n) {
    _print_D<<<1, 1>>>(xs, n);
}
