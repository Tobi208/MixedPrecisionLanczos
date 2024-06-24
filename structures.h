// CUDA 12.3.0
// https://docs.nvidia.com/cuda/archive/12.3.0/cublas/index.html
// https://docs.nvidia.com/cuda/archive/12.3.0/cusparse/index.html
// https://docs.nvidia.com/cuda/archive/12.3.0/cusolver/index.html


#ifndef __STRUCTURES_H_
#define __STRUCTURES_H_

// standard libraries
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>

// nvidia libraries
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_fp16.h>
#include <cusolverDn.h>
#include <cuda_profiler_api.h>
#include <nvml.h>


// -------------- CONSTANTS --------------

const double ZERO_D = 0.0;
const double ONE_D = 1.0;
const float ZERO_S = 0.0;
const float ONE_S = 1.0;
const half ZERO_H = 0.0;
const half ONE_H = 1.0;
const double MINUSONE_D = -1.0;
const float MINUSONE_S = -1.0;


// -------------- STRUCTURES --------------

/**
 * Wrapper for cusparse dense vector in multiple precisions
 */
typedef struct dense_vector {
    int n;                        // size
    double* h_x;                  // host values double
    bool use_S;                   // flag for deallocating
    bool use_H;                   // flag for deallocating 
    double* d_x_D;                // device values double
    float* d_x_S;                 // device values single
    half* d_x_H;                  // device values half
    cublasHandle_t* handle;       // shared cublas handle
    cusparseDnVecDescr_t descr_D; // cusparse desc double
    cusparseDnVecDescr_t descr_S; // cusparse desc single
    cusparseDnVecDescr_t descr_H; // cusparse desc half
} dv_t;

/**
 *  Wrapper for cusparse dense matrix in multiple precisions
 */ 
typedef struct dense_matrix {

    int n;   // rows
    int m;   // columns
    int k;   // elementary reflectors
    int lda; // leading dimension = #rows

    bool use_S; // flag for deallocating
    bool use_H; // flag for deallocating

    double* h_val;   // host values double
    double* d_val_D; // device values double
    float* d_val_S;  // device values single
    half* d_val_H;   // device values half

    cublasHandle_t* handle;       // shared cublas handle
    cusparseDnMatDescr_t descr_D; // cusparse desc double
    cusparseDnMatDescr_t descr_S; // cusparse desc single
    cusparseDnMatDescr_t descr_H; // cusparse desc half
    
    cusolverDnHandle_t solver_handle; // cusolver handle
    bool use_qr_D;                    // flag for deallocating
    bool use_qr_S;                    // flag for deallocating
    int* d_info;                      // cusolver debugging memory location

    int tau_size;             // qr tau size
    int geqrf_buffer_size;    // qr buffer size
    int orgqr_buffer_size;    // qr buffer size
    double* d_val_qr_D;       // device qr values double
    float* d_val_qr_S;        // device qr values single
    double* d_tau_D;          // device tau double
    float* d_tau_S;           // device tau single
    double* d_geqrf_buffer_D; // device qr buffer double
    float* d_geqrf_buffer_S;  // device qr buffer single
    double* d_orgqr_buffer_D; // device qr buffer double
    float* d_orgqr_buffer_S;  // device qr buffer single
} dm_t;

/**
 *  Wrapper for cusparse sparse matrix in multiple precisions
 */ 
typedef struct sparse_symmetric_matrix {
    
    int n;           // #cols, #rows
    int nnz;         // #non-zero values

    double* h_val;   // host nnz double
    int* h_col;      // host col indices
    int* h_row;      // host row pointer

    double* d_val_D; // device nnz double
    float* d_val_S;  // device nnz single
    half* d_val_H;   // device nnz half
    int* d_col;      // device col indices
    int* d_row;      // device row pointer

    bool use_S;      // flag for deallocation
    bool use_H;      // flag for deallocation
    bool use_HS;     // flag for deallocation
    bool use_dv;     // flag for deallocation
    bool use_dm;     // flag for deallocation

    cusparseHandle_t handle;      // cusparse handle
    cusparseSpMatDescr_t descr_D; // cusparse desc double
    cusparseSpMatDescr_t descr_S; // cusparse desc single
    cusparseSpMatDescr_t descr_H; // cusparse desc half
    void* spsm_dv_buffer_D;       // spsm-dv buffer double
    void* spsm_dv_buffer_S;       // spsm-dv buffer single
    void* spsm_dv_buffer_H;       // spsm-dv buffer half
    void* spsm_dv_buffer_HS;      // spsm-dv buffer half/single
    void* spsm_dm_buffer_D;       // spsm-dm buffer double
    void* spsm_dm_buffer_S;       // spsm-dm buffer single
    void* spsm_dm_buffer_H;       // spsm-dm buffer half
    void* spsm_dm_buffer_HS;      // spsm-dm buffer half/single
} spsm_t;

/**
 *  Wrapper for tridiagonal matrix
 */ 
typedef struct tridiagonal_matrix {
    double* alpha; // host values double
    double* beta;  // host values double
} tdm_t;


// -------------- LEVEL 1 ROUTINES --------------

/**
 * Scale dense vector
 * 
 * @param v dense vector in double precision
 */
void dv_scale_D(dv_t* v, const double alpha);

/**
 * Scale dense vector
 * 
 * @param v dense vector in double precision
 */
void dv_scale_S(dv_t* v, const float alpha);

/**
 * Compute norm of dense vector
 * 
 * @param v dense vector in double precision
 * @return norm of vector
 */
double dv_norm_D(dv_t* v);

/**
 * Compute norm of dense vector
 * 
 * @param v dense vector in single precision
 * @return norm of vector
 */
float dv_norm_S(dv_t* v);

/**
 * Compute norm of dense matrix
 * 
 * @param v dense matrix in double precision
 * @return norm of matrix
 */
double dm_norm_D(dm_t* A);

/**
 * Compute norm of dense matrix
 * 
 * @param v dense matrix in single precision
 * @return norm of matrix
 */
float dm_norm_S(dm_t* A);

/**
 * Compute norm of sparse matrix
 * 
 * @param v sparse matrix in double precision
 * @return norm of matrix
 */
double spsm_norm_D(spsm_t* A, cublasHandle_t* handle);

/**
 * Compute norm of sparse matrix
 * 
 * @param v sparse matrix in single precision
 * @return norm of matrix
 */
float spsm_norm_S(spsm_t* A, cublasHandle_t* handle);

/**
 * Compute axpy w = alpha * v + w
 * 
 * @param v dense vector in double precision
 * @param w dense vector in double precision
 */
void dv_axpy_D(dv_t* v, dv_t* w, const double alpha);

/**
 * Compute axpy w = alpha * v + w
 * 
 * @param v dense vector in single precision
 * @param w dense vector in single precision
 */
void dv_axpy_S(dv_t* v, dv_t* w, const float alpha);

/**
 * Compute axpy B = alpha * A + B
 * 
 * @param v dense matrix in double precision
 * @param w dense matrix in double precision
 */
void dm_axpy_D(dm_t* A, dm_t* B, const double alpha);

/**
 * Compute axpy B = alpha * A + B
 * 
 * @param v dense matrix in single precision
 * @param w dense matrix in single precision
 */
void dm_axpy_S(dm_t* A, dm_t* B, const float alpha);

/**
 * Compute dot product v x w
 * 
 * @param v dense vector in double precision
 * @param w dense vector in double precision
 * @return dot product in double precision
 */
double dv_dv_D(dv_t* v, dv_t* w);

/**
 * Compute dot product v x w
 * 
 * @param v dense vector in single precision
 * @param w dense vector in single precision
 * @return dot product in single precision
 */
float dv_dv_S(dv_t* v, dv_t* w);


// -------------- LEVEL 2 ROUTINES --------------

/**
 * Compute w = A * v in double precision
 * 
 * @param A sparse matrix in double precision
 * @param v dense vector in double precision
 * @param w dense vector in double precision
 */
void spsm_dv_D(spsm_t* A, dv_t* v, dv_t* w);

/**
 * Compute w = A * v in single precision
 * 
 * @param A sparse matrix in single precision
 * @param v dense vector in single precision
 * @param w dense vector in single precision
 */
void spsm_dv_S(spsm_t* A, dv_t* v, dv_t* w);

/**
 * Compute w = A * v in single precision
 * 
 * @param A sparse matrix in half precision
 * @param v dense vector in half precision
 * @param w dense vector in half precision
 */
void spsm_dv_H(spsm_t* A, dv_t* v, dv_t* w);

/**
 * Compute w = A * v in single precision
 * 
 * @param A sparse matrix in half precision
 * @param v dense vector in half precision
 * @param w dense vector in single precision
 */
void spsm_dv_HS(spsm_t* A, dv_t* v, dv_t* w);


// -------------- LEVEL 3 ROUTINES --------------

/**
 * Compute C = A * B
 * 
 * @param A dense matrix in double precision
 * @param B dense matrix in double precision
 * @param C dense matrix in double precision
 */
void dm_dm_D(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A^T * B
 * 
 * @param A dense matrix in double precision
 * @param B dense matrix in double precision
 * @param C dense matrix in double precision
 */
void dm_dm_transA_D(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B^T
 * 
 * @param A dense matrix in double precision
 * @param B dense matrix in double precision
 * @param C dense matrix in double precision
 */
void dm_dm_transB_D(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B
 * 
 * @param A dense matrix in single precision
 * @param B dense matrix in single precision
 * @param C dense matrix in single precision
 */
void dm_dm_S(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A^T * B
 * 
 * @param A dense matrix in single precision
 * @param B dense matrix in single precision
 * @param C dense matrix in single precision
 */
void dm_dm_transA_S(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B^T
 * 
 * @param A dense matrix in single precision
 * @param B dense matrix in single precision
 * @param C dense matrix in single precision
 */
void dm_dm_transB_S(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B
 * 
 * @param A dense matrix in half precision
 * @param B dense matrix in half precision
 * @param C dense matrix in half precision
 */
void dm_dm_H(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A^T * B
 * 
 * @param A dense matrix in half precision
 * @param B dense matrix in half precision
 * @param C dense matrix in half precision
 */
void dm_dm_transA_H(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B^T
 * 
 * @param A dense matrix in half precision
 * @param B dense matrix in half precision
 * @param C dense matrix in half precision
 */
void dm_dm_transB_H(dm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B in double precision
 * 
 * @param A sparse matrix in double precision
 * @param B dense matrix in double precision
 * @param C dense matrix in double precision
 */
void spsm_dm_D(spsm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B in single precision
 * 
 * @param A sparse matrix in single precision
 * @param B dense matrix in single precision
 * @param C dense matrix in single precision
 */
void spsm_dm_S(spsm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B in single precision
 * 
 * @param A sparse matrix in half precision
 * @param B dense matrix in half precision
 * @param C dense matrix in half precision
 */
void spsm_dm_H(spsm_t* A, dm_t* B, dm_t* C);

/**
 * Compute C = A * B in single precision
 * 
 * @param A sparse matrix in half precision
 * @param B dense matrix in half precision
 * @param C dense matrix in single precision
 */
void spsm_dm_HS(spsm_t* A, dm_t* B, dm_t* C);

/**
 * Compute A * V on host machine
 * 
 * @param A n x n csr matrix
 * @param V n x m matrix
 * @returns AV n x m matrix
 */
double* spsm_m(spsm_t* A, const double* V, const int n, const int m);

/**
 * Compute V * T on host machine
 * 
 * @param T m x m tridiagonal matrix
 * @param V n x m matrix
 * @returns VT n x m matrix
 */
double* tdm_m(tdm_t* T, const double* V, const int n, const int m);

// -------------- QR DECOMPOSITION --------------

/**
 * Compute QR factorization A * R = A
 * 
 * @param A m x n dense matrix in double precision
 * @param R n x n dense matrix in double precision
 */
void dm_geqrf_D(dm_t* A, dm_t* R);

/**
 * Compute QR factorization A * R = A
 * 
 * @param A m x n dense matrix in single precision
 * @param R n x n dense matrix in single precision
 */
void dm_geqrf_S(dm_t* A, dm_t* R);

/**
 * Compute Q from elementary reflectors in A
 * 
 * @param A m x n dense matrix in double precision
 * @param Q m x n dense matrix in double precision
 */
void dm_orgqr_D(dm_t* A, dm_t* Q);

/**
 * Compute Q from elementary reflectors in A
 * 
 * @param A m x n dense matrix in single precision
 * @param Q m x n dense matrix in single precision
 */
void dm_orgqr_S(dm_t* A, dm_t* Q);


// -------------- ENABLE MIXED PRECISION --------------

void dv_use_S(dv_t* v);
void dv_use_H(dv_t* v);
void dm_use_S(dm_t* A);
void dm_use_H(dm_t* A);
void dm_use_qr_D(dm_t* A);
void dm_use_qr_S(dm_t* A);
void spsm_use_S(spsm_t* A);
void spsm_use_H(spsm_t* A);
void spsm_use_HS(spsm_t* A);


// -------------- CASTING & MEMORY TRANSFER --------------

void S2H(const float* xs, half* ys, const int n);
void D2H(const double* xs, half* ys, const int n);
void D2S(const double* xs, float* ys, const int n);
void H2S(const half* xs, float* ys, const int n);
void H2D(const half* xs, double* ys, const int n);
void S2D(const float* xs, double* ys, const int n);

void dv_device_to_host(dv_t* v);
void dv_host_to_device(dv_t* v);
void dv_device_to_device_D(dv_t* v, dv_t* w);
void dv_device_to_device_S(dv_t* v, dv_t* w);
void dv_device_to_device_H(dv_t* v, dv_t* w);
void dm_host_to_device(dm_t* A);
void dm_device_to_host(dm_t* A);
void dm_device_to_device_D(dm_t* A, dm_t* B);
void dm_device_to_device_S(dm_t* A, dm_t* B);
void dm_device_to_device_H(dm_t* A, dm_t* B);

/**
 * Copy upper triangle from A to B
 * @param A col-major n x m matrix
 * @param B col-major m x m matrix
 */
void dm_to_upper_triangle_D(dm_t* A, dm_t* B);

/**
 * Copy upper triangle from A to B
 * @param A col-major n x m matrix
 * @param B col-major m x m matrix
 */
void dm_to_upper_triangle_S(dm_t* A, dm_t* B);


// -------------- OTHER UTILITY --------------

bool ends_with(const std::string &str, const std::string &suffix);
void spsm_load_csr(spsm_t* A, const std::string &filepath);
void spsm_load_mtx(spsm_t* A, const std::string &filepath);
void dv_swap(dv_t* &v, dv_t* &w);
void dm_swap(dm_t* &A, dm_t* &B);

/**
 * Construct T from alpha, beta
 * @param alpha array of b x b dense matrices of length m
 * @param beta array of b x b dense matrices of length m
 * @param Tm m*b x m*b dense matrix constructed from alpha, beta blocks
 */
void dm_assemble_blocks(dm_t** alpha, dm_t** beta, const int b, const int m, dm_t* Tm);


// -------------- (DE)ALLOCATION --------------

void spsm_dv_allocate_buffer_D(spsm_t* A, dv_t* v, dv_t* w);
void spsm_dv_allocate_buffer_S(spsm_t* A, dv_t* v, dv_t* w);
void spsm_dv_allocate_buffer_H(spsm_t* A, dv_t* v, dv_t* w);
void spsm_dv_allocate_buffer_HS(spsm_t* A, dv_t* v, dv_t* w);
void spsm_dm_allocate_buffer_D(spsm_t* A, dm_t* B, dm_t* C);
void spsm_dm_allocate_buffer_S(spsm_t* A, dm_t* B, dm_t* C);
void spsm_dm_allocate_buffer_H(spsm_t* A, dm_t* B, dm_t* C);
void spsm_dm_allocate_buffer_HS(spsm_t* A, dm_t* B, dm_t* C);
void spsm_dm_preprocess_D(spsm_t* A, dm_t* B, dm_t* C);
void spsm_dm_preprocess_S(spsm_t* A, dm_t* B, dm_t* C);
void spsm_dm_preprocess_H(spsm_t* A, dm_t* B, dm_t* C);
void spsm_dm_preprocess_HS(spsm_t* A, dm_t* B, dm_t* C);

dv_t* dv_init(cublasHandle_t* handle, const int n);
dv_t* dv_init_rand(cublasHandle_t* handle, const int n);
dm_t* dm_init(cublasHandle_t* handle, const int n, const int m);
dm_t* dm_init_rand(cublasHandle_t* handle, const int n, const int m);
spsm_t* spsm_init(const std::string &filepath);
tdm_t* tdm_init(const int m);

void dv_free(dv_t* v);
void dm_free(dm_t* A);
void spsm_free(spsm_t* A);
void tdm_free(tdm_t* T);


// -------------- DEBUGGING --------------

void print_matrix(const double* A, const int n, const int m);
void print_col_major(double* val, int n, int m);
void print_H(half* xs, const int n);
void print_S(float* xs, const int n);
void print_D(double* xs, const int n);

#define AD(error_code)                                                     \
if (error_code != cudaSuccess)                                             \
    {                                                                      \
    std::cout << "The cuda call in " << __FILE__ << " on line "            \
                << __LINE__ << " resulted in the error '"                  \
                << cudaGetErrorString(error_code) << "'" << std::endl;     \
    std::abort();                                                          \
    }

#define AB(error_code)                                                     \
if (error_code != CUBLAS_STATUS_SUCCESS)                                   \
    {                                                                      \
    std::cout << "The cuda call in " << __FILE__ << " on line "            \
                << __LINE__ << " resulted in the error '"                  \
                << error_code << "'" << std::endl;                         \
    std::abort();                                                          \
    }

#define AS(error_code)                                                     \
if (error_code != CUSPARSE_STATUS_SUCCESS)                                 \
    {                                                                      \
    std::cout << "The cuda call in " << __FILE__ << " on line "            \
                << __LINE__ << " resulted in the error '"                  \
                << error_code << "'" << std::endl;                         \
    std::abort();                                                          \
    }

#define AO(error_code)                                                     \
if (error_code != CUSOLVER_STATUS_SUCCESS)                                 \
    {                                                                      \
    std::cout << "The cuda call in " << __FILE__ << " on line "            \
                << __LINE__ << " resulted in the error '"                  \
                << error_code << "'" << std::endl;                         \
    std::abort();                                                          \
    }

#endif // __STRUCTURES_H_
