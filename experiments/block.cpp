// CUDA 12.3.0
// https://docs.nvidia.com/cuda/archive/12.3.0/cublas/index.html
// https://docs.nvidia.com/cuda/archive/12.3.0/cusparse/index.html
// $ nvcc block.cpp structures.cu -o block -O3 -lcublas -lcusparse -lcusolver -lnvidia-ml
// $ ./block bla bla bla bla


// #define deterministic
// #define verify
#define minimal_example
// #define profile
// #define bench_accuracy
// #define bench_time
// #define bench_energy

#include "structures.h"


int R = 1;

#ifdef bench_energy
nvmlReturn_t result;
nvmlDevice_t device;
#endif

/*
    n m * m k = n k

    A = n n
    W = n m
    Q = n m
    b = m m
    a = m m

    Q_0 * b = qr(W)
    W = A * Q_0
    a_0 = Q^T_0 * W
    W = W - Q_0 * a_0

    for 1 <= i < m:
        Q_i * b_i = qr(W)
        W = A * Q_i - Q_i-1 * b^T
        a = Q^T_i * W
        W = W - Q_i * a_i
*/

/**
 * ||AQ - QT||F / ((m - 1) * norm(A))
*/
double matrix_error(spsm_t* A, double* h_Qm, dm_t** alpha, dm_t** beta, const int n, const int b, const int m) {

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    dm_t* Qm = dm_init(&handle, n, m * b);
    AD(cudaMemcpy(Qm->d_val_D, h_Qm, n * m * b * sizeof(double), cudaMemcpyHostToDevice));    
    AD(cudaDeviceSynchronize());

    dm_t* Tm = dm_init(&handle, m * b, m * b);

    if (alpha[0]->use_S && !alpha[0]->use_H) {
        for (int i = 0; i < m; i++) {
            S2D(alpha[i]->d_val_S, alpha[i]->d_val_D, b * b);
            S2D(beta[i]->d_val_S, beta[i]->d_val_D, b * b);
        }
    }
    if (alpha[0]->use_H) {
        for (int i = 0; i < m; i++) {
            H2D(alpha[i]->d_val_H, alpha[i]->d_val_D, b * b);
            H2D(beta[i]->d_val_H, beta[i]->d_val_D, b * b);
        }
    }
    AD(cudaDeviceSynchronize());

    dm_assemble_blocks(alpha, beta, b, m, Tm);

    #ifdef minimal_example
        dm_device_to_host(Qm);
        dm_device_to_host(Tm);
        AD(cudaDeviceSynchronize());
        printf("Qm:\n");
        print_col_major(Qm->h_val, n, m * b);
        printf("Tm:\n");
        print_col_major(Tm->h_val, m * b, m * b);
    #endif

    dm_t* AQ = dm_init(&handle, n, m * b);
    dm_t* QT = dm_init(&handle, n, m * b);

    spsm_dm_allocate_buffer_D(A, Qm, AQ);
    spsm_dm_preprocess_D(A, Qm, AQ);
    spsm_dm_D(A, Qm, AQ);

    dm_dm_D(Qm, Tm, QT);

    dm_device_to_host(AQ);
    dm_device_to_host(QT);
    cudaDeviceSynchronize();

    int i, j;
    double sq, sqsum = 0.0;

    for (j = 0; j < (m - 1) * b; j++) {
        for (i = 0; i < n; i++) {
            sq = AQ->h_val[j * n + i] - QT->h_val[j * n + i];
            sqsum += sq * sq;
        }
    }

    dm_free(Qm);
    dm_free(Tm);
    dm_free(AQ);
    dm_free(QT);

    return std::sqrt(sqsum) / ((m - 1) * spsm_norm_D(A, &handle));
}

/**
 * Compute matrix error epsilon = ||A - B||F / m
 */
double matrix_error_full(const double* A, const double* B, const int n, const int m) {
    double C_square_sum = 0;
    for (int i = 0; i < n * m; i++) {
        C_square_sum += (A[i] - B[i]) * (A[i] - B[i]);
    }
    const double eps = std::sqrt(C_square_sum) / m;
    return eps;
}

#ifdef bench_accuracy
double* block_lanczos_D(spsm_t* A, const int m, const int b) {
#else
void block_lanczos_D(spsm_t* A, const int m, const int b) {
#endif

    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    #if defined(minimal_example)
        for (i = 0; i < n; i++) {
            W->h_val[i] = 1.0 / 3.0;
        }
        W->h_val[9] = 1.0 / std::sqrt(8);
        W->h_val[10] = -1.0 / std::sqrt(8);
        W->h_val[11] = 1.0 / std::sqrt(8);
        W->h_val[12] = -1.0 / std::sqrt(8);
        W->h_val[13] = 1.0 / std::sqrt(8);
        W->h_val[14] = -1.0 / std::sqrt(8);
        W->h_val[15] = 1.0 / std::sqrt(8);
        W->h_val[16] = -1.0 / std::sqrt(8);
        W->h_val[17] = 0;
        dm_host_to_device(W);
        cudaDeviceSynchronize();
    #endif

    dm_use_qr_D(W);

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        double* Qm = (double*) malloc(n * b * m * sizeof(double));
    #endif
    dm_t** alpha = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t** beta = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t* Q = dm_init(&cublas_handle, n, b);
    dm_t* Q_prev = dm_init(&cublas_handle, n, b);
    dm_t* Qmul = dm_init(&cublas_handle, n, b);

    for (i = 0; i < m; i++) {
        alpha[i] = dm_init(&cublas_handle, b, b);
        beta[i] = dm_init(&cublas_handle, b, b);
    }

    spsm_dm_allocate_buffer_D(A, Q_prev, W);
    spsm_dm_preprocess_D(A, Q_prev, W);

    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        int r;
        dm_t* W_start = dm_init(&cublas_handle, n, b);
        dm_device_to_device_D(W, W_start);
        AD(cudaDeviceSynchronize());

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            dm_device_to_device_D(W_start, W);
            AD(cudaDeviceSynchronize());

    #endif

    // Q * b = qr(W)
    dm_geqrf_D(W, beta[0]);
    dm_orgqr_D(W, Q_prev);

    // W = A * Q
    spsm_dm_D(A, Q_prev, W);

    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif

    // a = Q^T * W
    dm_dm_transA_D(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_D(Q_prev, alpha[0], Qmul);
    dm_axpy_D(W, Qmul, MINUSONE_D);

    for (i = 1; i < m; i++) {

        #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
            AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
            AD(cudaDeviceSynchronize());
        #endif

        // Q * b = qr(W)
        dm_geqrf_D(W, beta[i]);
        dm_orgqr_D(W, Q);

        // W = A * Q - Q_prev * b^T
        spsm_dm_D(A, Q, W);
        dm_dm_transB_D(Q_prev, beta[i], Qmul);
        dm_axpy_D(W, Qmul, MINUSONE_D);

        // a = Q^T * W
        dm_dm_transA_D(Q, W, alpha[i]);

        // W = W - Q * a
        dm_dm_D(Q, alpha[i], Qmul);
        dm_axpy_D(W, Qmul, MINUSONE_D);

        dm_swap(Q, Q_prev);
    }

    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        }

        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif

        #ifdef bench_energy
            unsigned int power_average = 0;
            for (r = 0; r < R; r++) power_average += power_levels[r];
            std::cout << (power_average / 1000.0) / R << " W" << std::endl;
        #endif

        #ifdef profile
            cudaProfilerStop();
        #endif
    #endif

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
        AD(cudaDeviceSynchronize());
    #endif

    #if defined(verify) || defined(minimal_example)
        std::cout << "AQ = QT + eps, eps = " << matrix_error(A, Qm, alpha, beta, n, b, m) << std::endl;
        free(Qm);
    #endif

    #ifdef bench_energy
        delete[] power_levels;
    #endif

    dm_free(W);
    for (i = 0; i < m; i++) {
        dm_free(alpha[i]);
        dm_free(beta[i]);
    }
    free(alpha);
    free(beta);
    dm_free(Q);
    dm_free(Q_prev);
    dm_free(Qmul);
    cublasDestroy_v2(cublas_handle);
    #ifdef bench_accuracy
        return Qm;
    #endif
}

#ifdef bench_accuracy
double* block_lanczos_S(spsm_t* A, const int m, const int b) {
#else
void block_lanczos_S(spsm_t* A, const int m, const int b) {
#endif

    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    spsm_use_S(A);

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    #if defined(minimal_example)
        for (i = 0; i < n; i++) {
            W->h_val[i] = 1.0 / 3.0;
        }
        W->h_val[9] = 1.0 / std::sqrt(8);
        W->h_val[10] = -1.0 / std::sqrt(8);
        W->h_val[11] = 1.0 / std::sqrt(8);
        W->h_val[12] = -1.0 / std::sqrt(8);
        W->h_val[13] = 1.0 / std::sqrt(8);
        W->h_val[14] = -1.0 / std::sqrt(8);
        W->h_val[15] = 1.0 / std::sqrt(8);
        W->h_val[16] = -1.0 / std::sqrt(8);
        W->h_val[17] = 0;
        dm_host_to_device(W);
        cudaDeviceSynchronize();
    #endif

    dm_use_S(W);
    dm_use_qr_S(W);

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        double* Qm = (double*) malloc(n * b * m * sizeof(double));
    #endif
    dm_t** alpha = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t** beta = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t* Q = dm_init(&cublas_handle, n, b);
    dm_t* Q_prev = dm_init(&cublas_handle, n, b);
    dm_t* Qmul = dm_init(&cublas_handle, n, b);

    dm_use_S(Q);
    dm_use_S(Q_prev);
    dm_use_S(Qmul);

    for (i = 0; i < m; i++) {
        alpha[i] = dm_init(&cublas_handle, b, b);
        beta[i] = dm_init(&cublas_handle, b, b);
        dm_use_S(alpha[i]);
        dm_use_S(beta[i]);
    }

    spsm_dm_allocate_buffer_S(A, Q_prev, W);
    spsm_dm_preprocess_S(A, Q_prev, W);

    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        int r;
        dm_t* W_start = dm_init(&cublas_handle, n, b);
        dm_use_S(W_start);
        dm_device_to_device_S(W, W_start);
        AD(cudaDeviceSynchronize());

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            dm_device_to_device_S(W_start, W);
            AD(cudaDeviceSynchronize());

    #endif

    // Q * b = qr(W)
    dm_geqrf_S(W, beta[0]);
    dm_orgqr_S(W, Q_prev);

    // W = A * Q
    spsm_dm_S(A, Q_prev, W);

    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif

    // a = Q^T * W
    dm_dm_transA_S(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_S(Q_prev, alpha[0], Qmul);
    dm_axpy_S(W, Qmul, MINUSONE_S);

    for (i = 1; i < m; i++) {

        #if defined(verify) || defined(minimal_example) ||  defined(bench_accuracy)
            S2D(Q_prev->d_val_S, Q_prev->d_val_D, Q_prev->n * Q_prev->m);
            AD(cudaDeviceSynchronize());
            AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
            AD(cudaDeviceSynchronize());
        #endif

        // Q * b = qr(W)
        dm_geqrf_S(W, beta[i]);
        dm_orgqr_S(W, Q);

        // W = A * Q - Q_prev * b^T
        spsm_dm_S(A, Q, W);
        dm_dm_transB_S(Q_prev, beta[i], Qmul);
        dm_axpy_S(W, Qmul, MINUSONE_S);

        // a = Q^T * W
        dm_dm_transA_S(Q, W, alpha[i]);

        // W = W - Q * a
        dm_dm_S(Q, alpha[i], Qmul);
        dm_axpy_S(W, Qmul, MINUSONE_S);

        dm_swap(Q, Q_prev);
    }

    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        }

        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif

        #ifdef bench_energy
            unsigned int power_average = 0;
            for (r = 0; r < R; r++) power_average += power_levels[r];
            std::cout << (power_average / 1000.0) / R << " W" << std::endl;
        #endif

        #ifdef profile
            cudaProfilerStop();
        #endif
    #endif

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        S2D(Q_prev->d_val_S, Q_prev->d_val_D, Q_prev->n * Q_prev->m);
        AD(cudaDeviceSynchronize());
        AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
        AD(cudaDeviceSynchronize());
    #endif

    #if defined(verify) || defined(minimal_example)
        std::cout << "AQ = QT + eps, eps = " << matrix_error(A, Qm, alpha, beta, n, b, m) << std::endl;
        free(Qm);
    #endif

    #ifdef bench_energy
        delete[] power_levels;
    #endif

    dm_free(W);
    for (i = 0; i < m; i++) {
        dm_free(alpha[i]);
        dm_free(beta[i]);
    }
    free(alpha);
    free(beta);
    dm_free(Q);
    dm_free(Q_prev);
    dm_free(Qmul);
    cublasDestroy_v2(cublas_handle);
    #ifdef bench_accuracy
        return Qm;
    #endif
}

#ifdef bench_accuracy
double* block_lanczos_H(spsm_t* A, const int m, const int b) {
#else
void block_lanczos_H(spsm_t* A, const int m, const int b) {
#endif

    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    spsm_use_H(A);

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    #if defined(minimal_example)
        for (i = 0; i < n; i++) {
            W->h_val[i] = 1.0 / 3.0;
        }
        W->h_val[9] = 1.0 / std::sqrt(8);
        W->h_val[10] = -1.0 / std::sqrt(8);
        W->h_val[11] = 1.0 / std::sqrt(8);
        W->h_val[12] = -1.0 / std::sqrt(8);
        W->h_val[13] = 1.0 / std::sqrt(8);
        W->h_val[14] = -1.0 / std::sqrt(8);
        W->h_val[15] = 1.0 / std::sqrt(8);
        W->h_val[16] = -1.0 / std::sqrt(8);
        W->h_val[17] = 0;
        dm_host_to_device(W);
        cudaDeviceSynchronize();
    #endif

    dm_use_S(W);
    dm_use_qr_S(W);
    dm_use_H(W);

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        double* Qm = (double*) malloc(n * b * m * sizeof(double));
    #endif
    dm_t** alpha = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t** beta = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t* Q = dm_init(&cublas_handle, n, b);
    dm_t* Q_prev = dm_init(&cublas_handle, n, b);
    dm_t* Qmul = dm_init(&cublas_handle, n, b);

    dm_use_S(Q);
    dm_use_S(Q_prev);
    dm_use_S(Qmul);
    dm_use_H(Q);
    dm_use_H(Q_prev);
    dm_use_H(Qmul);

    for (i = 0; i < m; i++) {
        alpha[i] = dm_init(&cublas_handle, b, b);
        beta[i] = dm_init(&cublas_handle, b, b);
        dm_use_S(alpha[i]);
        dm_use_S(beta[i]);
        dm_use_H(alpha[i]);
        dm_use_H(beta[i]);
    }

    spsm_dm_allocate_buffer_H(A, Q_prev, W);
    spsm_dm_preprocess_H(A, Q_prev, W);


    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        int r;
        dm_t* W_start = dm_init(&cublas_handle, n, b);
        dm_use_S(W_start);
        dm_use_H(W_start);
        dm_device_to_device_S(W, W_start);
        dm_device_to_device_H(W, W_start);
        AD(cudaDeviceSynchronize());

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            dm_device_to_device_S(W_start, W);
            dm_device_to_device_H(W, W_start);
            AD(cudaDeviceSynchronize());

    #endif

    // Q * b = qr(W)
    dm_geqrf_S(W, beta[0]);
    dm_orgqr_S(W, Q_prev);

    S2H(Q_prev->d_val_S, Q_prev->d_val_H, Q_prev->n * Q_prev->m);
    S2H(beta[0]->d_val_S, beta[0]->d_val_H, beta[0]->n * beta[0]->m);
    AD(cudaDeviceSynchronize());

    // W = A * Q
    spsm_dm_H(A, Q_prev, W);

    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif

    // a = Q^T * W
    dm_dm_transA_H(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_H(Q_prev, alpha[0], Qmul);
    H2S(Qmul->d_val_H, Qmul->d_val_S, Qmul->n * Qmul->m);
    H2S(W->d_val_H, W->d_val_S, W->n * W->m);
    AD(cudaDeviceSynchronize());
    dm_axpy_S(W, Qmul, MINUSONE_S);

    for (i = 1; i < m; i++) {

        #if defined(verify) || defined(minimal_example) ||  defined(bench_accuracy)
            H2D(Q_prev->d_val_H, Q_prev->d_val_D, Q_prev->n * Q_prev->m);
            AD(cudaDeviceSynchronize());
            AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
            AD(cudaDeviceSynchronize());
        #endif

        // Q * b = qr(W)
        dm_geqrf_S(W, beta[i]);
        dm_orgqr_S(W, Q);

        S2H(Q->d_val_S, Q->d_val_H, Q->n * Q->m);
        S2H(beta[i]->d_val_S, beta[i]->d_val_H, beta[i]->n * beta[i]->m);
        AD(cudaDeviceSynchronize());

        // W = A * Q - Q_prev * b^T
        spsm_dm_H(A, Q, W);
        dm_dm_transB_H(Q_prev, beta[i], Qmul);
        H2S(Qmul->d_val_H, Qmul->d_val_S, Qmul->n * Qmul->m);
        H2S(W->d_val_H, W->d_val_S, W->n * W->m);
        AD(cudaDeviceSynchronize());
        dm_axpy_S(W, Qmul, MINUSONE_S);

        S2H(W->d_val_S, W->d_val_H, W->n * W->m);
        AD(cudaDeviceSynchronize());

        // a = Q^T * W
        dm_dm_transA_H(Q, W, alpha[i]);

        // W = W - Q * a
        dm_dm_H(Q, alpha[i], Qmul);
        H2S(Qmul->d_val_H, Qmul->d_val_S, Qmul->n * Qmul->m);
        H2S(W->d_val_H, W->d_val_S, W->n * W->m);
        AD(cudaDeviceSynchronize());
        dm_axpy_S(W, Qmul, MINUSONE_S);

        dm_swap(Q, Q_prev);
    }

    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        }

        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif

        #ifdef bench_energy
            unsigned int power_average = 0;
            for (r = 0; r < R; r++) power_average += power_levels[r];
            std::cout << (power_average / 1000.0) / R << " W" << std::endl;
        #endif

        #ifdef profile
            cudaProfilerStop();
        #endif
    #endif

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        H2D(Q_prev->d_val_H, Q_prev->d_val_D, Q_prev->n * Q_prev->m);
        AD(cudaDeviceSynchronize());
        AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
        AD(cudaDeviceSynchronize());
    #endif

    #if defined(verify) || defined(minimal_example)
        std::cout << "AQ = QT + eps, eps = " << matrix_error(A, Qm, alpha, beta, n, b, m) << std::endl;
        free(Qm);
    #endif

    #ifdef bench_energy
        delete[] power_levels;
    #endif
    
    dm_free(W);
    for (i = 0; i < m; i++) {
        dm_free(alpha[i]);
        dm_free(beta[i]);
    }
    free(alpha);
    free(beta);
    dm_free(Q);
    dm_free(Q_prev);
    dm_free(Qmul);
    cublasDestroy_v2(cublas_handle);
    #ifdef bench_accuracy
        return Qm;
    #endif
}

#ifdef bench_accuracy
double* block_lanczos_HS_1(spsm_t* A, const int m, const int b) {
#else
void block_lanczos_HS_1(spsm_t* A, const int m, const int b) {
#endif

    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    spsm_use_H(A);
    spsm_use_HS(A);

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    #if defined(minimal_example)
        for (i = 0; i < n; i++) {
            W->h_val[i] = 1.0 / 3.0;
        }
        W->h_val[9] = 1.0 / std::sqrt(8);
        W->h_val[10] = -1.0 / std::sqrt(8);
        W->h_val[11] = 1.0 / std::sqrt(8);
        W->h_val[12] = -1.0 / std::sqrt(8);
        W->h_val[13] = 1.0 / std::sqrt(8);
        W->h_val[14] = -1.0 / std::sqrt(8);
        W->h_val[15] = 1.0 / std::sqrt(8);
        W->h_val[16] = -1.0 / std::sqrt(8);
        W->h_val[17] = 0;
        dm_host_to_device(W);
        cudaDeviceSynchronize();
    #endif

    dm_use_qr_D(W);
    dm_use_S(W);

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        double* Qm = (double*) malloc(n * b * m * sizeof(double));
    #endif
    dm_t** alpha = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t** beta = (dm_t**) malloc(m * sizeof(dm_t*));
    dm_t* Q = dm_init(&cublas_handle, n, b);
    dm_t* Q_prev = dm_init(&cublas_handle, n, b);
    dm_t* Qmul = dm_init(&cublas_handle, n, b);

    dm_use_H(Q);
    dm_use_H(Q_prev);

    for (i = 0; i < m; i++) {
        alpha[i] = dm_init(&cublas_handle, b, b);
        beta[i] = dm_init(&cublas_handle, b, b);
    }

    spsm_dm_allocate_buffer_HS(A, Q_prev, W);
    spsm_dm_preprocess_HS(A, Q_prev, W);


    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        int r;
        dm_t* W_start = dm_init(&cublas_handle, n, b);
        dm_device_to_device_D(W, W_start);
        AD(cudaDeviceSynchronize());

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            dm_device_to_device_D(W_start, W);
            AD(cudaDeviceSynchronize());

    #endif

    // Q * b = qr(W)
    dm_geqrf_D(W, beta[0]);
    dm_orgqr_D(W, Q_prev);

    D2H(Q_prev->d_val_D, Q_prev->d_val_H, Q_prev->n * Q_prev->m);
    AD(cudaDeviceSynchronize());

    // W = A * Q
    spsm_dm_HS(A, Q_prev, W);

    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif

    S2D(W->d_val_S, W->d_val_D, W->n * W->m);
    AD(cudaDeviceSynchronize());

    // a = Q^T * W
    dm_dm_transA_D(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_D(Q_prev, alpha[0], Qmul);
    dm_axpy_D(W, Qmul, MINUSONE_D);

    for (i = 1; i < m; i++) {

        #if defined(verify) || defined(minimal_example) ||  defined(bench_accuracy)
            AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
            AD(cudaDeviceSynchronize());
        #endif

        // Q * b = qr(W)
        dm_geqrf_D(W, beta[i]);
        dm_orgqr_D(W, Q);

        D2H(Q->d_val_D, Q->d_val_H, Q->n * Q->m);
        AD(cudaDeviceSynchronize());

        // W = A * Q - Q_prev * b^T
        spsm_dm_HS(A, Q, W);
        S2D(W->d_val_S, W->d_val_D, W->n * W->m);
        dm_dm_transB_D(Q_prev, beta[i], Qmul);
        AD(cudaDeviceSynchronize());
        dm_axpy_D(W, Qmul, MINUSONE_D);

        // a = Q^T * W
        dm_dm_transA_D(Q, W, alpha[i]);

        // W = W - Q * a
        dm_dm_D(Q, alpha[i], Qmul);
        dm_axpy_D(W, Qmul, MINUSONE_S);

        dm_swap(Q, Q_prev);
    }

    #if defined(profile) || defined(bench_time) || defined(bench_energy)
        }

        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif

        #ifdef bench_energy
            unsigned int power_average = 0;
            for (r = 0; r < R; r++) power_average += power_levels[r];
            std::cout << (power_average / 1000.0) / R << " W" << std::endl;
        #endif

        #ifdef profile
            cudaProfilerStop();
        #endif
    #endif

    #if defined(verify) || defined(minimal_example) || defined(bench_accuracy)
        AD(cudaMemcpy(Qm + n * b * (i - 1), Q_prev->d_val_D, n * b * sizeof(double), cudaMemcpyDeviceToHost));
        AD(cudaDeviceSynchronize());
    #endif

    #if defined(verify) || defined(minimal_example)
        std::cout << "AQ = QT + eps, eps = " << matrix_error(A, Qm, alpha, beta, n, b, m) << std::endl;
        free(Qm);
    #endif

    #ifdef bench_energy
        delete[] power_levels;
    #endif

    dm_free(W);
    for (i = 0; i < m; i++) {
        dm_free(alpha[i]);
        dm_free(beta[i]);
    }
    AD(cudaDeviceSynchronize());
    free(alpha);
    free(beta);
    dm_free(Q);
    dm_free(Q_prev);
    dm_free(Qmul);
    AD(cudaDeviceSynchronize());
    cublasDestroy_v2(cublas_handle);
    #ifdef bench_accuracy
        return Qm;
    #endif
}

int main(int argc, char *argv[]) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::string file = argv[1];
    const int m = std::stoi(argv[2]);
    const int b = std::stoi(argv[3]);
    if (argc >= 5) {
        R = std::stoi(argv[4]);
    }

    spsm_t* A = spsm_init(file);
    A->use_dm = true;
    const int n = A->n;

    #ifdef verify
        block_lanczos_S(A, m, b);
    #endif
    #ifdef minimal_example
        block_lanczos_D(A, m, b);
        block_lanczos_S(A, m, b);
        block_lanczos_H(A, m, b);
        block_lanczos_HS_1(A, m, b);
    #endif
    #ifdef profile
        block_lanczos_H(A, m, b);
    #endif
    #ifdef bench_accuracy
        double* Qm_D = block_lanczos_D(A, m, b);
        double* Qm_S = block_lanczos_S(A, m, b);
        std::cout << "S = " << matrix_error_full(Qm_D, Qm_S, n, b * m) << std::endl;
        free(Qm_S);
        double* Qm_H = block_lanczos_H(A, m, b);
        std::cout << "H = " << matrix_error_full(Qm_D, Qm_H, n, b * m) << std::endl;
        free(Qm_H);
        double* Qm_HS_1 = block_lanczos_HS_1(A, m, b);
        std::cout << "HS1 = " << matrix_error_full(Qm_D, Qm_HS_1, n, b * m) << std::endl;
        free(Qm_HS_1);
        free(Qm_D);
    #endif
    #ifdef bench_time
        std::cout << "D = ";
        block_lanczos_D(A, m, b);
        std::cout << "S = ";
        block_lanczos_S(A, m, b);
        std::cout << "H = ";
        block_lanczos_H(A, m, b);
        std::cout << "HS1 = ";
        block_lanczos_HS_1(A, m, b);
    #endif
    #ifdef bench_energy
        // Initialize NVML library
        result = nvmlInit();
        if (NVML_SUCCESS != result) {
            std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
            return 1;
        }
        // Get the first device (GPU 0)
        result = nvmlDeviceGetHandleByIndex(0, &device);
        if (NVML_SUCCESS != result) {
            std::cerr << "Failed to get handle for device 0: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 1;
        }
        unsigned int power = 0;
        result = nvmlDeviceGetPowerUsage(device, &power);
        if (NVML_SUCCESS != result) {
            std::cerr << "Failed to get power usage: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 1;
        }
        std::cout << "Average energy consumption in Watt:" << std::endl;
        std::cout << "D = ";
        block_lanczos_D(A, m, b);
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::cout << "S = ";
        block_lanczos_S(A, m, b);
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::cout << "H = ";
        block_lanczos_H(A, m, b);
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::cout << "HS1 = ";
        block_lanczos_HS_1(A, m, b);
    #endif

    spsm_free(A);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "total program time = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << " min" << std::endl;
    return 0;
}
