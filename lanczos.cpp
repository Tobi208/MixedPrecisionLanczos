#include "structures.h"


// -------------- BASIC LANCZOS --------------

/**
 * Basic lanczos in double precision
 */
void basic_lanczos_D(spsm_t* A, const int m) {
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    dv_t* v_prev = dv_init_rand(&handle, n);
    dv_t* w_prev = dv_init(&handle, n);
    dv_t* v = dv_init(&handle, n);
    dv_t* w = dv_init(&handle, n);

    spsm_dv_allocate_buffer_D(A, v, w);

    double alpha, beta;
    double beta_1, beta_prev, beta_test;

    // w = A * v
    spsm_dv_D(A, v_prev, w_prev);
    
    // alpha = w * v
    alpha = dv_dv_D(v_prev, w_prev);

    // beta = ||w_j-1||
    beta_1 = beta_prev = dv_dv_D(w_prev, w_prev);
    beta_1 = beta_prev = std::sqrt(beta_prev);

    // w = w - alpha * v
    dv_axpy_D(v_prev, w_prev, -alpha);

    // write to tridiagonal matrix
    T->alpha[0] = alpha;

    for (j = 1; j < m; j++) {

        // beta = ||w_j-1||
        beta = dv_dv_D(w_prev, w_prev);
        beta = std::sqrt(beta);

        // if beta is zero, no new subspace was spanned
        // by the vector -> breakdown, don't restart
        // if beta is really now or does not change much
        // after X iterations -> convergence criteria met
        beta_test = std::abs(beta_prev - beta) / beta_1;
        if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
            std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
            break;
        } else if (j % 10 == 0) {
            beta_prev = beta;
        }

        // v = w_j-1 / beta
        dv_device_to_device_D(w_prev, v);
        dv_scale_D(v, 1.0 / beta);

        // w = A * v
        spsm_dv_D(A, v, w);

        // alpha = w * v
        alpha = dv_dv_D(v, w);

        // w = w - alpha * v - beta * v_j-1
        // w = w - alpha * v
        dv_axpy_D(v, w, -alpha);
        // w = w - beta * v_j-1
        dv_axpy_D(v_prev, w, -beta);

        // write to tridiagonal matrix
        T->alpha[j] = alpha;
        T->beta[j] = beta;

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }

    AB(cublasDestroy_v2(handle));
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    tdm_free(T);
}

/**
 * Basic lanczos in single precision
 */
void basic_lanczos_S(spsm_t* A, const int m) {
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    dv_t* v_prev = dv_init_rand(&handle, n);
    dv_t* w_prev = dv_init(&handle, n);
    dv_t* v = dv_init(&handle, n);
    dv_t* w = dv_init(&handle, n);

    dv_use_S(v_prev);
    dv_use_S(w_prev);
    dv_use_S(v);
    dv_use_S(w);
    spsm_use_S(A);
    AD(cudaDeviceSynchronize());

    spsm_dv_allocate_buffer_S(A, v, w);

    double alpha, beta;
    double beta_1, beta_prev, beta_test;

    // w = A * v
    spsm_dv_S(A, v_prev, w_prev);

    // alpha = w * v
    alpha = dv_dv_S(v_prev, w_prev);

    // beta = ||w_j-1||
    beta_1 = beta_prev = dv_dv_S(w_prev, w_prev);
    beta_1 = beta_prev = std::sqrt(beta_prev);

    // w = w - alpha * v
    dv_axpy_S(v_prev, w_prev, -alpha);

    // write to tridiagonal matrix
    T->alpha[0] = alpha;

    for (j = 1; j < m; j++) {

        // beta = ||w_j-1||
        beta = dv_dv_S(w_prev, w_prev);
        beta = std::sqrt(beta);

        // if beta is zero, no new subspace was spanned
        // by the vector -> breakdown, don't restart
        // if beta is really now or does not change much
        // after X iterations -> convergence criteria met
        beta_test = std::abs(beta_prev - beta) / beta_1;
        if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
            std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
                break;
        } else if (j % 10 == 0) {
            beta_prev = beta;
        }

        // v = w_j-1 / beta
        dv_device_to_device_S(w_prev, v);
        dv_scale_S(v, 1.0 / beta);

        // w = A * v
        spsm_dv_S(A, v, w);

        // alpha = w * v
        alpha = dv_dv_S(v, w);

        // w = w - alpha * v - beta * v_j-1
        // w = w - alpha * v
        dv_axpy_S(v, w, -alpha);
        // w = w - beta * v_j-1
        dv_axpy_S(v_prev, w, -beta);

        // write to tridiagonal matrix
        T->alpha[j] = alpha;
        T->beta[j] = beta;

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }
 
    AB(cublasDestroy_v2(handle));
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    tdm_free(T);
}

/**
 * Basic lanczos in minimal mixed precision
 */
void basic_lanczos_Mmin(spsm_t* A, const int m) {
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    dv_t* v_prev = dv_init_rand(&handle, n);
    dv_t* w_prev = dv_init(&handle, n);
    dv_t* v = dv_init(&handle, n);
    dv_t* w = dv_init(&handle, n);

    dv_use_S(v_prev);
    dv_use_S(w_prev);
    dv_use_S(v);
    dv_use_S(w);
    dv_use_H(v_prev);
    dv_use_H(w_prev);
    dv_use_H(v);
    dv_use_H(w);
    spsm_use_H(A);
    AD(cudaDeviceSynchronize());

    spsm_dv_allocate_buffer_H(A, v, w);

    double alpha, beta;
    double beta_1, beta_prev, beta_test;

    // w = A * v
    spsm_dv_H(A, v_prev, w_prev);
    H2D(w_prev->d_x_H, w_prev->d_x_D, w_prev->n);
    AD(cudaDeviceSynchronize());

    // alpha = w * v
    alpha = dv_dv_S(v_prev, w_prev);

    // beta = ||w_j-1||
    beta_1 = beta_prev = dv_dv_S(w_prev, w_prev);
    beta_1 = beta_prev = std::sqrt(beta_prev);

    // w = w - alpha * v
    dv_axpy_S(v_prev, w_prev, -alpha);

    // write to tridiagonal matrix
    T->alpha[0] = alpha;

    for (j = 1; j < m; j++) {

        // beta = ||w_j-1||
        beta = dv_dv_S(w_prev, w_prev);
        beta = std::sqrt(beta);

        // if beta is zero, no new subspace was spanned
        // by the vector -> breakdown, don't restart
        // if beta is really now or does not change much
        // after X iterations -> convergence criteria met
        beta_test = std::abs(beta_prev - beta) / beta_1;
        if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
            std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
            break;
        } else if (j % 10 == 0) {
            beta_prev = beta;
        }

        // v = w_j-1 / beta
        dv_device_to_device_S(w_prev, v);
        dv_scale_S(v, 1.0 / beta);

        S2H(v->d_x_S, v->d_x_H, v->n);
        AD(cudaDeviceSynchronize());

        // w = A * v
        spsm_dv_H(A, v, w);

        H2S(w->d_x_H, w->d_x_S, w->n);
        AD(cudaDeviceSynchronize());

        // alpha = w * v
        alpha = dv_dv_S(v, w);

        // w = w - alpha * v - beta * v_j-1
        // w = w - alpha * v
        dv_axpy_S(v, w, -alpha);
        // w = w - beta * v_j-1
        dv_axpy_S(v_prev, w, -beta);

        // write to tridiagonal matrix
        T->alpha[j] = alpha;
        T->beta[j] = beta;

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }

    AB(cublasDestroy_v2(handle));
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    tdm_free(T);
}

/**
 * Basic lanczos in optimal mixed precision
 */
void basic_lanczos_Mopt(spsm_t* A, const int m) {
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    dv_t* v_prev = dv_init_rand(&handle, n);
    dv_t* w_prev = dv_init(&handle, n);
    dv_t* v = dv_init(&handle, n);
    dv_t* w = dv_init(&handle, n);

    dv_use_H(v_prev);
    dv_use_S(w_prev);
    dv_use_H(v);
    dv_use_S(w);
    spsm_use_H(A);
    spsm_use_HS(A);
    AD(cudaDeviceSynchronize());

    spsm_dv_allocate_buffer_HS(A, v, w);

    double alpha, beta;
    double beta_1, beta_prev, beta_test;

    // w = A * v
    spsm_dv_HS(A, v_prev, w_prev);
    S2D(w_prev->d_x_S, w_prev->d_x_D, w_prev->n);
    AD(cudaDeviceSynchronize());


    // alpha = w * v
    alpha = dv_dv_D(v_prev, w_prev);

    // beta = ||w_j-1||
    beta_1 = beta_prev = dv_dv_D(w_prev, w_prev);
    beta_1 = beta_prev = std::sqrt(beta_prev);

    // w = w - alpha * v
    dv_axpy_D(v_prev, w_prev, -alpha);

    // write to tridiagonal matrix
    T->alpha[0] = alpha;

    for (j = 1; j < m; j++) {

        // beta = ||w_j-1||
        beta = dv_dv_D(w_prev, w_prev);
        beta = std::sqrt(beta);

        // if beta is zero, no new subspace was spanned
        // by the vector -> breakdown, don't restart
        // if beta is really now or does not change much
        // after X iterations -> convergence criteria met
        beta_test = std::abs(beta_prev - beta) / beta_1;
        if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
            std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
            break;
        } else if (j % 10 == 0) {
            beta_prev = beta;
        }

        // v = w_j-1 / beta
        dv_device_to_device_D(w_prev, v);
        dv_scale_D(v, 1.0 / beta);

        D2H(v->d_x_D, v->d_x_H, v->n);
        AD(cudaDeviceSynchronize());

        // w = A * v
        spsm_dv_HS(A, v, w);

        S2D(w->d_x_S, w->d_x_D, w->n);
        AD(cudaDeviceSynchronize());

        // alpha = w * v
        alpha = dv_dv_D(v, w);

        // w = w - alpha * v - beta * v_j-1
        // w = w - alpha * v
        dv_axpy_D(v, w, -alpha);
        // w = w - beta * v_j-1
        dv_axpy_D(v_prev, w, -beta);

        // write to tridiagonal matrix
        T->alpha[j] = alpha;
        T->beta[j] = beta;

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }

    AB(cublasDestroy_v2(handle));
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    tdm_free(T);
}


// -------------- BLOCK LANCZOS --------------

/**
 * Block lanczos in double precision
 */
void block_lanczos_D(spsm_t* A, const int m, const int b) {
    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    dm_use_qr_D(W);

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

    // Q * b = qr(W)
    dm_geqrf_D(W, beta[0]);
    dm_orgqr_D(W, Q_prev);

    // W = A * Q
    spsm_dm_D(A, Q_prev, W);

    // a = Q^T * W
    dm_dm_transA_D(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_D(Q_prev, alpha[0], Qmul);
    dm_axpy_D(W, Qmul, MINUSONE_D);

    for (i = 1; i < m; i++) {

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
}

/**
 * Block lanczos in single precision
 */
void block_lanczos_S(spsm_t* A, const int m, const int b) {
    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    spsm_use_S(A);

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    dm_use_S(W);
    dm_use_qr_S(W);

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

    // Q * b = qr(W)
    dm_geqrf_S(W, beta[0]);
    dm_orgqr_S(W, Q_prev);

    // W = A * Q
    spsm_dm_S(A, Q_prev, W);

    // a = Q^T * W
    dm_dm_transA_S(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_S(Q_prev, alpha[0], Qmul);
    dm_axpy_S(W, Qmul, MINUSONE_S);

    for (i = 1; i < m; i++) {

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
}

/**
 * Block lanczos in minimal mixed precision
 */
void block_lanczos_Mmin(spsm_t* A, const int m, const int b) {
    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    spsm_use_H(A);

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    dm_use_S(W);
    dm_use_qr_S(W);
    dm_use_H(W);

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

    // Q * b = qr(W)
    dm_geqrf_S(W, beta[0]);
    dm_orgqr_S(W, Q_prev);

    S2H(Q_prev->d_val_S, Q_prev->d_val_H, Q_prev->n * Q_prev->m);
    S2H(beta[0]->d_val_S, beta[0]->d_val_H, beta[0]->n * beta[0]->m);
    AD(cudaDeviceSynchronize());

    // W = A * Q
    spsm_dm_H(A, Q_prev, W);

    // a = Q^T * W
    dm_dm_transA_H(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_H(Q_prev, alpha[0], Qmul);
    H2S(Qmul->d_val_H, Qmul->d_val_S, Qmul->n * Qmul->m);
    H2S(W->d_val_H, W->d_val_S, W->n * W->m);
    AD(cudaDeviceSynchronize());
    dm_axpy_S(W, Qmul, MINUSONE_S);

    for (i = 1; i < m; i++) {

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
}

/**
 * Block lanczos in optimal mixed precision
 */
void block_lanczos_Mopt(spsm_t* A, const int m, const int b) {
    const int n = A->n;
    int i, j;

    cublasHandle_t cublas_handle;
    AB(cublasCreate_v2(&cublas_handle));

    spsm_use_H(A);
    spsm_use_HS(A);

    dm_t* W = dm_init_rand(&cublas_handle, n, b);

    dm_use_qr_D(W);
    dm_use_S(W);
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

    // Q * b = qr(W)
    dm_geqrf_D(W, beta[0]);
    dm_orgqr_D(W, Q_prev);

    D2H(Q_prev->d_val_D, Q_prev->d_val_H, Q_prev->n * Q_prev->m);
    AD(cudaDeviceSynchronize());

    // W = A * Q
    spsm_dm_HS(A, Q_prev, W);

    S2D(W->d_val_S, W->d_val_D, W->n * W->m);
    AD(cudaDeviceSynchronize());

    // a = Q^T * W
    dm_dm_transA_D(Q_prev, W, alpha[0]);

    // W = W - Q * a
    dm_dm_D(Q_prev, alpha[0], Qmul);
    dm_axpy_D(W, Qmul, MINUSONE_D);

    for (i = 1; i < m; i++) {

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
}


// -------------- MAIN --------------

/**
 * Print usage and exit
 */
void usage() {
    cout << "Usage:" << cout::endl;
    cout << "  ./lanczos <file [symmetric coo | csr]> basic <precision [D|S|Mmin|Mopt]> <iterations>" << cout::endl;
    cout << "  ./lanczos <file [symmetric coo | csr]> block <precision [D|S|Mmin|Mopt]> <iterations> <blocksize>" << cout::endl;
    exit(-1);
}

/**
 * Basic or block lanczos in 4 different precision classes
 */
int main(int argc, char *argv[]) {
    // timings
    std::chrono::steady_clock::time_point total_start, alg_start, stop;
    total_start = std::chrono::steady_clock::now();

    // parse args
    const std::string file = argv[1];
    if (!file.compare("--help")) usage();
    const std::string method = argv[2];
    const std::string p = argv[3];
    const int m = std::stoi(argv[4]);
    int b;
    if (!method.compare("block")) b = std::stoi(argv[5]);

    // parse symmetric COO/CSR input matrix
    spsm_t* A = spsm_init(file);
    const int n = A->n;
    stop = std::chrono::steady_clock::now();
    std::cout << "parsed input matrix in " << std::chrono::duration_cast<std::chrono::milliseconds> (stop - total_start).count() << " ms" << std::endl;
    alg_start = std::chrono::steady_clock::now();

    // run basic lanczos
    if (!method.compare("basic")) {
        if (!p.compare("D")) {
            basic_lanczos_D(A, m);
        } else if (!p.compare("S")) {
            basic_lanczos_S(A, m);
        } else if (!p.compare("Mmin")) {
            basic_lanczos_Mmin(A, m);
        } else if (!p.compare("Mopt")) {
            basic_lanczos_Mopt(A, m);
        } else {
            usage();
        }
    // run block lanczos
    } else if (!method.compare("block")) {
        A->use_dm = true;
        if (!p.compare("D")) {
            block_lanczos_D(A, m, b);
        } else if (!p.compare("S")) {
            block_lanczos_S(A, m, b);
        } else if (!p.compare("Mmin")) {
            block_lanczos_Mmin(A, m, b);
        } else if (!p.compare("Mopt")) {
            block_lanczos_Mopt(A, m, b);
        } else {
            usage();
        }
    } else {
        usage();
    }

    spsm_free(A);
    stop = std::chrono::steady_clock::now();
    std::cout << "total time = " << std::chrono::duration_cast<std::chrono::seconds> (stop - total_start).count() << " min" << std::endl;
    return 0;
}
