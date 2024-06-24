// CUDA 12.3.0
// https://docs.nvidia.com/cuda/archive/12.3.0/cublas/index.html
// https://docs.nvidia.com/cuda/archive/12.3.0/cusparse/index.html
// $ nvcc basic.cpp structures.cu -o basic -O3 -lcublas -lcusparse -lcusolver
// $ ./basic bla bla bla bla

// #define verify
// #define deterministic
// #define find_convergence
// #define profile
// #define bench_accuracy
// #define bench_time
// #define bench_energy
#define bench_convergence

int R = 1;

#include "structures.h"

#ifdef bench_energy
nvmlReturn_t result;
nvmlDevice_t device;
#endif

/**
 * Compute matrix error epsilon = ||A - B||F / (m - 1)
 * while disregarding the last col of A and B.
 */
double matrix_error(const double* A, const double* B, const int n, const int m) {
    double C_square_sum = 0;
    for (int i = 0; i < n; i++) {
        const int offset = i * m;
        for (int j = 0; j < m - 1; j++) {
            C_square_sum += (A[offset + j] - B[offset + j]) * (A[offset + j] - B[offset + j]);
        }
    }
    const double eps = std::sqrt(C_square_sum) / (m - 1);
    return eps;
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

/**
 * Apply the basic lanczos algorithm in full precision to obtain a tridiagonalized matrix
 * 
 * @param A symmetric matrix
 * @param m maximum number of iterations
 * @param check lanczos iteration is checked for convergence/breakdown
 * @param R number of repititions for benching
 * @return tridiagonalized matrix
*/
#ifdef bench_accuracy
double* basic_lanczos_D(spsm_t* A, const int m) {
#else
tdm_t* basic_lanczos_D(spsm_t* A, const int m) {
#endif
    #if defined(verify) || defined(bench_accuracy)
        int i;
    #endif
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    #if defined(verify) || defined(bench_accuracy)
        double* V = new double[n * m];
        for (i = 0; i < n * m; i++) V[i] = 0;
    #endif

    dv_t* v_prev = dv_init_rand(&handle, n);
    dv_t* w_prev = dv_init(&handle, n);
    dv_t* v = dv_init(&handle, n);
    dv_t* w = dv_init(&handle, n);

    spsm_dv_allocate_buffer_D(A, v, w);

    double alpha, beta;
    double beta_1, beta_prev, beta_test;

    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        int r;
        dv_t* v_start = dv_init(&handle, n);
        dv_device_to_device_D(v_prev, v_start);
        cudaDeviceSynchronize();

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef bench_convergence
            unsigned int* convergence_loops = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            #ifdef bench_convergence
                dv_t* rand_start = dv_init_rand(&handle, n);
                dv_device_to_device_D(rand_start, v_prev);
                cudaDeviceSynchronize();
                dv_free(rand_start);
            #else
                dv_device_to_device_D(v_start, v_prev);
                cudaDeviceSynchronize();
            #endif
    #endif

    // w = A * v
    spsm_dv_D(A, v_prev, w_prev);

    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif
    
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

        #if defined(bench_convergence) || defined(find_convergence)
            // if beta is zero, no new subspace was spanned
            // by the vector -> breakdown, don't restart
            // if beta is really now or does not change much
            // after X iterations -> convergence criteria met
            beta_test = std::abs(beta_prev - beta) / beta_1;
            if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
                #ifdef find_convergence
                    std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
                #endif
                #ifdef bench_convergence
                    convergence_loops[r] = j;
                #endif
                break;
            } else if (j % 10 == 0) {
                beta_prev = beta;
            }
        #endif

        #if defined(verify) || defined(bench_accuracy)
            // copy values of v to V col-wise
            dv_device_to_host(v_prev);
            for (i = 0; i < n; i++) {
                V[i * m + (j - 1)] = v_prev->h_x[i];
            }
        #endif

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

        #ifdef verify
            if (m < 20 || j % (m / 20) == 0) {
                // verify that vi and vj are orthogonal -> should be close to 0
                const double ort = dv_dv_D(v, v_prev);
                // verify that vi are normalized -> should be 1
                double norm = dv_dv_D(v, v);
                norm = std::sqrt(norm);
                std::cout << j << " : beta = " << beta << " : ort = " << ort << " : norm = " << norm << std::endl; 
            }
        #endif

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }

    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        }
        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif
    #endif

    #ifdef profile
        cudaProfilerStop();
    #endif

    #if defined(verify) || defined(bench_accuracy)
        // number of real iterations in case of breakdown/convergence
        const int m2 = j;

        // append last vector (after pointer swap)
        for (i = 0; i < n; i++) {
            dv_device_to_host(v_prev);
            V[i * m + (m2 - 1)] = v_prev->h_x[i];
        }
    #endif

    #ifdef verify

        // copy to new matrix of correct size
        double* V2 = new double[n * m2];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m2; j++) {
                V2[i * m2 + j] = V[i * m + j];
            }
        }

        // construct A*V and V*T with correct dimensions
        double* AV = spsm_m(A, V2, n, m2);
        double* VT = tdm_m(T, V2, n, m2);

        // eps = ( ||AV - VT||F / (m - 1) ) / ||A||F
        double epsilon = matrix_error(AV, VT, n, m2) / spsm_norm_D(A, &handle);
        std::cout << "VA = VT + eps, eps = " << epsilon << std::endl;

        delete[] V;
        delete[] V2;
        delete[] AV;
        delete[] VT;
    #endif

    #ifdef bench_energy
        unsigned int power_average = 0;
        for (r = 0; r < R; r++) power_average += power_levels[r];
        std::cout << (power_average / 1000.0) / R << " W" << std::endl;
    #endif

    #ifdef bench_convergence
        unsigned int convergence_average = 0;
        for (r = 0; r < R; r++) convergence_average += convergence_loops[r];
        std::cout << (convergence_average * 1.0) / R << std::endl;
    #endif

    AB(cublasDestroy_v2(handle));
    #ifdef bench_time
        dv_free(v_start);
    #endif
    #ifdef bench_energy
        delete[] power_levels;
    #endif
    #ifdef bench_convergence
        delete[] convergence_loops;
    #endif
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    #ifdef bench_accuracy
        tdm_free(T);
        return V;
    #else
        return T;
    #endif
}

/**
 * Apply the basic lanczos algorithm in mixed precision to obtain a tridiagonalized matrix
 * 
 * @param A symmetric matrix
 * @param m maximum number of iterations
 * @return tridiagonalized matrix
*/
#ifdef bench_accuracy
double* basic_lanczos_S(spsm_t* A, const int m) {
#else
tdm_t* basic_lanczos_S(spsm_t* A, const int m) {
#endif
    #if defined(verify) || defined(bench_accuracy)
        int i;
    #endif
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    #if defined(verify) || defined(bench_accuracy)
        double* V = new double[n * m];
        for (i = 0; i < n * m; i++) V[i] = 0;
    #endif

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

    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        int r;
        dv_t* v_start = dv_init(&handle, n);
        dv_use_S(v_start);
        dv_device_to_device_D(v_prev, v_start);
        dv_device_to_device_S(v_prev, v_start);
        cudaDeviceSynchronize();

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef bench_convergence
            unsigned int* convergence_loops = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            #ifdef bench_convergence
                dv_t* rand_start = dv_init_rand(&handle, n);
                dv_use_S(rand_start);
                dv_device_to_device_D(rand_start, v_prev);
                dv_device_to_device_S(rand_start, v_prev);
                cudaDeviceSynchronize();
                dv_free(rand_start);
            #else
                dv_device_to_device_D(v_start, v_prev);
                dv_device_to_device_S(v_start, v_prev);
                cudaDeviceSynchronize();
            #endif
    #endif

    // w = A * v
    spsm_dv_S(A, v_prev, w_prev);

    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif

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

        #if defined(bench_convergence) || defined(find_convergence)
            // if beta is zero, no new subspace was spanned
            // by the vector -> breakdown, don't restart
            // if beta is really now or does not change much
            // after X iterations -> convergence criteria met
            beta_test = std::abs(beta_prev - beta) / beta_1;
            if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
                #ifdef find_convergence
                    std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
                #endif
                #ifdef bench_convergence
                    convergence_loops[r] = j;
                #endif
                break;
            } else if (j % 10 == 0) {
                beta_prev = beta;
            }
        #endif

        #if defined(verify) || defined(bench_accuracy)
            // copy values of v to V col-wise
            S2D(v_prev->d_x_S, v_prev->d_x_D, v_prev->n);
            AD(cudaDeviceSynchronize());
            dv_device_to_host(v_prev);
            for (i = 0; i < n; i++) {
                V[i * m + (j - 1)] = v_prev->h_x[i];
            }
        #endif

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

        #ifdef verify
            if (m < 20 || j % (m / 20) == 0) {
                // verify that vi and vj are orthogonal -> should be close to 0
                const double ort = dv_dv_S(v, v_prev);
                // verify that vi are normalized -> should be 1
                double norm = dv_dv_S(v, v);
                norm = std::sqrt(norm);
                std::cout << j << " : beta = " << beta << " : ort = " << ort << " : norm = " << norm << std::endl; 
            }
        #endif

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }

    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        }
        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif
    #endif

    #ifdef profile
        cudaProfilerStop();
    #endif

    #if defined(verify) || defined(bench_accuracy)
        // number of real iterations in case of breakdown/convergence
        const int m2 = j;

        // append last vector (after pointer swap)
        for (i = 0; i < n; i++) {
            S2D(v_prev->d_x_S, v_prev->d_x_D, v_prev->n);
            AD(cudaDeviceSynchronize());
            dv_device_to_host(v_prev);
            V[i * m + (m2 - 1)] = v_prev->h_x[i];
        }
    #endif

    #ifdef verify

        // copy to new matrix of correct size
        double* V2 = new double[n * m2];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m2; j++) {
                V2[i * m2 + j] = V[i * m + j];
            }
        }

        // construct A*V and V*T with correct dimensions
        double* AV = spsm_m(A, V2, n, m2);
        double* VT = tdm_m(T, V2, n, m2);

        // eps = ( ||AV - VT||F / (m - 1) ) / ||A||F
        double epsilon = matrix_error(AV, VT, n, m2) / spsm_norm_D(A, &handle);
        std::cout << "VA = VT + eps, eps = " << epsilon << std::endl;

        delete[] V;
        delete[] V2;
        delete[] AV;
        delete[] VT;
    #endif

    #ifdef bench_energy
        unsigned int power_average = 0;
        for (r = 0; r < R; r++) power_average += power_levels[r];
        std::cout << (power_average / 1000.0) / R << " W" << std::endl;
    #endif

    #ifdef bench_convergence
        unsigned int convergence_average = 0;
        for (r = 0; r < R; r++) convergence_average += convergence_loops[r];
        std::cout << (convergence_average * 1.0) / R << std::endl;
    #endif

    AB(cublasDestroy_v2(handle));
    #ifdef bench_time
        dv_free(v_start);
    #endif
    #ifdef bench_energy
        delete[] power_levels;
    #endif
    #ifdef bench_convergence
        delete[] convergence_loops;
    #endif
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    #ifdef bench_accuracy
        tdm_free(T);
        return V;
    #else
        return T;
    #endif
}


/**
 * Apply the basic lanczos algorithm in mixed precision to obtain a tridiagonalized matrix
 * 
 * @param A symmetric matrix
 * @param m maximum number of iterations
 * @return tridiagonalized matrix
*/
#ifdef bench_accuracy
double* basic_lanczos_H(spsm_t* A, const int m) {
#else
tdm_t* basic_lanczos_H(spsm_t* A, const int m) {
#endif
    #if defined(verify) || defined(bench_accuracy)
        int i;
    #endif
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    #if defined(verify) || defined(bench_accuracy)
        double* V = new double[n * m];
        for (i = 0; i < n * m; i++) V[i] = 0;
    #endif

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

    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        int r;
        dv_t* v_start = dv_init(&handle, n);
        dv_use_H(v_start);
        dv_device_to_device_D(v_prev, v_start);
        dv_device_to_device_H(v_prev, v_start);
        cudaDeviceSynchronize();

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef bench_convergence
            unsigned int* convergence_loops = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            #ifdef bench_convergence
                dv_t* rand_start = dv_init_rand(&handle, n);
                dv_use_H(rand_start);
                dv_device_to_device_D(rand_start, v_prev);
                dv_device_to_device_H(rand_start, v_prev);
                cudaDeviceSynchronize();
                dv_free(rand_start);
            #else
                dv_device_to_device_D(v_start, v_prev);
                dv_device_to_device_H(v_start, v_prev);
                cudaDeviceSynchronize();
            #endif
    #endif

    // w = A * v
    spsm_dv_H(A, v_prev, w_prev);
    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif
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

        #if defined(bench_convergence) || defined(find_convergence)
            // if beta is zero, no new subspace was spanned
            // by the vector -> breakdown, don't restart
            // if beta is really now or does not change much
            // after X iterations -> convergence criteria met
            beta_test = std::abs(beta_prev - beta) / beta_1;
            if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
                #ifdef find_convergence
                    std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
                #endif                
                #ifdef bench_convergence
                    convergence_loops[r] = j;
                #endif
                break;
            } else if (j % 10 == 0) {
                beta_prev = beta;
            }
        #endif

        #if defined(verify) || defined(bench_accuracy)
            // copy values of v to V col-wise
            H2D(v_prev->d_x_H, v_prev->d_x_D, v_prev->n);
            AD(cudaDeviceSynchronize());
            dv_device_to_host(v_prev);
            for (i = 0; i < n; i++) {
                V[i * m + (j - 1)] = v_prev->h_x[i];
            }
        #endif

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

        #ifdef verify
            if (m < 20 || j % (m / 20) == 0) {
                // verify that vi and vj are orthogonal -> should be close to 0
                const double ort = dv_dv_S(v, v_prev);
                // verify that vi are normalized -> should be 1
                double norm = dv_dv_S(v, v);
                norm = std::sqrt(norm);
                std::cout << j << " : beta = " << beta << " : ort = " << ort << " : norm = " << norm << std::endl; 
            }
        #endif

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }

    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        }
        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif
    #endif

    #ifdef profile
        cudaProfilerStop();
    #endif

    #if defined(verify) || defined(bench_accuracy)
        // number of real iterations in case of breakdown/convergence
        const int m2 = j;

        // append last vector (after pointer swap)
        for (i = 0; i < n; i++) {
            H2D(v_prev->d_x_H, v_prev->d_x_D, v_prev->n);
            AD(cudaDeviceSynchronize());
            dv_device_to_host(v_prev);
            V[i * m + (m2 - 1)] = v_prev->h_x[i];
        }
    #endif

    #ifdef verify

        // copy to new matrix of correct size
        double* V2 = new double[n * m2];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m2; j++) {
                V2[i * m2 + j] = V[i * m + j];
            }
        }

        // construct A*V and V*T with correct dimensions
        double* AV = spsm_m(A, V2, n, m2);
        double* VT = tdm_m(T, V2, n, m2);

        // eps = ( ||AV - VT||F / (m - 1) ) / ||A||F
        double epsilon = matrix_error(AV, VT, n, m2) / spsm_norm_D(A, &handle);
        std::cout << "VA = VT + eps, eps = " << epsilon << std::endl;

        delete[] V;
        delete[] V2;
        delete[] AV;
        delete[] VT;
    #endif

    #ifdef bench_energy
        unsigned int power_average = 0;
        for (r = 0; r < R; r++) power_average += power_levels[r];
        std::cout << (power_average / 1000.0) / R << " W" << std::endl;
    #endif

    #ifdef bench_convergence
        unsigned int convergence_average = 0;
        for (r = 0; r < R; r++) convergence_average += convergence_loops[r];
        std::cout << (convergence_average * 1.0) / R << std::endl;
    #endif

    AB(cublasDestroy_v2(handle));
    #ifdef bench_time
        dv_free(v_start);
    #endif
    #ifdef bench_energy
        delete[] power_levels;
    #endif
    #ifdef bench_convergence
        delete[] convergence_loops;
    #endif
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    #ifdef bench_accuracy
        tdm_free(T);
        return V;
    #else
        return T;
    #endif
}


/**
 * Apply the basic lanczos algorithm in mixed precision to obtain a tridiagonalized matrix
 * 
 * @param A symmetric matrix
 * @param m maximum number of iterations
 * @return tridiagonalized matrix
*/
#ifdef bench_accuracy
double* basic_lanczos_HS_1(spsm_t* A, const int m) {
#else
tdm_t* basic_lanczos_HS_1(spsm_t* A, const int m) {
#endif
    #if defined(verify) || defined(bench_accuracy)
        int i;
    #endif
    int j;
    const int n = A->n;

    cublasHandle_t handle;
    AB(cublasCreate_v2(&handle));

    tdm_t* T = tdm_init(m);

    #if defined(verify) || defined(bench_accuracy)
        double* V = new double[n * m];
        for (i = 0; i < n * m; i++) V[i] = 0;
    #endif

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


    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        int r;
        dv_t* v_start = dv_init(&handle, n);
        dv_use_H(v_start);
        dv_device_to_device_D(v_prev, v_start);
        dv_device_to_device_H(v_prev, v_start);
        cudaDeviceSynchronize();

        #ifdef bench_time
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        #ifdef bench_energy
            unsigned int* power_levels = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef bench_convergence
            unsigned int* convergence_loops = (unsigned int*) malloc(R * sizeof(unsigned int));
        #endif

        #ifdef profile
            cudaProfilerStart();
        #endif

        for (r = 0; r < R; r++) {
            #ifdef bench_convergence
                dv_t* rand_start = dv_init_rand(&handle, n);
                dv_use_H(rand_start);
                dv_device_to_device_D(rand_start, v_prev);
                dv_device_to_device_H(rand_start, v_prev);
                cudaDeviceSynchronize();
                dv_free(rand_start);
            #else
                dv_device_to_device_D(v_start, v_prev);
                dv_device_to_device_H(v_start, v_prev);
                cudaDeviceSynchronize();
            #endif
    #endif

    // w = A * v
    spsm_dv_HS(A, v_prev, w_prev);
    #ifdef bench_energy
        nvmlDeviceGetPowerUsage(device, power_levels + r);
    #endif
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

        #if defined(bench_convergence) || defined(find_convergence)
            // if beta is zero, no new subspace was spanned
            // by the vector -> breakdown, don't restart
            // if beta is really now or does not change much
            // after X iterations -> convergence criteria met
            beta_test = std::abs(beta_prev - beta) / beta_1;
            if (beta < 1e-8 || (j % 10 == 0 && beta_test < 1e-3)) {
                #ifdef find_convergence
                    std::cout << j << ": breaking loop, beta = " << beta << ", beta test = " << beta_test << std::endl;
                #endif                
                #ifdef bench_convergence
                    convergence_loops[r] = j;
                #endif
                break;
            } else if (j % 10 == 0) {
                beta_prev = beta;
            }
        #endif

        #if defined(verify) || defined(bench_accuracy)
            // copy values of v to V col-wise
            H2D(v_prev->d_x_H, v_prev->d_x_D, v_prev->n);
            AD(cudaDeviceSynchronize());
            dv_device_to_host(v_prev);
            for (i = 0; i < n; i++) {
                V[i * m + (j - 1)] = v_prev->h_x[i];
            }
        #endif

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

        #ifdef verify
            if (m < 20 || j % (m / 20) == 0) {
                // verify that vi and vj are orthogonal -> should be close to 0
                const double ort = dv_dv_D(v, v_prev);
                // verify that vi are normalized -> should be 1
                double norm = dv_dv_D(v, v);
                norm = std::sqrt(norm);
                std::cout << j << " : beta = " << beta << " : ort = " << ort << " : norm = " << norm << std::endl; 
            }
        #endif

        // iterate
        dv_swap(v, v_prev);
        dv_swap(w, w_prev);
    }

    #if defined(bench_time) || defined(bench_energy) || defined(bench_convergence) || defined(profile)
        }
        #ifdef bench_time
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        #endif
    #endif

    #ifdef profile
        cudaProfilerStop();
    #endif

    #if defined(verify) || defined(bench_accuracy)
        // number of real iterations in case of breakdown/convergence
        const int m2 = j;

        // append last vector (after pointer swap)
        for (i = 0; i < n; i++) {
            H2D(v_prev->d_x_H, v_prev->d_x_D, v_prev->n);
            AD(cudaDeviceSynchronize());
            dv_device_to_host(v_prev);
            V[i * m + (m2 - 1)] = v_prev->h_x[i];
        }
    #endif
    
    #ifdef verify

        // copy to new matrix of correct size
        double* V2 = new double[n * m2];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m2; j++) {
                V2[i * m2 + j] = V[i * m + j];
            }
        }

        // construct A*V and V*T with correct dimensions
        double* AV = spsm_m(A, V2, n, m2);
        double* VT = tdm_m(T, V2, n, m2);

        // eps = ( ||AV - VT||F / (m - 1) ) / ||A||F
        double epsilon = matrix_error(AV, VT, n, m2) / spsm_norm_D(A, &handle);
        std::cout << "VA = VT + eps, eps = " << epsilon << std::endl;

        delete[] V;
        delete[] V2;
        delete[] AV;
        delete[] VT;
    #endif

    #ifdef bench_energy
        unsigned int power_average = 0;
        for (r = 0; r < R; r++) power_average += power_levels[r];
        std::cout << (power_average / 1000.0) / R << " W" << std::endl;
    #endif

    #ifdef bench_convergence
        unsigned int convergence_average = 0;
        for (r = 0; r < R; r++) convergence_average += convergence_loops[r];
        std::cout << (convergence_average * 1.0) / R << std::endl;
    #endif

    AB(cublasDestroy_v2(handle));
    #ifdef bench_time
        dv_free(v_start);
    #endif
    #ifdef bench_energy
        delete[] power_levels;
    #endif
    #ifdef bench_convergence
        delete[] convergence_loops;
    #endif
    dv_free(v_prev);
    dv_free(w_prev);
    dv_free(v);
    dv_free(w);
    #ifdef bench_accuracy
        tdm_free(T);
        return V;
    #else
        return T;
    #endif
}


int main(int argc, char *argv[]) {
    std::string file = argv[1];
    const int m = std::stoi(argv[2]);
    if (argc >= 4) {
        R = std::stoi(argv[3]);
    }

    spsm_t* A = spsm_init(file);
    A->use_dv = true;

    #ifdef bench_time
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        tdm_t* T;
        std::cout << "D = ";
        T = basic_lanczos_D(A, m);
        tdm_free(T);
        std::cout << "S = ";
        T = basic_lanczos_S(A, m);
        tdm_free(T);
        std::cout << "H = ";
        T = basic_lanczos_H(A, m);
        tdm_free(T);
        std::cout << "HS_1 = ";
        T = basic_lanczos_HS_1(A, m);
        tdm_free(T);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "total time = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
    #endif
    #ifdef bench_accuracy
        double* V_D = basic_lanczos_D(A, m);
        double* V_S = basic_lanczos_S(A, m);
        std::cout << "MSE S    = " << matrix_error_full(V_D, V_S, A->n, m) << std::endl;
        delete[] V_S;
        double* V_H = basic_lanczos_H(A, m);
        std::cout << "MSE H    = " << matrix_error_full(V_D, V_H, A->n, m) << std::endl;
        delete[] V_H;
        double* V_HS_1 = basic_lanczos_HS_1(A, m);
        std::cout << "MSE HS 1 = " << matrix_error_full(V_D, V_HS_1, A->n, m) << std::endl;
        delete[] V_HS_1;
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
        tdm_t* T;
        std::cout << "Average energy consumption in Watt:" << std::endl;
        std::cout << "D = ";
        T = basic_lanczos_D(A, m);
        tdm_free(T);
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::cout << "S = ";
        T = basic_lanczos_S(A, m);
        tdm_free(T);
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::cout << "H = ";
        T = basic_lanczos_H(A, m);
        tdm_free(T);
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::cout << "HS 1 = ";
        T = basic_lanczos_HS_1(A, m);
        tdm_free(T);
    #endif
    #ifdef bench_convergence
        tdm_t* T;
        std::cout << "Average convergence loop:" << std::endl;
        std::cout << "D = ";
        T = basic_lanczos_D(A, m);
        tdm_free(T);
        std::cout << "S = ";
        T = basic_lanczos_S(A, m);
        tdm_free(T);
        std::cout << "H = ";
        T = basic_lanczos_H(A, m);
        tdm_free(T);
        std::cout << "HS 1 = ";
        T = basic_lanczos_HS_1(A, m);
        tdm_free(T);
    #endif
    #ifdef find_convergence
        tdm_t* T = basic_lanczos_D(A, m);
        tdm_free(T);
    #endif
    #ifdef profile
        // std::cout << "D = ";
        // tdm_t* T = basic_lanczos_D(A, m);
        // tdm_free(T);
        std::cout << "H = ";
        tdm_t* T = basic_lanczos_H(A, m);
        tdm_free(T);
    #endif

    spsm_free(A);

    return 0;
}
