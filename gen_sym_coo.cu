/**
 * Generate a sparse symmetric COO matrix
 * 
 * Usage: ./gen_sym_coo <size> <density> <min val> <max val> <precision[double|single|half]> <output>
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void d2h(double* xs, const int n) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) xs[i] = (double) __double2half(xs[i]);
}

bool compare(const int* a, const int* b) {
    return a[1] == b[1] ? a[0] < b[0] : a[1] < b[1];
}

int main(int argc, char *argv[]) {
    const int n = std::stoi(argv[1]);
    const float d = std::stof(argv[2]);
    const float min_val = std::stof(argv[3]);
    const float max_val = std::stof(argv[4]);
    const std::string precision = std::string(argv[5]);
    const std::string output = std::string(argv[6]);

    const int half_size = n * (n + 1) / 2;
    const int full_size = n * n;
    int** coords = (int**) malloc(half_size * sizeof(int*));
    int i, j, k = 0;
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            int* coord = new int[2];
            coord[0] = i;
            coord[1] = j;
            coords[k++] = coord;
        }
    }
    std::random_shuffle(coords, coords + half_size);

    int full_nnz = 0;
    for (i = 0; i < half_size && 1.0 * full_nnz / full_size < d; i++) {
        full_nnz += coords[i][0] == coords[i][1] ? 1 : 2;
    }
    const int half_nnz = i;
    std::sort(coords, coords + half_nnz, compare);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(min_val, max_val);

    double* val = new double[half_nnz];
    for (i = 0; i < half_nnz; i++) {
        val[i] = distribution(gen);
    }

    if (precision == "half") {
        double* d_val;
        cudaMalloc(&d_val, half_nnz * sizeof(double));
        cudaMemcpy(d_val, val, half_nnz * sizeof(double), cudaMemcpyHostToDevice);
        const int block_size = 256;
        const int num_blocks = (half_nnz + block_size - 1) / block_size;
        d2h<<<num_blocks, block_size>>>(d_val, half_nnz);
        cudaDeviceSynchronize();
        cudaMemcpy(val, d_val, half_nnz * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_val);
    }

    std::ofstream out_file(output);
    out_file << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);

    out_file << n << " " << n << " " << half_nnz << std::endl;
    for (i = 0; i < half_nnz; i++) {
        out_file << coords[i][0] << " " << coords[i][1] << " " << val[i] << std::endl;
    }
    out_file.close();

    for (i = 0; i < half_size; i++) {
        free(coords[i]);
    }
    free(coords);
    free(val);
    return 0;
}
