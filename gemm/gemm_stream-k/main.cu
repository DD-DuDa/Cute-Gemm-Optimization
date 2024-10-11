#include <cuda.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include "cutlass/workspace.h"
#include "gemm_kernel.hpp"

using T = cute::half_t;
using namespace cute;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define OFFSETCOL(row, col, ld) ((col) * (ld) + (row))

template <typename T>
void cpuF16F16Gemm(T *a, T *b, T *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSETCOL(k, n, K)];
            }
            c[OFFSET(m, n, N)] = (T)psum;
        }
    }
}

template <typename T>
void gemm_launch(T *a, T *b, T *c, int M, int N, int K) {
    static constexpr int kSMs = 108;
    using Blocks = Int<kSMs>;
    using Threads = Int<256>;
    using TileM = Int<32>;
    using TileN = Int<32>;
    using TileK = Int<64>;

    auto tiles_M = ceil_div(M, TileM{});
    auto tiles_N = ceil_div(N, TileN{});
    auto tiles_K = ceil_div(K, TileK{});
    auto tiles = tiles_M * tiles_N * tiles_K;

    int workspace_blocks;
    int workspace_accum_size;

    workspace_blocks = Blocks{}; // Int<108>{};

    // TODO: workspace_accum_size
    workspace_accum_size = sizeof(T) * 32 * 4;

    int workspace_size_partials = config::get_workspace_size_partials(workspace_blocks, Threads{}, workspace_accum_size);
    int workspace_size_barriers = config::get_workspace_size_barriers(workspace_blocks);
    int workspace_size = workspace_size_partials + workspace_size_barriers;

    // we only need to clear the workspace of barriers, and we assume that
    // the barriers workspace is at the beginning of the workspace
    cutlass::zero_workspace(d_workspace, workspace_size_barriers);
    
    gemm_host<
        T,
        Blocks,
        Threads,
        TileM,
        
    > (

    );

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("shape = (%d, %d, %d) err = %d, str = %s\n", M, N, K, err, cudaGetErrorString(err));
}


template <typename T>
float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm) (T *, T *, T *, int, int, int),
    int M, int N, int K) {

    size_t size_a = M * K * sizeof(T);
    size_t size_b = K * N * sizeof(T);
    size_t size_c = M * N * sizeof(T);

    T *h_a, *h_b, *d_a, *d_b;
    T *h_c, *d_c, *h_d_c;

    h_a = (T *)malloc(size_a);
    h_b = (T *)malloc(size_b);
    h_c = (T *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    h_d_c = (T *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = (T)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = (T)(rand() / float(RAND_MAX));

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c); 
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

    return max_error;
}

int main() {
    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];

    for (int i = 0; i < test_num; i++) {
        M_list[i] = (i + 1) * 256;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    const int outer_repeat = 10, inner_repeat = 1;

    printf("\nalgo = Stream-K \n");

    const int M = 1024, N = 1024, K = 1024;
    float max_error = testF16F16GemmMaxError<T>(
        gemm_launch, M, N, K);
    printf("Max Error = %f\n", max_error);


    return 0;
}