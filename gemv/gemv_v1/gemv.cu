#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>
#include "utils.h"

using T = cute::half_t;
using namespace cute;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define OFFSETCOL(row, col, ld) ((col) * (ld) + (row))

template <typename T>
void cpuGemv(T *a, T *b, T *c, int m, int n, int k) {
    for (int j = 0; j < n; j++) {
        float psum = 0.0;
        for (int l = 0; l < k; l++) {
            psum += (float)a[l] * (float)b[j * k + l];
        }
        
        c[j] = (T)psum;
    }
}




template <typename T, typename ThrLayout, int BN>
__global__ void gemv_kernel(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k) {
    int tid = threadIdx.x;
    int n_idx = blockIdx.x * BN + threadIdx.x;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(1, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(1, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;

    Tensor gB = local_tile(B, make_tile(Int<BN>{}, k), make_coord(_, 0)); // (BN, k, num_tile_N)
    Tensor gC = local_tile(C, make_tile(Int<1>{}, Int<BN>{}), make_coord(0, _)); // (BM, BN) 

    const auto num_iters = size<2>(gC);
    for (int i = 0; i < num_iters; i++) {
        auto c = gC(0,tid,i);
        auto a = A(0,_);
        auto b = gB(tid,_,i);

        float psum = 0.0;
        for (int j = 0; j < k; j++) {
            psum += (float)a(j) * (float)b(j);
        }
        c = (T)psum;
        gC(0,tid,i) = c;
    }

    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     PRINT("A", A.shape())
    //     PRINT("gB", gB.shape())  
    //     PRINT("gC", gC.shape())    
    // }
}

template <typename T>
void gemv_v1(T *a, T *b, T *c, int m, int n, int k) {
    // Launch the kernel
    constexpr int BN = 128;
    using thr_layout = decltype(make_layout(make_shape(Int<BN>{})));

    int numBlocks = (n + BN - 1) / BN;
    printf("numBlocks = %d\n", numBlocks);

    gemv_kernel<T, thr_layout, BN><<<numBlocks, BN>>>(a, b, c, 1, n, k);
    
    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}


template <typename T>
float testF16F16GemmMaxError(
    void (*gpuF16F16Gemv) (T *, T *, T *, int, int, int),
    int M, int N, int K) {

    size_t size_a = 1 * K * sizeof(T);
    size_t size_b = N * K * sizeof(T);
    size_t size_c = 1 * N * sizeof(T);

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
    for (int i = 0; i < N * K; i++)
        h_b[i] = (T)(rand() / float(RAND_MAX));

    // for (int ii = 0; ii < 10; ii++) {
    //     printf("a = %f, b = %f\n", (float)h_a[ii], (float)h_b[ii]);
    // }

    cpuGemv(h_a, h_b, h_c, M, N, K);

    printf("gpu \n");

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    gpuF16F16Gemv(d_a, d_b, d_c, M, N, K); // TODO

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
        M_list[i] = 1;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    const int outer_repeat = 10, inner_repeat = 1;

    printf("\nalgo = Cute_HGEMV_V1\n");

    const int M = 1, N = 256, K = 256;
    float max_error = testF16F16GemmMaxError<T>(
        gemv_v1, M, N, K);
    printf("Max Error = %f\n", max_error);

}