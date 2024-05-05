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

template <typename T, int BM, int BN, int BK, typename TiledMMA>
__global__ void gemm_cute_v1(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k) {

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _)); // (BM, BK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _)); // (BN, BK, num_tile_k)
    Tensor gC = local_tile(C, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix)); // (BM, BN) 
    

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
    
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {

    //     PRINT("gA", gA.shape())     
    //     PRINT("tAgA", tAgA.shape())
    //     print(layout<0>(tAgA));
    //     PRINT("tArA", tArA.shape()) 

    //     PRINT("gB", gB.shape())     
    //     PRINT("tBgB", tBgB.shape()) 
    //     print(layout<0>(tBgB));
    //     PRINT("tBrB", tBrB.shape()) 

    //     PRINT("gC", gC.shape())     
    //     PRINT("tCgC", tCgC.shape())
    //     print(layout<0>(tCgC)); 
    //     PRINT("tCrC", tCrC.shape()) 

    // }
    
    clear(tCrC);
    
    int num_tile_k = size<2>(gA);
#pragma unroll 1
    for(int itile = 0; itile < num_tile_k; ++itile) {
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC); 
}


template <typename T>
void gemm_v1(T *a, T *b, T *c, int m, int n, int k) {

    using mma_op = SM80_16x8x16_F16F16F16F16_TN; // A 行优先 B 列优先
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    // using MMA = decltype(make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
    //                                         Layout<Shape<_2, _4>,
    //                                         Stride<_4,_1>>{})
    //                                     ); 

    using MMA = decltype(make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                        Layout<Shape<_1, _1, _1>>{}));           // Tiler

    // TiledMMA mma1 = make_tiled_mma(mma_atom{},
    //                             Layout<Shape<_1,_1,_1>,S>{});    
                                
    // print_latex(mma1);

    constexpr int BM = 128; 
    constexpr int BN = 256; 
    constexpr int BK = 32; 

    int BX = (n + BN - 1) / BN;
    int BY = (m + BM - 1) / BM;

    // print(size(MMA{}));

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY);

    gemm_cute_v1<T, BM, BN, BK, MMA><<<grid, block>>>(a, b, c, m, n, k);
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

template <typename T>
float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm) (T *, T *, T *, int, int, int),
    int M, int N, int K, int repeat) {

    size_t size_a = M * K * sizeof(T);
    size_t size_b = K * N * sizeof(T);
    size_t size_c = M * N * sizeof(T);

    T *d_a, *d_b;
    T *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
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

    printf("\nalgo = Cute_HGEMM_V1\n");

    const int M = 256, N = 256, K = 256;
    float max_error = testF16F16GemmMaxError<T>(
        gemm_v1, M, N, K);
    printf("Max Error = %f\n", max_error);

    for (int j = 0; j < test_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance<T>(
                gemm_v1, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
        printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
    }


    return 0;
}