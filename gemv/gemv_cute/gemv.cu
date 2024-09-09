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
__global__ void gemv_kernel_v1(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k) {
    int tid = threadIdx.x;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(1, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(1, n), make_stride(n, Int<1>{}));

    Tensor gB = local_tile(B, make_tile(Int<BN>{}, k), make_coord(_, 0)); // (BN, k, num_tile_N)
    Tensor gC = local_tile(C, make_tile(Int<1>{}, Int<BN>{}), make_coord(0, _)); // (BM, BN) 

    auto a = A(0, _);
    const auto num_iters = size<2>(gC);
    for (int i = 0; i < num_iters; i++) {
        
        auto b = gB(tid, _, i);

        float psum = 0.0;
        for (int j = 0; j < k; j++) {
            psum += (float)a(j) * (float)b(j);
        }
        gC(0, tid, i) = (T)psum;
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
    constexpr int BK = 128;
    using thr_layout = decltype(make_layout(make_shape(Int<BN>{}, Int<1>{})));

    const int blocks_x = (n + BN - 1) / BN;
    const int blocks_y = (k + BK - 1) / BK;
    dim3 blocks(blocks_x, blocks_y);

    gemv_kernel_v1<T, thr_layout, BN><<<blocks, BN>>>(a, b, c, 1, n, k);
    
    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

template <typename T, typename ThrLayout, int BN, int BK,
          typename G2SCopyB>
__global__ void gemv_kernel_v2(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k) {
    int tid = threadIdx.x;
    int bix = blockIdx.x;
    int biy = blockIdx.y;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(1, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(1, n), make_stride(n, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<1>{}, Int<BK>{}), make_coord(0, biy));      // (1, k)
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bix, 0));     // (BN, k)
    Tensor gC = local_tile(C, make_tile(Int<1>{}, Int<BN>{}), make_coord(0, bix));      // (1, BN) 

    Tensor thr_tile_A = local_partition(gA, ThrLayout{}, threadIdx.x);   // (ThrValM, ThrValN)
    Tensor tCrA = make_fragment_like(thr_tile_A);                 // (ThrValM, ThrValN)

    Tensor tCrB = make_tensor_like(gB);                 // (ThrValM, ThrValN)

    // Vector dimensions
    G2SCopyB g2r_tiled_copy_b;              

    // Construct a Tensor corresponding to each thread's slice.
    auto g2r_thr_copy_b = g2r_tiled_copy_b.get_thread_slice(threadIdx.x);
    Tensor tBgB = g2r_thr_copy_b.partition_S(gB);                    // (CopyOp, CopyM, CopyN)
    Tensor tCrB_copy = g2r_thr_copy_b.partition_D(tCrB);             // (CopyOp, CopyM, CopyN)

    // copy A from GMEM to RMEM
    copy(thr_tile_A, tCrA);

    const int num_iters = size<2>(tBgB);
    float psum = 0.0;
    CUTE_UNROLL
    for (int i = 0; i < num_iters; ++i) {        
        copy(g2r_tiled_copy_b, tBgB(_, _, i), tCrB_copy(_, _, i));
        CUTE_UNROLL
        for (int j = i * 8; j < i * 8 + 8; ++j) {
            psum += (float)tCrA(0, j) * (float)tCrB(tid, j);
        }

    }
    gC(0, tid) = (T)psum;

    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     PRINT("A", A.shape())
    //     PRINT("gB", gB.shape())  
    //     PRINT("gC", gC.shape())    
    //     PRINT("thr_tile_A", thr_tile_A.shape())
    //     PRINT("tCrA", tCrA.shape())
    //     PRINT("tBgB", tBgB.shape())
    //     PRINT("tCrB", tCrB.shape())
    //     PRINT("tCrB_copy", tCrB_copy.shape())
    // }
}

template <typename T>
void gemv_v2(T *a, T *b, T *c, int m, int n, int k) {
    // Launch the kernel
    constexpr int BN = 128;
    constexpr int BK = 128;
    using thr_layout = decltype(make_layout(make_shape(Int<BN>{}, Int<1>{}))); 

    using g2r_copy_atom = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using G2RCopyB =
        decltype(make_tiled_copy(g2r_copy_atom{},
                                 thr_layout{},
                                 make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8


    const int blocks_x = (n + BN - 1) / BN;
    const int blocks_y = (k + BK - 1) / BK;
    dim3 blocks(blocks_x, blocks_y);

    gemv_kernel_v2<T, thr_layout, BN, BK, G2RCopyB><<<blocks, BN>>>(a, b, c, 1, n, k);
    
    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <typename T>
__global__ void gemv_kernel_v3(T *Aptr, T *Bptr, T *Cptr, int m, int n, int k) {
    // Block index
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 16;
    int laneId = tx % warp_size;

    // Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(1, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(1, n), make_stride(n, Int<1>{}));

    // Tensor gA = local_tile(A, make_tile(Int<1>{}, Int<128>{}), make_coord(0, 0)); 
    Tensor gB = local_tile(B, make_tile(Int<8>{}, Int<128>{}), make_coord(bx, 0));     // (BN, k)
    Tensor gC = local_tile(C, make_tile(Int<1>{}, Int<8>{}), make_coord(0, bx));      // (1, BN) 

    float res = 0;
    Bptr = &gB(ty, 0);

    int current_col_vec = laneId;
    float4 current_val= reinterpret_cast<float4 *>(Bptr)[current_col_vec];
    float4 current_x = reinterpret_cast<float4 *>(Aptr)[current_col_vec];
    const half2* vec_h1 = (half2*)&current_x.x;
    const half2* vec_h2 = (half2*)&current_x.y;
    const half2* vec_h3 = (half2*)&current_x.z;
    const half2* vec_h4 = (half2*)&current_x.w;
    const half2* mat_h1 = (half2*)&current_val.x;
    const half2* mat_h2 = (half2*)&current_val.y;
    const half2* mat_h3 = (half2*)&current_val.z;
    const half2* mat_h4 = (half2*)&current_val.w;
    res += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
    res += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
    res += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
    res += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
    res += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
    res += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
    res += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
    res += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);

    res = warpReduceSum<warp_size>(res);

    if(laneId == 0) gC(0, ty) = res;
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
    {
        // PRINT("gA", A.shape())
        PRINT("gB", gB.shape())  
        PRINT("gC", gC.shape())    
        print_tensor(gC(0, _));
    }
}

template <typename T>
void gemv_v3(T *a, T *b, T *c, int m, int n, int k) {
    dim3 dimGrid(n / 8);
    dim3 dimBlock(16, 8);
    gemv_kernel_v3<T><<<dimGrid, dimBlock>>>(a, b, c, 1, n, k);
    
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

    cpuGemv(h_a, h_b, h_c, M, N, K);

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

template <typename T>
float testF16F16GemvPerformance(
    void (*gpuF16F16Gemv) (T *, T *, T *, int, int, int),
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
        gpuF16F16Gemv(d_a, d_b, d_c, M, N, K);
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

    for (int i = 0; i < test_num; i++) {
        M_list[i] = 1;
        N_list[i] = (i + 1) * 256;
    }

    const int outer_repeat = 10, inner_repeat = 1;

    printf("\nalgo = Cute_HGEMV\n");

    const int M = 1, N = 4096, K = 128;
    float max_error = testF16F16GemmMaxError<T>(
        gemv_v3, M, N, K);
    printf("Max Error = %f\n", max_error);

    // for (int j = 0; j < test_num; j++) {
    //     int M = M_list[j], N = N_list[j], K = 128;

    //     double max_sec = 0.0;
    //     double min_sec = DBL_MAX;
    //     double total_sec = 0.0;

    //     for (int k = 0; k < outer_repeat; k++) {
    //         double this_sec = testF16F16GemvPerformance<T>(
    //             gemv_v3, M, N, K, inner_repeat);
    //         max_sec = max(max_sec, this_sec);
    //         min_sec = min(min_sec, this_sec);
    //         total_sec += this_sec;
    //     }

    //     double avg_sec = total_sec / outer_repeat;
    //     double avg_Gflops = ((double)M) * N * K * 2 / 1000 / 1000 / 1000 / avg_sec;

    //     double avg_msec = avg_sec * 1000;
    //     printf("M N K = %6d %6d %6d, ", M, N, K);
    //     printf("Time = %12.8lf ms, ", avg_msec);
    //     printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
    // }

}