#include <cuda.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

using T = half;

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

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

///////////////////////////// REDUCE SUM //////////////////////////////

__device__ __forceinline__ float warpReduceSum(float sum,
    unsigned int threadNum) {
    if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
    if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}

///////////////////////////// NORMAL //////////////////////////////
// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
__global__ void gemv_kernel(T* mat, T* vec, T* res, unsigned int n,
                          unsigned int num_per_thread) {
    float sum = 0;
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;
    float4* mat4 = reinterpret_cast<float4*>(mat);
    float4* vec4 = reinterpret_cast<float4*>(vec);

    #pragma unroll
    for (int iter = 0; iter < num_per_thread >> 3; iter++) {
        unsigned int j = start_idx + iter * blockDim.x;
        if (j < n >> 3) {
        float4 vec_val = vec4[j];
        float4 mat_val = mat4[row * (n >> 3) + j];
        const half2* vec_h1 = (half2*)&vec_val.x;
        const half2* vec_h2 = (half2*)&vec_val.y;
        const half2* vec_h3 = (half2*)&vec_val.z;
        const half2* vec_h4 = (half2*)&vec_val.w;
        const half2* mat_h1 = (half2*)&mat_val.x;
        const half2* mat_h2 = (half2*)&mat_val.y;
        const half2* mat_h3 = (half2*)&mat_val.z;
        const half2* mat_h4 = (half2*)&mat_val.w;
        sum += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
        sum += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
        sum += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
        sum += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
        sum += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
        sum += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
        sum += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
        sum += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);
        }
    }

    sum = warpReduceSum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE) {
        if (tid == 0) {
        res[row] = __float2half(sum);
        }
        return;
    }

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE)
                ? warpLevelSums[threadIdx.y][laneId]
                : 0.0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
    if (tid == 0) {
        res[row] = __float2half(sum);
    }
}

template <typename T>
void fast_gemv(T *a, T *b, T *c, int m, int n, int k) {
    unsigned int block_dim_x = 32;
    unsigned int block_dim_y = 4;
    // assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    // assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = n / block_dim_x;
    // assert(num_per_thread >= 8);
    dim3 grid_dim(1, k / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    gemv_kernel<<<grid_dim, block_dim>>>(b, a, c,
        n, num_per_thread);

    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

template <typename T>
float testF16F16GemvMaxError(
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
    // Test correct
    const int M = 1, N = 1024, K = 1024;
    float max_error = testF16F16GemvMaxError<T>(
        fast_gemv, M, N, K);
    printf("Max Error = %f\n", max_error);

    // Benchmark
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

    for (int j = 0; j < 10; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemvPerformance<T>(
                fast_gemv, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1000 / 1000 / 1000 / avg_sec;

        double avg_msec = avg_sec * 1000;
        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Time = %12.8lf ms, ", avg_msec);
        printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
    }

    return 0;
}