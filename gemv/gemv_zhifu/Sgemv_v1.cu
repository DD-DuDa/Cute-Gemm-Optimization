
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

using T = float;


// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

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

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

// if N>= 128
__global__ void Sgemv_v1( 
    float * __restrict__ A,
    float * __restrict__ x,
    float * __restrict__ y, 
    const int M,
    const int N) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size=32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if(current_row < M){
        float res=0;
        int kIteration = (N / warp_size) / 4; // 
        if(kIteration == 0) kIteration = 1;
        A = &A[current_row*N];
        #pragma unroll
        for(int i=0; i < kIteration; i++){
            int current_col_vec = (i*warp_size + laneId);
            float4 current_val= reinterpret_cast<float4 *>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4 *>(x)[current_col_vec];
            res += current_val.x*current_x.x;
            res += current_val.y*current_x.y;
            res += current_val.z*current_x.z;
            res += current_val.w*current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if(laneId==0) y[current_row]=res;
    }
}

template <typename T>
void gemv_zhihu(T *a, T *b, T *c, int m, int n, int k) {
    dim3 dimGrid(n / 4);
    dim3 dimBlock(32, 4);
    Sgemv_v1<<< dimGrid, dimBlock >>>(b, a, c, n, k);
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

int main(int argc, char** argv) {
    // Test correct
    const int M = 1, N = 2560, K = 128;
    float max_error = testF16F16GemvMaxError<T>(
        gemv_zhihu, M, N, K);
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
        int M = M_list[j], N = N_list[j], K = 128;

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemvPerformance<T>(
                gemv_zhihu, M, N, K, inner_repeat);
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
