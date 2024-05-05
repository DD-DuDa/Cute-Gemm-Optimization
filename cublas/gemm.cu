#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>
#include <cublas_v2.h>
#include <float.h>

inline cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const half* alpha,
     const half* A, int ldA,
     const half* B, int ldB,
     const half* beta,
     half* C, int ldC)
{
    // return cublasGemmEx(handle, transA, transB,
    //                   m, n, k,
    //                   reinterpret_cast<const __half*>(alpha),
    //                   reinterpret_cast<const __half*>(A), CUDA_R_16F, ldA,
    //                   reinterpret_cast<const __half*>(B), CUDA_R_16F, ldB,
    //                   reinterpret_cast<const __half*>(beta),
    //                   reinterpret_cast<      __half*>(C), CUDA_R_16F, ldC,
    //                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    return cublasHgemm(handle, transA, transB, m, n, k,
        alpha, (half *)B, ldB, (half *)A, ldA,
        beta, (half *)C, ldC);
}

float testF16F16GemmPerformance(int M, int N, int K, int repeat) {
    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half alpha = 1.f;
    half beta = 0.f;

    half *d_a, *d_b;
    half *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // warmup
    for (int i = 0; i < 10; ++i)
    {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_a, K, d_b, K, &beta, d_c, N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_a, K, d_b, K, &beta, d_c, N);
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

int main(int argc, char *argv[]) {
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

    printf("\nalgo = Cublas\n");

    for (int j = 0; j < test_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance(M, N, K, inner_repeat);
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