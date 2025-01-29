#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>
#include "utils.h"

#include "cutlass/cluster_launch.hpp"

using T = cute::half_t;
using namespace cute;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define OFFSETCOL(row, col, ld) ((col) * (ld) + (row))
#define DEBUG 0
#define BENCHMARK 0

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


template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;
};


template <typename T, int BM, int BN, int BK, typename TiledMMA, 
            typename G2SCopyA, typename G2SCopyB,
            typename SmemLayoutA, typename SmemLayoutB>
__global__ static
__launch_bounds__(decltype(size(TiledMMA{}))::value)
void
gemm_device(const T *Aptr, const T *Bptr, T *Dptr, int m, int n, int k) {
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<T, T, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    // Initilize thread block
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    // global memory
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mD = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));

    // local tile
    Tensor gA = local_tile(mA, make_tile(Int<BM>{}, Int<BK>{}), make_coord(ix, _)); // (BM, BK, num_tile_k)
    Tensor gB = local_tile(mB, make_tile(Int<BN>{}, Int<BK>{}), make_coord(iy, _)); // (BN, BK, num_tile_k)
    Tensor gD = local_tile(mD, make_tile(Int<BM>{}, Int<BN>{}), make_coord(ix, iy)); // (BM, BN) 

    // shared memory
    Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // allocate "fragments"
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgD = thr_mma.partition_C(gD);

    // allocate accumulators and clear them
    Tensor tCrD = thr_mma.make_fragment_C(tCgD);
    clear(tCrD);

    // allocate fragments
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    
    // copy, from global memory to shared memory
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)
    
    // loop over k: i. load tile, ii. mma
    int ntile = k / BK;
    CUTE_UNROLL
    for (int itile = 0; itile < ntile; ++itile) {
        // copy  (CPY, CPY_M, CPY_K, k) , async
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile), tAsA_copy(_, _, _));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile), tBsB_copy(_, _, _));
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrD, tCrA(_, _, _), tCrB(_, _, _), tCrD);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    __syncthreads();
    // register to global memory
    cute::copy(tCrD, tCgD);

    #if DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        PRINT("gA", gA.shape())
        PRINT("gB", gB.shape())
        PRINT("gD", gD.shape())
        PRINT("sA", sA.shape())
        PRINT("sB", sB.shape())
        PRINT("tCsA", tCsA.shape())
        PRINT("tCsB", tCsB.shape())
    }
    #endif
}

template <typename T>
void gemm_wgmma(T *a, T *b, T *c, int M, int N, int K) {
    auto BM = Int<128>{};
    auto BN = Int<256>{};
    auto BK = Int< 64>{}; // at least 64

    // Define the smem layouts
    using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(BM,BK)));
    using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(BN,BK)));

    // mma
    // using TiledMMA = decltype(make_tiled_mma(SM90_64x8x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{}));
    using TileShape_MNK = decltype(make_shape(BM, BN, BK));
    using TiledMMA = decltype(make_tiled_mma(cute::GMMA::ss_op_selector<T, T, T, TileShape_MNK>()));

    
    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), // Thr layout 32x4 k-major
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8
    using G2SCopyB = G2SCopyA;

    // Launch parameter setup
    int smem_size = int(sizeof(SharedStorage<T, T, SmemLayoutA, SmemLayoutB>));
    dim3 dimBlock(size(TiledMMA{}));

    // printf("num_threads = %d\n", dimBlock.x);

    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(M, BM)), dimCluster.x),
                 round_up(size(ceil_div(N, BN)), dimCluster.y));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
        &gemm_device<T, BM, BN, BK, TiledMMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB>
    );

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                               a, b, c, M, N, K);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
    
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

    const int outer_repeat = 10, inner_repeat = 3;

    printf("\nalgo = wgmma_ss\n");

    const int M = 1024, N = 1024, K = 1024;
    float max_error = testF16F16GemmMaxError<T>(
        gemm_wgmma, M, N, K);
    printf("Max Error = %f\n", max_error);

    #if BENCHMARK
    for (int j = 0; j < test_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance<T>(
                gemm_wgmma, M, N, K, inner_repeat);
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
    #endif
    
    return 0;
}



