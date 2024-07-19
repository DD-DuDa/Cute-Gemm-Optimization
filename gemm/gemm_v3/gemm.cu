#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
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

template <typename T, int BM, int BN, int BK, typename TiledMMA, 
            typename G2SCopyA, typename G2SCopyB,
            typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,
            typename S2RCopyAtomA, typename S2RCopyAtomB,
            typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void gemm_shm_v5(const T *Aptr, const T *Bptr, T *Dptr, int m, int n, int k) {
    // Initilize shared memory
    extern __shared__ T shm_data[];

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    // Initilize thread block
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));

    // Global Memory
    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _)); // (BM, BK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _)); // (BN, BK, num_tile_k)
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix)); // (BM, BN) 

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm),
                            SmemLayoutA{}); // (BM, BK)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (BN, BK)

    // register, use tiled_mma to partition register A/B/C
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)
    clear(tCrD);

    // from global memory to shared memory
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

    // from shared memory to register, use tiled_mma to generate tiled_copy
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)


    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     PRINT("tCrA", tCrA.shape())    
    //     PRINT("tCrB", tCrB.shape())   

    //     PRINT("tAgA_copy", tAgA_copy.shape())     
    //     PRINT("tAsA_copy", tAsA_copy.shape())
    //     // print(layout<0>(tAgA));
    //     // PRINT("tArA", tArA.shape()) 
    //     PRINT("tBgB_copy", tBgB_copy.shape())     
    //     PRINT("tBsB_copy", tBsB_copy.shape())

    //     PRINT("tAsA", tAsA.shape())     
    //     PRINT("tCrA_view", tCrA_view.shape()) 
    //     // print(layout<0>(tBgB));
    //     // PRINT("tBrB", tBrB.shape()) 

    //     PRINT("tBsB", tBsB.shape())     
    //     PRINT("tCrB_view", tCrB_view.shape()) 
    // }

    // loop over k: i. load tile, ii. mma
    int ntile = k / BK;
    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile)
    {
        // copy  (CPY, CPY_M, CPY_K) , async
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile),
                tAsA_copy(_, _, _));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile),
                tBsB_copy(_, _, _));
        cp_async_fence();

        cp_async_wait<0>();
        __syncthreads();

        int nk = size<2>(tCrA);
    #pragma unroll
        for (int ik = 0; ik < nk; ++ik)
        {
            // copy  (CPY, CPY_M), sync
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik),
                        tCrA_view(_, _, ik));
            // copy  (CPY, CPY_N)
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik),
                        tCrB_view(_, _, ik));
            // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        } // for ik
    } // itile

  
    // use less shared memory as a scratchpad tile to use large wide instuction
    // Dreg -> shm -> reg -> global
    auto sC = make_tensor(sA(_, _).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N) ((_2,(_2,_2)),_4,_8)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe) ((_2,(_2,_2)),_1,_1,_2)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe) ((_2,(_2,_2)),_1,_1,_2)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N) ((_8,_1),_4,_8)

    
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN) ((_2,(_2,_2)),(_4,_8))
    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN) ((_8,_1),(_4,_8))

    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     PRINT("tCrC_r2s", tCrC_r2s.shape()) 
    //     PRINT("tCsC_r2s", tCsC_r2s.shape()) 

    //     // PRINT("tCsC_s2g", tCsC_s2g.shape()) 
    //     // PRINT("tCgC_s2g", tCgC_s2g.shape()) 

    //     // PRINT("tCgC_s2gx", tCgC_s2gx.shape()) 
    //     PRINT("tCrC_r2sx", tCrC_r2sx.shape()) 
    //     PRINT("size<1>(tCrC_r2sx)", size<1>(tCrC_r2sx))
    // }

    int step = size<3>(tCsC_r2s);  // pipe
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        // reg -> shm
#pragma unroll
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
            cute::copy(tCrC_r2sx(_, i + j), t);

            cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

    #pragma unroll
        // shm -> global
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }

        __syncthreads();
    }
}

template <typename T>
void gemm_v3(T *a, T *b, T *c, int M, int N, int K) {
    auto BM = Int<128>{};
    auto BN = Int<256>{};
    auto BK = Int< 32>{};
    auto kSmemLayoutCBatch = Int<2>{};
    // Define the smem layouts
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<BM>{}, Int<BK>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<BN>{}, Int<BK>{})));                    // (m,n) -> smem_idx

    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 2 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

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

    // copy from shared memory to register
    // use mma tiled ,so no tiled here
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // epilogue: register to global via shared memory
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                        make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{}))));

    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY);

    // C_shm is shared with A_shm and B_shm
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize =
        cute::max(shm_size_AB, shm_size_C) * sizeof(T);

    int shm_size = kShmSize;

    cudaFuncSetAttribute(gemm_shm_v5<T, BM, BN, BK, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    
    gemm_shm_v5<T, BM, BN, BK, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>
               <<<grid, block, shm_size>>>(a, b, c, M, N, K);
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

    printf("\nalgo = Cute_HGEMM_V2\n");

    const int M = 256, N = 256, K = 256;
    float max_error = testF16F16GemmMaxError<T>(
        gemm_v3, M, N, K);
    printf("Max Error = %f\n", max_error);

    // double this_sec = testF16F16GemmPerformance<T>(
    //     gemm_v2, 8192, 8192, 8192, inner_repeat);
    for (int j = 0; j < test_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance<T>(
                gemm_v3, M, N, K, inner_repeat);
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

