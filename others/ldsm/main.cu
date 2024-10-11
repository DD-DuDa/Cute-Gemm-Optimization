#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

using namespace cute;

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

#define PRINTTENSOR(name, content) \
    print(name);                   \
    print(" : ");                  \
    print_tensor(content);         \
    print("\n");
    


template<class T, class SmemLayoutA, class G2SCopyA, int m,int k>
__global__ void gemm_device(const T* Aptr) {
    
    int tidx = threadIdx.x + blockDim.x * threadIdx.y;
    int sx = tidx % 16;
    int sy = tidx / 16 * 8;

    Tensor gA = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<m>{}, Int<k>{}), make_stride(Int<k>{}, Int<1>{}));

    // Shared memory buffers
    __shared__ T smemA[size(SmemLayoutA{})];
    Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});        // (BLK_M,BLK_K)


    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
    const auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY_M, CPY_K)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY_M, CPY_K)

    cute::copy(G2SCopyA{}, tAgA_copy, tAsA_copy);

    __syncthreads();

    // Got the shared memory data sA
    
    uint32_t my_register[4];
    uint32_t smem = __cvta_generic_to_shared(&sA(sx, sy));
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
    : "=r"(my_register[0]), "=r"(my_register[1]), "=r"(my_register[2]), "=r"(my_register[3]) 
    : "r"(smem)
    );


    // __syncthreads();

    if (tidx == 0) {

        PRINT("gA", gA.layout());
        PRINT("sA", sA.layout());
        PRINT("tAgA_copy", tAgA_copy.layout());
        PRINT("tAsA_copy", tAsA_copy.layout());
    
        printf("\n\n#### sA ####");
        for (int i = 0; i < sA.size(); i++) {
            if(i % k == 0)
                printf("\n\n");
            printf("%f ", __half2float(sA.data()[i]));
        }
        printf("\n");

        printf("\n#### Register for thread 0 \n");
        for (int i = 0; i < 4; i++) {
            half * tmp = (half*)(&(my_register[i]));
            printf("%f\n", (float)(tmp[0]));
            printf("%f\n", (float)(tmp[1]));
        }
    }
}

int main(int argc, char** argv) {

    using namespace cute;
    using T = half;
    
    const int M = 16;
    const int K = 16;
    thrust::host_vector<T> h_A(M*K);

    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<T>(i);
    }

    thrust::device_vector<T> d_A = h_A;
    const T* Aptr = thrust::raw_pointer_cast(d_A.data());

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    static constexpr int kMmaVRepeatM = 1;
    static constexpr int kMmaVRepeatN = 1;
    static constexpr int kMmaVRepeatK = 1;

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_V_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaVRepeatM>{}, Int<kMmaVRepeatN>{}, Int<kMmaVRepeatK>{})));
    using TiledMMA =
        decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{}));

    using g2s_copy_op = UniversalCopy<cute::uint64_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                make_layout(make_shape(Int<8>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<4>{}))));

    // shared memory to register copy

    // 使用 Swizzle 语义的 smem layout
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<16>{}),
                    make_stride(Int<16>{}, Int<1>{}))));
    // using SmemLayoutAtom = decltype(
    //   make_layout(make_shape(Int<16>{}, Int<16>{}),
    //               make_stride(Int<16>{}, Int<1>{})));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<16>{}, Int<16>{})));

    dim3 gridDim(1);
    dim3 blockDim(size(TiledMMA{}));

    print(size(TiledMMA{})); printf("\n");
    gemm_device<T, SmemLayoutA, G2SCopyA, M, K>
              <<<gridDim, blockDim>>>(Aptr);
    cudaDeviceSynchronize();

    return 0;
}