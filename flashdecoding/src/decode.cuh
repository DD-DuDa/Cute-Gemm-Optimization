#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "flash.h"
#include "kernel_traits.h"
#include "utils.h"
#include "block_info.h"
#include "softmax.h"

namespace flash {
using namespace cute;

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
        typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
        typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

} // namespace flash

/*Set params*/
template <typename T>
void set_params_fprop(Flash_fwd_params &params,
                        const int batch_size, 
                        const int seq_len_q, const int seq_len_kv, 
                        const int num_heads, const int head_dim, 
                        const int bs_stride, const int head_stride, const int seq_len_stride, const int dim_stride,
                        T *q, T *k, T *v, T *out, const float softmax_scale) {
    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.q_ptr = q;
    params.o_ptr = out;
    params.k_ptr = k;
    params.v_ptr = v;
    
    // All stride are in elements, not bytes.
    params.q_row_stride = seq_len_stride;
    params.o_row_stride = seq_len_stride;
    params.k_row_stride = seq_len_stride;
    params.v_row_stride = seq_len_stride;

    params.q_head_stride = head_stride;
    params.o_head_stride = head_stride;
    params.k_head_stride = head_stride;
    params.v_head_stride = head_stride;

    params.q_batch_stride = seq_len_q * seq_len_stride;
    params.o_batch_stride = seq_len_q * seq_len_stride;
    params.k_batch_stride = bs_stride;
    params.v_batch_stride = bs_stride;

    // Set the dimensions.
    params.b = batch_size;
    params.h = num_heads;
    params.seqlen_q = seq_len_q;
    params.seqlen_k = seq_len_kv;
    params.d = head_dim;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
}


/*
Find the number of splits that maximizes the occupancy. For example, if we have
batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
better than having 3 splits (efficiency = 0.67). However, we also don't want too many
splits as that would incur more HBM reads/writes.
So we find the best efficiency, then find the smallest number of splits that gets 85%
of the best efficiency.
*/
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min(max_splits, std::min(num_SMs, num_n_blocks));
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

/*Set split-k params*/
template <typename T>
void set_params_splitkv(Flash_fwd_params &params, int batch_size,
                        int num_heads, int head_size, int max_seqlen_k, int max_seqlen_q,
                        int num_splits, cudaDeviceProp *dprops) {

    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64); 
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    if (num_splits < 1) {
        // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount * 2, num_n_blocks, 128);
    } 
    if (params.num_splits > 1) {
        std::vector<T> out_accum_host(num_splits * batch_size * num_heads * max_seqlen_q * head_size);
        thrust::device_vector<T> out_accum(out_accum_host);
        params.oaccum_ptr = thrust::raw_pointer_cast(out_accum.data());
    }
}

/* Kernel */
template <typename Kernel_traits, typename Params>
__global__ void flash_fwd_splitkv_kernel(const Params params, bool Is_even_MN = true, bool Split = true) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("### flash_fwd_splitkv_kernel \n");
    }
    
    // Blcok
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyO = typename Kernel_traits::GmemTiledCopyOaccum;
    using ElementO = ElementAccum;

    // TODO: const Is_even_MN
    const BlockInfo</*Varlen=*/false> binfo(params, bidb);
    // if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = n_split_idx * n_blocks_per_split;
    int n_block_max = cute::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = bidb;
    const int *block_table = nullptr;
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + bidh * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + bidh * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // TODO Is_even_MN, Is_even_K
    flash::copy<true, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // TODO Is_even_MN, Is_even_K
    flash::copy<true, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;
}

/* Launch Kernel */
template<typename Kernel_traits>
void run_mha_fwd_splitkv(Flash_fwd_params &params, cudaStream_t stream) {
    printf("### run_mha_fwd_splitkv \n");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    bool Split = params.num_splits > 1;
    bool is_even_MN = params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    bool is_even_K = params.d == Kernel_traits::kHeadDim;
    
    auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Flash_fwd_params>;
    if (smem_size >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    dim3 block(size(Kernel_traits::kNThreads));
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, is_even_MN, Split);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * shape of QO: (batch_size, 1, num_heads, head_dim)
 * shape of KV: (batch_size, seq_len, num_heads, head_dim)
 */
template <typename T, int num_heads, int head_dim>
int mha_fwd_kvcache(T *q, T *kcache, T *vcache, T *out,
                    int batch_size, int seq_len,
                    float softmax_scale,
                    int num_splits
                    ) {
    auto seq_len_q = 1;
    auto seq_len_kv = seq_len;

    auto bs_stride = seq_len_kv * num_heads * head_dim;
    auto head_stride = head_dim;
    auto seq_len_stride = num_heads * head_dim;
    auto dim_stride = head_dim;

    Flash_fwd_params params;
    set_params_fprop<T>(params,
                     batch_size, 
                     seq_len_q, seq_len_kv, 
                     num_heads, head_dim, 
                     bs_stride, head_stride, seq_len_stride, dim_stride,
                     q, kcache, vcache, out, softmax_scale);

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get current CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    set_params_splitkv<T>(params, batch_size, num_heads,
                       head_dim, seq_len_kv, seq_len_q, num_splits, &deviceProp);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = head_dim <= 64 ? 256 : (head_dim <= 128 ? 128 : 64);
    run_mha_fwd_splitkv<Flash_fwd_kernel_traits<head_dim, kBlockM, kBlockN, 4, T>>(params, stream);

    return 0;
}