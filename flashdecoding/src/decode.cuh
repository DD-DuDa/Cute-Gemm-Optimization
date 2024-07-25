#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <sys/cdefs.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "flash.h"
#include "kernel_traits.h"
#include "utils.h"
#include "block_info.h"
#include "softmax.h"
#include "mask.h"
#include "static_switch.h"

using namespace cute;
using namespace flash;

// Error checking macro
#define CUDA_CHECK_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(err); \
    } \
}

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
        std::vector<float> out_accum_host(params.num_splits * batch_size * num_heads * max_seqlen_q * head_size);
        params.out_accum = thrust::device_vector<float>(out_accum_host);
        params.oaccum_ptr = thrust::raw_pointer_cast(params.out_accum.data());

        std::vector<float> softmax_lse_accum_host(params.num_splits * batch_size * num_heads * max_seqlen_q);
        params.softmax_lse_accum = thrust::device_vector<float>(softmax_lse_accum_host);
        params.softmax_lseaccum_ptr = thrust::raw_pointer_cast(params.softmax_lse_accum.data());
    }
}

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params, typename final_layout>
__global__ void flash_fwd_splitkv_combine_kernel(const Params params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t lse_size = params.b * params.h * params.seqlen_q;

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(lse_size, _1{}));

    // LSE format is different depending on params.unpadded_lse and params.seqlenq_ngroups_swapped, see comment in get_lse_tile.
    // This tensor's layout maps row_offset_lse to {bidb, bidh, q_offset}.
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // This layout maps row_offset_lse to {bidh, q_offset, bidb} or {bidh, bidb, q_offset}.
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
    auto transposed_stride = make_stride(1, params.seqlen_q * params.b, params.seqlen_q);
    Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
    // Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));
    Tensor gLSE_unpadded = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)), final_layout{});

    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then transpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = cute::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM && tidx / kRowsPerLoadTranspose < 32) {
        gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
    }
    // // Store the scales exp(lse - lse_logsum) in shared memory.
    // #pragma unroll
    // for (int l = 0; l < kNLsePerThread; ++l) {
    //     const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
    //     const int col = tidx / kRowsPerLoadTranspose;
    //     if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    // }
    // __syncthreads();

    // const index_t row_offset_oaccum = bidx * kBlockM * params.d;
    // Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
    //                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                              Stride<Int<kHeadDim>, _1>{});
    // constexpr int kBlockN = kNThreads / kBlockM;
    // using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    // using GmemTiledCopyOaccum = decltype(
    //     make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
    //                     GmemLayoutAtomOaccum{},
    //                     Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    // GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    // auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    // Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    // Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    // Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    // clear(tOrO);

    // // Predicates
    // Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // // Repeat the partitioning with identity layouts
    // Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    // Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // if (!Is_even_K) {
    //     #pragma unroll
    //     for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    // }
    // // Load Oaccum in then scale and accumulate to O
    // for (int split = 0; split < params.num_splits; ++split) {
    //     flash::copy</*Is_even_MN=*/false, Is_even_K>(
    //         gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
    //     );
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(tOrOaccum); ++m) {
    //         int row = get<0>(tOcOaccum(0, m, 0));
    //         ElementAccum lse_scale = sLSE[split][row];
    //         #pragma unroll
    //         for (int k = 0; k < size<2>(tOrOaccum); ++k) {
    //             #pragma unroll
    //             for (int i = 0; i < size<0>(tOrOaccum); ++i) {
    //                 tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
    //             }
    //         }
    //     // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); }
    //     }
    //     tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d;
    // }
    // // if (cute::thread0()) { print_tensor(tOrO); }

    // Tensor rO = flash::convert_type<Element>(tOrO);
    // // Write to gO
    // #pragma unroll
    // for (int m = 0; m < size<1>(rO); ++m) {
    //     const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
    //     if (idx < params.b * params.h * params.seqlen_q) {
    //         const int batch_idx = idx / (params.h * params.seqlen_q);
    //         const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
    //         // The index to the rows of Q
    //         const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
    //         auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
    //             + head_idx * params.o_head_stride + row * params.o_row_stride;
    //         #pragma unroll
    //         for (int k = 0; k < size<2>(rO); ++k) {
    //             if (Is_even_K || tOpOaccum(k)) {
    //                 const int col = get<1>(tOcOaccum(0, m, k));
    //                 Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
    //                                         Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    //                 // TODO: Should check if this is using vectorized store, but it seems pretty fast
    //                 copy(rO(_, m, k), gO);
    //                 // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
    //                 // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
    //             }
    //         }
    //     }
    // }
}


/* Kernel */
template <typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Split, typename Params>
__global__ void flash_fwd_splitkv_kernel(const Params params) {
    const int m_block = blockIdx.x;
    // Split-K Block
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;

    // get datatype of element and index (int64_t) 
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
    constexpr int kNWarps = Kernel_traits::kNWarps; // 4

    // the memory transition of tensor O 
    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = n_split_idx * n_blocks_per_split;
    int n_block_max = cute::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);

    const int bidb_cache = bidb;
    const int *block_table = nullptr;
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + bidh * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + bidh * params.v_head_stride;

    // Global memory tensors
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    // Shared memory tensors
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // Copy, global memory to shared memory
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // Register memory tensors
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    // Copy, shared memory to register memory
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

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("k_row_stride: %d \n", params.k_row_stride);
        printf("scale_softmax_log2: %f \n", params.scale_softmax_log2);
        printf("num_splits: %d \n", params.num_splits);
        PRINT("gQ", gQ.shape())     
        PRINT("gK", gK.shape())
        PRINT("gV", gV.shape())
        PRINT("sQ", sQ.shape())
        PRINT("sK", sK.shape())
        PRINT("sV", sV.shape())
        PRINT("tQgQ", tQgQ.shape())
        PRINT("tQsQ", tQsQ.shape())
        PRINT("tKgK", tKgK.shape())
        PRINT("tKsK", tKsK.shape())
        PRINT("tVgV", tVgV.shape())
        PRINT("tVsV", tVsV.shape())
        PRINT("acc_o", acc_o.shape())
    }

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue
    // Copy TensorQ, global memory to shared memory
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);

    // Copy TensorK, global memory to shared memory
    int n_block = n_block_max - 1;
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    clear(acc_o);
 
    flash::Softmax<2 * size<1>(acc_o)> softmax;
    const float alibi_slope = 0.0f;
    flash::Mask<false, false, false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, -1, -1, alibi_slope);

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.

    constexpr int n_masking_steps = 1;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } 
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence();

        flash::gemm(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        mask.template apply_mask<false, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } 
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // We have key_padding_mask so we'll need to Check_inf
        // TODO
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/!Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/!Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16 
        Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();
        // Advance gV
        if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } 
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } 
            // TODO
            flash::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );
        // TODO
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);

        // TODO
        Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
    // if (cute::thread0()) { print(lse); }

    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    // TODO
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;

    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    if constexpr (Split) { __syncthreads(); }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d;
    const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});


    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    
}


/* Launch Kernel */
template<typename Kernel_traits>
void run_mha_fwd_splitkv(Flash_fwd_params &params, cudaStream_t stream) {
    printf("### run_mha_fwd_splitkv \n");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    dim3 block(size(Kernel_traits::kNThreads));

    const bool is_even_MN = params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, IsEvenMNConst, IsEvenKConst, Split, Flash_fwd_params>;
                // when shared memory is large, we need this
                if (smem_size >= 48 * 1024) {
                    CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                }
            });
        });
    });

    const int lse_size = params.b * params.h * params.seqlen_q;
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
    auto transposed_stride = make_stride(1, params.seqlen_q * params.b, params.seqlen_q);
    Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
    Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));
    using Final_layout = decltype(final_layout);


    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst, Flash_fwd_params, Final_layout><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            // if (params.num_splits <= 2) {
            //     flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst, Flash_fwd_params><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            // } else if (params.num_splits <= 4) {
            //     flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst, Flash_fwd_params><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);}
            // } else if (params.num_splits <= 8) {
            //     flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst, Flash_fwd_params><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            // } else if (params.num_splits <= 16) {
            //     flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst, Flash_fwd_params><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            // } else if (params.num_splits <= 32) {
            //     flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst, Flash_fwd_params><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            // } else if (params.num_splits <= 64) {
            //     flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst, Flash_fwd_params><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            // } else if (params.num_splits <= 128) {
            //     flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst, Flash_fwd_params><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            // }
            CUDA_CHECK_ERROR();
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
        });
    }

    std::vector<cutlass::half_t> Q_host(params.h * params.d);
    cudaMemcpy(thrust::raw_pointer_cast(Q_host.data()), params.q_ptr, params.h * params.d * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);
    std::cout << "Q_host:" << std::endl;
    for (int i = 0; i < 128; ++i) {
        std::cout << Q_host[i] << " ";
    }
    std::cout << std::endl;

    size_t output_size = params.h * params.d; // Adjust this based on actual size
    thrust::host_vector<float> o_host(output_size);
    // Copy data from device to host
    cudaMemcpy(thrust::raw_pointer_cast(o_host.data()), params.oaccum_ptr, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "O_host_accum" << std::endl;
    for (int i = 0; i < 128; ++i) {
        std::cout << o_host[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * shape of QO: (batch_size, 1, num_heads, head_dim)
 * shape of KV: (batch_size, seq_len, num_heads, head_dim)
 */
template <typename T, int num_heads, int head_dim>
int mha_fwd_kvcache(T *q, T *kcache, T *vcache, T *out,
                    const int batch_size, const int seq_len,
                    const float softmax_scale,
                    int num_splits
                    ) {
    auto seq_len_q = 1;
    auto seq_len_kv = seq_len;

    auto bs_stride = seq_len_kv * num_heads * head_dim;
    auto head_stride = head_dim;
    auto seq_len_stride = num_heads * head_dim;
    auto dim_stride = head_dim;

    Flash_fwd_params params;
    std::vector<float> softmax_lse_host(batch_size * num_heads * seq_len_q);
    params.softmax_lse = thrust::device_vector<float>(softmax_lse_host);
    params.softmax_lse_ptr = thrust::raw_pointer_cast(params.softmax_lse.data());
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