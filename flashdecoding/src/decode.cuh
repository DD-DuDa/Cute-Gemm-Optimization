#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "flash.h"
#include "kernel_traits.h"

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

template<typename Kernel_traits>
void run_mha_fwd_splitkv(Flash_fwd_params &params, cudaStream_t stream) {
    printf("run_mha_fwd_splitkv\n");
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