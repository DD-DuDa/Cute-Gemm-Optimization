#pragma once

#include <cuda.h>
#include <vector>

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;

    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    
    // The number of heads.
    int h;
};

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The dimensions.
    int b;
    int seqlen_q;
    int seqlen_k;
    int d;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // For split-KV version
    int num_splits;  
};
