#include <iostream>
#include <cuda_fp16.h> // for __half (if using CUDA's half type)
#include "cpu_reference.h"
#include "utils.h"
#include "decode.cuh"

template <typename DTypeQO, typename DTypeKV, int num_heads, int head_dim>
void TestDecodingKernelCorrectness(int seq_len) {
    // shape of Q: (1, 1, num_heads, head_dim)
    // shape of KV: (1, seq_len, num_heads, head_dim)

    std::vector<DTypeQO> Q_host(num_heads * head_dim);
    std::vector<DTypeKV> K_host(seq_len * num_heads * head_dim);
    std::vector<DTypeKV> V_host(seq_len * num_heads * head_dim);
    std::vector<DTypeQO> O_host(num_heads * head_dim);
    std::vector<DTypeQO> O_host_ref;

    utils::vec_normal_(Q_host, 0, 1, 0);
    utils::vec_normal_(K_host, 0, 1, 1);
    utils::vec_normal_(V_host, 0, 1, 2);
    utils::vec_zero_(O_host);
    
    O_host_ref = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(Q_host, K_host, V_host, 1, seq_len, num_heads, head_dim);

    /* CUDA Device */
    thrust::device_vector<DTypeQO> Q(Q_host);
    thrust::device_vector<DTypeKV> K(K_host);
    thrust::device_vector<DTypeKV> V(V_host);
    thrust::device_vector<DTypeQO> O(O_host);

    const float sm_scale =1.f / std::sqrt(float(head_dim));

    mha_fwd_kvcache<DTypeQO, num_heads, head_dim>(thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()), 
                             thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()), 
                             1, seq_len, sm_scale, 0);

    thrust::host_vector<DTypeQO> o_host = O;
    size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
    bool nan_detected = false;
    for (size_t i = 0; i < num_heads * head_dim; ++i) {
        if (isnan(float(o_host[i]))) {
            nan_detected = true;
        }
        num_result_errors_atol_1e_3_rtol_1e_3 +=
            (!utils::isclose(float(o_host[i]), float(O_host_ref[i]), 1e-2, 1e-2));
    }
    float result_accuracy =
        1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(num_heads * head_dim);
    std::cout << "num_qo_heads=" << num_heads << ", num_kv_heads=" << num_heads
                << ", seq_len=" << seq_len << ", head_dim=" << head_dim
                << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << std::endl;

}


int main() {
    TestDecodingKernelCorrectness<half, half, 32, 128>(1024);
    std::cout << "Run finish!" << std::endl;
    return 0;
}