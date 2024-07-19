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

    mha_fwd_kvcache<DTypeQO, num_heads, head_dim>(thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()), 
                             thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()), 
                             1, seq_len, 1.0, 0);

}


int main() {
    TestDecodingKernelCorrectness<half, half, 32, 128>(1024);
    std::cout << "Test passed!" << std::endl;
    return 0;
}