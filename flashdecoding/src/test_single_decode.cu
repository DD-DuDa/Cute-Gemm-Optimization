#include "flash_api.h"
#include <cstdio>

template <typename T, typename U>
bool all_close(T *A, U *B, int total_size, float tolerance = 1e-2) {
    for (int i = 0; i < total_size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, (float)B[i]);
            return false;
        }
    }
    return true;
}

torch::Tensor single_mha(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, int head_dim) {
    const float sm_scale = 1.f / std::sqrt(float(head_dim));
    auto scaled_q = q * sm_scale;
    auto scores = torch::einsum("bthd,bshd->bhts", {scaled_q, k});
    auto attention = torch::softmax(scores, -1).to(v.dtype());
    auto output = torch::einsum("bhts,bshd->bthd", {attention, v});
    return output;
}

template <int num_heads, int head_dim>
void TestDecodingKernelCorrectness(int seqlen_kv) {
    const int bs = 1;
    const int seqlen_q = 1;

    torch::manual_seed(42);
    torch::Tensor Q_host = torch::randn({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor K_host = torch::randn({bs, seqlen_kv, num_heads, head_dim}, torch::dtype(torch::kHalf));
    // #pragma unroll
    // for (int i = 0; i < seqlen_kv; ++i) {
    //     for (int j = 0; j < num_heads; ++j) {
    //         for (int m = 0; m < head_dim; ++m) {
    //             // Use .index({indices}) for indexing
    //             K_host.index({0, i, j, m}) = static_cast<torch::Half>(m);
    //         }
    //     }
    // }
    torch::Tensor V_host = torch::randn({bs, seqlen_kv, num_heads, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor Q_device = Q_host.to(torch::kCUDA);
    torch::Tensor K_device = K_host.to(torch::kCUDA);
    torch::Tensor V_device = V_host.to(torch::kCUDA);

    // mha_fwd_kvcache
    const float sm_scale = 1 / std::sqrt(float(head_dim));
    torch::Tensor out = mha_fwd_kvcache(Q_device, K_device, V_device, sm_scale);
    torch::Tensor out_cpu = out.to(torch::kCPU);

    // CPU reference
    torch::Tensor out_ref = single_mha(Q_host, K_host, V_host, head_dim);

    // Compute the difference
    torch::Tensor diff = out_cpu - out_ref;
    float mean_absolute_error = diff.abs().mean().item<float>();
    float mean_squared_error = diff.pow(2).mean().item<float>();

    printf("num_heads: %d seqlen_kv: %d head_dim: %d \n", num_heads, seqlen_kv, head_dim);
    if (mean_absolute_error < 2e-2 && mean_squared_error < 2e-2) {
        printf("test pass ! \n");
    } else {
        printf("test fail ! \n");
    }
        
}

int main() {
    const int num_heads = 32;
    const int head_dim = 128;
    const int seqlen_kv = 1024;
    TestDecodingKernelCorrectness<num_heads, head_dim>(seqlen_kv);
    return 0;
}