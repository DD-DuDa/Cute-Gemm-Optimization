#include "flash_api.h"
#include <cstdio>
#include <optional>

torch::Tensor single_mha(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, int head_dim) {
    const float sm_scale = 1.f / std::sqrt(float(head_dim));
    auto scaled_q = q * sm_scale;
    auto scores = torch::einsum("bthd,bshd->bhts", {scaled_q, k});
    auto attention = torch::softmax(scores, -1).to(v.dtype());
    auto output = torch::einsum("bhts,bshd->bthd", {attention, v});
    return output;
}

template <int num_heads, int head_dim, bool new_kv>
void TestDecodingKernelCorrectness(int seqlen_kv) {
    const int bs = 1;
    const int seqlen_q = 1;

    torch::manual_seed(42);
    torch::Tensor Q_host = torch::randn({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor K_host = torch::rand({bs, seqlen_kv, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor V_host = torch::rand({bs, seqlen_kv, num_heads, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor Q_device = Q_host.to(torch::kCUDA);
    torch::Tensor K_device = K_host.to(torch::kCUDA);
    torch::Tensor V_device = V_host.to(torch::kCUDA);

    at::Tensor K_new_host, V_new_host, K_new_device, V_new_device, seqlens_k;

    if (new_kv) {
        auto seqlen_new = seqlen_q;
        seqlens_k = torch::full({bs}, seqlen_kv, torch::dtype(torch::kInt32).device(torch::kCUDA));
        K_new_host = torch::rand({bs, seqlen_new, num_heads, head_dim}, torch::dtype(torch::kHalf));
        V_new_host = torch::rand({bs, seqlen_new, num_heads, head_dim}, torch::dtype(torch::kHalf));
        K_new_device = K_new_host.to(torch::kCUDA);
        V_new_device = V_new_host.to(torch::kCUDA);
    }

    // mha_fwd_kvcache
    const float sm_scale = 1 / std::sqrt(float(head_dim));
    std::optional<const at::Tensor> opt_K_new_device = new_kv ? std::make_optional(K_new_device) : std::nullopt;
    std::optional<const at::Tensor> opt_V_new_device = new_kv ? std::make_optional(V_new_device) : std::nullopt;
    std::optional<const at::Tensor> opt_seqlens_k = std::make_optional(seqlens_k);

    torch::Tensor out = mha_fwd_kvcache(Q_device, K_device, V_device, opt_K_new_device, opt_V_new_device, opt_seqlens_k, sm_scale);
    torch::Tensor out_cpu = out.to(torch::kCPU);

    // CPU reference
    if (new_kv) {
        K_host = torch::cat({K_host, K_new_host}, 1);
        V_host = torch::cat({V_host, V_new_host}, 1);
    }
    torch::Tensor out_ref = single_mha(Q_host, K_host, V_host, head_dim);

    // Compute the difference
    torch::Tensor diff = out_cpu - out_ref;
    float mean_absolute_error = diff.abs().mean().item<float>();
    float mean_squared_error = diff.pow(2).mean().item<float>();

    printf("num_heads: %d seqlen_kv: %d head_dim: %d \n", num_heads, seqlen_kv, head_dim);
    if (mean_absolute_error < 2e-3 && mean_squared_error < 2e-3) {
        printf("test pass ! \n");
    } else {
        printf("test fail ! \n");
        printf("mean_absolute_error: %.6f mean_squared_error: %.6f\n", mean_absolute_error, mean_squared_error);
    }

    printf("\nFirst 10 elements of out_cpu:\n");
    auto out_cpu_accessor = out_cpu.flatten().data_ptr<at::Half>();
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", static_cast<float>(out_cpu_accessor[i]));
    }
    
    printf("\n\nFirst 10 elements of out_ref:\n"); 
    auto out_ref_accessor = out_ref.flatten().data_ptr<at::Half>();
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", static_cast<float>(out_ref_accessor[i]));
    }
    printf("\n");
        
}

int main() {
    const int num_heads = 32;
    const int head_dim = 128;
    const int seqlen_kv = 1024;
    const bool new_kv = true;
    TestDecodingKernelCorrectness<num_heads, head_dim, new_kv>(seqlen_kv);
    return 0;
}