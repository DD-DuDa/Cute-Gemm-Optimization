#include "flash_api.h"
#include <cstdio>

template <int num_heads, int head_dim>
double TestDecodingKernelPerformance(int seqlen_kv, int repeat) {
    const int bs = 1;
    const int seqlen_q = 1;

    torch::Tensor Q_host = torch::randn({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor K_host = torch::randn({bs, seqlen_kv, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor V_host = torch::randn({bs, seqlen_kv, num_heads, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor Q_device = Q_host.to(torch::kCUDA);
    torch::Tensor K_device = K_host.to(torch::kCUDA);
    torch::Tensor V_device = V_host.to(torch::kCUDA);

    // Warm up
    for (int i = 0; i < 5; ++i)
        mha_fwd_kvcache(Q_device, K_device, V_device);

    // Benchmark
    float sm_scale;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        sm_scale = 1 / std::sqrt(float(head_dim));
        torch::Tensor out = mha_fwd_kvcache(Q_device, K_device, V_device, sm_scale);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    msec = msec / repeat;

    return msec;
        
}

int main() {
    const int num_heads = 32;
    const int head_dim = 128;

    const int test_num = 10;
    int len_list[test_num];
    len_list[0] = 1024;
    for (int i = 1; i < test_num; i++) {
        len_list[i] = len_list[i - 1] * 2;
    }

    const int outer_repeat = 10, inner_repeat = 1;
    printf("\n######## Benchmark single decode ########\n");
    for (int j = 0; j < test_num; j++) {
        int seqlen_kv = len_list[j];
        double max_msec = 0.0;
        double min_msec = DBL_MAX;
        double total_msec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = TestDecodingKernelPerformance<num_heads, head_dim>(seqlen_kv, inner_repeat);
            max_msec = max(max_msec, this_sec);
            min_msec = min(min_msec, this_sec);
            total_msec += this_sec;
        }

        double avg_msec = total_msec / outer_repeat;
        printf("seqlen_kv num_heads head_dim = %6d %6d %6d, ", seqlen_kv, num_heads, head_dim);
        printf("Time = %12.8lf %12.8lf %12.8lf ms, \n", min_msec, avg_msec, max_msec);
    }

    return 0;
}