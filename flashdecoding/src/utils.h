/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <random>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in line %d of file %s: %s\n",       \
                    __LINE__, __FILE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }
  
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
    
namespace utils {

template <typename T>
void vec_normal_(std::vector<T>& vec, float mean = 0.f, float std = 1.f, unsigned int seed = 42) {
  // std::random_device rd{};
  // std::mt19937 gen{rd()};

  std::mt19937 gen{seed};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = T(d(gen));
  }
}

template <typename T>
void vec_uniform_(std::vector<T>& vec, float a = 0.f, float b = 1.f) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution d{a, b};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = T(d(gen));
  }
}

template <typename T>
void vec_zero_(std::vector<T>& vec) {
  std::fill(vec.begin(), vec.end(), T(0));
}

template <typename T>
void vec_fill_(std::vector<T>& vec, T val) {
  std::fill(vec.begin(), vec.end(), val);
}

template <typename T>
void vec_randint_(std::vector<T>& vec, int low, int high) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_int_distribution d{low, high};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = T(d(gen));
  }
}

template <typename T>
size_t vec_bytes(const T& vec) {
  return vec.size() * sizeof(typename T::value_type);
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  return fabs(a - b) <= (atol + rtol * fabs(b));
}

template <typename T>
std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<int32_t>>>
create_shared_prefix_testcase_data(size_t batch_size, size_t shared_prefix_length,
                                   size_t unique_kv_length, size_t qo_append_length,
                                   size_t num_qo_heads, size_t num_kv_heads, size_t head_dim,
                                   size_t page_size) {
  uint32_t num_pages = ((shared_prefix_length + unique_kv_length * batch_size) / page_size);
  std::vector<T> shared_k_h(shared_prefix_length * num_kv_heads * head_dim);
  std::vector<T> shared_v_h(shared_prefix_length * num_kv_heads * head_dim);
  std::vector<T> q_h((batch_size * qo_append_length) * num_qo_heads * head_dim);

  utils::vec_normal_(shared_k_h);
  utils::vec_normal_(shared_v_h);
  utils::vec_normal_(q_h);

  std::vector<int32_t> qo_indptr{0};
  std::vector<int32_t> kv_indptr_combined_h{0};
  std::vector<int32_t> kv_indptr_unique_h{0};
  std::vector<int32_t> kv_last_page_len_combined_h;
  std::vector<int32_t> kv_last_page_len_unique_h;

  for (uint32_t request_id = 0; request_id < batch_size; ++request_id) {
    qo_indptr.push_back(qo_indptr.back() + qo_append_length);
    kv_indptr_combined_h.push_back(kv_indptr_combined_h.back() +
                                   (shared_prefix_length + unique_kv_length) / page_size);
    kv_indptr_unique_h.push_back(kv_indptr_unique_h.back() + unique_kv_length / page_size);
    kv_last_page_len_combined_h.push_back(page_size);
    kv_last_page_len_unique_h.push_back(page_size);
  }

  std::vector<int32_t> kv_indices_combined_h(kv_indptr_combined_h.back());
  std::vector<int32_t> kv_indices_unique_h(kv_indptr_unique_h.back());

  std::vector<T> kv_data_h(num_pages * 2 * num_kv_heads * page_size * head_dim);
  uint32_t page_id = 0;

  for (; page_id < (shared_prefix_length / page_size); page_id++) {
    for (uint32_t entry_idx = 0; entry_idx < page_size; entry_idx++) {
      for (uint32_t head_idx = 0; head_idx < num_kv_heads; head_idx++) {
        std::copy(
            shared_k_h.begin() +
                ((page_id * page_size + entry_idx) * num_kv_heads + head_idx) * head_dim,
            shared_k_h.begin() +
                ((page_id * page_size + entry_idx) * num_kv_heads + head_idx + 1) * head_dim,
            kv_data_h.begin() +
                (((page_id * 2 + 0) * num_kv_heads + head_idx) * page_size + entry_idx) * head_dim);
        std::copy(
            shared_v_h.begin() +
                ((page_id * page_size + entry_idx) * num_kv_heads + head_idx) * head_dim,
            shared_v_h.begin() +
                ((page_id * page_size + entry_idx) * num_kv_heads + head_idx + 1) * head_dim,
            kv_data_h.begin() +
                (((page_id * 2 + 1) * num_kv_heads + head_idx) * page_size + entry_idx) * head_dim);
      }
    }
    for (uint32_t request_id = 0; request_id < batch_size; ++request_id) {
      kv_indices_combined_h[request_id * ((shared_prefix_length + unique_kv_length) / page_size) +
                            page_id] = page_id;
    }
  }

  for (uint32_t request_id = 0; request_id < batch_size; ++request_id) {
    for (uint32_t page_iter = 0; page_iter < (unique_kv_length / page_size);
         ++page_iter, ++page_id) {
      for (uint32_t entry_idx = 0; entry_idx < page_size; entry_idx++) {
        for (uint32_t head_idx = 0; head_idx < num_kv_heads; head_idx++) {
          std::vector<T> k(head_dim), v(head_dim);
          utils::vec_normal_(k);
          utils::vec_normal_(v);
          std::copy(k.begin(), k.end(),
                    kv_data_h.begin() +
                        (((page_id * 2 + 0) * num_kv_heads + head_idx) * page_size + entry_idx) *
                            head_dim);
          std::copy(v.begin(), v.end(),
                    kv_data_h.begin() +
                        (((page_id * 2 + 1) * num_kv_heads + head_idx) * page_size + entry_idx) *
                            head_dim);
        }
      }
      kv_indices_combined_h[request_id * ((shared_prefix_length + unique_kv_length) / page_size) +
                            (shared_prefix_length / page_size) + page_iter] = page_id;
      kv_indices_unique_h[request_id * (unique_kv_length / page_size) + page_iter] = page_id;
    }
  }
  return std::make_tuple<std::vector<std::vector<T>>, std::vector<std::vector<int32_t>>>(
      {std::move(q_h), std::move(shared_k_h), std::move(shared_v_h), std::move(kv_data_h)},
      {std::move(qo_indptr), std::move(kv_indices_combined_h), std::move(kv_indices_unique_h),
       std::move(kv_indptr_combined_h), std::move(kv_indptr_unique_h),
       std::move(kv_last_page_len_combined_h), std::move(kv_last_page_len_unique_h)});
}

}  // namespace utils


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

using namespace cute;
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n" 
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// TODO:
// https://github.com/NVIDIA/cutlass/issues/802
// TODO: convert出来后数据是否在寄存器?
template <typename Fragment>
inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32) {
  Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
  {
    Tensor acc_fp32x2 = recast< float2>(acc_fp32);
    Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
    for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
  }
  return acc_fp16;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = flash::convert_type<To_type>(tensor);
    flash::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash