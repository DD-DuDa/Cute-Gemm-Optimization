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

#include <stdexcept>

#include "utils.h"

namespace cpu_reference {

template <typename dtype_q, typename dtype_kv, typename dtype_out>
std::vector<dtype_out> single_mha(const std::vector<dtype_q>& q, const std::vector<dtype_kv>& k,
                                  const std::vector<dtype_kv>& v, size_t qo_len, size_t kv_len,
                                  size_t num_heads, size_t head_dim) {
    assert(qo_len <= kv_len);
    float sm_scale = 1.f / std::sqrt(float(head_dim));
    std::vector<dtype_out> o(qo_len * num_heads * head_dim);
    std::vector<float> att(kv_len);

    for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
            float max_val = -5e4;
            for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                att[kv_idx] = 0.;
                for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                    att[kv_idx] += float(q[q_idx * num_heads * head_dim + head_idx * head_dim + feat_idx]) *
                                   float(k[kv_idx * num_heads * head_dim + head_idx * head_dim + feat_idx]) *
                                   sm_scale;
                }
                max_val = std::max(max_val, att[kv_idx]);
            }
            // exp minus max
            float denom = 0;
            for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                att[kv_idx] = std::exp(att[kv_idx] - max_val);
                denom += att[kv_idx];
            }

            // divide by denom
            for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                att[kv_idx] /= denom;
            }

            for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                float o_float = 0.;
                for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                    o_float += att[kv_idx] * float(v[kv_idx * num_heads * head_dim + head_idx * head_dim + feat_idx]);
                }
                o[q_idx * num_heads * head_dim + head_idx * head_dim + feat_idx] = dtype_out(o_float);
            }
        }
    }
    return std::move(o);
}

} // namespace cpu_reference