#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>

using T = float;
using namespace cute;

int main() {
    // TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
    //                               Layout<Shape<_1, _4, _1>>{},
    //                               Tile<_16, _64, _16>{});           // Tiler
    // // Tensor acc_s = partition_fragment_C(mma, Shape<Int<64>, Int<128>>{});  // (MMA=4, MMA_M, MMA_N)
    // print_latex(mma);

    // 这是未使用 Swizzle 语义的 smem layout
    using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<32>{}),
                  make_stride(Int<32>{}, Int<1>{}))));
    // using SmemLayoutAtom = decltype(
    //     make_layout(make_shape(Int<8>{}, Int<32>{}),
    //                 make_stride(Int<32>{}, Int<1>{})));
    using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<8>{}, Int<32>{})));

    print_latex(SmemLayoutA{});

    return 0;
}