#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>

using T = float;
using namespace cute;

int main() {
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                  Layout<Shape<_4, _1, _1>>{},
                                  Tile<Int<16 * 4>, _16, _16>{});           // Tiler
    // print_latex(mma);

    // auto tSsK_shape = make_shape(_1{},_8{},_8{});
    // print_latex(tSsK_shape);

    // constexpr int tpb = 128;
    // auto thr_layout = make_layout(make_shape(Int<tpb>{}));
    // print_latex(thr_layout);

    using g2r_copy_atom = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    auto G2RCopyB = make_tiled_copy(g2r_copy_atom{},
                                 make_layout(make_shape(Int<4>{}, Int<32>{}), // Thr layout 32x4 k-major
                                            make_stride(Int<32>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<4>{}))); // Val layout 1x8

    print_latex(G2RCopyB);

    return 0;
}