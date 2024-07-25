#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>

using T = cute::half_t;
using namespace cute;

int main() {
    // TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
    //                               Layout<Shape<_4, _1, _1>>{},
    //                               Tile<Int<16 * 4>, _16, _16>{});           // Tiler
    // print_latex(mma);

    const int lse_size = 1 * 32 * 1;
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(1, 32, 1));
    auto transposed_stride = make_stride(1, 1, 1);
    Layout remapped_layout = make_layout(make_shape(1, 32, 1), transposed_stride);
    Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));
    print(final_layout);
    return 0;
}