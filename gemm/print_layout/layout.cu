#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>

using T = cute::half_t;
using namespace cute;

int main() {
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                  Layout<Shape<_4, _1, _1>>{},
                                  Tile<Int<16 * 4>, _16, _16>{});           // Tiler
    print_latex(mma);

    return 0;
}