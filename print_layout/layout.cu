#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>

using T = cute::half_t;
using namespace cute;

int main() {
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                  Layout<Shape<_1, _1, _1>>{});           // Tiler
    print_latex(mma);

    return 0;
}