#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

#include "helper.h"
#include "unfused_weight_dequantize.hpp"

using namespace cute;


// This is just an example, so we use a regular enum so we can compare directly to the command-line int.
enum GemmMode {
  ConvertOnly,
  ScaleOnly,
  ScaleWithZeroPoint
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
using MmaType = cutlass::half_t;
using QuantType = cutlass::int8_t;

// A matrix configuration
using         ElementA    = MmaType;                                        // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand

// B matrix configuration
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using ElementZero = cutlass::half_t;
using ElementScale = cutlass::half_t;
using LayoutScale = cutlass::layout::RowMajor;

// =================================================================================================================================================================

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;


//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
uint64_t seed;

// Scale and Zero share a stride since the layout and shapes must be the same.
using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
using StrideS = StrideScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;
StrideS stride_S;
StrideS_ref stride_S_ref;

cutlass::HostTensor<MmaType, LayoutA> tensor_A;
cutlass::HostTensor<QuantType, LayoutB> tensor_B;
cutlass::HostTensor<MmaType, LayoutB> tensor_B_dq;
cutlass::HostTensor<ElementScale, LayoutScale> tensor_scale;
cutlass::HostTensor<ElementZero, LayoutScale> tensor_zero;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 1000;
  int mode = 2;
  int m = 128, n = 1024, k = 4096;
  int g = 128;
  int l = 1;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("g", g);
    cmd.get_cmd_line_argument("mode", mode);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "55_hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   The number of independent gemm problems with mnk shape\n"
      << "  --g=<int>                   The size of each group for the scales and zeros. To broadcast a vector of scales or zeros, set the group size to K.\n"
      << "  --mode=<int>                The mode to run the gemm. 0 does (A @ B), 1 means A @ (scale * B), 2 means A @ (scale * B + zero-point).\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "55_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 -g 0 --l=10 --alpha=2 --mode=2 --beta=0.707 \n\n";

    return out;
  }

};



/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element, class Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed=2023) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min);

  return true;
}

template <typename Element, typename Layout>
bool initialize_quant_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed=2023) {
  
  float scope_min = 0.f;
  float scope_max = 7.f;

  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min);

  return true;
}

template <class Element, class Layout>
bool initialize_scale(
  cutlass::TensorView<Element, Layout> view, 
  const Options &options) {
  
  if (options.mode == GemmMode::ConvertOnly) {
    // No scales, so just initialize with 1 so we can use the same kernel to dequantize the data.
    cutlass::reference::host::TensorFill(view, Element(1.0f));
  } 
  else {
    float elt_max_f = float(cutlass::platform::numeric_limits<QuantType>::max());
    const float max_dequant_val = 4.f;
    const float min_dequant_val = 0.5f;

    float scope_max(max_dequant_val / elt_max_f);
    float scope_min(min_dequant_val / elt_max_f);

    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min);
  }
  return true;
}

template <class Element, class Layout>
bool initialize_zero(
  cutlass::TensorView<Element, Layout> view,
  const Options &options) {
  
  if (options.mode == GemmMode::ScaleWithZeroPoint) {
    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, 2.0f, -2.0f);
  } else {
    // No bias, so just initialize with 1 so we can use the same kernel to dequantize the data.
    cutlass::reference::host::TensorFill(view, Element(0.0f));
  }
  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  auto shape_b = cute::make_shape(options.n, options.k, options.l);
  const int scale_k = (options.k + options.g - 1) / options.g;
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_b);

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);

  tensor_A.resize(a_coord);
  // tensor_B.resize(b_coord);
  tensor_B_dq.resize(b_coord);

  tensor_scale.resize({scale_k * options.l, options.n});
  tensor_zero.resize({scale_k * options.l, options.n});

  initialize_tensor(tensor_A.host_view(), seed + 2022);
  // initialize_quant_tensor(tensor_B.host_view(), seed + 2021);
  initialize_scale(tensor_scale.host_view(), options);
  initialize_zero(tensor_zero.host_view(), options);

  tensor_A.sync_device();
  // tensor_B.sync_device();
  tensor_scale.sync_device();
  tensor_zero.sync_device();

  auto layout_B = make_layout(shape_b, stride_B);

  auto shape_scale_zero = cute::make_shape(options.n, scale_k, options.l);
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(options.n, scale_k, options.l));
  stride_S_ref = cutlass::make_cute_packed_stride(StrideS_ref{}, cute::make_shape(options.n, scale_k, options.l));
  auto layout_scale_zero = make_layout(shape_scale_zero, stride_S_ref);

  dequantize_weight(tensor_B_dq.device_data(), tensor_B.device_data(), layout_B, tensor_scale.device_data(), tensor_zero.device_data(), layout_scale_zero, options.g);
  tensor_B_dq.sync_host();
}

/// Execute a given example GEMM computation
int run(Options &options)
{
  initialize(options);

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //
  run(options);
 

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
