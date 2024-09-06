#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

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

/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
    using namespace cute;

    // Slice the tiled tensors
    Tensor tile_S = S(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

    // Construct a partitioning of the tile among threads with the given thread arrangement.

    // Concept:                         Tensor  ThrLayout       ThrIndex
    Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
    Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_tensor to try to match the layout of thr_tile_S
    Tensor fragment = make_tensor_like(thr_tile_S);               // (ThrValM, ThrValN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(thr_tile_S, fragment);
    copy(fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout)
{
    using namespace cute;
    using Element = typename TensorS::value_type;

    // Slice the tensors to obtain a view into each tile.
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

    // Define `AccessType` which controls the size of the actual memory access.
    using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;

    // A copy atom corresponds to one hardware memory access.
    using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

    // Construct tiled copy, a tiling of copy atoms.
    //
    // Note, this assumes the vector and thread layouts are aligned with contigous data
    // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
    // reads. Alternative vector layouts are also possible, though incompatible layouts
    // will result in compile time errors.
    auto tiled_copy =
        make_tiled_copy(
        Atom{},                       // access size
        ThreadLayout{},               // thread layout
        VecLayout{});                 // vector layout (e.g. 4x1)

    // Construct a Tensor corresponding to each thread's slice.
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
    Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        PRINT("tile_S", tile_S.layout())
        PRINT("tile_D", tile_D.layout())
        PRINT("thr_tile_S", thr_tile_S.layout())
        PRINT("thr_tile_D", thr_tile_D.layout())
    }

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}

/// Main function
int main(int argc, char** argv)
{
    using namespace cute;
    using Element = float;

    // Define a tensor shape with dynamic extents (m, n)
    auto tensor_shape = make_shape(256, 512);

    //
    // Allocate and initialize
    //

    thrust::host_vector<Element> h_S(size(tensor_shape));
    thrust::host_vector<Element> h_D(size(tensor_shape));

    for (size_t i = 0; i < h_S.size(); ++i) {
        h_S[i] = static_cast<Element>(i);
        h_D[i] = Element{};
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    //
    // Make tensors
    //

    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));


    //
    // Tile tensors
    //

    // Define a statically sized block (M, N).
    // Note, by convention, capital letters are used to represent static modes.
    auto block_shape = make_shape(Int<128>{}, Int<64>{});

    // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
    // shape, and modes (m', n') correspond to the number of tiles.
    //
    // These will be used to determine the CUDA kernel grid dimensions.
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n') -> ((_128,_64),2,8)
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n') -> ((_128,_64),2,8)((_128,_64),2,8)

    PRINT("tiled_tensor_S", tiled_tensor_S.shape())

    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));

    // Vector dimensions
    Layout vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

    //
    // Determine grid and block dimensions
    //

    dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(thr_layout));

    //
    // Launch the kernel
    //
    copy_kernel<<< gridDim, blockDim >>>(
        tiled_tensor_S,
        tiled_tensor_D,
        thr_layout);

    cudaError result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
        return -1;
    }

    //
    // Verify
    //

    h_D = d_D;

    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    for (size_t i = 0; i < h_D.size(); ++i) {
        if (h_S[i] != h_D[i]) {
        std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

        if (++errors >= kErrorLimit) {
            std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
            return -1;
        }
        }
    }

    std::cout << "Success." << std::endl;

    return 0;
}