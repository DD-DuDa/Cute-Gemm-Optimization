#pragma once

#include <cuda.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include "cutlass/array.h"
#include "cutlass/barrier.h"
#include "cutlass/fast_math.h"
#include "cutlass/block_striped.h"


namespace config {

using namespace cute;

// Pad the given allocation size up to the nearest cache line
CUTE_HOST_DEVICE static
size_t
cacheline_align_up(size_t size)
{
    static const int CACHELINE_SIZE = 128;
    return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
}

// Get the workspace size needed for intermediate partial sums
CUTE_HOST_DEVICE
size_t
get_workspace_size_partials(int sk_blocks, int threads, int accum_size)
{
    return cacheline_align_up(sk_blocks * threads * accum_size);
}

// Get the workspace size needed for barrier
CUTE_HOST_DEVICE
size_t
get_workspace_size_barriers(int sk_blocks)
{
    // For atomic reduction, each SK-block needs a synchronization flag.  For parallel reduction,
    // each reduction block needs its own synchronization flag.
    return cacheline_align_up(sizeof(typename cutlass::Barrier::T) * sk_blocks);
}


} // namespace config