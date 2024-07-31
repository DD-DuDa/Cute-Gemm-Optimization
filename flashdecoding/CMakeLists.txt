cmake_minimum_required(VERSION 3.23.1)
project(flashdecoding CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/cutlass/include)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


message(STATUS "Compile testing kernel.")
add_executable(test_single_decode 
    ${PROJECT_SOURCE_DIR}/src/test_single_decode.cu
    ${PROJECT_SOURCE_DIR}/src/flash_fwd_hdim128_fp16_sm80.cu
    ${PROJECT_SOURCE_DIR}/src/flash_fwd_split_hdim128_fp16_sm80.cu
)
target_link_libraries(test_single_decode "${TORCH_LIBRARIES}")
target_include_directories(test_single_decode PRIVATE ${INCLUDE_DIR})
target_compile_options(test_single_decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)

# message(STATUS "Compile benchmarking kernel.")
# add_executable(bench_single_decode 
#     ${PROJECT_SOURCE_DIR}/src/bench_single_decode.cu
#     ${PROJECT_SOURCE_DIR}/src/flash_fwd_hdim128_fp16_sm80.cu
#     ${PROJECT_SOURCE_DIR}/src/flash_fwd_split_hdim128_fp16_sm80.cu
# )
# target_link_libraries(bench_single_decode "${TORCH_LIBRARIES}")
# target_include_directories(bench_single_decode PRIVATE ${INCLUDE_DIR})
# target_compile_options(bench_single_decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)