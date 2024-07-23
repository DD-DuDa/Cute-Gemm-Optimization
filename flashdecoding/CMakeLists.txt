cmake_minimum_required(VERSION 3.23.1)
project(flashdecoding CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -g -G")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/cutlass/include)

message(STATUS "Compile testing kernel.")
add_executable(test_single_decode 
    ${PROJECT_SOURCE_DIR}/src/test_single_decode.cu
)

target_include_directories(test_single_decode PRIVATE ${INCLUDE_DIR})
target_compile_options(test_single_decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)