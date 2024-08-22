default:
	nvcc -o gemm gemm.cu -O2 -arch=sm_80 -std=c++17 -I/home/v-dayoudu/projects/BitAttn/3rdparty/cutlass/include -I/home/v-dayoudu/projects/BitAttn/3rdparty/cutlass/tools/util/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas && ./gemm

