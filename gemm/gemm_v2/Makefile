default:
	nvcc -o gemm gemm.cu -O2 -arch=sm_80 -std=c++17 -I/home/zhichen/dayou/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none && CUDA_VISIBLE_DEVICES=0 ./gemm