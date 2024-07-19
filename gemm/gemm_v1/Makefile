default:
	nvcc -o gemm gemm.cu -O2 -arch=sm_89 -std=c++17 -I/home/dudayou/dayou/repo/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none && CUDA_VISIBLE_DEVICES=4 ./gemm