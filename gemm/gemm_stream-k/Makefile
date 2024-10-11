default:
	nvcc -o gemm main.cu -O2 -arch=sm_80 -std=c++17 -I../../flashdecoding/src/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none && CUDA_VISIBLE_DEVICES=0 ./gemm