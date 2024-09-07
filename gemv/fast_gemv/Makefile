default:
	nvcc -o gemv gemv.cu -O2 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -cudart shared --cudadevrt none && CUDA_VISIBLE_DEVICES=0 ./gemv