default:
	nvcc -arch=sm_89 gemm.cu -lcublas -o gemm && CUDA_VISIBLE_DEVICES=5 ./gemm
