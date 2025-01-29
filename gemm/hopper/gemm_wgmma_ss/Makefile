default:
	nvcc -o gemm gemm.cu -O2 -arch=compute_90a -code=sm_90a -std=c++17 \
	-I/home/zhichen/dayou/cutlass/include \
	-I/home/zhichen/dayou/cutlass/tools/util/include \
	--expt-relaxed-constexpr -cudart shared --cudadevrt none \
	&& CUDA_VISIBLE_DEVICES=3 ./gemm