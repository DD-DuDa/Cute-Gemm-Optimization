default:
	nvcc -o gemv Sgemv_v1.cu -O2 -arch=sm_80 -std=c++17 --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublas