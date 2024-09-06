default:
	nvcc -o layout layout.cu -arch=sm_80 -std=c++17 -I../../flashdecoding/src/cutlass/include && CUDA_VISIBLE_DEVICES=0 ./layout