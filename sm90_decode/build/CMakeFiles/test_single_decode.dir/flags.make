# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# compile CUDA with /usr/local/cuda-12.4/bin/nvcc
CUDA_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE

CUDA_INCLUDES = --options-file CMakeFiles/test_single_decode.dir/includes_CUDA.rsp

CUDA_FLAGS =  -DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_90,code=sm_90 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -std=c++17 -maxrregcount=255 -gencode arch=compute_90a,code=sm_90a -w -D_GLIBCXX_USE_CXX11_ABI=0

