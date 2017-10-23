#!/usr/bin/env bash
if hash nvidia-smi 2>/dev/null; then
  export HAS_GPU=true
else 
  echo "No GPU found."
  exit
fi

CUDA_PATH=/n/regal/rush_lab/sw/usr/local/cuda-7.5/

# clean everything from before
rm -f *.o *.so
nvcc -x cu -arch=sm_35 -Xcompiler -fPIC -lcudadevrt -lcudart -std=c++11 main.cpp functions_cuda_kernel.cu -o main
