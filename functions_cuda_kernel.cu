#include <stdbool.h>
#include <stdio.h>
#include "functions_cuda_kernel.h"
#include <math.h>
#include <unistd.h>
#include <stdlib.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define real float
#define NUM_BLOCKS 256

__device__ curandState_t global_states[256];

// there's a way to write shorter code by templating float/double, but without knowing much about template overhead (which I think is small, but not certain) I'm just going to reimplement + vim
__global__ void init(unsigned int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &global_states[idx]);     
    __syncthreads();
}

// just for compilation purposes - this is Marsaglia's algorithm
__global__ void sample_gamma_dbl_kernel(int height, int width, double *a_data, double *output_data) {
  for (int addr = threadIdx.x; addr < width * height; addr += blockDim.x) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double d = a_data[addr]  - (1./3.);
    double c = 1./sqrt(9. * d);
    double u, v, x = 0;
    do {
      x = curand_normal(&global_states[idx]);
      v = (1 + c * x) * (1 + c * x) * (1 + c * x);
      u = curand_uniform(&global_states[idx]);
    } while (v <= 0. || (log(u) >= 0.5 * x * x + d * (1 - v + log(v))));
    output_data[addr] = d * v;
  }
}

__device__ double log_target(double x) {
    // -0.5 log(2pi) - log(1)
    return -0.91893853320467267 + 0 - 0.5 * (x * x);
}

__global__ void mh_kernel(float *samples, int length) {
    int addr = threadIdx.x;
    // replace with loop
    if (addr != 0) {
        return;
    }
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t *rand_state = &global_states[idx];
    for (int i = 1; i < length; i++) {
        float proposal = curand_normal(rand_state) + samples[i - 1];
        double p_pdf = log_target(proposal);
        double s_pdf = log_target(samples[i - 1]);
        double coin = log(curand_uniform(rand_state));
        if (coin < (p_pdf - s_pdf)) {
            samples[i] = proposal;
        }
        else {
            samples[i] = samples[i - 1];
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

int run_mh_with_kernel(float *samples, int length) {
    mh_kernel<<<1, NUM_BLOCKS>>>(samples, length);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in mh_kernel: %s\n", cudaGetErrorString(err));
        return 2;
    }
    return 0;
}

void init_rand(void) {
    init<<<1, NUM_BLOCKS>>>(time(NULL));
}

#ifdef __cplusplus
}
#endif
