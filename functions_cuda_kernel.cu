#include <stdbool.h>
#include <stdio.h>
#include "functions_cuda_kernel.h"
#include <math.h>
#include <unistd.h>
#include <stdlib.h>

#include <utility>

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

__device__ double log_norm_pdf(double x, double mu, double sigma) {
    // -0.5 log(2pi) hardcoded
    return -0.91893853320467267 - log(sigma) - 0.5 * ((x - mu) * (x - mu)) / (sigma * sigma);
}

__device__ void max_coupling_kernel(float mu1, float mu2, float sigma1, float sigma2, float *x_out, float *y_out, curandState_t *rand_state) {
    float x = mu1 + (curand_normal(rand_state) * sigma1);
    float coin = log(curand_uniform(rand_state));
    if (log_norm_pdf(x, mu1, sigma1) + coin < log_norm_pdf(x, mu2, sigma2)) {
        *x_out = x;
        *y_out = x;
    }
    else {
        float y = 0;
        do {
            y = mu2 + (curand_normal(rand_state) * sigma2); 
            coin = log(curand_uniform(rand_state));
        } while (log_norm_pdf(y, mu2, sigma2) + coin < log_norm_pdf(y, mu1, sigma1));
        *x_out = x;
        *y_out = y;
    }
}

// we can run this function in parallel, but for now let's do the slow thing, to test
__global__ void sample_from_max_coupling(float *samples, int length, int pitch, float mu1, float mu2, float sigma1, float sigma2) {
    int addr = threadIdx.x;
    // replace with loop when parallelizing
    if (addr != 0) {
        return;
    }
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t *rand_state = &global_states[idx];
    
    // calculate the positions of the two blocks that we care about
    float *x_chain = (float *) ((char *) samples);
    float *y_chain = (float *) ((char *) samples + pitch);
  
    for (int i = 0; i < length; i++) {
        max_coupling_kernel(mu1, mu2, sigma1, sigma2, &x_chain[i], &y_chain[i], rand_state);
    }
}

__device__ void mh_kernel(float *samples, int i, curandState_t *rand_state) {
    float proposal = curand_normal(rand_state) + samples[i - 1];
    double p_pdf = log_norm_pdf(proposal, 0., 1.);
    double s_pdf = log_norm_pdf(samples[i - 1], 0., 1.);
    double coin = log(curand_uniform(rand_state));
    if (coin < (p_pdf - s_pdf)) {
        samples[i] = proposal;
    }
    else {
        samples[i] = samples[i - 1];
    }
}

__device__ void coupled_mh_kernel(float chain_x_state, float chain_y_state, float std_1, float std_2, float *ret_x, float *ret_y) {
    int addr = threadIdx.x;
    // replace with loop when parallelizing
    if (addr != 0) {
        return;
    }
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t *rand_state = &global_states[idx];
    float proposal_x;
    float proposal_y;
    max_coupling_kernel(chain_x_state, chain_y_state, std_1, std_2, &proposal_x, &proposal_y, rand_state);
    double p_pdf_x = log_norm_pdf(proposal_x, 0., 1.);
    double p_pdf_y = log_norm_pdf(proposal_y, 0., 1.);
    double s_pdf_x = log_norm_pdf(chain_x_state, 0., 1.);
    double s_pdf_y = log_norm_pdf(chain_y_state, 0., 1.);
    double coin = log(curand_uniform(rand_state));
    *ret_x = chain_x_state;
    *ret_y = chain_y_state;
    if (coin < (p_pdf_x - s_pdf_x)) {
        *ret_x = proposal_x;
    } 
    if (coin < (p_pdf_y - s_pdf_y)) {
        *ret_y = proposal_y;
    }
}

__global__ void run_mh_chain(float *samples, int length) {
    int addr = threadIdx.x;
    // replace with loop when parallelizing
    if (addr != 0) {
        return;
    }
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t *rand_state = &global_states[idx];
    for (int i = 1; i < length; i++) {
        mh_kernel(samples, i, rand_state);
    }
}

// samples is a preallocated chain of [max_iterations] floats
__global__ void run_mh_coupled_chain(float *samples, int length, int pitch, int max_iterations, int K=1) {
    int addr = threadIdx.x;
    // replace with loop when parallelizing
    if (addr != 0) {
        return;
    }
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t *rand_state = &global_states[idx];
    
    // calculate the positions of the two blocks that we care about
    float *x_chain = (float *) ((char *) samples);
    float *y_chain = (float *) ((char *) samples + pitch);

    // we can allow passing this in, hard coded for now
    x_chain[0] = (curand_normal(rand_state) * 0.1) + 0.4;
    y_chain[0] = (curand_normal(rand_state) * 0.9) - 0.8;

    // sample the x from the marginal kernel
    mh_kernel(x_chain, 1, rand_state);

    int meeting_time = max_iterations + 1;
    bool met = false;
  
    for (int i = 1; i < max_iterations - 1; i++) {
        if (met) {
            mh_kernel(x_chain, i + 1, rand_state);
            y_chain[i] = x_chain[i + 1];
        }
        else {
            coupled_mh_kernel(x_chain[i], y_chain[i - 1], 0.2, 0.3, &x_chain[i + 1], &y_chain[i]);
            if (!met && (x_chain[i + 1] == y_chain[i])) {
                met = true;
                meeting_time = i;
            }
        }
        if (i >= max(meeting_time, K)) {
            break;
        }
    }
    printf("meeting_time: %d, (%f, %f)", meeting_time, x_chain[meeting_time + 1], y_chain[meeting_time]);
}

#ifdef __cplusplus
extern "C" {
#endif

int run_mh_with_kernel(float *samples, int length) {
    run_mh_chain<<<1, NUM_BLOCKS>>>(samples, length);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in mh_kernel: %s\n", cudaGetErrorString(err));
        return 2;
    }
    return 0;
}

int run_mh_with_coupled_kernel(float *samples, int length, int pitch) {
    run_mh_coupled_chain<<<1, NUM_BLOCKS>>>(samples, length, pitch, length, 100);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in mh_kernel: %s\n", cudaGetErrorString(err));
        return 2;
    }
    return 0;
}

int run_mh_coupling_sampler(float *samples, int length, int pitch) {
    sample_from_max_coupling<<<1, NUM_BLOCKS>>>(samples, length, pitch, 0.2, 0.8, 0.4, 1.7);
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
