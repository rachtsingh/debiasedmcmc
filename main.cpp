#include <stdbool.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include "functions_cuda_kernel.h"

using namespace std;
std::default_random_engine generator;
std::normal_distribution<float> init_distribution(1.0, 1.0);

#define real float

float run_init(void) {
    return init_distribution(generator);
}

// this is confirmed as working
void run_chain(int length, int num_chains) {
    float *samples;
    int err = cudaMalloc(&samples, length * sizeof(float));
    if (err) {
        cout << "cudaMalloc failed;" << endl;
    }
    
    float init = run_init();
    cudaMemcpy(samples, &init, sizeof(float), cudaMemcpyHostToDevice);

    // run the CUDA sampler
    err = run_mh_with_kernel(samples, length);
    if (err) {
        cout << "mh_with_kernel failed" << endl;
    }

    float *local_samples = new float[length];
    cudaMemcpy(local_samples, samples, length * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate the mean (variance is hard)
    float sum = 0.;
    for (int i = 0; i < length; i++) {
        sum += local_samples[i];
    }
    cout << "mean: " << (sum / ((float) length)) << endl;
    cout << endl;

    for (int i = 0;  i < length; i++) {
        cout << local_samples[i] << endl;
    }

    cudaFree(samples);
    delete[] local_samples;
}

// this is confirmed as working
void sample_from_coupling(const int length, int num_chains) {
    float *samples;
    size_t pitch;
    int err = cudaMallocPitch(&samples, &pitch, length * sizeof(float), 2);
    if (err) {
        cout << "cudaMalloc failed;" << endl;
    }
    
    // run the CUDA sampler
    err = run_mh_coupling_sampler(samples, length, pitch);
    if (err) {
        cout << "mh_with_kernel failed" << endl;
    }

    float local_samples[2][length];
    cudaMemcpy2D(local_samples, length * sizeof(float), samples, pitch, length * sizeof(float), 2, cudaMemcpyDeviceToHost);

    for (int i = 0;  i < length; i++) {
        cout << local_samples[0][i] << ',' << local_samples[1][i] << endl;
    }

    cudaFree(samples);
}

// looks like this works as well
void run_coupled_chains(const int length, int num_chains) {
    float *samples;
    size_t pitch;
    int err = cudaMallocPitch(&samples, &pitch, length * sizeof(float), 2);
    if (err) {
        cout << "cudaMalloc failed;" << endl;
    }
    
    // run the CUDA sampler
    err = run_mh_with_coupled_kernel(samples, length, pitch);
    if (err) {
        cout << "mh_with_kernel failed" << endl;
    }

    float local_samples[2][length];
    cudaMemcpy2D(local_samples, length * sizeof(float), samples, pitch, length * sizeof(float), 2, cudaMemcpyDeviceToHost);

    for (int i = 0;  i < length; i++) {
        cout << local_samples[0][i] << ',' << local_samples[1][i] << endl;
    }

    cudaFree(samples);
}

int main (void) {
    init_rand();
    run_coupled_chains(500, 1);
}
