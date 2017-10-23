#ifdef __cplusplus
extern "C" {
#endif

// float
void init_rand(void);
int run_mh_with_kernel(float* samples, int length);
int run_mh_coupling_sampler(float *samples, int length, int pitch);

#ifdef __cplusplus
}
#endif
