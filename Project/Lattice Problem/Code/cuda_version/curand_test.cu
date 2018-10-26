
#include <stdio.h>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
// #define ITER 10

__device__ int RandP(curandState rng_state, double lambda) {

  double L = exp(-lambda);
  double p = 1.0;
  int k = 0;
  while (p > L) {
    k++;
    double u = curand_uniform_double(&rng_state);
    p *= u;
  }
  return k - 1;

}


__global__ void setup_kernel(curandState *state){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void generate_kernel(curandState *my_curandstate, int *result, int *resultp){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    result[idx] = RandP(my_curandstate[idx],0.1);
    resultp[idx] = curand_poisson(&my_curandstate[idx], 0.1);
}

int main(){
  int ITER = 1000;

  curandState *d_state;
  cudaMalloc(&d_state, ITER*sizeof(curandState));

  int *d_result;
  int *d_resultp;
  int *h_result  = (int*) malloc(ITER*sizeof(int));
  int *h_resultp = (int*) malloc(ITER*sizeof(int));
  cudaMalloc(&d_result,  ITER*sizeof(int));
  cudaMalloc(&d_resultp, ITER*sizeof(int));
  setup_kernel<<<1,ITER>>>(d_state);

  generate_kernel<<<1,ITER>>>(d_state, d_result, d_resultp);
  cudaMemcpy(h_result,  d_result,  ITER*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_resultp, d_resultp, ITER*sizeof(int), cudaMemcpyDeviceToHost);

  // Set limit on distribution
  std::mt19937 rng;
  std::poisson_distribution <long long> distr(0.1);


  printf("\n\nRandP:\n");
  for(int i = 0; i < ITER; i++){
    if (h_result[i] > 0) {
     printf("%d, ",h_result[i]);
    }
  }
  printf("\n\nstd library:\n");


  for(int i = 0; i < ITER; i++){
    int k = (int)distr(rng);
    if (k>0) {
     printf("%d, ",k);
    }
  }
  printf("\n");




  return 0;
}