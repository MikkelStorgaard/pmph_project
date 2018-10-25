
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
// #define ITER 10

__global__ void setup_kernel(curandState *state){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void generate_kernel(curandState *my_curandstate, float *result, int *resultp){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    result[idx] = curand_uniform(&my_curandstate[idx])+1;
    resultp[idx] = curand_poisson(&my_curandstate[idx], 0);
}

int main(){
  int ITER = 10;

  curandState *d_state;
  cudaMalloc(&d_state, ITER*sizeof(curandState));

  float *d_result;
  int *d_resultp;
  float *h_result = (float*) malloc(ITER*sizeof(float));
  int *h_resultp = (int*) malloc(ITER*sizeof(int));
  cudaMalloc(&d_result, ITER*sizeof(float));
  cudaMalloc(&d_resultp, ITER*sizeof(int));
  setup_kernel<<<1,ITER>>>(d_state);

  generate_kernel<<<1,ITER>>>(d_state, d_result, d_resultp);
  cudaMemcpy(h_result, d_result, ITER*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_resultp, d_resultp, ITER*sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < ITER; i++){
    printf("result : %f \n" , h_result[i]);
    printf("resultp: %d \n" , h_resultp[i]);
  }

  return 0;
}