
#include <stdio.h>
#include <iostream>         // Input and output
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>


__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(0, idx, 0, &state[idx]);
}

__global__ void generateAll(double *result, curandState *state){

  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  result[idx] = curand_uniform_double(&state[idx]);
}

__global__ void generateSingle(double* N, curandState *state, int i){

  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (i == idx) {
    *N = curand_uniform_double(&state[idx]);
  }
}

int main(){
  int BlockSize = 10;

  curandState *d_state1, *d_state2;
  cudaMalloc((void**)&d_state1, BlockSize*sizeof(curandState));
  cudaMalloc((void**)&d_state2, BlockSize*sizeof(curandState));

  setup_kernel<<<1,BlockSize>>>(d_state1);
  setup_kernel<<<1,BlockSize>>>(d_state2);

  double *d_result;
  double *h_result = new double[BlockSize];
  cudaMalloc((void**)&d_result,  BlockSize*sizeof(double));


  generateAll<<<1,BlockSize>>>(d_result, d_state1);
  cudaMemcpy(h_result,  d_result,  BlockSize*sizeof(double), cudaMemcpyDeviceToHost);


  std::cout << "Generating all at once:" << std::endl;
  for(int i = 0; i < BlockSize; i++){
    std::cout << h_result[i] << ", ";
  }
  std::cout << std::endl;


  std::cout << "Generating one at a time:" << std::endl;
  double* d_N;
  double* N = new double;
  cudaMalloc((void**)&d_N,sizeof(double));


  for(int i = 0; i < BlockSize; i++){
    generateSingle<<<1,BlockSize>>>(d_N, d_state2,i);
    cudaMemcpy(N, d_N, sizeof(double), cudaMemcpyDeviceToHost);

    // std::cout << *N << ", ";
  }
  std::cout << std::endl;

  return 0;
}
