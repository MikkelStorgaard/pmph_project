
#include <stdio.h>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
// #define ITER 10

__device__ int RandP(curandState rng_state ,double lambda) {

  // double lambdaLeft = lambda;
  // int k = 0;
  // double p = 0;
  // double STEP = 500;

  // do {
  //   k++;
  //   double u = curand_uniform(&rng_state);

  //   p *= u;

  //   while ((p < 1) && (lambdaLeft > 0)){
  //     if (lambdaLeft > STEP) {
  //       p *= exp(STEP);
  //       lambdaLeft -= STEP;
  //     } else {
  //       p = exp(lambdaLeft);
  //       lambdaLeft = 0;
  //     }
  //   }
  // } while (p > 1);

  // return k - 1;


  double L = exp(-lambda);
  double p = 1;
  double k = 0;
  do {
    k++;
    double u = curand_uniform(&rng_state);
    p *= u;
  } while (p > L);
  return k - 1;

}


__global__ void setup_kernel(curandState *state){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void generate_kernel(curandState *my_curandstate, int *result, int *resultp){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    result[idx] = RandP(my_curandstate[idx],0.01);
    resultp[idx] = curand_poisson(&my_curandstate[idx], 0.01);
}

int main(){
  int ITER = 10000;

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
  std::poisson_distribution <long long> distr(0.01);

  printf("cuRand:\n");
  for(int i = 0; i < ITER; i++){
    // printf("result : %f \n" , h_result[i]);
    // printf("resultp: %d \n" , h_resultp[i]);
    if (h_resultp[i] > 0) {
     printf("%d, ",h_resultp[i]);
    }
  }
  printf("\n\nRandP:\n");
  for(int i = 0; i < ITER; i++){
    // printf("result : %f \n" , h_result[i]);
    // printf("resultp: %d \n" , h_resultp[i]);
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




  return 0;
}