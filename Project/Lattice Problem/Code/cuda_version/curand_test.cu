
#include <stdio.h>
#include <iostream>         // Input and output
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <map>
#include <assert.h>


__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(0, idx, 0, &state[idx]);
}

__global__ void RandP_cuda(double *result, double l, curandState *state){

  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx != 0) return;

  double L = exp(-l);
  double p = 1.0;
  *result = 0;
  while (p > L) {
      *result++;
      double u = curand_uniform_double(&state[idx]);
      p *= u;
  }
  *result--;
}

double RandP(double l, std::mt19937 *rng) {

  std::uniform_real_distribution <double> distr(0, 1);

  double L = exp(-l);
  double p = 1.0;
  double n = 0;
  while (p > L) {
    n++;
    double u = distr(*rng);
    p *= u;
  }
  return n - 1;
}


int main() {
  double lambda  = 4;
  std::mt19937 rng;

  curandState *d_state;
  cudaMalloc((void**)&d_state, sizeof(curandState));

  double *d_result;
  double *h_result = new double;
  cudaMalloc((void**)&d_result, sizeof(double));

  setup_kernel<<<1,1>>>(d_state);

  std::map<int, int> hist_std;
  std::map<int, int> hist_cuda;
  for(int n=0; n<10000; ++n) {
      ++hist_std[(int)RandP(lambda,&rng)];

      RandP_cuda<<<1,1>>>(d_result, lambda, d_state);
      cudaMemcpy(h_result,  d_result,  sizeof(double), cudaMemcpyDeviceToHost);

      ++hist_cuda[static_cast<int>(*h_result)];
  }

  std::cout << "Output from RandP" << std::endl;
  for(auto p : hist_std) {
      std::cout << p.first <<
              ' ' << std::string(p.second/100, '*') << '\n';
  }
  std::cout << std::endl << std::endl;
  std::cout << "Output from RandP_cuda:" << std::endl;
  for(auto p : hist_cuda) {
    std::cout << p.first <<
            ' ' << std::string(p.second/100, '*') << '\n';
  }
}