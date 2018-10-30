
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

__global__ void RandP_cuda(double *result, double l, curandState *state){

  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx != 0) continue;

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

double RandP(double l, std::mt19937 rng) {

  // // Set limit on distribution
  // poisson_distribution <long long> distr(l);

  // return distr(arr_rng[i*nGridXY*nGridZ + j*nGridZ + k]);
  double L = exp(-l);
  double p = 1.0;
  double n = 0;
  while (p > L) {
    n++;
    double u = rand(rng);
    p *= u;
  }
  return n - 1;
}






int main() {
  double lambda  = 4;
  std::mt19937 rng;
  std::poisson_distribution<double> distr(lambda);

  curandState *d_state
  cudaMalloc((void**)&d_state, BlockSize*sizeof(curandState));

  double *d_result;
  double *h_result = new double;
  cudaMalloc((void**)&d_result, sizeof(double));

  setup_kernel<<<1,BlockSize>>>(d_state);

  std::map<int, int> hist_std;
  std::map<int, int> hist_cuda;
  for(int n=0; n<10000; ++n) {
      ++hist_std[static_cast<int>distr(rng)];

      RandP_cuda<<<1,1>>>(d_result, lambda, d_state);
      cudaMemcpy(h_result,  d_result,  sizeof(double), cudaMemcpyDeviceToHost);

      ++hist_cuda[static_cast<int>(*h_result)];
  }

  for(auto p : hist_std) {
      std::cout << p.first <<
              ' ' << std::string(p.second/100, '*') << '\n';
  }

  for(auto p : hist_cuda) {
    std::cout << p.first <<
            ' ' << std::string(p.second/100, '*') << '\n';
}
}
















int main(){
  int BlockSize = 10;








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
