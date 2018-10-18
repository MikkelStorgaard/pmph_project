#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS

__global__ void FirstKernel(double* arr_Occ, double* arr_nC, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  bool outOfBounds = (i >= N);

  if (outOfBounds){
    return;
  }

  if (arr_Occ[i] < arr_nC[i]){
      arr_nC[i] = arr_Occ[i];
  }
}

__global__ void SecondKernel(double* arr_Occ, double* arr_nC, double* maxOcc, int N){

  extern __shared__ double shared[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  bool outOfBounds = (i >= N);
  double nC = outOfBounds ? 0.0 : arr_nC[tid];
  double Occ = outOfBounds ? 0.0 : arr_Occ[tid];
  bool active = nC >= 1.0 || Occ >= 1.0;

  shared[tid] = (outOfBounds || !active) ? 0.0 : Occ;

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      shared[tid] = max(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if(tid == 0){
    maxOcc[blockIdx.x] = shared[0];
  }
}

__global__ void ThirdKernel(double* arr_Occ,
                            double* arr_P,
                            double* arr_nutrient,
                            double* arr_B,
                            double* arr_B_new,
                            double* , int N){
}

#endif
