#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS
#define IJK i*gridX*gridZ+j*gridZ+k

__global__ void FirstKernel(double* arr_Occ, double* arr_nC, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  bool outOfBounds = (i >= N);

  if (outOfBounds){
    return;
  }

  if (arr_Occ[IJK] < arr_nC[IJK]){
      arr_nC[IJK] = arr_Occ[IJK];
  }
}

__global__ void SecondKernel(double* arr_Occ, double* arr_nC, double* maxOcc,
                            int gridX, int gridY, int gridZ) {

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
    d_maxOccupancy[blockIdx.x] = shared[0];
  }
}

#endif
