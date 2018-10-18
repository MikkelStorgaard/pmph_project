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


// Birth

// Compute p

// Return warning

// Update B_new
// Update Nutrient

}

__global__ void FourhtKernel(){

// Increase Infections

// Compute Beta

// Compute p

// Update I9
// Update Occ
// Update P_new
// Update M

// Compute N
// Update I8,I9

// Compute N
// Update I7,I8

// Compute N
// Update I6,I7

// Compute N
// Update I5,I6

// Compute N
// Update I4,I5

// Compute N
// Update I3,I4

// Compute N
// Update I2,I3

// Compute N
// Update I1,I2

}

__global__ void FifthKernel(){

// New infections

// Compute s
// Compute n

// Compute N

// Compute S

// Compute p

// Compute N

// Update B_new
// Update I0_new / P_bew

}

__global__ void SixthKernel(){

// Phage decay

// Compute p
// Compute n

// Update P

}

__global__ void SeventhKernel(){

// Phage decay

// Compute p
// Compute n

// Update P

}


#endif
