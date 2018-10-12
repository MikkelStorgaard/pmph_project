#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS

// widthA = heightB
__global__ void FirstKernel(double* arr_Occ, double* arr_nC, double* maxOcc,
                            int gridX, int gridY, int gridZ) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  bool outOfBounds = ((i >= gridX) || (j >= gridY) || (k >= gridZ));

  if (!(outOfBounds)){
    if (arr_Occ[i][j][k] < arr_nC[i][j][k]){
      arr_nC[i][j][k] = arr_Occ[i][j][k];
    }
  }

}

__global__ void SecondKernel(double*** arr_Occ, double*** arr_nC, double* maxOcc,
                             int gridX, int gridY, int gridZ) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  bool outOfBounds = ((i >= gridX) || (j >= gridY) || (k >= gridZ));

  if (arr_Occ[i][j][k] < arr_nC[i][j][k]){
    arr_nC[i][j][k] = arr_Occ[i][j][k];
  }
}

#endif
