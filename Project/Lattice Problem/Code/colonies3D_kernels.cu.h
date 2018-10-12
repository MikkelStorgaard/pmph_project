#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS
#define IJK i*gridX*gridZ+j*gridZ+k


// widthA = heightB
__global__ void FirstKernel(double* arr_Occ, double* arr_nC, double* maxOcc,
                            int gridX, int gridY, int gridZ) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  bool outOfBounds = ((i >= gridX) || (j >= gridY) || (k >= gridZ));

  if (!(outOfBounds)){
    if (arr_Occ[IJK] < arr_nC[IJK]){
      arr_nC[IJK] = arr_Occ[IJK];
    }
  }

}

template <class T>
__global__ void ReduceMax(T* A, T* max, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  bool outOfBounds = ((i >= gridX) || (j >= gridY) || (k >= gridZ));

  if (arr_Occ[i][j][k] < arr_nC[i][j][k]){
    arr_nC[i][j][k] = arr_Occ[i][j][k];
  }
}

#endif
