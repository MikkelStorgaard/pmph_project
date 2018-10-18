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
                            double* arr_nC,
                            double* arr_P,
                            double* arr_P_new,
                            double* arr_nutrient,
                            double* arr_B,
                            double* arr_B_new,
                            double* arr_M,
                            double* arr_I0_new,
                            bool reducedBeta,
                            bool clustering,
                            bool shielding,
                            double K,
                            double alpha,
                            double beta,
                            double eta,
                            double zeta,
                            double dT,
                            double r,
                            int N){

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  bool outOfBounds = (tid >= N);
  double B = outOfBounds ? 0.0 : arr_B[tid];
  double nC = outOfBounds ? 0.0 : arr_nC[tid];
  double Occ = outOfBounds ? 0.0 : arr_Occ[tid];
  double P = outOfBounds ? 0.0 : arr_P[tid];
  double M = outOfBounds ? 0.0 : arr_M[tid];
  double tmp;

  bool active = nC >= 1.0 || Occ >= 1.0;
  if (!active){
    return;
  }

	// Compute the growth modifier
	double growthModifier =
     arr_nutrient[tid] / (arr_nutrient[tid] + K);
  ///////////// should the growth modifier have been an array instead?
  // Compute beta
  double Beta = beta;
  if (reducedBeta) {
    Beta *= growthModifier;
  }

  double p;
  double s;
  double n;

  // KERNEL THIS
  if ((Occ >= 1) && (P >= 1)) {
    if (clustering) {   // Check if clustering is enabled
      s = pow(Occ / nC, 1.0 / 3.0);
      n = nC;
    } else {            // Else use mean field computation
      s = 1.0;
      n = Occ;
    }

    // Compute the number of hits
    if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
      tmp = P;
    } else {
      p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
      // tmp = ComputeEvents(P, p, 4, tid);           // Number of targets hit //
      tmp = 1;
      // TODO: replace ComputeEvents with something that works
      /* ComputeEvents used to be (..., i, j, k), but in this flat kernel,
         tid is equal to i * j * k */
    }

    if (tmp + M >= 1) {
      // If bacteria were hit, update events
      arr_P[tid] = max(0.0, P - tmp); // Update count

      double S;

      if (shielding) {
        // Absorbing medium model
        double d =
          pow(Occ / nC, 1.0 / 3.0) - pow(B / nC, 1.0 / 3.0);
        S = exp(-zeta * d); // Probability of hitting succebtible target

      } else {
        // Well mixed model
        S = B / Occ;
      }

      p = max(0.0, min(B / Occ, S)); // Probability of hitting succebtible target
      // TODO:
      //tmp = ComputeEvents(tmp + M, p, 4, tid); // Number of targets hit
      tmp = 1;

      tmp = min(tmp, B); // If more bacteria than present are set to be infeced, round down

      // Update the counts
      arr_B[tid] = max(0.0, B - N);
      if (r > 0.0) {
        arr_I0_new[tid] += tmp;
      } else {
        arr_P_new[tid] += tmp * (1 - alpha) * Beta;
      }
    }
  }
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
