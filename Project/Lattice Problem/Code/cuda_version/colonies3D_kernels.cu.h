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

__global__ void SetIsActive(double* arr_Occ, double* arr_nC, bool* arr_IsActive, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  bool insideBounds = (i < N);

  double nC =  insideBounds ? arr_nC[i] : 0.0;
  double Occ = insideBounds ? arr_Occ[i] : 0.0;
  arr_IsActive[i] = insideBounds && (nC >= 1.0 || Occ >= 1.0);

}

__global__ void SecondKernel(double* arr_Occ, double* arr_nC, double* maxOcc,
                             bool* arr_IsActive, int N){

  extern __shared__ double shared[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  shared[tid] = arr_IsActive[i] ? arr_Occ[i] : 0.0;

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

__global__ void UpdateCountKernel(double* arr_GrowthModifier,
                                  double* arr_I9,
                                  double* arr_Occ,
                                  double* arr_P_new,
                                  double* arr_M,
                                  double* arr_p,
                                  bool* arr_IsActive,
                                  double alpha,
                                  double beta,
                                  double r,
                                  double dT,
                                  bool* Warn_r,
                                  bool reducedBeta
                                  ){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (!(arr_IsActive[i])){
    return;
  }

  double p;
  double tmp;

  // Compute the growth modifier
  double growthModifier = arr_GrowthModifier[i];

  // Compute beta
  double Beta = beta;
  if (reducedBeta) {
    Beta *= growthModifier;
  }

  /* BEGIN tredje Map-kernel */

  p = r*growthModifier*dT;
  if ((p > 0.25) and (!(*Warn_r))) {
    *Warn_r = true;
  }
  arr_p[i] = p;

  //tmp = ComputeEvents(arr_I9[i], p, 2, i);  // Bursting events
  tmp = 1.0;
  // Update count
  arr_I9[i]    = max(0.0, arr_I9[i] - tmp);
  arr_Occ[i]   = max(0.0, arr_Occ[i] - tmp);
  arr_P_new[i] += round( (1 - alpha) * Beta * tmp);  // Phages which escape the colony
  arr_M[i] = round(alpha * Beta * tmp); // Phages which reinfect the colony
}


__global__ void NonBurstingEventsKernel(double* arr_A, double* arr_B, double* arr_p, bool* arr_IsActive){
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (!(arr_IsActive[i])){
    return;
  }

  double tmp;
  double A = arr_A[i];
  double p = arr_p[i];

  // TODO: FIX ComputeEvents
  //tmp = ComputeEvents(A, p, 2, i);
  tmp = 1.0;
  arr_A[i] = max(0.0, A - tmp);
  arr_B[i] += tmp;
}

__global__ void NewInfectionsKernel(double* arr_Occ,
                                    double* arr_nC,
                                    double* arr_P,
                                    double* arr_P_new,
                                    double* arr_GrowthModifier,
                                    double* arr_B,
                                    double* arr_B_new,
                                    double* arr_M,
                                    double* arr_I0_new,
                                    bool* arr_IsActive,
                                    bool reducedBeta,
                                    bool clustering,
                                    bool shielding,
                                    double K,
                                    double alpha,
                                    double beta,
                                    double eta,
                                    double zeta,
                                    double dT,
                                    double r){

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  bool isInactive = (!(arr_IsActive[tid]));
  if (isInactive){
    return;
  }

  double B = arr_B[tid];
  double nC = arr_nC[tid];
  double Occ = arr_Occ[tid];
  double P = arr_P[tid];
  double M = arr_M[tid];
  double tmp;


	// Compute the growth modifier
	double growthModifier = arr_GrowthModifier[tid];

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
      arr_B[tid] = max(0.0, B - tmp);
      if (r > 0.0) {
        arr_I0_new[tid] += tmp;
      } else {
        arr_P_new[tid] += tmp * (1 - alpha) * Beta;
      }
    }
  }
}


__global__ void SixthKernel(double* arr_P, double p, bool *warn_delta, int numberOfElements){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= numberOfElements) return;

    // TODO: skip empty sites or not?? (Not included here)
    double N = 0;
 // TODO: figure out a shared variable warn      
 //   if ((p > 0.1) and (!warn_delta)) &warn_delta = true;
    

    // TODO: do proper Compute events:
//    N = ComputeEvents(arr_P[i], p, 5, i); 
    N = 1.0  
    arr_P[i]    = max(0.0, arr_P[i] - N);

}

__global__ void SeventhKernel(){

// Phage decay

// Compute p
// Compute n

// Update P

}

#endif

