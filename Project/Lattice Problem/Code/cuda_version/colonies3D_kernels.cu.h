// #include "colonies3D_helpers.cu"
#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS

__device__ double RandP(curandState rng_state, double lambda) {

  double L = exp(-lambda);
  double p = 1.0;
  double k = 0;
  while (p > L) {
    k++;
    double u = curand_uniform_double(&rng_state);
    p *= u;
  }
  return k - 1;

}

__global__ void ComputeEvents_seq(double *N, double n, double p, curandState* rng_state, int index){
    // Trivial cases
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i == index) {

      *N = 0.0;

      if (p == 1) return;
      if (p == 0) return;
      if (n < 1)  return;

      *N = round(RandP(rng_state[i], n*p));

    }
}

__device__ double ComputeEvents(double n, double p, curandState rng_state){
    // Trivial cases

    if (p == 1) return n;
    if (p == 0) return 0.0;
    if (n < 1)  return 0.0;

    // double N = (double)curand_poisson(&rng_state, n*p);
    return round(RandP(rng_state, n*p));

}

__global__ void initRNG(curandState *state, int N){

  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < N) {
    curand_init(0, i, 0, &state[i]);
  }
}

__global__ void FirstKernel(double* arr_Occ, double* arr_nC, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  // Out of bounds check
  if (i>= N){
    return;
  }

  if (arr_Occ[i] < arr_nC[i]){
      arr_nC[i] = arr_Occ[i];
  }
  arr_nC[i] = min(arr_nC[i],arr_Occ[i]);
}

__global__ void SetIsActive(double* arr_Occ, double* arr_P, bool* arr_IsActive, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  bool insideBounds = (i < N);

  double Occ = insideBounds ? arr_Occ[i] : 0.0;
  double P   = insideBounds ? arr_P[i]   : 0.0;
  arr_IsActive[i] = insideBounds && ((P >= 1.0) && (Occ >= 1.0));

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

__global__ void SequentialReduce(double* A, int A_len){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > 0){
    return;
  }
  double tmp = 0.0;
  double current_max = 0.0;

  // the little thread that could
  for (unsigned int ind=0; ind<A_len; ind++) {
    tmp = A[ind];
    if(tmp > current_max){
      current_max = tmp;
    }
  }
  A[0] = current_max;
}


__global__ void ComputeBirthEvents(double* arr_B, double* arr_B_new, double* arr_nutrient, double* arr_GrowthModifier, double K, double g, double dT, bool* Warn_g, bool* Warn_fastGrowth, curandState *rng_state, bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (!arr_IsActive[i]){
    return;
  }

  // Compute the growth modifier
  double growthModifier = arr_nutrient[i] / (arr_nutrient[i] + K);
  if (arr_nutrient[i] < 1) {
    growthModifier = 0;
  }
  arr_GrowthModifier[i] = growthModifier;

  // Compute birth probability
  double p = g * growthModifier * dT;

  // Produce warning
  if ((p > 0.1) and (!(*Warn_g))){
    *Warn_g = true;
  }

  // Compute the number of births
  double N = ComputeEvents(arr_B[i], p, rng_state[i]);

  // Ensure there is enough nutrient
	if ( N > arr_nutrient[i] ) {

    if (!(*Warn_fastGrowth)){
      *Warn_fastGrowth = true;
    }

    N = round( arr_nutrient[i] );
  }

  // Update count
  arr_B_new[i] += N;
  arr_nutrient[i] = max(0.0, arr_nutrient[i] - N);

}


__global__ void BurstingEvents(double* arr_I9, double* arr_P_new, double* arr_Occ, double* arr_GrowthModifier, double* arr_M, double* arr_p, double alpha, double beta, double r, double dT, bool* Warn_r, curandState *rng_state, bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (!arr_IsActive[i]){
    return;
  }

  // Fetch growthModifier
  double growthModifier = arr_GrowthModifier[i];
  double Beta = beta*growthModifier;

  // Compute infection increse probability
  double p = r * growthModifier *dT;

  // Produce warning
  if ((p > 0.25) and (!(*Warn_r))){
    *Warn_r = true;
  }

  // Compute the number of bursts
  double N = ComputeEvents(arr_I9[i], p, rng_state[i]);

  // Update count
  arr_I9[i]    = max(0.0, arr_I9[i] - N);
  arr_Occ[i]   = max(0.0, arr_Occ[i] - N);
  arr_P_new[i] += round( (1 - alpha) * Beta * N);
  arr_M[i]     = round(alpha * Beta * N);
  arr_p[i]     = p;
}

__global__ void NonBurstingEvents(double* arr_I, double* arr_In, double* arr_p, curandState *rng_state, bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  // Out of bounds check
  if (!arr_IsActive[i]){
    return;
  }

  // Compute the number of bursts
  double N = ComputeEvents(arr_I[i], arr_p[i], rng_state[i]);

  // Update count
  arr_I[i]     = max(0.0, arr_I[i] - N);
  arr_In[i]    += N;
}


//Kernel 3.2 Birth 2


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
                                  bool reducedBeta,
                                  curandState* rng_state
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

  tmp = ComputeEvents(arr_I9[i], p, rng_state[i]);  // Bursting events

  // Update count
  arr_I9[i]    = max(0.0, arr_I9[i] - tmp);
  arr_Occ[i]   = max(0.0, arr_Occ[i] - tmp);
  arr_P_new[i] += round( (1 - alpha) * Beta * tmp);  // Phages which escape the colony
  arr_M[i] = round(alpha * Beta * tmp); // Phages which reinfect the colony
}


/*
__global__ void ThirdTwoKernel(bool* arr_IsActive, double* arr_nutrient, double* arr_B_new, bool* warn_fastGrowth){
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Skip empty sites
    if (!(arr_IsActive[i])) return;


        double N = 0;

    // TODO: Proper compute events
    //N = ComputeEvents(arr_B[i*nGridXY*nGridZ + j*nGridZ + k], p, 1, i, j, k);
    N = 1;

    // Ensure there is enough nutrient
    if ( N > arr_nutrient[i] ) {
        if (!*warn_fastGrowth) { *warn_fastGrowth = true;    }
        N = round( arr_nutrient[i] );
    }

    // Update count
    arr_B_new[i] += N;
    arr_nutrient[i] = max(0.0, arr_nutrient[i] - N);
}

*/
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
                                    double r,
                                    curandState* rng_state
                                    ){

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
      tmp = ComputeEvents(P, p, rng_state[tid]);           // Number of targets hit //
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

      tmp = ComputeEvents(tmp + M, p, rng_state[tid]); // Number of targets hit
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
__global__ void PhageDecay(double* arr_P, double p,
                           bool *warn_delta, curandState* rng_state,
                           bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (!(arr_IsActive[i])){
    return;
  }

 double N = ComputeEvents(arr_P[i], p, rng_state[i]);

 if ((p > 0.1) && (!(*warn_delta))){
   *warn_delta = true;
 }

 arr_P[i] = max(0.0, arr_P[i] - N);
}

#endif

