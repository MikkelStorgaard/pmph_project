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
  // return lambda;

}

__global__ void ComputeEvents_seq(double *N, double n, double p, curandState* rng_state, int index){
    // Trivial cases
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i == index) {

      *N = n;
      if (p == 1) return;

      *N = 0.0;
      if (p == 0) return;
      if (n < 1)  return;

      *N = RandP(rng_state[i], n*p);

    }
}

__device__ double ComputeEvents(double n, double p, curandState rng_state){
    // Trivial cases

    if (p == 1) return n;
    if (p == 0) return 0.0;

		// DETERMINITIC CHANGE
		if (n < 1)  return 0.0;

		return RandP(rng_state, n*p);
		// return n*min(1.0,p);

}

__global__ void initRNG(curandState *state, int N){

  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < N) {
    curand_init(i, 0, 0, &state[i]);
  }
}

__device__ void ComputeDiffusion(curandState state, double n, double lambda, double* n_0, double* n_u, double* n_d, double* n_l, double* n_r, double* n_f, double* n_b, int i, int j, int k, int nGridXY) {

		// Reset positions
		*n_0 = 0.0;
		*n_u = 0.0;
		*n_d = 0.0;
		*n_l = 0.0;
		*n_r = 0.0;
		*n_f = 0.0;
		*n_b = 0.0;

		// DETERMINITIC CHANGE
		if (n < 1) return;

		// Check if diffusion should occur
		if ((lambda == 0) or (nGridXY == 1)) {
				*n_0 = n;
				return;
		}

    // DETERMINITIC CHANGE
		if (lambda*n < 5) {   // Compute all movement individually

				for (int l = 0; l < round(n); l++) {

						double r = curand_uniform(&state);

						if       (r <    lambda)                     (*n_u)++;  // Up movement
						else if ((r >=   lambda) and (r < 2*lambda)) (*n_d)++;  // Down movement
						else if ((r >= 2*lambda) and (r < 3*lambda)) (*n_l)++;  // Left movement
						else if ((r >= 3*lambda) and (r < 4*lambda)) (*n_r)++;  // Right movement
						else if ((r >= 4*lambda) and (r < 5*lambda)) (*n_f)++;  // Forward movement
						else if ((r >= 5*lambda) and (r < 6*lambda)) (*n_b)++;  // Backward movement
						else                                         (*n_0)++;  // No movement

				}


		} else {

				// Compute the number of agents which move
				double N = RandP(state, 3*lambda*n); // Factor of 3 comes from 3D

				*n_u = RandP(state, N/6);
				*n_d = RandP(state, N/6);
				*n_l = RandP(state, N/6);
				*n_r = RandP(state, N/6);
				*n_f = RandP(state, N/6);
				*n_b = RandP(state, N/6);
				*n_0 = n - (*n_u + *n_d + *n_l + *n_r + *n_f + *n_b);
		}

		*n_u = round(*n_u);
		*n_d = round(*n_d);
		*n_l = round(*n_l);
		*n_r = round(*n_r);
		*n_f = round(*n_f);
		*n_b = round(*n_b);
		*n_0 = n - (*n_u + *n_d + *n_l + *n_r + *n_f + *n_b);

    // *n_u = 0.5*lambda*n;
    // *n_d = 0.5*lambda*n;
    // *n_l = 0.5*lambda*n;
    // *n_r = 0.5*lambda*n;
    // *n_f = 0.5*lambda*n;
    // *n_b = 0.5*lambda*n;
    // *n_0 = n - (*n_u + *n_d + *n_l + *n_r + *n_f + *n_b);

		// assert(*n_0 >= 0);
		// assert(*n_u >= 0);
		// assert(*n_d >= 0);
		// assert(*n_l >= 0);
		// assert(*n_r >= 0);
		// assert(*n_f >= 0);
		// assert(*n_b >= 0);
		// assert(fabs(n - (*n_0 + *n_u + *n_d + *n_l + *n_r + *n_f + *n_b)) < 1);

}

__global__ void ComputeDiffusionWeights(curandState* state, double* arr, double lambda, double* arr_n_0, double* arr_n_u, double* arr_n_d, double* arr_n_l, double* arr_n_r, double* arr_n_f, double* arr_n_b, int nGridXY, bool* arr_IsActive) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (!(arr_IsActive[i])){
    return;
  }

  double tmp = arr[i];
  double n_0 = 0.0;
  double n_u = 0.0;
  double n_d = 0.0;
  double n_l = 0.0;
  double n_r = 0.0;
  double n_f = 0.0;
  double n_b = 0.0;

  // DETERMINITIC CHANGE
  // if (n < 1) return;

  // Check if diffusion should occur
  if ((lambda == 0) or (nGridXY == 1)) {
      n_0 = tmp;
  } else {

  // DETERMINITIC CHANGE
  // if (lambda*n < 5) {   // Compute all movement individually

  // 		for (int l = 0; l < round(n); l++) {

  // 				double r = curand_uniform(&state);

  // 				if       (r <    lambda)                     (n_u)++;  // Up movement
  // 				else if ((r >=   lambda) and (r < 2*lambda)) (n_d)++;  // Down movement
  // 				else if ((r >= 2*lambda) and (r < 3*lambda)) (n_l)++;  // Left movement
  // 				else if ((r >= 3*lambda) and (r < 4*lambda)) (n_r)++;  // Right movement
  // 				else if ((r >= 4*lambda) and (r < 5*lambda)) (n_f)++;  // Forward movement
  // 				else if ((r >= 5*lambda) and (r < 6*lambda)) (n_b)++;  // Backward movement
  // 				else                                         (n_0)++;  // No movement

  // 		}


  // } else {

      // // Compute the number of agents which move
      // double N = RandP(state[i], 3*lambda*n); // Factor of 3 comes from 3D

      // n_u = RandP(state[i], N/6);
      // n_d = RandP(state[i], N/6);
      // n_l = RandP(state[i], N/6);
      // n_r = RandP(state[i], N/6);
      // n_f = RandP(state[i], N/6);
      // n_b = RandP(state[i], N/6);
      // n_0 = n - (n_u + n_d + n_l + n_r + n_f + n_b);
  // }

  // n_u = round(n_u);
  // n_d = round(n_d);
  // n_l = round(n_l);
  // n_r = round(n_r);
  // n_f = round(n_f);
  // n_b = round(n_b);
  // n_0 = n - (n_u + n_d + n_l + n_r + n_f + n_b);

  n_u = 0.5*lambda*tmp;
  n_d = 0.5*lambda*tmp;
  n_l = 0.5*lambda*tmp;
  n_r = 0.5*lambda*tmp;
  n_f = 0.5*lambda*tmp;
  n_b = 0.5*lambda*tmp;
  n_0 = tmp - (n_u + n_d + n_l + n_r + n_f + n_b);

  // assert(n_0 >= 0);
  // assert(n_u >= 0);
  // assert(n_d >= 0);
  // assert(n_l >= 0);
  // assert(n_r >= 0);
  // assert(n_f >= 0);
  // assert(n_b >= 0);
  // assert(fabs(n - (n_0 + n_u + n_d + n_l + n_r + n_f + n_b)) < 1);
  }

  // Write the weights
  arr_n_0[i] = n_0;
  arr_n_u[i] = n_u;
  arr_n_d[i] = n_d;
  arr_n_l[i] = n_l;
  arr_n_r[i] = n_r;
  arr_n_f[i] = n_f;
  arr_n_b[i] = n_b;

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

  if (!(arr_IsActive[i])){
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

__global__ void BurstingEvents(double* arr_I9, double* arr_P_new, double* arr_Occ, double* arr_GrowthModifier, double* arr_M, double* arr_p, double alpha, double beta, double r, double dT, bool reducedBeta, bool* Warn_r, curandState *rng_state, bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (!(arr_IsActive[i])){
    return;
  }

  // Fetch growthModifier
  double growthModifier = arr_GrowthModifier[i];
  if (reducedBeta) {
    beta *= growthModifier;
  }

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
  // DETERMINITIC CHANGE
  // arr_P_new[i] += round( (1 - alpha) * beta * N);
  // arr_M[i]     = round(alpha * beta * N);
  arr_P_new[i] += (1 - alpha) * beta * N;
  arr_M[i]     = alpha * beta * N;
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


    // If bacteria were hit, update events
    if (tmp + M >= 1) {

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

__global__ void ApplyMovement(double* arr_new,
                              double* arr_n_0,
                              double* arr_n_u,
                              double* arr_n_d,
                              double* arr_n_l,
                              double* arr_n_r,
                              double* arr_n_f,
                              double* arr_n_b,
                              int nGridZ,
                              int nGridXY,
                              bool experimentalConditions,
                              bool* arr_IsActive) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    // Skip empty sites
    if (!arr_IsActive[tid]){
        return;
    }

    int k = tid % nGridZ;
    int j = ( (tid - k) / nGridZ ) % nGridXY;
    int i = ( (tid - k) / nGridZ ) / nGridXY;

    int ip, jp, kp, im, jm, km;

    if (i + 1 >= nGridXY) ip = i + 1 - nGridXY;
    else ip = i + 1;

    if (i == 0) im = nGridXY - 1;
    else im = i - 1;

    if (j + 1 >= nGridXY) jp = j + 1 - nGridXY;
    else jp = j + 1;

    if (j == 0) jm = nGridXY - 1;
    else jm = j - 1;

    if (not experimentalConditions) {   // Periodic boundaries in Z direction

      if (k + 1 >= nGridZ) kp = k + 1 - nGridZ;
      else kp = k + 1;

      if (k == 0) km = nGridZ - 1;
      else km = k - 1;

    } else {    // Reflective boundaries in Z direction
      if (k + 1 >= nGridZ) kp = k - 1;
      else kp = k + 1;

      if (k == 0) km = k + 1;
      else km = k - 1;

    }

    // Update counts
    arr_new[tid] += arr_n_0[ i*nGridXY*nGridZ +  j*nGridZ + k ];
    arr_new[tid] += arr_n_u[ip*nGridXY*nGridZ +  j*nGridZ + k ];
    arr_new[tid] += arr_n_d[im*nGridXY*nGridZ +  j*nGridZ + k ];
    arr_new[tid] += arr_n_r[ i*nGridXY*nGridZ + jp*nGridZ + k ];
    arr_new[tid] += arr_n_l[ i*nGridXY*nGridZ + jm*nGridZ + k ];
    arr_new[tid] += arr_n_f[ i*nGridXY*nGridZ +  j*nGridZ + kp];
    arr_new[tid] += arr_n_b[ i*nGridXY*nGridZ +  j*nGridZ + km];

}


// first movement kernel (if nGridXY > 1)
__global__ void Movement1(curandState *rng_state,
                          double* arr,
                          double* arr_new,
                          bool* arr_IsActive,
                          int nGridZ,
                          int nGridXY,
                          bool experimentalConditions,
                          double lambda){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    // Skip empty sites
    if (!arr_IsActive[tid]){
        return;
    }

    int k = tid%nGridZ;
    int j = ((tid - k)/nGridZ)%nGridXY;
    int i = ((tid -k) /nGridZ)/nGridXY;

    int ip, jp, kp, im, jm, km;

    if (i + 1 >= nGridXY) ip = i + 1 - nGridXY;
    else ip = i + 1;

    if (i == 0) im = nGridXY - 1;
    else im = i - 1;

    if (j + 1 >= nGridXY) jp = j + 1 - nGridXY;
    else jp = j + 1;

    if (j == 0) jm = nGridXY - 1;
    else jm = j - 1;

    if (not experimentalConditions) {   // Periodic boundaries in Z direction

      if (k + 1 >= nGridZ) kp = k + 1 - nGridZ;
      else kp = k + 1;

      if (k == 0) km = nGridZ - 1;
      else km = k - 1;

    } else {    // Reflective boundaries in Z direction
      if (k + 1 >= nGridZ) kp = k - 1;
      else kp = k + 1;

      if (k == 0) km = k + 1;
      else km = k - 1;

    }

    // Update counts
    double n_0; // No movement
    double n_u; // Up
    double n_d; // Down
    double n_l; // Left
    double n_r; // Right
    double n_f; // Front
    double n_b; // Back

    ComputeDiffusion(rng_state[tid], arr[i*nGridXY*nGridZ + j*nGridZ + k], lambda, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, i, j, k, nGridXY);
        arr_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0;
        arr_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u;
        arr_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d;
        arr_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r;
        arr_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l;
        arr_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f;
        arr_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

}

__global__ void Movement2(double* arr,
                          double* arr_new,
                          bool* arr_IsActive){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    // CELLS
	// Skip empty sites
    if (!arr_IsActive[tid]){
        return;
    }

    arr_new[tid] += arr[tid];


}

///////////////////////////////
// Simple end of loop kernels.

__global__ void SwapArrays(double* arr1, double* arr2, int size){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid<size){
        double tmp;
        tmp = arr1[tid];
        arr1[tid] = arr2[tid];
        arr2[tid] = tmp;
    }

}

__global__ void ZeroArray(double* arr, int size){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid<size){
        arr[tid] = 0.0;
    }
}
__global__ void UpdateOccupancy(double* arr_Occ,
                                double* arr_B,
                                double* arr_I0,
                                double* arr_I1,
                                double* arr_I2,
                                double* arr_I3,
                                double* arr_I4,
                                double* arr_I5,
                                double* arr_I6,
                                double* arr_I7,
                                double* arr_I8,
                                double* arr_I9,
                                int vol){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid<vol){
        arr_Occ[tid] = arr_B[tid] + arr_I0[tid] + arr_I1[tid] + arr_I2[tid] + arr_I3[tid] + arr_I4[tid] + arr_I5[tid] + arr_I6[tid] + arr_I7[tid] + arr_I8[tid] + arr_I9[tid];
    }

}
__global__ void NutrientDiffusion(double* arr_nutrient,
                                  double* arr_nutrient_new,
                                  double alphaXY,
                                  double alphaZ,
                                  int nGridXY,
                                  int nGridZ,
                                  bool experimentalConditions,
                                  int vol) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if(tid < vol) {


    int k = tid%nGridZ;
    int j = ((tid - k)/nGridZ)%nGridXY;
    int i = ((tid -k) /nGridZ)/nGridXY;


    // Update positions
    int ip, jp, kp, im, jm, km;

    if (i + 1 >= nGridXY) ip = i + 1 - nGridXY;
    else ip = i + 1;

    if (i == 0) im = nGridXY - 1;
    else im = i - 1;

    if (j + 1 >= nGridXY) jp = j + 1 - nGridXY;
    else jp = j + 1;

    if (j == 0) jm = nGridXY - 1;
    else jm = j - 1;

    if (not experimentalConditions) {   // Periodic boundaries in Z direction

      if (k + 1 >= nGridZ) kp = k + 1 - nGridZ;
      else kp = k + 1;

      if (k == 0) km = nGridZ - 1;
      else km = k - 1;

    } else {    // Reflective boundaries in Z direction
      if (k + 1 >= nGridZ) kp = k - 1;
      else kp = k + 1;

      if (k == 0) km = k + 1;
      else km = k - 1;
    }

    double tmp = arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k];
    arr_nutrient_new[i*nGridXY*nGridZ + j*nGridZ + k]  += tmp - (4 * alphaXY + 2 * alphaZ) * tmp;
    arr_nutrient_new[ip*nGridXY*nGridZ + j*nGridZ + k] += alphaXY * tmp;
    arr_nutrient_new[im*nGridXY*nGridZ + j*nGridZ + k] += alphaXY * tmp;
    arr_nutrient_new[i*nGridXY*nGridZ + jp*nGridZ + k] += alphaXY * tmp;
    arr_nutrient_new[i*nGridXY*nGridZ + jm*nGridZ + k] += alphaXY * tmp;
    arr_nutrient_new[i*nGridXY*nGridZ + j*nGridZ + kp] += alphaZ  * tmp;
    arr_nutrient_new[i*nGridXY*nGridZ + j*nGridZ + km] += alphaZ  * tmp;
  }
}

#endif

