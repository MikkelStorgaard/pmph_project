inline __device__ numtype gpu_round(numtype x){
 #if NUMTYPE_IS_FLOAT
  return roundf(x);
 #else
  return round(x);
 #endif
}

inline __device__ numtype gpu_exp(numtype x){
 #if NUMTYPE_IS_FLOAT
  return expf(x);
 #else
  return exp(x);
 #endif
}

inline __device__ numtype gpu_pow(numtype x, numtype y){
 #if NUMTYPE_IS_FLOAT
  return powf(x,y);
 #else
  return pow(x,y);
 #endif
}

inline __device__ numtype gpu_max(numtype x, numtype y){
 #if NUMTYPE_IS_FLOAT
  return fmaxf(x,y);
 #else
  return fmax(x,y);
 #endif
}

__device__ numtype RandP(curandState rng_state, numtype lambda) {

  numtype L = gpu_exp(-lambda);
  numtype p = 1.0;
  numtype k = 0;
  while (p > L) {
    k++;
    #if NUMTYPE_IS_FLOAT
        numtype u = curand_uniform(&rng_state);
    #else
        numtype u = curand_uniform_double(&rng_state);
    #endif
    p *= u;
  }
  return k - 1;
}

__global__ void ComputeEvents_seq(numtype *N, numtype n, numtype p, curandState *rng_state, int index){
    // Trivial cases
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i == index) {

      if (p == 1) {
        *N = n;
      } else if (p == 0) {
        *N = 0.0;
      } else if (n < 1) {
        *N = 0.0;
      } else {
        *N = gpu_round(RandP(rng_state[index], n*p));
        // *N = n*min(1.0,p);
      }
    }
}

__device__ numtype ComputeEvents(numtype n, numtype p, curandState rng_state){
    // Trivial cases

    // if (p >= 1) return n;
    // if (p == 0) return 0.0;

		// DETERMINITIC CHANGE
		// if (n < 1)  return 0.0;

		// return gpu_round(RandP(rng_state, n*p));
		return n*min(1.0,p);

}

__global__ void initRNG(curandState *state, int N){

  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < N) {
    curand_init(i, 0, 0, &state[i]);
  }
}

__global__ void ComputeDiffusionWeights(curandState* state, numtype* arr, numtype lambda, numtype* arr_n_0, numtype* arr_n_u, numtype* arr_n_d, numtype* arr_n_l, numtype* arr_n_r, numtype* arr_n_f, numtype* arr_n_b, int nGridXY, bool* arr_IsActive, int totalElements) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;


  if (!(arr_IsActive[i])){

    // If thread is inside of bounds, reset the arrays, the return.
    if (i < totalElements){
      arr_n_0[i] = 0.0;
      arr_n_u[i] = 0.0;
      arr_n_d[i] = 0.0;
      arr_n_l[i] = 0.0;
      arr_n_r[i] = 0.0;
      arr_n_f[i] = 0.0;
      arr_n_b[i] = 0.0;
    }

    return;
  }

  numtype tmp = arr[i];
  numtype n_0 = 0.0;
  numtype n_u = 0.0;
  numtype n_d = 0.0;
  numtype n_l = 0.0;
  numtype n_r = 0.0;
  numtype n_f = 0.0;
  numtype n_b = 0.0;

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

__global__ void FirstKernel(numtype* arr_Occ, numtype* arr_nC, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  // Out of bounds check
  if (i >= N){
    return;
  }

  if (arr_Occ[i] < arr_nC[i]) {
    arr_nC[i] = arr_Occ[i];
  }
  // arr_nC[i] = min(arr_nC[i],arr_Occ[i]);
}

__global__ void SetIsActive(numtype* arr_Occ, numtype* arr_P, bool* arr_IsActive, int N){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  bool insideBounds = (i < N);

  numtype Occ = insideBounds ? arr_Occ[i] : 0.0;
  numtype P   = insideBounds ? arr_P[i]   : 0.0;
  arr_IsActive[i] = insideBounds && ((P >= 1.0) || (Occ >= 1.0));

}

__global__ void SecondKernel(numtype* arr_Occ, numtype* arr_nC, numtype* maxOcc,
                             bool* arr_IsActive, int N){

  extern __shared__ numtype shared[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  shared[tid] = arr_IsActive[i] ? arr_Occ[i] : 0.0;
  __syncthreads();

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

__global__ void MaxOccupancyOpt(numtype* arr_Occ, bool* Warn_density, numtype threshold, bool* arr_IsActive){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ((!(arr_IsActive[i])) || *Warn_density){
        return;
    }
    if(arr_Occ[i] > threshold) *Warn_density = true;
}


__global__ void PartialMax(numtype* arr, numtype* partialMax, int N){

  extern __shared__ numtype shared[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (i < N) {
    shared[tid] = arr[i];
  } else {
    shared[tid] = 0.0;
  }
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      shared[tid] = gpu_max(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if(tid == 0){
    partialMax[blockIdx.x] = shared[0];
  }
}

__global__ void SequentialReduceMax(numtype* A, int A_len){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > 0){
    return;
  }
  numtype tmp = 0.0;
  numtype current_max = 0.0;

  // the little thread that could
  for (unsigned int ind=0; ind<A_len; ind++) {
    tmp = A[ind];
    if(tmp > current_max){
      current_max = tmp;
    }
  }
  A[0] = current_max;
}

__global__ void ComputeBirthEvents(numtype* arr_B, numtype* arr_B_new, numtype* arr_nutrient, numtype* arr_GrowthModifier, numtype K, numtype g, numtype dT, bool* Warn_g, bool* Warn_fastGrowth, curandState *rng_state, bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (!(arr_IsActive[i])){
    return;
  }

  // Compute the growth modifier
  numtype growthModifier = arr_nutrient[i] / (arr_nutrient[i] + K);
  arr_GrowthModifier[i] = growthModifier;

  // Compute birth probability
  numtype p = g * growthModifier * dT;

  if (arr_nutrient[i] < 1) {
    p = 0;
  }

  // Produce warning
  if ((p > 0.1) and (!(*Warn_g))){
    *Warn_g = true;
  }

  // Compute the number of births
  numtype N = ComputeEvents(arr_B[i], p, rng_state[i]);

  // Ensure there is enough nutrient
	if ( N > arr_nutrient[i] ) {

    if (!(*Warn_fastGrowth)){
      *Warn_fastGrowth = true;
    }

    // N = round( arr_nutrient[i] );
    N = arr_nutrient[i];
  }

  // Update count
  arr_B_new[i] = N;
  arr_nutrient[i] = gpu_max(0.0, arr_nutrient[i] - N);

}

__global__ void BurstingEvents(numtype* arr_I9, numtype* arr_P_new, numtype* arr_Occ, numtype* arr_GrowthModifier, numtype* arr_M, numtype* arr_p, numtype alpha, numtype beta, numtype r, numtype dT, bool reducedBeta, bool* Warn_r, curandState *rng_state, bool* arr_IsActive, int totalElements){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (!(arr_IsActive[i])){
	  if(i < totalElements) arr_P_new[i] = 0.0;
    return;
  }

  // Fetch growthModifier
  numtype growthModifier = arr_GrowthModifier[i];
  if (reducedBeta) {
    beta *= growthModifier;
  }

  // Compute infection increse probability
  numtype p = r * growthModifier *dT;

  // Produce warning
  if ((p > 0.25) and (!(*Warn_r))){
    *Warn_r = true;
  }

  // Compute the number of bursts
  numtype N = ComputeEvents(arr_I9[i], p, rng_state[i]);

  // Update count
  arr_I9[i]    = gpu_max(0.0, arr_I9[i] - N);
  arr_Occ[i]   = gpu_max(0.0, arr_Occ[i] - N);
  // DETERMINITIC CHANGE
  // arr_P_new[i] += round( (1 - alpha) * beta * N);
  // arr_M[i]     = round(alpha * beta * N);
  arr_P_new[i] = (1 - alpha) * beta * N;
  arr_M[i]     = alpha * beta * N;
  arr_p[i]     = p;
}

__global__ void NonBurstingEvents(numtype* arr_I, numtype* arr_In, numtype* arr_p, curandState *rng_state, bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  // Out of bounds check
  if (!arr_IsActive[i]){
    return;
  }

  // Compute the number of bursts
  numtype N = ComputeEvents(arr_I[i], arr_p[i], rng_state[i]);

  // Update count
  arr_I[i]     = gpu_max(0.0, arr_I[i] - N);
  arr_In[i]    += N;
}

__global__ void NewInfectionsKernel(numtype* arr_Occ,
                                    numtype* arr_nC,
                                    numtype* arr_P,
                                    numtype* arr_P_new,
                                    numtype* arr_GrowthModifier,
                                    numtype* arr_B,
                                    numtype* arr_M,
                                    numtype* arr_I0_new,
                                    bool* arr_IsActive,
                                    bool reducedBeta,
                                    bool clustering,
                                    bool shielding,
                                    numtype K,
                                    numtype alpha,
                                    numtype beta,
                                    numtype eta,
                                    numtype zeta,
                                    numtype dT,
                                    numtype r,
                                    curandState* rng_state,
									                  int totalElements
                                    ){

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  bool isInactive = (!(arr_IsActive[tid]));
  if (isInactive){
	  if (tid < totalElements)  arr_I0_new[tid] = 0.0;
    return;
  }

  numtype B = arr_B[tid];
  numtype nC = arr_nC[tid];
  numtype Occ = arr_Occ[tid];
  numtype P = arr_P[tid];
  numtype M = arr_M[tid];
  numtype tmp;


	// Compute the growth modifier
	numtype growthModifier = arr_GrowthModifier[tid];

  // Compute beta
  numtype Beta = beta;
  if (reducedBeta) {
    Beta *= growthModifier;
  }

  numtype p;
  // numtype s;
  // numtype n;

  // KERNEL THIS
  if ((Occ >= 1) && (P >= 1)) {
    // if (clustering) {   // Check if clustering is enabled
    //   s = gpu_pow(Occ / nC, 1.0 / 3.0);
    //   n = nC;
    // } else {            // Else use mean field computation
    //   s = 1.0;
    //   n = Occ;
    // }

    // Compute the number of hits
    // if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
      tmp = P;
    // } else {
      // p = 1 - gpu_pow(1 - eta * s * dT, n);        // Probability hitting any target
      // tmp = ComputeEvents(P, p, rng_state[tid]);           // Number of targets hit //
    // }


    // If bacteria were hit, update events
    // if (tmp + M >= 1) {

      arr_P[tid] = gpu_max(0.0, P - tmp); // Update count

      numtype S;

      if (shielding) {
        // Absorbing medium model
        numtype d = gpu_pow(Occ / nC, 1.0 / 3.0) - gpu_pow(B / nC, 1.0 / 3.0);
        S = gpu_exp(-zeta * d); // Probability of hitting succebtible target

      } else {
        // Well mixed model
        S = B / Occ;
      }

      p = gpu_max(0.0, min(B / Occ, S)); // Probability of hitting succebtible target

      tmp = ComputeEvents(tmp + M, p, rng_state[tid]); // Number of targets hit
      tmp = min(tmp, B); // If more bacteria than present are set to be infeced, round down

      // Update the counts
      arr_B[tid] = gpu_max(0.0, B - tmp);
      if (r > 0.0) {
        arr_I0_new[tid] = tmp;
      } else {
        arr_P_new[tid] += tmp * (1 - alpha) * Beta;
      }
    // }
  }
  else arr_I0_new[tid] = 0.0;
}

__global__ void PhageDecay(numtype* arr_P, numtype p,
                           bool *warn_delta, curandState* rng_state,
                           bool* arr_IsActive){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (!(arr_IsActive[i])){
    return;
  }

 numtype N = ComputeEvents(arr_P[i], p, rng_state[i]);

 if ((p > 0.1) && (!(*warn_delta))){
   *warn_delta = true;
 }

 arr_P[i] = gpu_max(0.0, arr_P[i] - N);
}

__global__ void ApplyMovement(numtype* arr_new,
                              numtype lambda,
                              numtype* arr_n_0,
                              numtype* arr_n_u,
                              numtype* arr_n_d,
                              numtype* arr_n_l,
                              numtype* arr_n_r,
                              numtype* arr_n_f,
                              numtype* arr_n_b,
                              int nGridZ,
                              int nGridXY,
                              bool experimentalConditions,
                              bool* arr_IsActive,
							                bool zero,
							                int totalElements) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    // Skip empty sites
    if (!arr_IsActive[tid]){
		    if (tid < totalElements)  arr_new[tid] = 0.0;
        return;
    }

    if (lambda > 0) {
      int k = tid % nGridZ;
      int j = ( (tid - k) / nGridZ ) % nGridXY;
      int i = (((tid - k) / nGridZ ) - j) / nGridXY;

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
      numtype tmp;
      if(zero) tmp = 0.0;
      else tmp = arr_new[tid];
      tmp += arr_n_0[ i*nGridXY*nGridZ +  j*nGridZ + k ];
      tmp += arr_n_u[ip*nGridXY*nGridZ +  j*nGridZ + k ];
      tmp += arr_n_d[im*nGridXY*nGridZ +  j*nGridZ + k ];
      tmp += arr_n_r[ i*nGridXY*nGridZ + jp*nGridZ + k ];
      tmp += arr_n_l[ i*nGridXY*nGridZ + jm*nGridZ + k ];
      tmp += arr_n_f[ i*nGridXY*nGridZ +  j*nGridZ + kp];
      tmp += arr_n_b[ i*nGridXY*nGridZ +  j*nGridZ + km];
      arr_new[tid] = tmp;

    } else {

      if(zero) arr_new[tid] = arr_n_0[tid];
      else arr_new[tid] += arr_n_0[tid];
    }

}

///////////////////////////////
// Simple end of loop kernels.

__global__ void SwapArrays(numtype* arr1, numtype* arr2, int size){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid<size){
        numtype tmp;
        tmp = arr1[tid];
        arr1[tid] = arr2[tid];
        arr2[tid] = tmp;
    }

}

__global__ void ZeroArray(numtype* arr, int size){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid<size){
        arr[tid] = 0.0;
    }
}

__global__ void UpdateOccupancy(numtype* arr_Occ,
                                numtype* arr_B,
                                numtype* arr_I0,
                                numtype* arr_I1,
                                numtype* arr_I2,
                                numtype* arr_I3,
                                numtype* arr_I4,
                                numtype* arr_I5,
                                numtype* arr_I6,
                                numtype* arr_I7,
                                numtype* arr_I8,
                                numtype* arr_I9,
                                int vol){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid<vol){
        arr_Occ[tid] = arr_B[tid] + arr_I0[tid] + arr_I1[tid] + arr_I2[tid] + arr_I3[tid] + arr_I4[tid] + arr_I5[tid] + arr_I6[tid] + arr_I7[tid] + arr_I8[tid] + arr_I9[tid];
    }

}

__global__ void NutrientDiffusion(numtype* arr_nutrient,
                                  numtype* arr_nutrient_new,
                                  numtype alphaXY,
                                  numtype alphaZ,
                                  int nGridXY,
                                  int nGridZ,
                                  bool experimentalConditions,
                                  int vol) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid < vol) {

    int k = tid % nGridZ;
    int j = ( (tid - k) / nGridZ ) % nGridXY;
    int i = (((tid - k) / nGridZ ) - j) / nGridXY;

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



    // Writing order has been reversed compared with non-CUDA version
    #if GPU_COPY_TO_SHARED

      // // 1D abstraction
      // int T = blockDim.x+2; // Tile length
      // int memSize = 5*T;    // Number of tiles

      // extern __shared__ numtype shared[memSize];
      extern __shared__ numtype shared[];

      int thr = threadIdx.x;

      // Copy to shared
      // Compute the neighbour indicies
      int ind_0 =  i*nGridXY*nGridZ +  j*nGridZ + k ;
      int ind_u = ip*nGridXY*nGridZ +  j*nGridZ + k ;
      int ind_d = im*nGridXY*nGridZ +  j*nGridZ + k ;
      int ind_r =  i*nGridXY*nGridZ + jp*nGridZ + k ;
      int ind_l =  i*nGridXY*nGridZ + jm*nGridZ + k ;
      int ind_f =  i*nGridXY*nGridZ +  j*nGridZ + kp;
      int ind_b =  i*nGridXY*nGridZ +  j*nGridZ + km;

      // Copy valies
      shared[thr + 1]       = arr_nutrient[ind_0]; // Own value
      shared[thr + 1 +   T] = arr_nutrient[ind_u]; // Upper neighbour
      shared[thr + 1 + 2*T] = arr_nutrient[ind_d]; // Upper neighbour
      shared[thr + 1 + 3*T] = arr_nutrient[ind_r]; // Upper neighbour
      shared[thr + 1 + 4*T] = arr_nutrient[ind_l]; // Upper neighbour
      if (tid == 0)           shared[thr]   = arr_nutrient[ind_b];
      if (tid == blockDim.x)  shared[thr+1] = arr_nutrient[ind_f];
      __syncthreads();

      numtype inflow = 0.0;
      inflow += alphaXY * shared[thr + 1 + T];
      inflow += alphaXY * shared[thr + 1 + 2*T];
      inflow += alphaXY * shared[thr + 1 + 3*T];
      inflow += alphaXY * shared[thr + 1 + 4*T];
      inflow += alphaZ  * shared[thr + 2];
      inflow += alphaZ  * shared[thr];

      numtype tmp = shared[thr + 1];
      numtype outflow = (4 * alphaXY + 2 * alphaZ) * tmp;

      arr_nutrient_new[i *nGridXY*nGridZ + j *nGridZ + k] = tmp + inflow - outflow;

    #else

      numtype inflow = 0.0;
      inflow += alphaXY * arr_nutrient[ip*nGridXY*nGridZ + j *nGridZ + k ];
      inflow += alphaXY * arr_nutrient[im*nGridXY*nGridZ + j *nGridZ + k ];
      inflow += alphaXY * arr_nutrient[i *nGridXY*nGridZ + jp*nGridZ + k ];
      inflow += alphaXY * arr_nutrient[i *nGridXY*nGridZ + jm*nGridZ + k ];
      inflow += alphaZ  * arr_nutrient[i *nGridXY*nGridZ + j *nGridZ + kp];
      inflow += alphaZ  * arr_nutrient[i *nGridXY*nGridZ + j *nGridZ + km];

      numtype tmp = arr_nutrient[i *nGridXY*nGridZ + j *nGridZ + k];
      numtype outflow = (4 * alphaXY + 2 * alphaZ) * tmp;

      arr_nutrient_new[i *nGridXY*nGridZ + j *nGridZ + k] = tmp + inflow - outflow;

    #endif
  }
}

__global__ void PartialSum(numtype* arr, numtype* partialSum, int N){

  extern __shared__ numtype shared[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (i < N) {
    shared[tid] = arr[i];
  } else {
    shared[tid] = 0.0;
  }
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      shared[tid] = shared[tid] + shared[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if(tid == 0){
    partialSum[blockIdx.x] = shared[0];
  }
}

__global__ void SequentialReduceSum(numtype* A, int A_len){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > 0){
    return;
  }
  numtype tmp = 0.0;

  // the little thread that could
  for (unsigned int ind=0; ind<A_len; ind++) {
    tmp += A[ind];
  }
  A[0] = tmp;
}

__global__ void PartialNonZero(numtype* arr, numtype* PartialNonZero, int N){

  extern __shared__ numtype shared[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (i < N) {
    shared[tid] = (arr[i] > 0) ? 1.0 : 0.0;
  } else {
    shared[tid] = 0.0;
  }
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      shared[tid] = shared[tid] + shared[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if(tid == 0){
    PartialNonZero[blockIdx.x] = shared[0];
  }
}
