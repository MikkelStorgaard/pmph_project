// #include "colonies3D_helpers.cu"

#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS

__global__ void FirstKernel(double* arr_Occ, double* arr_nC, int N){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;

	bool outOfBounds = (id >= N);

	if (outOfBounds){
		return;
	}

	if (arr_Occ[id] < arr_nC[id]){
			arr_nC[id] = arr_Occ[id];
	}
}

__global__ void SetIsActive(double* arr_Occ, double* arr_nC, bool* arr_IsActive, int N){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;

	bool insideBounds = (id < N);

	double nC =  insideBounds ? arr_nC[id] : 0.0;
	double Occ = insideBounds ? arr_Occ[id] : 0.0;
	arr_IsActive[id] = insideBounds && (nC >= 1.0 || Occ >= 1.0);

}

__global__ void SecondKernel(double* arr_Occ, double* arr_nC, double* maxOcc,
														 bool* arr_IsActive, int N){

	extern __shared__ double shared[];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;

	int tid = threadIdx.x;

	shared[tid] = arr_IsActive[id] ? arr_Occ[id] : 0.0;

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
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;


	if (!(arr_IsActive[id])){
		return;
	}

	double p;
	double tmp;

	// Compute the growth modifier
	double growthModifier = arr_GrowthModifier[id];

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
	arr_p[id] = p;

	//tmp = ComputeEvents(arr_I9[i], p, 2, i);  // Bursting events
	tmp = 1.0;
	// Update count
	arr_I9[id]    = max(0.0, arr_I9[id] - tmp);
	arr_Occ[id]   = max(0.0, arr_Occ[id] - tmp);
	arr_P_new[id] += round( (1 - alpha) * Beta * tmp);  // Phages which escape the colony
	arr_M[id] = round(alpha * Beta * tmp); // Phages which reinfect the colony
}

//Kernel 3.1: Birth
__global__ void ThirdKernel(bool* arr_IsActive, double* arr_GrowthModifier, double* arr_nutrient, double K, double g, double dT, bool* warn_g){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;


	// Skip empty sites
	if (!(arr_IsActive[id])) return;

	double p = 0;

	// Compute the growth modifier
	double growthModifier = arr_nutrient[id] / (arr_nutrient[id] + K);
	arr_GrowthModifier[id] = growthModifier;
	p = g * growthModifier*dT;

	if (arr_nutrient[id] < 1) { p = 0; }

	if ((p > 0.1) and (!*warn_g)) { *warn_g = true; }

}
//Kernel 3.2 Birth 2

__global__ void ThirdTwoKernel(bool* arr_IsActive, double* arr_nutrient, double* arr_B_new, bool* warn_fastGrowth){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;


	// Skip empty sites
	if (!(arr_IsActive[id])) return;

	double N = 0;

	// TODO: Proper compute events
	//N = ComputeEvents(arr_B[i*nGridXY*nGridZ + j*nGridZ + k], p, 1, i, j, k);
	N = 1;

	// Ensure there is enough nutrient
	if ( N > arr_nutrient[id] ) {
			if (!*warn_fastGrowth) { *warn_fastGrowth = true;    }
			N = round( arr_nutrient[id] );
	}

	// Update count
	arr_B_new[id] += N;
	arr_nutrient[id] = max(0.0, arr_nutrient[id] - N);
}



__global__ void NonBurstingEventsKernel(double* arr_A, double* arr_B, double* arr_p, bool* arr_IsActive){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;


	if (!(arr_IsActive[id])){
		return;
	}

	double tmp;
	double A = arr_A[id];
	double p = arr_p[id];

	// TODO: FIX ComputeEvents
	// tmp = ComputeEvents(A, p, 2, i);
	tmp = 1.0;
	arr_A[id] = max(0.0, A - tmp);
	arr_B[id] += tmp;
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

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;

	bool isInactive = (!(arr_IsActive[id]));
	if (isInactive){
		return;
	}

	double B = arr_B[id];
	double nC = arr_nC[id];
	double Occ = arr_Occ[id];
	double P = arr_P[id];
	double M = arr_M[id];
	double tmp;


	// Compute the growth modifier
	double growthModifier = arr_GrowthModifier[id];

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
			//tmp = ComputeEvents(P, p, 4, tid);           // Number of targets hit //
			tmp = 1;
			// TODO: replace ComputeEvents with something that works
			/* ComputeEvents used to be (..., i, j, k), but in this flat kernel,
				 tid is equal to i * j * k */
		}

		if (tmp + M >= 1) {
			// If bacteria were hit, update events
			arr_P[id] = max(0.0, P - tmp); // Update count

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
			arr_B[id] = max(0.0, B - tmp);
			if (r > 0.0) {
				arr_I0_new[id] += tmp;
			} else {
				arr_P_new[id] += tmp * (1 - alpha) * Beta;
			}
		}
	}
}

/*
__global__ void SixthKernel(double* arr_P, double p, bool *warn_delta, int numberOfElements){
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if(i >= numberOfElements) return;

		// TODO: skip empty sites or not?? (Not included here)
		double N = 0;
 // TODO: figure out a shared variable warn
 //   if ((p > 0.1) and (!warn_delta)) &warn_delta = true;


		// TODO: do proper Compute events:
		N = ComputeEvents(arr_P[i], p, 5, i);

		arr_P[i]    = max(0.0, arr_P[i] - N);

}
*/
__global__ void SeventhKernel(){

// Phage decay

// Compute p
// Compute n

// Update P

}

__global__ void MovementKernel(double* arr, bool* arr_IsActive, int nGridXY, int nGridZ, int N){

	// Alocate shared memory
	extern __shared__ double shared[blockDim.x+2][blockDim.y+2][blockDim.z+2]();

	// Get index of thread
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	int id = i*gridDim.x*gridDim.y + j*gridDim.y + k;

	// Skip sites outside
	if (id < N) {

		// Copy the values into shared
		shared[i][j][k] = arr[i*gridDim.x*gridDim.y + j*gridDim.y + k];

		// Border points copy the extra values
		// X Border
		if (threadIdx.x == 0) && (i == 0) {	// Periodic copy
			shared[i-1][j][k] = arr[nGridXY*gridDim.x*gridDim.y + j*gridDim.y + k]; // Copy from the other side
		} else if (threadIdx.x == 0) { // Simple copy
			shared[i-1][j][k] = arr[(i-1)*gridDim.x*gridDim.y + j*gridDim.y + k]; // Copy from the left
		}

		if (i == nGridXY) { // Periodic copy
			shared[i+1][j][k] = arr[j*gridDim.y + k]; // Copy from the other side
		} else if (threadIdx.x == blockDim.x) { // Simple copy
			shared[i+1][j][k] = arr[(i+1)*gridDim.x*gridDim.y + j*gridDim.y + k]; // Copy from the right
		}

		// Y Border
		if (threadIdx.y == 0) && (j == 0) {	// Periodic copy
			shared[i][j-1][k] = arr[i*gridDim.x*gridDim.y + nGridXY*gridDim.y + k]; // Copy from the other side
		} else if (threadIdx.y == 0) { // Simple copy
			shared[i][j-1][k] = arr[(i*gridDim.x*gridDim.y + (j-1)*gridDim.y + k]; // Copy from the left
		}

		if (j == nGridXY) {	// Periodic copy
			shared[i][j+1][k] = arr[i*gridDim.x*gridDim.y + k]; // Copy from the other side
		} else if (threadIdx.y == blockDim.y) // Simple copy
			shared[i][j+1][k] = arr[(i*gridDim.x*gridDim.y + (j+1)*gridDim.y + k]; // Copy from the right
		}

		// Z Border
		if (threadIdx.z == 0) && (k == 0) {	// Periodic copy
			shared[i][j][k-1] = arr[i*gridDim.x*gridDim.y + j*gridDim.y + nGridZ]; // Copy from the other side
		} else if (threadIdx.z == 0) { // Simple copy
			shared[i][j][k-1] = arr[i*gridDim.x*gridDim.y + j*gridDim.y + (k-1)]; // Copy from the left
		}

		if (k == nGridZ) {	// Periodic copy
			shared[i][j][k+1] = arr[i*gridDim.x*gridDim.y + j*gridDim.y]; // Copy from the other side
		} else if (threadIdx.z == blockDim.z) // Simple copy
			shared[i][j][k+1] = arr[i*gridDim.x*gridDim.y + j*gridDim.y + (k+1)]; // Copy from the right
		}
	}

	// Make sure all data has been copied
	__syncthreads();







}

#endif

