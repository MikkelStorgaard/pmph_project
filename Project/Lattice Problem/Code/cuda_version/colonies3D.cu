#include "colonies3D.hpp"
#include "colonies3D_kernels.cu.h"
// #include "colonies3D_helpers.cu"

#define GPU_NC true
#define GPU_MAXOCCUPANCY true
#define GPU_BIRTH true
#define GPU_INFECTIONS true
#define GPU_NEWINFECTIONS false
#define GPU_PHAGEDECAY false
#define GPU_MOVEMENT false


using namespace std;

// Constructers /////////////////////////////////////////////////////////////////////////
// Direct constructer
Colonies3D::Colonies3D(double B_0, double P_0){

	// Store the initial densities
	this->B_0 = B_0;
	this->P_0 = P_0;

	// Set some default parameters (initlize some default objects)
	K                       = 1.0 / 5.0;//          Half-Speed constant
	n_0                     = 1e9;      // [1/ml]   Initial nutrient level (Carrying capacity per ml)

	L                       = 1e4;      // [µm]     Side-length of simulation array
	H                       = L;        // [µm]     Height of the simulation array
	nGridXY                 = 50;       //          Number of gridpoints
	nGridZ                  = nGridXY;  //          Number of gridpoints

	nSamp                   = 10;       //          Number of samples to save per simulation hour

	g                       = 2;        // [1/h]    Doubling rate for the cells

	alpha                   = 0.5;      //          Percentage of phages which reinfect the colony upon lysis
	beta                    = 100;      //          Multiplication factor phage
	eta                     = 1e4;      // [µm^3/h] Adsorption coefficient
	delta                   = 1.0/10.0; // [1/h]    Rate of phage decay
	r                       = 10.0/0.5; //          Constant used in the time-delay mechanism
	zeta                    = 1.0;      //          permeability of colony surface

	D_P                     = 1e4;      // [µm^2/h] Diffusion constant for the phage
	D_B                     = 0;//D_P/20;   // [µm^2/h] Diffusion constant for the cells
	D_n                     = 25e5;     // [µm^2/h] Diffusion constant for the nutrient

	T                       = 0;        // [h]      Current time
	dT                      = -1;       // [h]      Time-step size (-1 to compute based on fastest diffusion rate)
	T_end                   = 0;        // [h]      End time of simulation
	T_i                     = -1;       // [h]      Time when the phage infections begins (less than 0 disables phage infection)

	initialOccupancy        = 0;        // Number of gridpoints occupied initially;

	exit                    = false;    // Boolean to control early exit

	Warn_g                  = false;    //
	Warn_r                  = false;    //
	Warn_eta                = false;    // Booleans to keep track of warnings
	Warn_delta              = false;    //
	Warn_density            = false;    //
	Warn_fastGrowth         = false;    //

	experimentalConditions  = false;    // Booleans to control simulation type

	clustering              = true;     // When false, the ((B+I)/nC)^(1/3) factor is removed.
	shielding               = true;     // When true the simulation uses the shielding function (full model)
	reducedBeta             = false;    // When true the simulation modifies the burst size by the growthfactor

	reducedBoundary         = false;    // When true, bacteria are spawned at X = 0 and Y = 0. And phages are only spawned within nGrid boxes from (0,0,z).
	s                       = 1;

	fastExit                = false;     // Stop simulation when all cells are dead

	exportAll               = false;    // Boolean to export everything, not just populationsize

	rngSeed                 = -1;       // Random number seed  ( set to -1 if unused )



};

///////////////////////////////////////////////////////////////////////
// CPU Loop start
//////////////////////////////////////////////////////////////////////


int Colonies3D::Run_LoopDistributed_CPU(double T_end) {
	std::string filename_suffix = "loopDistributedCPU";

	this->T_end = T_end;

	// Get start time
	time_t  tic;
	time(&tic);

	// Generate a path
	path = GeneratePath();

	// Initilize the simulation matrices
	Initialize();

	// Export data
	ExportData_arr(T,filename_suffix);

	// Determine the number of samples to take
	int nSamplings = nSamp*T_end;

	// Loop over samplings
	for (int n = 0; n < nSamplings; n++) {
		if (exit) break;

		// Determine the number of timesteps between sampings
		int nStepsPerSample = static_cast<int>(round(1 / (nSamp *  dT)));

		for (int t = 0; t < nStepsPerSample; t++) {
			if (exit) break;

			// Increase time
			T += dT;

			// Spawn phages
			if ((T_i >= 0) and (abs(T - T_i) < dT / 2)) {
				spawnPhages();
				T_i = -1;
			}

			// Reset density counter
			double maxOccupancy = 0.0;

			/////////////////////////////////////////////////////
			// Main loop start //////////////////////////////////
			/////////////////////////////////////////////////////

			// Kernel 1-2: nC update and maxOccupancy //////////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Ensure nC is updated
						if (arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < arr_nC[i*nGridXY*nGridZ + j*nGridZ + k]){
								arr_nC[i*nGridXY*nGridZ + j*nGridZ + k] = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
						}
					}
				}
			}

			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

						// Record the maximum observed density
						if (arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] > maxOccupancy) maxOccupancy = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];

					}
				}
			}
			// Kernel 3: Birth //////////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

						double p = 0; // privatize
						double N = 0; // privatize

						// Compute the growth modifier
						double growthModifier = arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] / (arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] + K);
						arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k] = growthModifier;

						p = g * growthModifier*dT;
						if (arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] < 1) {
							p = 0;
						}

						if ((p > 0.1) and (!Warn_g)) {
							cout << "\tWarning: Birth Probability Large!" << "\n";
							f_log  << "Warning: Birth Probability Large!" << "\n";
							Warn_g = true;
						}

						/* BEGIN anden Map-kernel */
						N = ComputeEvents(arr_B[i*nGridXY*nGridZ + j*nGridZ + k], p, 1, i, j, k);
						// Ensure there is enough nutrient
						if ( N > arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] ) {
								if (!Warn_fastGrowth) {
										cout << "\tWarning: Colonies growing too fast!" << "\n";
										f_log  << "Warning: Colonies growing too fast!" << "\n";
										Warn_fastGrowth = true;
								}

								N = round( arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] );
						}

						// Update count
						arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += N;
						arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] - N);
						/* END anden Map-kernel */
					}
				}
			}

			// Kernel 4: Increase Infections ////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

						double p = 0; // privatize
						double N = 0; // privatize

						// Compute the growth modifier
						double growthModifier = arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k];

						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= growthModifier;
						}

				 		if (r > 0.0) {
							/* BEGIN tredje Map-kernel */

							p = r*growthModifier*dT;
							if ((p > 0.25) and (!Warn_r)) {
								cout << "\tWarning: Infection Increase Probability Large!" << "\n";
								f_log  << "Warning: Infection Increase Probability Large!" << "\n";
								Warn_r = true;
							}
							N = ComputeEvents(arr_I9[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);  // Bursting events

							// Update count
							arr_I9[i*nGridXY*nGridZ + j*nGridZ + k]    = max(0.0, arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k]   = max(0.0, arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += round( (1 - alpha) * Beta * N);   // Phages which escape the colony
							arr_M[i*nGridXY*nGridZ + j*nGridZ + k] = round(alpha * Beta * N);                        // Phages which reinfect the colony

							// Non-bursting events
							N = ComputeEvents(arr_I8[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I7[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I6[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I5[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I4[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I3[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I2[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I1[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = ComputeEvents(arr_I0[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
							arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							/* END tredje Map-kernel */
						}
					}
				}
			}

			// Kernel 5: New infections ///////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


						double p = 0; // privatize
						double N = 0; // privatize
												// double M = 0; // privatize

						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k];
						}

						// PRIVATIZE BOTH OF THESE
						double s;   // The factor which modifies the adsorption rate
						double n;   // The number of targets the phage has
												// Infectons


												// KERNEL THIS
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] >= 1)) {
							if (clustering) {   // Check if clustering is enabled
								s = pow(arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0);
								n = arr_nC[i*nGridXY*nGridZ + j*nGridZ + k];
							} else {            // Else use mean field computation
								s = 1.0;
								n = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
							}
						}

						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] >= 1)) {
							// Compute the number of hits
							if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
								N = arr_P[i*nGridXY*nGridZ + j*nGridZ + k];
							} else {
								p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
								N = ComputeEvents(arr_P[i*nGridXY*nGridZ + j*nGridZ + k], p, 4, i, j, k);     // Number of targets hit
							}

							if (N + arr_M[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) {
								// If bacteria were hit, update events
								arr_P[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_P[i*nGridXY*nGridZ + j*nGridZ + k] - N);     // Update count

								double S;
								if (shielding) {
									// Absorbing medium model
									double d = pow(arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0) -
										pow(arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0);
									S = exp(-zeta * d); // Probability of hitting succebtible target

								} else {
									// Well mixed model
									S = arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
								}

								p = max(0.0, min(arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k],
																	S)); // Probability of hitting succebtible target
								N = ComputeEvents(N + arr_M[i*nGridXY*nGridZ + j*nGridZ + k], p, 4, i, j, k);                  // Number of targets hit

								if (N > arr_B[i*nGridXY*nGridZ + j*nGridZ + k])
									N = arr_B[i*nGridXY*nGridZ + j*nGridZ + k];              // If more bacteria than present are set to be infeced, round down

								// Update the counts
								arr_B[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_B[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								if (r > 0.0) {
									arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += N;
								} else {
									arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += N * (1 - alpha) * Beta;
								}
							}
						}
 					}
				}
			}

			// Kernel 6: Phage decay ///////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


						double p = 0; // privatize
						double N = 0; // privatize

						// KERNEL BEGIN
						p = delta*dT;
						if ((p > 0.1) and (!Warn_delta)) {
								cout << "\tWarning: Decay Probability Large!" << "\n";
								f_log  << "Warning: Decay Probability Large!" << "\n";
								Warn_delta = true;
						}
						N = ComputeEvents(arr_P[i*nGridXY*nGridZ + j*nGridZ + k], p, 5, i, j, k);

						// Update count
						arr_P[i*nGridXY*nGridZ + j*nGridZ + k]    = max(0.0, arr_P[i*nGridXY*nGridZ + j*nGridZ + k] - N);
						// KERNEL END

					}
				}
			}


			// Movement ///////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


						if (nGridXY > 1) {
							// Update positions

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

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

							// CELLS
							ComputeDiffusion(arr_B[i*nGridXY*nGridZ + j*nGridZ + k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 1, i, j, k);
							arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_B_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_B_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_B_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_B_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_B_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_B_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

							if (r > 0.0) {
								ComputeDiffusion(arr_I0[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I0_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I0_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I0_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I0_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I1[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I1_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I1_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I1_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I1_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I2[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I2_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I2_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I2_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I2_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I3[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I3_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I3_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I3_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I3_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I4[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I4_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I4_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I4_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I4_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I5[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I5_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I5_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I5_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I5_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I6[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I6_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I6_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I6_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I6_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I7[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I7_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I7_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I7_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I7_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I8[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I8_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I8_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I8_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I8_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I9[i*nGridXY*nGridZ + j*nGridZ + k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I9_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I9_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I9_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I9_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;
							}

							// PHAGES
							ComputeDiffusion(arr_P[i*nGridXY*nGridZ + j*nGridZ + k], lambdaP, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 3, i, j, k);
							arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_P_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_P_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_P_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_P_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_P_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_P_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

							// KERNEL END



						} else {
							// KERNEL BEGIN
							// CELLS
							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

							arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_B[i*nGridXY*nGridZ + j*nGridZ + k];

							if (r > 0.0) {
								arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I0[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I1[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I2[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I3[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I4[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I5[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I6[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I7[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I8[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
							}

							// PHAGES
							arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_P[i*nGridXY*nGridZ + j*nGridZ + k];
					}
				}
			}
		}

		/////////////////////////////////////////////////////
		// Main loop end ////////////////////////////////////
		/////////////////////////////////////////////////////

			// Swap pointers
			std::swap(arr_B, arr_B_new);
			std::swap(arr_I0, arr_I0_new);
			std::swap(arr_I1, arr_I1_new);
			std::swap(arr_I2, arr_I2_new);
			std::swap(arr_I3, arr_I3_new);
			std::swap(arr_I4, arr_I4_new);
			std::swap(arr_I5, arr_I5_new);
			std::swap(arr_I6, arr_I6_new);
			std::swap(arr_I7, arr_I7_new);
			std::swap(arr_I8, arr_I8_new);
			std::swap(arr_I9, arr_I9_new);
			std::swap(arr_P, arr_P_new);

			// Zero the _new arrays
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
						arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
					}
				}
			}


				// Update occupancy
				for (int i = 0; i < nGridXY; i++) {
					for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {
							arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] = arr_B[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
						}
					}
				}


				// NUTRIENT DIFFUSION
				double alphaXY = D_n * dT / pow(L / (double)nGridXY, 2);
				double alphaZ  = D_n * dT / pow(H / (double)nGridZ, 2);

				for (int i = 0; i < nGridXY; i++) {
					for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {

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
				}

				std::swap(arr_nutrient, arr_nutrient_new);

				// Zero the _new arrays
				for (int i = 0; i < nGridXY; i++) {
					for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {
								arr_nutrient_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
						}
					}
				}

				if ((maxOccupancy > L * L * H / (nGridXY * nGridXY * nGridZ)) and (!Warn_density)) {
					cout << "\tWarning: Maximum Density Large!" << "\n";
					f_log  << "Warning: Maximum Density Large!" << "\n";
					Warn_density = true;
				}
		}

		// Fast exit conditions
		// 1) There are no more sucebtible cells
		// -> Convert all infected cells to phages and stop simulation
		double accuB = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					accuB += arr_B[i*nGridXY*nGridZ + j*nGridZ + k];
				}
			}
		}
		if ((fastExit) and (accuB < 1)) {
			// Update the P array
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_P[i*nGridXY*nGridZ + j*nGridZ + k] += (1-alpha)*beta * (arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k]);
					}
				}
			}


			// Zero the I arrays
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
					}
				}
			}
			exit = true;
		}

		// 2) There are no more alive cells
		// -> Stop simulation

		double accuOcc = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					accuOcc += arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
				}
			}
		}

		if ((fastExit) and (accuOcc < 1)) {
				exit = true;
		}

		// 3) The food is on average less than one per gridpoint
		// and the maximal nutrient at any point in space is less than 1

		double accuNutrient = 0.0;
		double maxNutrient  = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					double tmpN = arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k];
					accuNutrient += tmpN;

					if (tmpN > maxNutrient) {
						maxNutrient = tmpN;
					}
				}
			}
		}

		if (fastExit) {
			if  ((accuNutrient < nGridZ*pow(nGridXY,2)) && (maxNutrient < 0.5)) {
				exit = true;
			}
		}

		// Store the state
		ExportData_arr(T,filename_suffix);

		// Check for nutrient stability
		assert(accuNutrient >= 0);
		assert(accuNutrient <= n_0 * L * L * H);
	}

	// Get stop time
	time_t  toc;
	time(&toc);

	// Calculate time difference
	float seconds = difftime(toc, tic);
	float hours   = floor(seconds/3600);
	float minutes = floor(seconds/60);
	minutes -= hours*60;
	seconds -= minutes*60 + hours*3600;

	cout << "\n";
	cout << "\tSimulation complete after ";
	if (hours > 0.0)   cout << hours   << " hours and ";
	if (minutes > 0.0) cout << minutes << " minutes and ";
	cout  << seconds << " seconds." << "\n";

	std::ofstream f_out;
	f_out.open(GetPath() + "/Completed_LOOP_DISTRIBUTED.txt",fstream::trunc);
	f_out << "\tSimulation complete after ";
	if (hours > 0.0)   f_out << hours   << " hours and ";
	if (minutes > 0.0) f_out << minutes << " minutes and ";
	f_out  << seconds << " seconds." << "\n";
	f_out.flush();
	f_out.close();

	// Write sucess to log
	if (exit) {
			f_log << ">>Simulation completed with exit flag<<" << "\n";
	} else {
			f_log << ">>Simulation completed without exit flag<<" << "\n";
	}

	std::ofstream f_timing;
	f_timing << "\t"       << setw(3) << difftime(toc, tic) << " s of total time" << "\n";

	f_timing.flush();
	f_timing.close();

	if (exit) {
		return 1;
	} else {
		return 0;
	}
}

int Colonies3D::Run_LoopDistributed_CPU_cuRand(double T_end) {
	std::string filename_suffix = "loopDistributedCPU_cuRand";

	this->T_end = T_end;

	// Get start time
	time_t  tic;
	time(&tic);

	// Generate a path
	path = GeneratePath();

	// Initilize the simulation matrices
	Initialize();

	// Export data
	ExportData_arr(T,filename_suffix);

	// Determine the number of samples to take
	int nSamplings = nSamp*T_end;

	/* Allocate arrays on the device */
	int errC = 1;

	int totalElements = nGridXY * nGridXY * nGridZ;
	int blockSize = 256;
	int gridSize = (totalElements + blockSize - 1) / blockSize;

	cudaError_t err = cudaSuccess;

	err = cudaMalloc((void**)&d_rng_state, sizeof(curandState)*totalElements);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate d_rng_state on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMemcpy(d_rng_state, rng_state, sizeof(curandState)*totalElements, cudaMemcpyHostToDevice);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy rng_state to the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	double *d_N;
	err = cudaMalloc((void**)&d_N, sizeof(double));
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate d_N on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	initRNG<<<gridSize,blockSize>>>(d_rng_state, totalElements);



	// Loop over samplings
	for (int n = 0; n < nSamplings; n++) {
		if (exit) break;

		// Determine the number of timesteps between sampings
		int nStepsPerSample = static_cast<int>(round(1 / (nSamp *  dT)));

		for (int t = 0; t < nStepsPerSample; t++) {
			if (exit) break;

			// Increase time
			T += dT;

			// Spawn phages
			if ((T_i >= 0) and (abs(T - T_i) < dT / 2)) {
				spawnPhages();
				T_i = -1;
			}

			// Reset density counter
			double maxOccupancy = 0.0;

			/////////////////////////////////////////////////////
			// Main loop start //////////////////////////////////
			/////////////////////////////////////////////////////

			// Kernel 1-2: nC update and maxOccupancy //////////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Ensure nC is updated
						if (arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < arr_nC[i*nGridXY*nGridZ + j*nGridZ + k]){
								arr_nC[i*nGridXY*nGridZ + j*nGridZ + k] = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
						}
					}
				}
			}

			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

						// Record the maximum observed density
						if (arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] > maxOccupancy) maxOccupancy = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];

					}
				}
			}
			// Kernel 3: Birth //////////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

						double p = 0; // privatize
						double N = -1; // privatize

						// Compute the growth modifier
						double growthModifier = arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] / (arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] + K);
						arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k] = growthModifier;

						p = g * growthModifier*dT;
						if (arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] < 1) {
							p = 0;
						}

						if ((p > 0.1) and (!Warn_g)) {
							cout << "\tWarning: Birth Probability Large!" << "\n";
							f_log  << "Warning: Birth Probability Large!" << "\n";
							Warn_g = true;
						}

						/* BEGIN anden Map-kernel */
						ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_B[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
						cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
						assert(N != -1);

						// Ensure there is enough nutrient
						if ( N > arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] ) {
							if (!Warn_fastGrowth) {
								cout << "\tWarning: Colonies growing too fast!" << "\n";
								f_log  << "Warning: Colonies growing too fast!" << "\n";
								Warn_fastGrowth = true;
							}

							N = round( arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] );
						}

						// Update count
						arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += N;
						arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] - N);
						/* END anden Map-kernel */
					}
				}
			}

			// Kernel 4: Increase Infections ////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

						double p = 0; // privatize
						double N = -1; // privatize

						// Compute the growth modifier
						double growthModifier = arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k];

						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= growthModifier;
						}

				 		if (r > 0.0) {
							/* BEGIN tredje Map-kernel */

							p = r*growthModifier*dT;
							if ((p > 0.25) and (!Warn_r)) {
								cout << "\tWarning: Infection Increase Probability Large!" << "\n";
								f_log  << "Warning: Infection Increase Probability Large!" << "\n";
								Warn_r = true;
							}

							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I9[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);

							// Update count
							arr_I9[i*nGridXY*nGridZ + j*nGridZ + k]    = max(0.0, arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k]   = max(0.0, arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += round( (1 - alpha) * Beta * N);   // Phages which escape the colony
							arr_M[i*nGridXY*nGridZ + j*nGridZ + k] = round(alpha * Beta * N);                        // Phages which reinfect the colony

							// Non-bursting events
							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I8[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I7[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I6[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I5[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I4[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I3[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I2[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I1[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							N = -1;
							ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_I0[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
							cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
							assert(N != -1);
							arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] += N;

							/* END tredje Map-kernel */
						}
					}
				}
			}

			// Kernel 5: New infections ///////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


						double p = 0; // privatize
						double N = -1; // privatize
												// double M = 0; // privatize

						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k];
						}

						// PRIVATIZE BOTH OF THESE
						double s;   // The factor which modifies the adsorption rate
						double n;   // The number of targets the phage has
												// Infectons


												// KERNEL THIS
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] >= 1)) {
							if (clustering) {   // Check if clustering is enabled
								s = pow(arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0);
								n = arr_nC[i*nGridXY*nGridZ + j*nGridZ + k];
							} else {            // Else use mean field computation
								s = 1.0;
								n = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
							}
						}

						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] >= 1)) {
							// Compute the number of hits
							if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
								N = arr_P[i*nGridXY*nGridZ + j*nGridZ + k];
							} else {
								p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
								ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_P[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
								cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
								assert(N != -1);
							}

							if (N + arr_M[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) {
								// If bacteria were hit, update events
								arr_P[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_P[i*nGridXY*nGridZ + j*nGridZ + k] - N);     // Update count

								double S;
								if (shielding) {
									// Absorbing medium model
									double d = pow(arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0) -
										pow(arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0);
									S = exp(-zeta * d); // Probability of hitting succebtible target

								} else {
									// Well mixed model
									S = arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
								}

								p = max(0.0, min(arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k],S)); // Probability of hitting succebtible target

								double tmp = N;
								N = -1;
								ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, tmp + arr_M[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
								cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
								assert(N != -1);

								if (N > arr_B[i*nGridXY*nGridZ + j*nGridZ + k])
									N = arr_B[i*nGridXY*nGridZ + j*nGridZ + k];              // If more bacteria than present are set to be infeced, round down

								// Update the counts
								arr_B[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_B[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								if (r > 0.0) {
									arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += N;
								} else {
									arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += N * (1 - alpha) * Beta;
								}
							}
						}
 					}
				}
			}

			// Kernel 6: Phage decay ///////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


						double p = 0; // privatize
						double N = -1; // privatize

						// KERNEL BEGIN
						p = delta*dT;
						if ((p > 0.1) and (!Warn_delta)) {
								cout << "\tWarning: Decay Probability Large!" << "\n";
								f_log  << "Warning: Decay Probability Large!" << "\n";
								Warn_delta = true;
						}
						ComputeEvents_seq<<<gridSize,blockSize>>>(d_N, arr_P[i*nGridXY*nGridZ + j*nGridZ + k], p, d_rng_state, i*nGridXY*nGridZ + j*nGridZ + k);
						cudaMemcpy(&N, d_N, sizeof(double),cudaMemcpyDeviceToHost);
						assert(N != -1);

						// Update count
						arr_P[i*nGridXY*nGridZ + j*nGridZ + k]    = max(0.0, arr_P[i*nGridXY*nGridZ + j*nGridZ + k] - N);
						// KERNEL END

					}
				}
			}


			// Movement ///////////////////////////////////////////////////////////////////
			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Skip empty sites
						if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


						if (nGridXY > 1) {
							// Update positions

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

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

							// CELLS
							ComputeDiffusion(arr_B[i*nGridXY*nGridZ + j*nGridZ + k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 1, i, j, k);
							arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_B_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_B_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_B_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_B_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_B_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_B_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

							if (r > 0.0) {
								ComputeDiffusion(arr_I0[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I0_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I0_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I0_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I0_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I1[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I1_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I1_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I1_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I1_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I2[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I2_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I2_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I2_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I2_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I3[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I3_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I3_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I3_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I3_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I4[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I4_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I4_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I4_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I4_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I5[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I5_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I5_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I5_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I5_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I6[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I6_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I6_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I6_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I6_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I7[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I7_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I7_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I7_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I7_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I8[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I8_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I8_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I8_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I8_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								ComputeDiffusion(arr_I9[i*nGridXY*nGridZ + j*nGridZ + k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
								arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I9_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I9_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I9_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I9_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;
							}

							// PHAGES
							ComputeDiffusion(arr_P[i*nGridXY*nGridZ + j*nGridZ + k], lambdaP, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 3, i, j, k);
							arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_P_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_P_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_P_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_P_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_P_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_P_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

							// KERNEL END



						} else {
							// KERNEL BEGIN
							// CELLS
							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

							arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_B[i*nGridXY*nGridZ + j*nGridZ + k];

							if (r > 0.0) {
								arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I0[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I1[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I2[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I3[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I4[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I5[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I6[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I7[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I8[i*nGridXY*nGridZ + j*nGridZ + k];
								arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
							}

							// PHAGES
							arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_P[i*nGridXY*nGridZ + j*nGridZ + k];
					}
				}
			}
		}

		/////////////////////////////////////////////////////
		// Main loop end ////////////////////////////////////
		/////////////////////////////////////////////////////

			// Swap pointers
			std::swap(arr_B, arr_B_new);
			std::swap(arr_I0, arr_I0_new);
			std::swap(arr_I1, arr_I1_new);
			std::swap(arr_I2, arr_I2_new);
			std::swap(arr_I3, arr_I3_new);
			std::swap(arr_I4, arr_I4_new);
			std::swap(arr_I5, arr_I5_new);
			std::swap(arr_I6, arr_I6_new);
			std::swap(arr_I7, arr_I7_new);
			std::swap(arr_I8, arr_I8_new);
			std::swap(arr_I9, arr_I9_new);
			std::swap(arr_P, arr_P_new);

			// Zero the _new arrays
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
						arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
					}
				}
			}


				// Update occupancy
				for (int i = 0; i < nGridXY; i++) {
					for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {
							arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] = arr_B[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
						}
					}
				}


				// NUTRIENT DIFFUSION
				double alphaXY = D_n * dT / pow(L / (double)nGridXY, 2);
				double alphaZ  = D_n * dT / pow(H / (double)nGridZ, 2);

				for (int i = 0; i < nGridXY; i++) {
					for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {

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
				}

				std::swap(arr_nutrient, arr_nutrient_new);

				// Zero the _new arrays
				for (int i = 0; i < nGridXY; i++) {
					for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {
								arr_nutrient_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
						}
					}
				}

				if ((maxOccupancy > L * L * H / (nGridXY * nGridXY * nGridZ)) and (!Warn_density)) {
					cout << "\tWarning: Maximum Density Large!" << "\n";
					f_log  << "Warning: Maximum Density Large!" << "\n";
					Warn_density = true;
				}
		}

		// Fast exit conditions
		// 1) There are no more sucebtible cells
		// -> Convert all infected cells to phages and stop simulation
		double accuB = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					accuB += arr_B[i*nGridXY*nGridZ + j*nGridZ + k];
				}
			}
		}
		if ((fastExit) and (accuB < 1)) {
			// Update the P array
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_P[i*nGridXY*nGridZ + j*nGridZ + k] += (1-alpha)*beta * (arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k]);
					}
				}
			}


			// Zero the I arrays
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
					}
				}
			}
			exit = true;
		}

		// 2) There are no more alive cells
		// -> Stop simulation

		double accuOcc = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					accuOcc += arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
				}
			}
		}

		if ((fastExit) and (accuOcc < 1)) {
				exit = true;
		}

		// 3) The food is on average less than one per gridpoint
		// and the maximal nutrient at any point in space is less than 1

		double accuNutrient = 0.0;
		double maxNutrient  = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					double tmpN = arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k];
					accuNutrient += tmpN;

					if (tmpN > maxNutrient) {
						maxNutrient = tmpN;
					}
				}
			}
		}

		if (fastExit) {
			if  ((accuNutrient < nGridZ*pow(nGridXY,2)) && (maxNutrient < 0.5)) {
				exit = true;
			}
		}

		// Store the state
		ExportData_arr(T,filename_suffix);

		// Check for nutrient stability
		assert(accuNutrient >= 0);
		assert(accuNutrient <= n_0 * L * L * H);
	}

	// Get stop time
	time_t  toc;
	time(&toc);

	// Calculate time difference
	float seconds = difftime(toc, tic);
	float hours   = floor(seconds/3600);
	float minutes = floor(seconds/60);
	minutes -= hours*60;
	seconds -= minutes*60 + hours*3600;

	cout << "\n";
	cout << "\tSimulation complete after ";
	if (hours > 0.0)   cout << hours   << " hours and ";
	if (minutes > 0.0) cout << minutes << " minutes and ";
	cout  << seconds << " seconds." << "\n";

	std::ofstream f_out;
	f_out.open(GetPath() + "/Completed_LOOP_DISTRIBUTED.txt",fstream::trunc);
	f_out << "\tSimulation complete after ";
	if (hours > 0.0)   f_out << hours   << " hours and ";
	if (minutes > 0.0) f_out << minutes << " minutes and ";
	f_out  << seconds << " seconds." << "\n";
	f_out.flush();
	f_out.close();

	// Write sucess to log
	if (exit) {
			f_log << ">>Simulation completed with exit flag<<" << "\n";
	} else {
			f_log << ">>Simulation completed without exit flag<<" << "\n";
	}

	std::ofstream f_timing;
	f_timing << "\t"       << setw(3) << difftime(toc, tic) << " s of total time" << "\n";

	f_timing.flush();
	f_timing.close();

	if (exit) {
		return 1;
	} else {
		return 0;
	}
}

////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//  Just a separator between CPU and GPU to make it easier to spot when scrolling
///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////



int Colonies3D::Run_LoopDistributed_GPU(double T_end) {
	std::string filename_suffix = "loopDistributedGPU";

    int errC = 1;

	this->T_end = T_end;

	// Get start time
	time_t  tic;
	time(&tic);

	// Generate a path
	path = GeneratePath();

	// Initilize th e simulation matrices
	Initialize();

	// Export data
	ExportData_arr(T,filename_suffix);

	// Determine the number of samples to take
	int nSamplings = nSamp*T_end;



	/* Allocate arrays on the device */
	int totalElements = nGridXY * nGridXY * nGridZ;
	int totalMemSize = totalElements * sizeof(double);
	int blockSize = 256;
	int gridSize = (totalElements + blockSize - 1) / blockSize;

	cudaError_t err = cudaSuccess;

	// Allocate on GPU
	double *arr_maxOccupancy = new double[gridSize]();
	double *d_arr_maxOccupancy;

	err = cudaMalloc((void**)&d_arr_nC , totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_nC on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_Occ, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_Occ on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_IsActive, blockSize*gridSize*sizeof(bool));
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_IsActive on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_maxOccupancy, sizeof(double)*gridSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_maxOccupancy on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMemcpy(d_arr_maxOccupancy, arr_maxOccupancy, sizeof(double)*gridSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy arr_maxOccupancy to the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_rng_state, sizeof(curandState)*totalElements);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate d_rng_state on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMemcpy(d_rng_state, rng_state, sizeof(curandState)*totalElements, cudaMemcpyHostToDevice);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy rng_state to the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_B, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_B on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_B_new, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_B_new on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_P, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_P on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_P_new, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_P_new on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_P, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_P on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_P_new, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_P_new on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I0, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I0 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I1, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I1 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I2, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I2 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I3, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I3 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I4, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I4 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I5, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I5 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I6, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I6 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I7, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I7 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I8, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I8 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_I9, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_I9 on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_M, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_M on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_p, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_p to the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_nutrient, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_nutrient on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_arr_GrowthModifier, totalMemSize);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate arr_GrowthModifier to the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_Warn_g, sizeof(bool));
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate Warn_g on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_Warn_fastGrowth, sizeof(bool));
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate d_Warn_fastGrowth on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMalloc((void**)&d_Warn_r, sizeof(bool));
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to allocate Warn_r on the device! error = %s\n", cudaGetErrorString(err)); errC--;}

	err = cudaMemcpy(d_Warn_r, &this->Warn_r, sizeof(bool), cudaMemcpyHostToDevice);
	if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy Warn_r to the device! error = %s\n", cudaGetErrorString(err)); errC--;}


	initRNG<<<gridSize,blockSize>>>(d_rng_state, totalElements);


	// cudaMemCpy to device

	// Loop over samplings
	for (int n = 0; n < nSamplings; n++) {
		if (exit) break;

		// Determine the number of timesteps between sampings
		int nStepsPerSample = static_cast<int>(round(1 / (nSamp *  dT)));

		for (int t = 0; t < nStepsPerSample; t++) {
			if (exit) break;

			// Increase time
			T += dT;

			// Spawn phages
			if ((T_i >= 0) and (abs(T - T_i) < dT / 2)) {
				spawnPhages();
				T_i = -1;
			}

			// Reset density counter
			double maxOccupancy = 0.0;

			// /////////////////////////////////////////////////////
			// // Main loop start //////////////////////////////////
			// /////////////////////////////////////////////////////

			/* Do all the allocations and other CUDA device stuff here
			 * remember to do them outside the nSamplings loop afterwards
			 */

			if (GPU_NC){

				// Copy to the device
				CopyAllToDevice();

				// Run first Kernel
				FirstKernel<<<gridSize, blockSize>>>(d_arr_Occ, d_arr_nC, totalElements);
				err = cudaGetLastError();
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failure in FirstKernel! error = %s\n", cudaGetErrorString(err)); errC--;}

				// Copy data back from device
				if (!GPU_MAXOCCUPANCY) 	CopyAllToHost();

			} else {
				for (int i = 0; i < nGridXY; i++) {
					if (exit) break;

					for (int j = 0; j < nGridXY; j++) {
						if (exit) break;

						for (int k = 0; k < nGridZ; k++) {
							if (exit) break;

							// Ensure nC is updated
							if (arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < arr_nC[i*nGridXY*nGridZ + j*nGridZ + k]){
								arr_nC[i*nGridXY*nGridZ + j*nGridZ + k] = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
							}
						}
					}
				}
			}

			if (GPU_MAXOCCUPANCY) {

				// Copy to the device
				if (!GPU_NC) CopyAllToDevice();

				// set active flags
				SetIsActive<<<gridSize, blockSize>>>(d_arr_Occ, d_arr_IsActive, totalElements);
				err = cudaGetLastError();
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failure in SetIsActive! error = %s\n", cudaGetErrorString(err)); errC--;}

				// Run second Kernel
				SecondKernel<<<gridSize, blockSize, blockSize*sizeof(double)>>>(d_arr_Occ, d_arr_nC, d_arr_maxOccupancy, d_arr_IsActive, blockSize);
				err = cudaGetLastError();
				if (err != cudaSuccess && errC > 0) {fprintf(stderr, "Failure in SecondKernel! error = %s\n", cudaGetErrorString(err)); errC--;}

				// Copy data back from device
				if (!GPU_BIRTH) CopyAllToHost();

				err = cudaMemcpy(arr_maxOccupancy, d_arr_maxOccupancy, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy arr_maxOccupancy to the host! error = %s\n", cudaGetErrorString(err));
					errC--; }


				// excuse this for-loop
				for (int i = 0; i < gridSize; i++){
					maxOccupancy = max(maxOccupancy, arr_maxOccupancy[i]);
				}
			} else {
				for (int i = 0; i < nGridXY; i++) {
					if (exit) break;

					for (int j = 0; j < nGridXY; j++) {
						if (exit) break;

						for (int k = 0; k < nGridZ; k++) {
							if (exit) break;

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

							// Record the maximum observed density
							if (arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] > maxOccupancy) maxOccupancy = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];

						}
					}
				}
			}


			// Birth //////////////////////////////////////////////////////////////////////
			if (GPU_BIRTH){

				// Copy to the device
				if (!GPU_MAXOCCUPANCY) CopyAllToDevice();

				ComputeBirthEvents<<<gridSize, blockSize>>>(d_arr_B, d_arr_B_new, d_arr_nutrient, d_arr_GrowthModifier, K, g, dT, d_Warn_g, d_Warn_fastGrowth, d_rng_state, d_arr_IsActive);
				err = cudaGetLastError();
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failure in ComputeBirthEvents! error = %s\n", cudaGetErrorString(err)); errC--;}

				// Copy data back from device
				if (!GPU_INFECTIONS) CopyAllToHost();

				err = cudaMemcpy(&this->Warn_g, d_Warn_g, sizeof(bool), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy Warn_g to the host! error = %s\n", cudaGetErrorString(err)); errC--;}

				err = cudaMemcpy(&this->Warn_fastGrowth, d_Warn_fastGrowth, sizeof(bool), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy Warn_fastGrowth to the host! error = %s\n", cudaGetErrorString(err)); errC--;}


			} else {
				for (int i = 0; i < nGridXY; i++) {
					if (exit) break;

					for (int j = 0; j < nGridXY; j++) {
						if (exit) break;

						for (int k = 0; k < nGridZ; k++) {
							if (exit) break;

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

							double p = 0; // privatize
							double N = 0; // privatize

							// Compute the growth modifier
							double growthModifier = arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] / (arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] + K);
							arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k] = growthModifier;

							p = g * growthModifier*dT;
							if (arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] < 1) {		//
								p = 0;
							}

							if ((p > 0.1) and (!Warn_g)) {
								cout << "\tWarning: Birth Probability Large!" << "\n";
								f_log  << "Warning: Birth Probability Large!" << "\n";
								Warn_g = true;
						 }

							/* BEGIN anden Map-kernel */
							N = ComputeEvents(arr_B[i*nGridXY*nGridZ + j*nGridZ + k], p, 1, i, j, k);
							// Ensure there is enough nutrient
							if ( N > arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] ) {
								if (!Warn_fastGrowth) {
									cout << "\tWarning: Colonies growing too fast!" << "\n";
									f_log  << "Warning: Colonies growing too fast!" << "\n";
									Warn_fastGrowth = true;
								}

								N = round( arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] );
							}

							// Update count
							arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += N;
							arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							/* END anden Map-kernel */
						}
					}
				}
			}

			if (GPU_INFECTIONS){

				// Copy to the device
				if (!GPU_BIRTH) CopyAllToDevice();

				// Infections kernels
				BurstingEvents<<<gridSize, blockSize>>>(d_arr_I9, d_arr_P_new, d_arr_Occ, d_arr_GrowthModifier, d_arr_M, d_arr_p, alpha, beta, r, dT, d_Warn_r, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I8, d_arr_I9, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I7, d_arr_I8, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I6, d_arr_I7, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I5, d_arr_I6, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I4, d_arr_I5, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I3, d_arr_I4, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I2, d_arr_I3, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I1, d_arr_I2, d_arr_p, d_rng_state, d_arr_IsActive);
				NonBurstingEvents<<<gridSize, blockSize>>>(d_arr_I0, d_arr_I1, d_arr_p, d_rng_state, d_arr_IsActive);
				err = cudaGetLastError();
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failure in BurstingEvents or NonBurstingEvents! error = %s\n", cudaGetErrorString(err)); errC--;}

				// Copy data back from device
				if(!GPU_NEWINFECTIONS) CopyAllToHost();

				err = cudaMemcpy(&this->Warn_r, d_Warn_r, sizeof(bool), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess && errC > 0)	{fprintf(stderr, "Failed to copy Warn_r to the host! error = %s\n", cudaGetErrorString(err)); errC--;}


			} else {

				for (int i = 0; i < nGridXY; i++) {
					if (exit) break;

					for (int j = 0; j < nGridXY; j++) {
						if (exit) break;

						for (int k = 0; k < nGridZ; k++) {
							if (exit) break;

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

							double p = 0; // privatize
							double N = 0; // privatize

							// Compute the growth modifier
							double growthModifier = arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k];

							// Compute beta
							double Beta = beta;

							if (reducedBeta) {
								Beta *= growthModifier;
							}

							if (r > 0.0){

								p = r*growthModifier*dT;
								if ((p > 0.25) and (!Warn_r)) {
									cout << "\tWarning: Infection Increase Probability Large!" << "\n";
									f_log  << "Warning: Infection Increase Probability Large!" << "\n";
									Warn_r = true;
								}
								N = ComputeEvents(arr_I9[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);  // Bursting events

								// Update count
								arr_I9[i*nGridXY*nGridZ + j*nGridZ + k]    = max(0.0, arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k]   = max(0.0, arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += round( (1 - alpha) * Beta * N);   // Phages which escape the colony
								arr_M[i*nGridXY*nGridZ + j*nGridZ + k] = round(alpha * Beta * N);                        // Phages which reinfect the colony

								// Non-bursting events
								N = ComputeEvents(arr_I8[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I7[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I6[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I5[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I4[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I3[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I2[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I1[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								N = ComputeEvents(arr_I0[i*nGridXY*nGridZ + j*nGridZ + k], p, 2, i, j, k);
								arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] - N);
								arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] += N;

								/* END tredje Map-kernel */

							}
						}
					}
				}
			}

			if (GPU_NEWINFECTIONS) {

				// Copy to the device
				if (!GPU_INFECTIONS) CopyAllToDevice();

				NewInfectionsKernel<<<gridSize, blockSize>>>(d_arr_Occ, d_arr_nC, d_arr_P, d_arr_P_new,
																										 d_arr_GrowthModifier, d_arr_B, d_arr_B_new,
																										 d_arr_M, d_arr_I0_new, d_arr_IsActive,
																										 reducedBeta, clustering, shielding,
																										 K, alpha, beta, eta, zeta, dT, r, d_rng_state);
				// Copy data back from device
				if (!GPU_PHAGEDECAY) CopyAllToHost();
			} else {
				// Kernel 5: New infections ///////////////////////////////////////////////////////////////////
				for (int i = 0; i < nGridXY; i++) {
					if (exit) break;

					for (int j = 0; j < nGridXY; j++) {
						if (exit) break;

						for (int k = 0; k < nGridZ; k++) {
							if (exit) break;

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


							double p = 0; // privatize
							double N = 0; // privatize
							// double M = 0; // privatize

							// Compute beta
							double Beta = beta;
							if (reducedBeta) {
								Beta *= arr_GrowthModifier[i*nGridXY*nGridZ + j*nGridZ + k];
							}

							// PRIVATIZE BOTH OF THESE
							double s;   // The factor which modifies the adsorption rate
							double n;   // The number of targets the phage has
							// Infectons


							// KERNEL THIS
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] >= 1)) {
								if (clustering) {   // Check if clustering is enabled
									s = pow(arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0);
									n = arr_nC[i*nGridXY*nGridZ + j*nGridZ + k];
								} else {            // Else use mean field computation
									s = 1.0;
									n = arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
								}
							}

							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] >= 1)) {
								// Compute the number of hits
								if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
									N = arr_P[i*nGridXY*nGridZ + j*nGridZ + k];
								} else {
									p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
									N = ComputeEvents(arr_P[i*nGridXY*nGridZ + j*nGridZ + k], p, 4, i, j, k);     // Number of targets hit
								}

								if (N + arr_M[i*nGridXY*nGridZ + j*nGridZ + k] >= 1) {
									// If bacteria were hit, update events
									arr_P[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_P[i*nGridXY*nGridZ + j*nGridZ + k] - N);     // Update count

									double S;
									if (shielding) {
										// Absorbing medium model
										double d = pow(arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0) -
											pow(arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_nC[i*nGridXY*nGridZ + j*nGridZ + k], 1.0 / 3.0);
										S = exp(-zeta * d); // Probability of hitting succebtible target

									} else {
										// Well mixed model
										S = arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
									}

									p = max(0.0, min(arr_B[i*nGridXY*nGridZ + j*nGridZ + k] / arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k],
																	 S)); // Probability of hitting succebtible target
									N = ComputeEvents(N + arr_M[i*nGridXY*nGridZ + j*nGridZ + k], p, 4, i, j, k);                  // Number of targets hit

									if (N > arr_B[i*nGridXY*nGridZ + j*nGridZ + k])
										N = arr_B[i*nGridXY*nGridZ + j*nGridZ + k];              // If more bacteria than present are set to be infeced, round down

									// Update the counts
									arr_B[i*nGridXY*nGridZ + j*nGridZ + k] = max(0.0, arr_B[i*nGridXY*nGridZ + j*nGridZ + k] - N);
									if (r > 0.0) {
										arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += N;
									} else {
										arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += N * (1 - alpha) * Beta;
									}
								}
							}
						}
					}
				}
			}


			// Phage decay ///////////////////////////////////////////////////////////////////
			if (GPU_PHAGEDECAY) {
			/*
				// If previous kernel is not used, copy arrays to device
				if(!GPU_NEWINFECTIONS){
					CopyAllToDevice;
				}

				// Så vidt jeg kan se, er p på dette tidspunkt udregnet fra konstanter, og behøver derfor ikke være i kernel.
				double p;
				p = delta*dT;

				SixthKernel<<<gridSize, blockSize>>>(d_arr_P, p, &Warn_delta, totalElements, d_rng_state);
				cudaMemcpy(arr_P, d_arr_P, totalMemSize, cudaMemcpyDeviceToHost);

				// logging of warning (cout <<... and f_log << ... moved outside
				// loop to ensure it is only triggered once


				// Do Stuff
				*/
			} else {
				for (int i = 0; i < nGridXY; i++) {
					if (exit) break;

					for (int j = 0; j < nGridXY; j++) {
						if (exit) break;

						for (int k = 0; k < nGridZ; k++) {
							if (exit) break;

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;


							double p = 0; // privatize
							double N = 0; // privatize

							// KERNEL BEGIN
							p = delta*dT;

							// TODO: Check below is in the kernel.
							// TODO: But put the cout outside kernel (check if warn_delta is true)
							if ((p > 0.1) and (!Warn_delta)) {
								cout << "\tWarning: Decay Probability Large!" << "\n";
								f_log  << "Warning: Decay Probability Large!" << "\n";
								Warn_delta = true;
							}


							N = ComputeEvents(arr_P[i*nGridXY*nGridZ + j*nGridZ + k], p, 5, i, j, k);

							// Update count
							arr_P[i*nGridXY*nGridZ + j*nGridZ + k]    = max(0.0, arr_P[i*nGridXY*nGridZ + j*nGridZ + k] - N);
							// KERNEL END

						}
					}
				}
			}





			// Movement ///////////////////////////////////////////////////////////////////
			if (GPU_MOVEMENT) {
				// Do
			} else {
				for (int i = 0; i < nGridXY; i++) {
					if (exit) break;

					for (int j = 0; j < nGridXY; j++) {
						if (exit) break;

						for (int k = 0; k < nGridZ; k++) {
							if (exit) break;

							// Skip empty sites
							if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

							if (nGridXY > 1) {
								// KERNEL BEGIN
								// Update positions

								// Skip empty sites
								if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

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

								// CELLS
								ComputeDiffusion(arr_B[i*nGridXY*nGridZ + j*nGridZ + k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b,1, i, j, k);
								arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_B_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_B_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_B_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_B_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_B_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_B_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								if (r > 0.0) {
									ComputeDiffusion(arr_I0[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I0_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I0_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I0_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I0_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I1[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I1_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I1_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I1_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I1_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I2[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I2_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I2_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I2_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I2_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I3[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I3_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I3_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I3_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I3_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I4[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I4_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I4_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I4_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I4_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I5[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I5_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I5_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I5_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I5_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I6[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I6_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I6_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I6_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I6_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I7[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I7_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I7_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I7_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I7_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I8[i*nGridXY*nGridZ + j*nGridZ + k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I8_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I8_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I8_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I8_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

									ComputeDiffusion(arr_I9[i*nGridXY*nGridZ + j*nGridZ + k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2, i, j, k);
									arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_I9_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_I9_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_I9_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_I9_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;
								}

								// PHAGES
								ComputeDiffusion(arr_P[i*nGridXY*nGridZ + j*nGridZ + k], lambdaP, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 3, i, j, k);
								arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += n_0; arr_P_new[ip*nGridXY*nGridZ + j*nGridZ + k] += n_u; arr_P_new[im*nGridXY*nGridZ + j*nGridZ + k] += n_d; arr_P_new[i*nGridXY*nGridZ + jp*nGridZ + k] += n_r; arr_P_new[i*nGridXY*nGridZ + jm*nGridZ + k] += n_l; arr_P_new[i*nGridXY*nGridZ + j*nGridZ + kp] += n_f; arr_P_new[i*nGridXY*nGridZ + j*nGridZ + km] += n_b;

								// KERNEL END



							} else {
								// KERNEL BEGIN
								// CELLS
								// Skip empty sites
								if ((arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] < 1) and (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] < 1)) continue;

								arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_B[i*nGridXY*nGridZ + j*nGridZ + k];

								if (r > 0.0) {
									arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I0[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I1[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I2[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I3[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I4[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I5[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I6[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I7[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I8[i*nGridXY*nGridZ + j*nGridZ + k];
									arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
								}

								// PHAGES
								arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k] += arr_P[i*nGridXY*nGridZ + j*nGridZ + k];
								// KERNEL END
							}
						}
					}
				}
			}

			/////////////////////////////////////////////////////
			// Main loop end ////////////////////////////////////
			/////////////////////////////////////////////////////

			// Swap pointers
			std::swap(arr_B, arr_B_new);
			std::swap(arr_I0, arr_I0_new);
			std::swap(arr_I1, arr_I1_new);
			std::swap(arr_I2, arr_I2_new);
			std::swap(arr_I3, arr_I3_new);
			std::swap(arr_I4, arr_I4_new);
			std::swap(arr_I5, arr_I5_new);
			std::swap(arr_I6, arr_I6_new);
			std::swap(arr_I7, arr_I7_new);
			std::swap(arr_I8, arr_I8_new);
			std::swap(arr_I9, arr_I9_new);
			std::swap(arr_P, arr_P_new);

			// Zero the _new arrays
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_B_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
						arr_I0_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I1_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I2_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I3_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I4_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I5_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I6_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I7_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I8_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I9_new[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_P_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
					}
				}
			}


			// Update occupancy
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] = arr_B[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
					}
				}
			}


			// NUTRIENT DIFFUSION
			double alphaXY = D_n * dT / pow(L / (double)nGridXY, 2);
			double alphaZ  = D_n * dT / pow(H / (double)nGridZ, 2);

			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {

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
			}

			std::swap(arr_nutrient, arr_nutrient_new);

			// Zero the _new arrays
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_nutrient_new[i*nGridXY*nGridZ + j*nGridZ + k]  = 0.0;
					}
				}
			}

			if ((maxOccupancy > L * L * H / (nGridXY * nGridXY * nGridZ)) and (!Warn_density)) {
				cout << "\tWarning: Maximum Density Large!" << "\n";
				f_log  << "Warning: Maximum Density Large!" << "\n";
				Warn_density = true;
			}
		}

		// Fast exit conditions
		// 1) There are no more sucebtible cells
		// -> Convert all infected cells to phages and stop simulation
		double accuB = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					accuB += arr_B[i*nGridXY*nGridZ + j*nGridZ + k];
				}
			}
		}
		if ((fastExit) and (accuB < 1)) {
			// Update the P array
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_P[i*nGridXY*nGridZ + j*nGridZ + k] += (1-alpha)*beta * (arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k]);
					}
				}
			}


			// Zero the I arrays
			for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
					for (int k = 0; k < nGridZ; k++ ) {
						arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
						arr_I9[i*nGridXY*nGridZ + j*nGridZ + k] = 0.0;
					}
				}
			}
			exit = true;
		}

		// 2) There are no more alive cells
		// -> Stop simulation

		double accuOcc = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					accuOcc += arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k];
				}
			}
		}

		if ((fastExit) and (accuOcc < 1)) {
			exit = true;
		}

		// 3) The food is on average less than one per gridpoint
		// and the maximal nutrient at any point in space is less than 1

		double accuNutrient = 0.0;
		double maxNutrient  = 0.0;
		for (int i = 0; i < nGridXY; i++) {
			for (int j = 0; j < nGridXY; j++ ) {
				for (int k = 0; k < nGridZ; k++ ) {
					double tmpN = arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k];
					accuNutrient += tmpN;

					if (tmpN > maxNutrient) {
						maxNutrient = tmpN;
					}
				}
			}
		}

		if (fastExit) {
			if  ((accuNutrient < nGridZ*pow(nGridXY,2)) && (maxNutrient < 0.5)) {
				exit = true;
			}
		}

		// cudaMemCpy to host


		// Store the state
		ExportData_arr(T,filename_suffix);

		// Check for nutrient stability
		assert(accuNutrient >= 0);
		assert(accuNutrient <= n_0 * L * L * H);
	}
	// TODO: Husk at kommenter nedenstående tilbage ind.
	/*
	// logging of warning moved outside loop to ensure.
	// it is only triggered once.
	if(Warn_delta) {
	cout << "\tWarning: Decay Probability Large!" << "\n";
	f_log  << "Warning: Decay Probability Large!" << "\n";
	}
	if(Warn_g) {
		cout << "\tWarning: Birth Probability Large!" << "\n";
		f_log  << "Warning: Birth Probability Large!" << "\n";
	}
	if(Warn_fastGrowth){
		cout << "\tWarning: Colonies growing too fast!" << "\n";
		f_log  << "Warning: Colonies growing too fast!" << "\n";
	}

	*/
	// Get stop time
	time_t  toc;
	time(&toc);

	// Calculate time difference
	float seconds = difftime(toc, tic);
	float hours   = floor(seconds/3600);
	float minutes = floor(seconds/60);
	minutes -= hours*60;
	seconds -= minutes*60 + hours*3600;

	cout << "\n";
	cout << "\tSimulation complete after ";
	if (hours > 0.0)   cout << hours   << " hours and ";
	if (minutes > 0.0) cout << minutes << " minutes and ";
	cout  << seconds << " seconds." << "\n";

	std::ofstream f_out;
	f_out.open(GetPath() + "/Completed_LOOP_DISTRIBUTED.txt",fstream::trunc);
	f_out << "\tSimulation complete after ";
	if (hours > 0.0)   f_out << hours   << " hours and ";
	if (minutes > 0.0) f_out << minutes << " minutes and ";
	f_out  << seconds << " seconds." << "\n";
	f_out.flush();
	f_out.close();

	// Write sucess to log
	if (exit) {
		f_log << ">>Simulation completed with exit flag<<" << "\n";
	} else {
		f_log << ">>Simulation completed without exit flag<<" << "\n";
	}

	std::ofstream f_timing;
	f_timing << "\t"       << setw(3) << difftime(toc, tic) << " s of total time" << "\n";

	f_timing.flush();
	f_timing.close();

	// cudaFree here!!
	if (GPU_NC) {
		cudaFree(d_arr_nC);
		cudaFree(d_arr_Occ);
	}
	if (GPU_MAXOCCUPANCY) {
		cudaFree(d_arr_maxOccupancy);
	}
	if (GPU_BIRTH || GPU_INFECTIONS || GPU_NEWINFECTIONS || GPU_PHAGEDECAY || GPU_MOVEMENT) {
		cudaFree(d_rng_state);
		cudaFree(d_arr_P);
	}


	if (exit) {
		return 1;
	} else {
		return 0;
	}
}

/////////////////////////////////////////////////////////////////////
// GPU loop end
/////////////////////////////////////////////////////////////////////

// GPU copy helper functions
void Colonies3D::CopyToHost(double* hostArray, double* deviceArray, int failCode, int gridsz){
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(hostArray, deviceArray, sizeof(double)*gridsz, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)	fprintf(stderr, "Failed to copy to the host! Code %d error = %s\n", failCode, cudaGetErrorString(err));
}

///////
void Colonies3D::CopyAllToHost(){

	CopyToHost(arr_nC, 				d_arr_nC, 				1, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_Occ, 			d_arr_Occ, 				2, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_B, 				d_arr_B, 				3, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_B_new, 			d_arr_B_new, 			4, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_P, 				d_arr_P, 				5, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_P_new, 			d_arr_P_new, 			6, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_P, 				d_arr_P,				7, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_P_new,			d_arr_P_new, 			8, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I0, 				d_arr_I0,				9, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I1, 				d_arr_I1, 				10, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I2, 				d_arr_I2, 				11, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I3, 				d_arr_I3, 				12, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I4, 				d_arr_I4, 				13, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I5, 				d_arr_I5, 				14, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I6, 				d_arr_I6, 				15, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I7, 				d_arr_I7, 				16, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I8, 				d_arr_I8, 				17, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_I9, 				d_arr_I9, 				18, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_M, 				d_arr_M, 				19, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_p, 				d_arr_p, 				20, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_nutrient, 		d_arr_nutrient, 		21, nGridXY*nGridXY*nGridZ );
	CopyToHost(arr_GrowthModifier, 	d_arr_GrowthModifier, 	22, nGridXY*nGridXY*nGridZ );

}



////
void Colonies3D::CopyToDevice(double* hostArray, double* deviceArray, int failCode, int gridsz){
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(deviceArray, hostArray, sizeof(double)*gridsz, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)	fprintf(stderr, "Failed to copy to the device! Code %d error = %s\n", failCode, cudaGetErrorString(err));
}

////
void Colonies3D::CopyAllToDevice(){

	CopyToDevice(arr_nC, 				d_arr_nC, 				1, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_Occ, 				d_arr_Occ, 				2, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_B, 				d_arr_B, 				3, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_B_new, 			d_arr_B_new, 			4, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_P, 				d_arr_P, 				5, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_P_new, 			d_arr_P_new, 			6, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_P, 				d_arr_P,				7, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_P_new,				d_arr_P_new, 			8, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I0, 				d_arr_I0,				9, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I1, 				d_arr_I1, 				10, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I2, 				d_arr_I2, 				11, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I3, 				d_arr_I3, 				12, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I4, 				d_arr_I4, 				13, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I5, 				d_arr_I5, 				14, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I6, 				d_arr_I6, 				15, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I7, 				d_arr_I7, 				16, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I8, 				d_arr_I8, 				17, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_I9, 				d_arr_I9, 				18, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_M, 				d_arr_M, 				19, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_p, 				d_arr_p, 				20, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_nutrient, 			d_arr_nutrient, 		21, nGridXY*nGridXY*nGridZ );
	CopyToDevice(arr_GrowthModifier, 	d_arr_GrowthModifier, 	22, nGridXY*nGridXY*nGridZ );

};

// Initialize the simulation
void Colonies3D::Initialize() {

		// Set the random number generator seed
		if (rngSeed >= 0.0) {
				rng.seed( rngSeed );
		} else {
				static std::random_device rd;
				rng.seed(rd());
		}

		// Compute nGridZ
		if (L != H) {
				nGridZ = round(H / L * nGridXY);
				H = nGridZ * L / nGridXY;
		} else {
				nGridZ = nGridXY;
		}

		// Allocate the arrays
		// Compute the step size
		double dXY = L / nGridXY;
		double dZ  = H / nGridZ;
		double dV  = dXY * dXY * dZ;

		// Allocate arrays
		arr_B   = new double[nGridXY*nGridXY*nGridZ]();
		arr_I0  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I1  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I2  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I3  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I4  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I5  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I6  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I7  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I8  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I9  = new double[nGridXY*nGridXY*nGridZ]();
		arr_P   = new double[nGridXY*nGridXY*nGridZ]();
		arr_nC  = new double[nGridXY*nGridXY*nGridZ]();

		arr_B_new   = new double[nGridXY*nGridXY*nGridZ]();
		arr_I0_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I1_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I2_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I3_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I4_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I5_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I6_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I7_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I8_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_I9_new  = new double[nGridXY*nGridXY*nGridZ]();
		arr_P_new   = new double[nGridXY*nGridXY*nGridZ]();

		arr_nutrient = new double[nGridXY*nGridXY*nGridZ]();
		arr_Occ      = new double[nGridXY*nGridXY*nGridZ]();
		arr_nutrient_new = new double[nGridXY*nGridXY*nGridZ]();

		arr_rng = new std::mt19937[nGridXY*nGridXY*nGridZ];

		rng_state = new curandState[nGridXY*nGridXY*nGridZ];

		arr_M = new double[nGridXY*nGridXY*nGridZ]();
		arr_GrowthModifier = new double[nGridXY*nGridXY*nGridZ]();
		arr_p = new double[nGridXY*nGridXY*nGridZ]();

		// Initialize arrays
		for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++) {
					 for (int k = 0; k < nGridZ; k++) {
								arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k] = n_0 / 1e12 * dV;
								arr_rng[i*nGridXY*nGridZ + j*nGridZ + k].seed(i*nGridXY*nGridZ + j*nGridZ + k);
						}
				}
		}

		// Compute the size of the time step
		ComputeTimeStep();

		// Store the parameters
		WriteLog();

		// Convert parameters to match gridpoint ///////
		// Adjust eta to match volume
		// eta/V is the number of collisions per hour for a single target
		eta = eta / dV;    // Number of collisions per gridpoint per hour

		// Adjust carrying capacity
		K = K * n_0 / 1e12 * dV;   // Deterine the Monod growth factor used in n(i,j)/(n(i,j)+K)

		// Initialize the bacteria and phage populations
		spawnBacteria();

		if (T_i <= dT) {
				spawnPhages();
				T_i = -1;
		}
}


// Spawns the bacteria
void Colonies3D::spawnBacteria() {

		// Determine the number of cells to spawn
		double nBacteria = round(L * L * H * B_0 / 1e12);

		// Average bacteria per gridpoint
		double avgBacteria = nBacteria / (nGridXY * nGridXY * nGridZ);

		// Keep track of the number of cells spawned
		double numB = 0;

		// Initialize cell and phage populations
		if (nBacteria > (nGridXY * nGridXY * nGridZ)) {
				for (int k = 0; k < nGridZ; k++) {
						for (int j = 0; j < nGridXY; j++) {
								for (int i = 0; i < nGridXY; i++) {

										// Compute the number of bacteria to land in this gridpoint
										double BB = RandP(avgBacteria);
										if (BB < 1) continue;

										// Store the number of clusters in this gridpoint
										arr_nC[i*nGridXY*nGridZ + j*nGridZ + k] = BB;

										// Add the bacteria
										arr_B[i*nGridXY*nGridZ + j*nGridZ + k] = BB;
										numB += BB;
								}
						}
				}
		}

		// Correct for underspawning
		while (numB < nBacteria) {

				// Choose random point in space
				int i = RandI(nGridXY - 1);
				int j = RandI(nGridXY - 1);
				int k = RandI(nGridZ  - 1);

				if (reducedBoundary) {
						i = 0;
						j = 0;
				}

				// Add the bacteria
				arr_B[i*nGridXY*nGridZ + j*nGridZ + k]++;
				arr_nC[i*nGridXY*nGridZ + j*nGridZ + k]++;

				numB++;


		}

		// Correct for overspawning
		while (numB > nBacteria) {
				int i = RandI(nGridXY - 1);
				int j = RandI(nGridXY - 1);
				int k = RandI(nGridZ  - 1);

				if (arr_B[i*nGridXY*nGridZ + j*nGridZ + k] < 1) continue;

				arr_B[i*nGridXY*nGridZ + j*nGridZ + k]--;
				numB--;
				arr_nC[i*nGridXY*nGridZ + j*nGridZ + k]--;
		}

		// Count the initial occupancy
		for (int k = 0; k < nGridZ; k++ ) {
				for (int j = 0; j < nGridXY; j++ ) {
						for (int i = 0; i < nGridXY; i++) {
								if (arr_B[i*nGridXY*nGridZ + j*nGridZ + k] > 0.0) {
										initialOccupancy++;
								}
						}
				}
		}

		// Determine the occupancy
		for (int k = 0; k < nGridZ; k++ ) {
				for (int j = 0; j < nGridXY; j++ ) {
						for (int i = 0; i < nGridXY; i++) {
								arr_Occ[i*nGridXY*nGridZ + j*nGridZ + k] = arr_B[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
						}
				}
		}
}


// Spawns the phages
void Colonies3D::spawnPhages() {

		 // Determine the number of phages to spawn
		double nPhages = (double)round(L * L * H * P_0 / 1e12);

		// Apply generic spawning
		if (not experimentalConditions) {

				double numP = 0;
				if (nPhages <= nGridXY * nGridXY * nGridZ ) {
						for (double n = 0; n < nPhages; n++) {
								int i = RandI(nGridXY - 1);
								int j = RandI(nGridXY - 1);
								int k = RandI(nGridZ  - 1);
								arr_P[i*nGridXY*nGridZ + j*nGridZ + k]++;
								numP++;
						}
				} else {
						for (int k = 0; k < nGridZ; k++ ) {
								for (int j = 0; j < nGridXY; j++ ) {
										for (int i = 0; i < nGridXY; i++) {
												double PP = RandP(nPhages / (double)(nGridXY * nGridXY * nGridZ));

												if (PP < 1) continue;
												arr_P[i*nGridXY*nGridZ + j*nGridZ + k] = PP;
												numP += PP;
										}
								}
						}
						// Correct for overspawning
						while (numP > nPhages) {
								int i = RandI(nGridXY - 1);
								int j = RandI(nGridXY - 1);
								int k = RandI(nGridZ - 1);

								if (arr_P[i*nGridXY*nGridZ + j*nGridZ + k] > 0) {
										arr_P[i*nGridXY*nGridZ + j*nGridZ + k]--;
										numP--;
								}
						}
						// Correct for underspawning
						while (numP < nPhages) {
								int i = RandI(nGridXY - 1);
								int j = RandI(nGridXY - 1);
								int k = RandI(nGridZ - 1);

								arr_P[i*nGridXY*nGridZ + j*nGridZ + k]++;
								numP++;
						}
				}

		} else { // Apply scenario specific settings

				// Determine the number of phages to spawn
				double nPhages = (double)round(L * L * H * P_0 / 1e12);
				double numP = 0;
				if (nPhages <= nGridXY * nGridXY) {
						for (double n = 0; n < nPhages; n++) {
								arr_P[RandI(nGridXY - 1)*nGridXY*nGridZ + RandI(nGridXY - 1)*nGridZ + nGridZ - 1]++;
								numP++;
						}
				} else {
						for (int j = 0; j < nGridXY; j++ ) {
								for (int i = 0; i < nGridXY; i++ ) {
										arr_P[i*nGridXY*nGridZ + j*nGridZ + nGridZ - 1] = RandP(nPhages / (double)(nGridXY * nGridXY * nGridZ));
										numP += arr_P[i*nGridXY*nGridZ + j*nGridZ + nGridZ - 1];
								}
						}
						// Correct for overspawning
						while (numP > nPhages) {
								int i = RandI(nGridXY - 1);
								int j = RandI(nGridXY - 1);

								if (arr_P[i*nGridXY*nGridZ + j*nGridZ + nGridZ - 1] > 0) {
										arr_P[i*nGridXY*nGridZ + j*nGridZ + nGridZ - 1]--;
										numP--;
								}
						}
						// Correct for underspawning
						while (numP < nPhages) {
								int i = RandI(nGridXY - 1);
								int j = RandI(nGridXY - 1);

								arr_P[i*nGridXY*nGridZ + j*nGridZ + nGridZ - 1]++;
								numP++;
						}
				}
		}
}

// Computes the size of the time-step needed
void Colonies3D::ComputeTimeStep() {

		if (this->dT > 0) return;

		// Compute the step size
		double dXY = L / (double)nGridXY;
		double dZ  = H / (double)nGridZ;
		assert(dXY == dZ);
		double dx  = dXY;

		// Compute the time-step size
		int limiter = 0;

		double dT = min(pow(10,-2), 1 / nSamp);
		double dt;

		// Compute time-step limit set by D_P (LambdaP < 0.1)
		if (D_P > 0) {
				dt = pow(dx, 2) * 0.1 / (2 * D_P);
				if (dt < dT) {
						dT = dt;
						limiter = 1;
				}
		}

		// Compute time-step limit set by D_B (LambdaP < 0.1)
		if (D_B > 0) {
				dt = pow(dx, 2) * 0.1 / (2 * D_B);
				if (dt < dT) {
						dT = dt;
						limiter = 2;
				}
		}

		// Compute time-step limit set by D_n (D_n *dT/pow(dx,2) < 1/8)
		dt = pow(dx, 2) / (8 * D_n);
		if (dt < dT) {

				dT = dt;
				limiter = 3;
		}

		// Compute time-step limit set by r (r*dT < 0.25)
		if (r > 0.0) {
				dt = 0.25 / r;
				if (dt < dT) {
						dT = dt;
						limiter = 4;
				}
		}

		// Compute time-step limit set by g (g*dT < 0.1)
		dt = 0.1 / g;
		if (dt < dT) {
				dT = dt;
				limiter = 5;
		}


		// Compute time-step limit set by delta (delta*dT < 0.1)
		dt = 0.1 / delta;
		if (dt < dT) {

				dT = dt;
				limiter = 6;
		}

		// Get the order of magnitude of the timestep
		double m = floor(log10(dT));

		// Round remainder to 1, 2 or 5
		double r = round(dT * pow(10, -m));
		if (r >= 5)      dT = 5*pow(10, m);
		else if (r >= 2) dT = 2*pow(10, m);
		else             dT =   pow(10, m);

		if (this->dT != dT) {
				this->dT = dT;

				switch(limiter){
						case 1:
								cout << "\tdT is Limited by D_P" << "\n";
								break;
						case 2:
								cout << "\tdT is Limited by D_B" << "\n";
								break;
						case 3:
								cout << "\tdT is Limited by D_n" << "\n";
								break;
						case 4:
								cout << "\tdT is Limited by r" << "\n";
								break;
						case 5:
								cout << "\tdT is Limited by g" << "\n";
								break;
						case 6:
								cout << "\tdT is Limited by delta" << "\n";
								break;
				}
		}

		// Compute the jumping probabilities
		lambdaB = 2 * D_B * dT / pow(dx, 2);
		if (lambdaB > 0.1) {
				cout << "lambdaB = " << lambdaB << "\n";
				assert(lambdaB <= 0.1);
		}

		lambdaP = 2 * D_P * dT / pow(dx, 2);
		if (lambdaP > 0.1) {
				cout << "lambdaP = " << lambdaP << "\n";
				assert(lambdaP <= 0.1);
		}

}

// Returns the number of events ocurring for given n and p
double Colonies3D::ComputeEvents(double n, double p, int flag, int i, int j, int k) {

		// Trivial cases
		if (p == 1) return n;
		if (p == 0) return 0.0;
		if (n < 1)  return 0.0;

		double N = RandP(n*p, i, j, k);

		return round(N);
}

// Returns the number of events ocurring for given n and p, flat array
double Colonies3D::ComputeEvents(double n, double p, int flag, int i) {

		// Trivial cases
		if (p == 1) return n;
		if (p == 0) return 0.0;
		if (n < 1)  return 0.0;

		double N = RandP(n*p, i);

		return round(N);
}

// Computes how many particles has moved to neighbouing points
void Colonies3D::ComputeDiffusion(double n, double lambda, double* n_0, double* n_u, double* n_d, double* n_l, double* n_r, double* n_f, double* n_b, int flag, int i, int j, int k) {

		// Reset positions
		*n_0 = 0.0;
		*n_u = 0.0;
		*n_d = 0.0;
		*n_l = 0.0;
		*n_r = 0.0;
		*n_f = 0.0;
		*n_b = 0.0;

		// Trivial case
		if (n < 1) return;

		// Check if diffusion should occur
		if ((lambda == 0) or (nGridXY == 1)) {
				*n_0 = n;
				return;
		}

		if (lambda*n < 5) {   // Compute all movement individually

				for (int l = 0; l < round(n); l++) {

						double r = Rand(arr_rng[i*nGridXY*nGridZ + j*nGridZ + k]);

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
				double N = RandP(3*lambda*n, i, j, k); // Factor of 3 comes from 3D

				*n_u = RandP(N/6, i, j, k);
				*n_d = RandP(N/6, i, j, k);
				*n_l = RandP(N/6, i, j, k);
				*n_r = RandP(N/6, i, j, k);
				*n_f = RandP(N/6, i, j, k);
				*n_b = RandP(N/6, i, j, k);
				*n_0 = n - (*n_u + *n_d + *n_l + *n_r + *n_f + *n_b);
		}

		*n_u = round(*n_u);
		*n_d = round(*n_d);
		*n_l = round(*n_l);
		*n_r = round(*n_r);
		*n_f = round(*n_f);
		*n_b = round(*n_b);
		*n_0 = n - (*n_u + *n_d + *n_l + *n_r + *n_f + *n_b);

		assert(*n_0 >= 0);
		assert(*n_u >= 0);
		assert(*n_d >= 0);
		assert(*n_l >= 0);
		assert(*n_r >= 0);
		assert(*n_f >= 0);
		assert(*n_b >= 0);
		assert(fabs(n - (*n_0 + *n_u + *n_d + *n_l + *n_r + *n_f + *n_b)) < 1);

}


// Settings /////////////////////////////////////////////////////////////////////////////
void Colonies3D::SetLength(double L){this->L=L;}                                 // Set the side-length of the simulation
void Colonies3D::SetHeight(double H) {this->H=H;}                                // Set the height of the simulation}
void Colonies3D::SetGridSize(double nGrid){this->nGridXY=nGrid;}                 // Set the number of gridpoints
void Colonies3D::SetTimeStep(double dT){this->dT=dT;}                            // Set the time step size
void Colonies3D::SetSamples(int nSamp){this->nSamp=nSamp;}                       // Set the number of output samples

void Colonies3D::PhageInvasionStartTime(double T_i){this->T_i=T_i;}              // Sets the time when the phages should start infecting

void Colonies3D::CellGrowthRate(double g){this->g=g;}                            // Sets the maximum growthrate
void Colonies3D::CellCarryingCapacity(double K){this->K=K;}                      // Sets the carrying capacity
void Colonies3D::CellDiffusionConstant(double D_B){this->D_B=D_B;}               // Sets the diffusion constant of the phages

void Colonies3D::PhageBurstSize(int beta){this->beta=beta;}                      // Sets the size of the bursts
void Colonies3D::PhageAdsorptionRate(double eta){this->eta=eta;}                 // sets the adsorption parameter eta
void Colonies3D::PhageDecayRate(double delta){this->delta=delta;}                // Sets the decay rate of the phages
void Colonies3D::PhageInfectionRate(double r){this->r=r;}                        // Sets rate of the infection increaasing in stage
void Colonies3D::PhageDiffusionConstant(double D_P){this->D_P=D_P;}              // Sets the diffusion constant of the phages

// Sets latency time of the phage (r and tau are related by r = 10 / tau)
void Colonies3D::PhageLatencyTime(double tau) {
		if (tau > 0.0) r = 10 / tau;
		else r = 0.0;
}

void Colonies3D::SurfacePermeability(double zeta){this->zeta=zeta;}             // Sets the permeability of the surface

void Colonies3D::InitialNutrient(double n_0){this->n_0=n_0;}                    // Sets the amount of initial nutrient
void Colonies3D::NutrientDiffusionConstant(double D_n){this->D_n=D_n;}          // Sets the nutrient diffusion rate

void Colonies3D::SimulateExperimentalConditions(){experimentalConditions=true;} // Sets the simulation to spawn phages at top layer and only have x-y periodic boundaries

void Colonies3D::DisableShielding(){shielding=false;}                           // Sets shielding bool to false
void Colonies3D::DisablesClustering(){clustering=false;}                        // Sets clustering bool to false
void Colonies3D::ReducedBurstSize(){reducedBeta=true;}                          // Sets the simulation to limit beta as n -> 0

// Sets the reduced boundary bool to true and the value of s
void Colonies3D::ReducedBoundary(int s) {
		this->s = s;
		reducedBoundary = true;
}

void Colonies3D::SetAlpha(double alpha){this->alpha=alpha;}                     // Sets the value of alpha

// Helping functions ////////////////////////////////////////////////////////////////////
// Returns random integter between 0 and n
int Colonies3D::RandI(int n) {

		// Set limit on distribution
		uniform_int_distribution <int> distr(0, n);

		return distr(rng);
}

// Returns random double between 0 and 1
double Colonies3D::Rand(std::mt19937 rng) {

		// Set limit on distribution
		uniform_real_distribution <double> distr(0, 1);

		return distr(rng);
}

// Returns random normal dist. number with mean m and variance s^2
double Colonies3D::RandN(double m, double s) {

		// Set limit on distribution
		normal_distribution <double> distr(m, s);

		return distr(rng);
}

// Returns poisson dist. number with mean l
double Colonies3D::RandP(double l, int i, int j, int k) {

		// Set limit on distribution
		poisson_distribution <long long> distr(l);

		return distr(arr_rng[i*nGridXY*nGridZ + j*nGridZ + k]);
}

// Returns poisson dist. number with mean l, flat array
double Colonies3D::RandP(double l, int i) {

		// Set limit on distribution
		poisson_distribution <long long> distr(l);

		return distr(arr_rng[i]);
}

// Returns poisson dist. number with mean l
double Colonies3D::RandP(double l) {

		// Set limit on distribution
		poisson_distribution <long long> distr(l);

		return distr(rng);
}

// Returns poisson dist. number with mean l
double Colonies3D::RandP_fast(double l) {

		double N;

		if (l < 60) {

				double L = exp(-l);
				double p = 1;
				N = 0;
				do {
						N++;
						p *= drand48();
				} while (p > L);
				N--;

		} else {

				double r;
				double x;
				double pi = 3.14159265358979;
				double sqrt_l = sqrt(l);
				double log_l = log(l);
				double g_x;
				double f_m;

				do {
						do {
								x = l + sqrt_l*tan(pi*(drand48()-1/2.0));
						} while (x < 0);

						g_x = sqrt_l/(pi*((x-l)*(x-l) + l));
						N = floor(x);

						double xx = N + 1;
						double pi = 3.14159265358979;
						double xx2 = xx*xx;
						double xx3 = xx2*xx;
						double xx5 = xx3*xx2;
						double xx7 = xx5*xx2;
						double xx9 = xx7*xx2;
						double xx11 = xx9*xx2;
						double lgxx = xx*log(xx) - xx - 0.5*log(xx/(2*pi)) +
						1/(12*xx) - 1/(360*xx3) + 1/(1260*xx5) - 1/(1680*xx7) +
						1/(1188*xx9) - 691/(360360*xx11);

						f_m = exp(N*log_l - l - lgxx);
						r = f_m / g_x / 2.4;
				} while (drand48() > r);
		}

		return round(N);

}

// Sets the seed of the random number generator
void Colonies3D::SetRngSeed(int n) {
		rngSeed = n;
}

// Write a log.txt file
void Colonies3D::WriteLog() {
		if ((not f_log.is_open()) and (not exit)) {

				// Open the file stream and write the command
				f_log.open(path + "/log.txt", fstream::trunc);

				// Store the initial densities
				f_log << "B_0 = " << fixed << setw(12)  << B_0      << "\n";    // Initial density of bacteria
				f_log << "P_0 = " << fixed << setw(12)  << P_0      << "\n";    // Intiial density of phages
				f_log << "n_0 = " << fixed << setw(12)  << n_0      << "\n";    // Intiial density of nutrient
				f_log << "K = "   << fixed << setw(12)  << K        << "\n";    // Carrying capacity
				f_log << "L = "                         << L        << "\n";    // Side-length of simulation array
				f_log << "H = "                         << H        << "\n";    // height of simulation array
				f_log << "nGridXY = "                   << nGridXY  << "\n";    // Number of gridpoints
				f_log << "nGridZ = "                    << nGridZ   << "\n";    // Number of gridpoints
				f_log << "nSamp = "                     << nSamp    << "\n";    // Number of samples to save per simulation hour
				f_log << "g = "                         << g        << "\n";    // Growth rate for the cells
				f_log << "alpha = "                     << alpha    << "\n";    // Reinfection Percentage
				f_log << "beta = "                      << beta     << "\n";    // Multiplication factor phage
				f_log << "eta = "                       << eta      << "\n";    // Adsorption coefficient
				f_log << "delta = "                     << delta    << "\n";    // Rate of phage decay
				f_log << "r = "                         << r        << "\n";    // Constant used in the time-delay mechanism
				f_log << "zeta = "                      << zeta     << "\n";    // Permeability of surface
				f_log << "D_B = "                       << D_B      << "\n";    // Diffusion constant for the cells
				f_log << "D_P = "                       << D_P      << "\n";    // Diffusion constant for the phage
				f_log << "D_n = "                       << D_n      << "\n";    // Diffusion constant for the nutrient
				f_log << "dT = "                        << dT       << "\n";    // Time-step size
				f_log << "T_end = "                     << T_end    << "\n";    // Time when the simulation stops

				f_log << "rngSeed = "                   << rngSeed  << "\n";    // Random number seed  ( set to -1 if unused )

				f_log << "s = "                         << s        << "\n";    // The reduction of the phage boundary                       = 1;

				f_log << "experimentalConditions = "    << experimentalConditions   << "\n";
				f_log << "clustering = "                << clustering               << "\n";
				f_log << "shielding = "                 << shielding                << "\n";
				f_log << "reducedBeta = "               << reducedBeta              << "\n";
				f_log << "reducedBoundary = "           << reducedBoundary          << endl;

		}
}

// File outputs /////////////////////////////////////////////////////////////////////////

// Stop simulation when all cells are dead
void Colonies3D::FastExit(){fastExit=true;}

// Sets the simulation to export everything
void Colonies3D::ExportAll(){exportAll=true;}

// Master function to export the data
void Colonies3D::ExportData_arr(double t, std::string filename_suffix){

		// Verify the file stream is open
		string fileName = "PopulationSize_"+filename_suffix;
		OpenFileStream(f_N, fileName);


		double accuB = 0.0;
		double accuI = 0.0;
		double accuP = 0.0;
		double accuNutrient = 0.0;
		double accuClusters = 0.0;
		for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {
								accuB += arr_B[i*nGridXY*nGridZ + j*nGridZ + k];
								accuI += arr_I0[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I1[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I2[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I3[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I4[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I5[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I6[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I7[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I8[i*nGridXY*nGridZ + j*nGridZ + k] + arr_I9[i*nGridXY*nGridZ + j*nGridZ + k];
								accuP += arr_P[i*nGridXY*nGridZ + j*nGridZ + k];
								accuNutrient += arr_nutrient[i*nGridXY*nGridZ + j*nGridZ + k];
								accuClusters += arr_nC[i*nGridXY*nGridZ + j*nGridZ + k];
						}
				}
		}

		// Writes the time, number of cells, number of infected cells, number of phages
		f_N << fixed    << setprecision(2);
		f_N << setw(6)  << t       << "\t";
		f_N << setw(12) << round(accuB)    << "\t";
		f_N << setw(12) << round(accuI)    << "\t";
		f_N << setw(12) << round(accuP)    << "\t";

		int nz = 0;
		for (int i = 0; i < nGridXY; i++) {
				for (int j = 0; j < nGridXY; j++ ) {
						for (int k = 0; k < nGridZ; k++ ) {
								if (arr_B[i*nGridXY*nGridZ + j*nGridZ + k] > 0) {
										nz++;
								}
						}
				}
		}

		f_N << setw(12) << static_cast<double>(nz) / initialOccupancy << "\t";
		f_N << setw(12) << n_0 / 1e12 * pow(L, 2) * H - accuNutrient << "\t";
		f_N << setw(12) << accuClusters << endl;

		if (exportAll) {
				// Save the position data
				// Verify the file stream is open
				fileName = "CellDensity_"+filename_suffix;
				OpenFileStream(f_B, fileName);

				fileName = "InfectedDensity_"+filename_suffix;
				OpenFileStream(f_I, fileName);

				fileName = "PhageDensity_"+filename_suffix;
				OpenFileStream(f_P, fileName);

				fileName = "NutrientDensity_"+filename_suffix;
				OpenFileStream(f_n, fileName);

				// Write file as MATLAB would a 3D matrix!
				// row 1 is x_vector, for y_1 and z_1
				// row 2 is x_vector, for y_2 and z_1
				// row 3 is x_vector, for y_3 and z_1
				// ...
				// When y_vector for x_n has been printed, it goes:
				// row n+1 is x_vector, for y_1 and z_2
				// row n+2 is x_vector, for y_2 and z_2
				// row n+3 is x_vector, for y_3 and z_2
				// ... and so on

				// Loop over z
				for (int z = 0; z < nGridZ; z++) {

						// Loop over x
						for (int x = 0; x < nGridXY; x++) {

								// Loop over y
								for (int y = 0; y < nGridXY - 1; y++) {
										#define XYZ x*nGridXY*nGridZ+y*nGridZ+z

										f_B << setw(6) << arr_B[x*nGridXY*nGridZ + y*nGridZ + z] << "\t";
										f_P << setw(6) << arr_P[x*nGridXY*nGridZ + y*nGridZ + z] << "\t";
										double nI = round(arr_I0[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I1[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I2[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I3[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I4[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I5[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I6[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I7[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I8[x*nGridXY*nGridZ + y*nGridZ + z] + arr_I9[x*nGridXY*nGridZ + y*nGridZ + z]);
										f_I << setw(6) << nI       << "\t";
										f_n << setw(6) << arr_nutrient[x*nGridXY*nGridZ + y*nGridZ + z] << "\t";
								}

								#define XnGridXYZ x*nGridXY*nGridZ+(nGridXY-1)*nGridZ+z
								// Write last line ("\n" instead of tab)
								f_B << setw(6) << round(arr_B[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z]) << "\n";
								f_P << setw(6) << round(arr_P[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z]) << "\n";
								double nI = round(arr_I0[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I1[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I2[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I3[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I4[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I5[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I6[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I7[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I8[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z] + arr_I9[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z]);
								f_I << setw(6) << nI                        << "\n";
								f_n << setw(6) << round(arr_nutrient[x*nGridXY*nGridZ + (nGridXY - 1)*nGridZ + z]) << "\n";
						}
				}
		}
}

// Open filstream if not allready opened
void Colonies3D::OpenFileStream(ofstream& stream, string& fileName) {

		// Check that if file stream is open.
		if ((not stream.is_open()) and (not exit)) {

				// Debug info
				cout << "\tSaving data to file: " << path << "/" << fileName << ".txt" << "\n";

				// Check if the output file exists
				time_t theTime = time(NULL);
				struct tm *aTime = localtime(&theTime);

				string streamPath;
				streamPath = path+"/"+fileName+"_"+std::to_string(aTime->tm_hour)+"_"+std::to_string(aTime->tm_min)+".txt";

				// Open the file stream
				stream.open(streamPath, fstream::trunc);

				// Check stream is open
				if ((not exit) and (not stream.is_open())) {
						cerr << "\t>>Could not open filestream \"" << streamPath << "\"! Exiting..<<" << "\n";
						f_log <<  ">>Could not open filestream \"" << streamPath << "\"! Exiting..<<" << "\n";
						exit = true;
				};

				// Write meta data to the data file
				stream << "Datatype: "  << fileName << "\n";
		}
}

// Generates a save path for datafiles
string Colonies3D::GeneratePath() {

			// Generate a directory path
		string prefix = "data";    // Data folder name

		// Create the path variable
		string path_s = prefix;

		// Check if user has specified numbered folder
		if (path.empty()) {

				// Get current date
				time_t t = time(0);                               // Get time now
				struct tm tstruct;                                // And format the date
				tstruct = *localtime(&t);                         // as "MNT_DD_YY" for folder name
				char buffer[80];                                  // Create a buffer to store the date
				strftime(buffer, sizeof(buffer), "%F", &tstruct); // Store the formated foldername in buffer
				string dateFolder(buffer);

				// Add datefolder to path
				path_s += "/";
				path_s += dateFolder;

				// Check if path exists
				struct stat info;
				if (not(stat(path_s.c_str(), &info) == 0 && S_ISDIR(info.st_mode))) {
						// Create path if it does not exist
						mkdir(path_s.c_str(), 0700);
				}

				// Loop over folders in date folder, to find current number
				int currentNumerateFolder = 1;
				DIR *dir;
				if ((dir = opendir (path_s.c_str())) != NULL) {
						struct dirent *ent;
						while ((ent = readdir (dir)) != NULL) {
								if (ent->d_type == DT_DIR) {
										// Skip . or ..
										if (ent->d_name[0] == '.') {continue;}
										currentNumerateFolder++;        // Increment folder number
								}
						}
						closedir (dir);
				}

				// Append numerate folder
				path_s += "/";
				path_s += to_string(currentNumerateFolder);

				// Check if path exists
				if (not(stat(path_s.c_str(), &info) == 0 && S_ISDIR(info.st_mode))) {
						// Create path if it does not exist
						mkdir(path_s.c_str(), 0700);
				}

		} else {    // User has specified a path

				// This path maybe more than one layer deep, so attempt to make it recursively
				int len = path.length();

				// Boolean to see name of first folder
				bool firstFolder = true;

				string folder = "";
				for (int i = 0; i < len; i++) {
						folder += path[i]; // Append char to folder name

						// If seperator is found or if end of path is reached, construct folder
						if ((path[i] == '/') or (i == len - 1)) {

								// If seperator is found, remove it:
								if (path[i] == '/') folder.pop_back();

								// Check if this is the first subfolder
								if (firstFolder) {
										firstFolder = false;

										// Check if first folder contains date format
										if (not ((folder.length() == 10) and(folder[4] == '-') and (folder[7] == '-'))) {

												// Get current date
												time_t t = time(0);                               // Get time now
												struct tm tstruct;                                // And format the date
												tstruct = *localtime(&t);                         // as "MNT_DD_YY" for folder name
												char buffer[80];                                  // Create a buffer to store the date
												strftime(buffer, sizeof(buffer), "%F", &tstruct); // Store the formated foldername in buffer
												string dateFolder(buffer);

												// Add datefolder to path
												path_s += "/";
												path_s += dateFolder;

												// Check if path exists
												struct stat info;
												if (not(stat(path_s.c_str(), &info) == 0 && S_ISDIR(info.st_mode))) {
														// Create path if it does not exist
														mkdir(path_s.c_str(), 0700);
												}
										}
								}

								// Append folder to path
								path_s += "/";
								path_s += folder;

								// Make folder
								struct stat info;
								if (not(stat(path_s.c_str(), &info) == 0 && S_ISDIR(info.st_mode)))
								{ // Create path if it does not exist
										mkdir(path_s.c_str(), 0700);
								}

								folder = ""; // Reset folder
						}
				}
		}

		return path_s;
}

// Sets the folder number (useful when running parralel code)
void Colonies3D::SetFolderNumber(int number) {path = to_string(number);}

// Sets the folder path (useful when running parralel code)
void Colonies3D::SetPath(std::string& path) {this->path = path;}

// Returns the save path
std::string Colonies3D::GetPath() {
		return path;
}


// Clean up /////////////////////////////////////////////////////////////////////////////

// Delete the data folder
void Colonies3D::DeleteFolder() {
		DeleteFolderTree(path.c_str());
}

// Delete folders recursively
void Colonies3D::DeleteFolderTree(const char* directory_name) {

		DIR*            dp;
		struct dirent*  ep;
		char            p_buf[512] = {0};


		dp = opendir(directory_name);

		while ((ep = readdir(dp)) != NULL) {
				// Skip self dir "."
				if (strcmp(ep->d_name, ".") == 0 || strcmp(ep->d_name, "..") == 0) continue;

				sprintf(p_buf, "%s/%s", directory_name, ep->d_name);

				// Is the path a folder?
				struct stat s_buf;
				int IsDirectory = -1;
				if (stat(p_buf, &s_buf)){
						IsDirectory = 0;
				} else {
						IsDirectory = S_ISDIR(s_buf.st_mode);
				}

				// If it is a folder, go recursively into
				if (IsDirectory) {
						DeleteFolderTree(p_buf);
				} else {    // Else delete the file
						unlink(p_buf);
				}
		}

		closedir(dp);
		rmdir(directory_name);
}

// Destructor
Colonies3D::~Colonies3D() {

		// Close filestreams
		if (f_B.is_open()) {
				f_B.flush();
				f_B.close();
		}
		if (f_I.is_open()) {
				f_I.flush();
				f_I.close();
		}
		if (f_P.is_open()) {
				f_P.flush();
				f_P.close();
		}
		if (f_N.is_open()) {
				f_N.flush();
				f_N.close();
		}
		if (f_log.is_open()) {
				f_log.flush();
				f_log.close();
		}

		 // Delete arrays
		 delete[] arr_B;
		 delete[] arr_I0;
		 delete[] arr_I1;
		 delete[] arr_I2;
		 delete[] arr_I3;
		 delete[] arr_I4;
		 delete[] arr_I5;
		 delete[] arr_I6;
		 delete[] arr_I7;
		 delete[] arr_I8;
		 delete[] arr_I9;
		 delete[] arr_P;
		 delete[] arr_nC;

		 delete[] arr_B_new;
		 delete[] arr_I0_new;
		 delete[] arr_I1_new;
		 delete[] arr_I2_new;
		 delete[] arr_I3_new;
		 delete[] arr_I4_new;
		 delete[] arr_I5_new;
		 delete[] arr_I6_new;
		 delete[] arr_I7_new;
		 delete[] arr_I8_new;
		 delete[] arr_I9_new;
		 delete[] arr_P_new;

		 delete[] arr_nutrient;
		 delete[] arr_Occ;
		 delete[] arr_nutrient_new;

		 delete[] arr_rng;

		 delete[] arr_M;
		 delete[] arr_GrowthModifier;
}
