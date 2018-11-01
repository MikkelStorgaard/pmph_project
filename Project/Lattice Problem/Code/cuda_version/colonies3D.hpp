#ifndef COLONIES3DDEF
#define COLONIES3DDEF

#define ARMA_NO_DEBUG

#include <iostream>         // Input and output
#include <iomanip>          // Input and output formatting
#include <fstream>          // File streams

#include <random>           // Random numbers
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>           // Mathmatical constants
#include <algorithm>        // Mathmatical constants

#include <vector>           // C++ standard vector
#include <string.h>         // Strings

#include <cassert>          // Assertions

#include <sys/types.h>      // Packages for the directory
#include <sys/stat.h>       //    information,
#include <dirent.h>         //    handling and etc.
#include <unistd.h>
#include <ctime>            // Time functions


#define NUMTYPE_IS_FLOAT false
#if NUMTYPE_IS_FLOAT
typedef float numtype;
#else
typedef double numtype;
#endif


/* Class to contain simulation parameters, and which drives the simulation forward */
class Colonies3D {
 private:

  numtype B_0;                     // [CFU/mL] Initial density of bacteria
  numtype P_0;                     // [PFU/mL] Initial density of phages

  numtype K;                       //          Carrying capacity
  numtype n_0;                     // [1/ml]   Initial nutrient level (number of bacteria per ml)

  numtype L;                       // [µm]     Side-length of simulation array
  numtype H;                       // [µm]     Height of the simulation array
  int    nGridXY;                 //          Number of gridpoints
  int    nGridZ;                  //          Number of gridpoints
  int    volume;

  numtype nSamp;                   //          Number of samples to save per simulation hour

  numtype g;                       // [1/hour] Growth rate for the cells

  numtype alpha;                   //          Percentage of phages which escape the colony upon lysis
  int    beta;                    //          Multiplication factor phage
  numtype eta;                     //          Adsorption coefficient
  numtype delta;                   // [1/hour] Rate of phage decay
  numtype r;                       //          Constant used in the time-delay mechanism
  numtype zeta;                    //          permeability of colony surface

  numtype D_B;                     // [µm^2/hour] Diffusion constant for the cells
  numtype D_P;                     // [µm^2/hour] Diffusion constant for the phage
  numtype D_n;                     // [µm^2/hour] Diffusion constant for the nutrient

  numtype lambdaB;                 // Probability of cell to jump to neighbour point
  numtype lambdaP;                 // Probability of phage to jump to neighbour point

  numtype T;                       // [hours]  Current time
  numtype dT;                      // [hours]  Time-step size
  numtype T_end;                   // [hours]  End time of simulation
  numtype T_i;                     // [hours]  Time when the phage infections begins (less than 0 disables phage infection)

  numtype initialOccupancy;        // Number of gridpoints occupied initially;

  bool   exit;                    // Boolean to control early exit

  bool   Warn_g;                  //
  bool   Warn_r;                  //
  bool   Warn_eta;                // Booleans to keep track of warnings
  bool   Warn_delta;              //
  bool   Warn_density;            //
  bool   Warn_fastGrowth;         //

  bool   experimentalConditions;  // Booleans to control simulation type

  bool   clustering;              // When false, the ((B+I)/nC)^(1/3) factor is removed.
  bool   shielding;               // When true the simulation uses the shielding function (full model)
  bool   reducedBeta;             // When true the simulation modifies the burst size by the growthfactor

  bool   reducedBoundary;         // When true, bacteria are spawned at X = 0 and Y = 0. And phages are only spawned within nGrid/s boxes from (0,0,z).
  int    s;

  int    timeScaleSeperation;     // Indicates the difference in time scale between diffusion of nutrient

  bool   fastExit;                // Stop simulation when all cells are dead

  bool   exportAll;               // Boolean to export everything, not just populationsize

  numtype rngSeed;                 // The seed for the random number generator
  std::mt19937 rng;               // Mersenne twister, random number generator
  std::mt19937* arr_rng;          // Mersenne twister, random number generator
  std::mt19937* d_arr_rng;        // Mersenne twister, random number generator

	curandState *rng_state;
  curandState *d_rng_state;

  std::uniform_real_distribution  <numtype> rand;
  std::normal_distribution        <numtype> randn;

  std::ofstream f_B;              // Filestream to save configuration of sucebtible cells
  std::ofstream f_I;              // Filestream to save configuration of infected cells
  std::ofstream f_P;              // Filestream to save configuration of phages
  std::ofstream f_n;              // Filestream to save configuration of nutrient
  std::ofstream f_N;              // Filestream to save number agents
  std::ofstream f_log;            // Filestream to save log.txt
  std::ofstream f_kerneltimings;  // Filestream to save GPU kernel timings

  std::string path;               // Sets the path to store in

  // Coordinates of agents in the simulation
  numtype* arr_B;           // Sensitive bacteria
  numtype* arr_P;           // Phages
  numtype* arr_I0;          // Infected bacteria
  numtype* arr_I1;          // Infected bacteria
  numtype* arr_I2;          // Infected bacteria
  numtype* arr_I3;          // Infected bacteria
  numtype* arr_I4;          // Infected bacteria
  numtype* arr_I5;          // Infected bacteria
  numtype* arr_I6;          // Infected bacteria
  numtype* arr_I7;          // Infected bacteria
  numtype* arr_I8;          // Infected bacteria
  numtype* arr_I9;          // Infected bacteria
  numtype* arr_nC;          // Number of colonies in gridpoint

  numtype* arr_B_new;       // Sensitive bacteria
  numtype* arr_P_new;       // Phages
  numtype* arr_I0_new;      // Infected bacteria
  numtype* arr_I1_new;      // Infected bacteria
  numtype* arr_I2_new;      // Infected bacteria
  numtype* arr_I3_new;      // Infected bacteria
  numtype* arr_I4_new;      // Infected bacteria
  numtype* arr_I5_new;      // Infected bacteria
  numtype* arr_I6_new;      // Infected bacteria
  numtype* arr_I7_new;      // Infected bacteria
  numtype* arr_I8_new;      // Infected bacteria
  numtype* arr_I9_new;      // Infected bacteria

  // allocations for array
  numtype* d_arr_B;           // Sensitive bacteria
  numtype* d_arr_P;           // Phages
  numtype* d_arr_I0;          // Infected bacteria
  numtype* d_arr_I1;          // Infected bacteria
  numtype* d_arr_I2;          // Infected bacteria
  numtype* d_arr_I3;          // Infected bacteria
  numtype* d_arr_I4;          // Infected bacteria
  numtype* d_arr_I5;          // Infected bacteria
  numtype* d_arr_I6;          // Infected bacteria
  numtype* d_arr_I7;          // Infected bacteria
  numtype* d_arr_I8;          // Infected bacteria
  numtype* d_arr_I9;          // Infected bacteria
  numtype* d_arr_nC;          // Number of colonies in gridpoint

  numtype* d_arr_B_new;       // Sensitive bacteria
  numtype* d_arr_P_new;       // Phages
  numtype* d_arr_I0_new;      // Infected bacteria
  numtype* d_arr_I1_new;      // Infected bacteria
  numtype* d_arr_I2_new;      // Infected bacteria
  numtype* d_arr_I3_new;      // Infected bacteria
  numtype* d_arr_I4_new;      // Infected bacteria
  numtype* d_arr_I5_new;      // Infected bacteria
  numtype* d_arr_I6_new;      // Infected bacteria
  numtype* d_arr_I7_new;      // Infected bacteria
  numtype* d_arr_I8_new;      // Infected bacteria
  numtype* d_arr_I9_new;      // Infected bacteria

	// Privatized array.
	numtype* arr_M;
  numtype* arr_GrowthModifier;
  numtype* arr_p;

  // Nutrient matrix
  numtype* arr_nutrient;
  numtype* arr_nutrient_new;

  // Occupancy of grid
  numtype* arr_Occ;

	// Privatized array.
	numtype* d_arr_M;
  numtype* d_arr_GrowthModifier;
  numtype* d_arr_p;

  // Nutrient matrix
  numtype* d_arr_nutrient;
  numtype* d_arr_nutrient_new;

  // Occupancy of grid
  numtype* d_arr_Occ;

  numtype* d_arr_n_0;
  numtype* d_arr_n_u;
  numtype* d_arr_n_d;
  numtype* d_arr_n_l;
  numtype* d_arr_n_r;
  numtype* d_arr_n_f;
  numtype* d_arr_n_b;

  // Active-array
  bool* d_arr_IsActive;

  bool* skipArray;

  // Active-array
  bool* d_Warn_g;
  bool* d_Warn_fastGrowth;
  bool* d_Warn_r;
  bool* d_Warn_delta;
  bool* d_Warn_density;

  int errC;

 public:
  // Constructers
  explicit    Colonies3D(numtype B_0, numtype P_0);                           // Direct constructer

  // Driver
  int         Run_LoopDistributed_CPU(numtype T_end);                          // Controls the evaluation of the simulation
  int         Run_LoopDistributed_GPU(numtype T_end);                          // Controls the evaluation of the simulation
  int         Run_LoopDistributed_CPU_cuRand(numtype T_end);                          // Controls the evaluation of the simulation

 private:
	void 		    CopyToHost(numtype* hostArray, numtype* deviceArray, int failCode, int gridsz);
//	void 		    CopyToHost(bool hostElement, bool deviceElement, int failCode);
	void		    CopyToDevice(numtype* hostArray, numtype* deviceArray, int failCode, int gridsz);
//	void		    CopyToDevice(bool hostElement, bool deviceElement, int failCode);
	void 		    CopyAllToHost();
	void		    CopyAllToDevice();
  void        Initialize();                                                   // Initialize the simulation
  void        spawnBacteria();                                                // Spawns the bacteria
  void        spawnPhages();                                                  // Spawns the phages
  void        ComputeTimeStep();                                              // Computes the size of the time-step needed
  numtype      ComputeEvents(numtype n, numtype p, int flag, int i, int j, int k);                    // Returns the number of events ocurring for given n and p
  numtype      ComputeEvents(numtype n, numtype p, int flag, int i);                    // Returns the number of events ocurring for given n and p, flat array
  numtype      ComputeEvents_cuRand(numtype n, numtype p, int flag, int i, int j, int k);                    // Returns the number of events ocurring for given n and p
  void        ComputeDiffusion(numtype n, numtype lambda,                       // Computes how many particles has moved to neighbouing points
                    numtype* n_0, numtype* n_u, numtype* n_d, numtype* n_l, numtype* n_r, numtype* n_f, numtype* n_b, int flag, int i, int j, int k);
 public:
  void        SetLength(numtype L);                                            // Set the side-length of the simulation
  void        SetHeight(numtype H);                                            // Set the height of the simulation
  void        SetGridSize(numtype nGrid);                                      // Set the number of gridpoints
  void        SetTimeStep(numtype dT);                                         // Set the time step size
  void        SetSamples(int nSamp);                                          // Set the number of output samples

  void        PhageInvasionStartTime(numtype T_i);                             // Sets the time when the phages should start infecting

  void        CellGrowthRate(numtype g);                                       // Sets the maximum growthrate
  void        CellCarryingCapacity(numtype K);                                 // Sets the carrying capacity
  void        CellDiffusionConstant(numtype D_B);                              // Sets the diffusion constant of the phages

  void        PhageBurstSize(int beta);                                       // Sets the size of the bursts
  void        PhageAdsorptionRate(numtype eta);                                // sets the adsorption parameter gamma
  void        PhageDecayRate(numtype delta);                                   // Sets the decay rate of the phages
  void        PhageInfectionRate(numtype r);                                   // Sets rate of the infection increaasing in stage
  void        PhageDiffusionConstant(numtype D_P);                             // Sets the diffusion constant of the phages
  void        PhageLatencyTime(numtype tau);                                   // Sets latency time of the phage (r and tau are related by r = 10 / tau)

  void        SurfacePermeability(numtype zeta);                               // Sets the permeability of the surface

  void        InitialNutrient(numtype n_0);                                    // Sets the amount of initial nutrient
  void        NutrientDiffusionConstant(numtype D_n);                          // Sets the nutrient diffusion rate

  void        SimulateExperimentalConditions();                               // Sets the simulation to spawn phages at top layer and only have x-y periodic boundaries

  void        DisableShielding();                                             // Sets shielding bool to false
  void        DisablesClustering();                                           // Sets clustering bool to false
  void        ReducedBurstSize();                                             // Sets the simulation to limit beta as n -> 0

  void        ReducedBoundary(int s);                                         // Sets the reduced boundary bool to true and the value of s

  void        SetAlpha(numtype alpha);                                         // Sets the value of alpha

 private:
  // Helping functions
  int         RandI(int n);                                                   // Returns random integer less than n
  numtype      Rand(std::mt19937);                                             // Returns random numtype less than n
  numtype      RandN(numtype m, numtype s);                                      // Returns random normal dist. number with mean m and variance s^2
  numtype      RandP(numtype l, int i, int j, int k);                           // Returns poisson dist. number with mean l
  numtype      RandP(numtype l, int i);                           // Returns poisson dist. number with mean l, flat array
  numtype      RandP(numtype l);                                                // Returns poisson dist. number with mean l
  numtype      RandP_fast(numtype l);                                           // Returns poisson dist. number with mean l

 public:
  void        SetRngSeed(int n);                                              // Sets the seed of the random number generator

 private:
  void        WriteLog();                                                     // Write a log.txt file

 public:
  void        FastExit();                                                     // Stop simulation when all cells are dead
  void        ExportAll();                                                    // Sets the simulation to export everything

 private:
  void        ExportData_arr(numtype t, std::string filename_suffix);              // Master function to export the data
  void        ExportData_arr_reduced(numtype t, numtype nz, std::string filename_suffix);

  // Data handling
  void        OpenFileStream(std::ofstream& stream,                           // Open filstream if not allready opened
                  std::string& fileName);
  std::string GeneratePath();                                                 // Generates a save path for datafiles

 public:
  void        SetFolderNumber(int number);                                    // Sets the folder number (useful when running parralel code)
  void        SetPath(std::string& path);                                     // Sets the folder path (useful when running parralel code)

  // Get properties
  std::string GetPath();                                                      // Returns the save path
  int         GetTime();                                                      // Returns the time
  int         GetDeltaT();                                                    // Returns the time-step dT


  // Clean up
  void        DeleteFolder();                                                 // Delete the data folder
 private:
  void        DeleteFolderTree(const char* directory_name);                   // Delete folders recursively

 public:
    ~Colonies3D();                                                            // Destructor

};

#endif

