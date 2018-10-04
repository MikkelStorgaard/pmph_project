#include "Simulation.hpp"

using namespace std;

int main(int argc, char** argv){

	// Default parameters
	int N = 1;			// Number of cells
	double T = 1;		// Simulation time length
	double T_i = 0; 	// Time phage-infection begins
	double P_0 = -1; 	// Concentration of Phage
	double L =   65; 	// Concentration of bacteria

	for (int i = 0; i < argc; i++) {
		if (i==1) {
			N = atoi(argv[i]);
		}
		else if (i==2) {
			T = atof(argv[i]);
		}
		else if (i==3) {
			T_i = atof(argv[i]);
		}
		else if (i==4) {
			P_0 = atof(argv[i]);
			// P_0 = atof(argv[i])/pow(L,3);
		}
		else if (i==5) {
			L = atof(argv[i]);
		}
	}

	// Load simulation module
	Simulation s(N);
	// s.Quiet();
	s.SetRngSeed(0);

	// Set the infection time
	s.PhageInvasionStartTime(T_i);

	// Set initial phage density
	s.PhageInvasionType(3);			// 1: Single Infected cell, 2: Planar Phage Invasion, 3: Uniform Phage Invasion
	s.PhageInitialDensity(P_0);

	// Set initial bacteria
	s.CellInitialCount(1);

	// Set the lysogen frequency and burst size
	s.PhageLysogenFrequency(0.00);
	s.PhageBurstSize(100);

	// Set the infection time
	s.PhageInfectionRate(0.6);

	//s.SetSamples(1);

	// Set the data to export
	s.ExportCellData();
	// s.ExportPhageData();
	s.ExportColonySize();
	// s.ExportCellDensity2D();
	// s.ExportNutrient();

	// s.SetSamples(1);

	// Start the simulation
	s.Run(T);

	return 0;
}
