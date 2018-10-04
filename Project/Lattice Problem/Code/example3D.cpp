#include "colonies3D.hpp"

using namespace std;

int main(int argc, char** argv){

     // Default parameters
    int nGrid 	 = 50;
    int beta  	 = 100;
    double delta = 1.0/10.0;
    double eta   = 1e4;
    double tau   = 0.5;

    for (int i = 0; i < argc; i++) {
        if (i == 1) {
            nGrid = atoi(argv[i]);
        } else if (i == 2) {
            beta  = atoi(argv[i]);
        } else if (i == 3) {
            delta = atof(argv[i]);
        } else if (i == 4) {
            eta   = atof(argv[i]);
        } else if (i == 5) {
            tau   = atof(argv[i]);
        }
    }

    double B_0 = pow(10,4);
    double P_0 = pow(10,5);
    double T   = 20;

    string pathName = "3D_Example_Full_Model";

    // Load simulation module
    Colonies3D s(B_0, P_0);

    s.SetPath(pathName);

    s.SetRngSeed(5);

    s.SetGridSize(nGrid);
    s.PhageBurstSize(beta);
    s.PhageDecayRate(delta);
    s.PhageAdsorptionRate(eta);
    s.PhageLatencyTime(tau);
    s.SetSamples(10);

    s.ExportAll();

    // Start the simulation
    s.Run(T);

    return 0;
}
