#include "colonies3D.hpp"

using namespace std;

int main(int argc, char** argv){

    // Default parameters
    int nGrid 	 = 10;
    int beta  	 = 100;
    double delta = 1.0/10.0;
    double eta   = 1e4;
    double tau   = 0.5;

    int k = 0;

    for (int i = 0; i < argc; i++) {
        if (i == 1) {
            k = atoi(argv[i]);
        }
        if (i == 2) {
            nGrid = atoi(argv[i]);
        }
    }

    double B_0 = pow(10,4);
    double P_0 = pow(10,5);
    double T   = 5;

    string pathName = "3D_Example";

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

    // s.ExportAll();

    // Start the simulation
    if ( k==0 ) {
        s.Run_LoopDistributed_GPU(T);
    } else if (k == 1) {
        s.Run_LoopDistributed_CPU(T);
    }

    return 0;
}
