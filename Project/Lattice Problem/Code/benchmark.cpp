#include "colonies3D.hpp"

using namespace std;

int main(int argc, char** argv){

     // Default parameters
    int nGrid 	 = 50;
    int beta  	 = 100;
    double delta = 1.0/10.0;
    double eta   = 1e4;
    double tau   = 0.5;
    double B_0 = pow(10,4);
    double P_0 = pow(10,5);
    double T   = 20;

    string pathName = "3D_Example_Full_Model";


    for (int k = 0; k < 3; k++) {
    for (int nGrid = 1; nGrid <= 50; nGrid += 10) {

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
    if (k == 0) {
        s.Run_Original(T);  // Uses Armadillo
    } else if (k == 1) {
        s.Run_NoMatrixMatrixMultiplication(T);  // Uses Arrays only
    }

    }
    }

    return 0;
}
