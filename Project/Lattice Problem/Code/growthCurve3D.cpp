#include "colonies3D.hpp"
#include <omp.h>

using namespace std;

int main(int argc, char** argv){

    // Default parameters
    int nGrid = 100;

    double T = 5; 	// Simulation length

    char buffer[80];                                  // Create a buffer to store the date
    time_t t = time(0);                               // Get time now
    struct tm tstruct;                                // And format the date
    tstruct = *localtime(&t);                         // as "MNT_DD_YY" for folder name
    strftime(buffer, sizeof(buffer), "%F", &tstruct); // Store the formated foldername in buffer
    string dateFolder(buffer);

    for (int k = 0; k < 1; k++) {

        Colonies3D s(1e12/4e10,0);

        string pathName = "";
        pathName += dateFolder;
        pathName += "/";

        pathName += "Growthcurve_Original";

        pathName += "/nGrid_";
        pathName += to_string(nGrid);

        pathName += "/";
        pathName += to_string(k);

        // Check if data is already made
        string path_s = "data/"; // Data folder name
        path_s += pathName;
        path_s += "/Completed.txt";

        // Check if run exists and is completed
        struct stat info;
        if (stat(path_s.c_str(), &info) == 0 && S_ISREG(info.st_mode)) {    // Create path if it does not exist
            continue;
        }

        s.SetPath(pathName);

        s.SetRngSeed(k);

        s.SetGridSize(nGrid);
        s.SetLength(1e4);
        s.SetHeight(400);

        s.PhageDecayRate(0.0);
        s.CellGrowthRate(2.0);
        s.PhageInfectionRate(0.0);
        s.PhageBurstSize(0);

        s.SetSamples(100);

        // Start the simulation
        s.Run_Original(T);
        // s.Run_NoMatrixMatrixMultiplication_with_arma(T);
    }

    return 0;
}
