#include "colonies3D.cu"
#include "cstdlib"
#include <algorithm>    // std::random_shuffle
#include <vector>

using namespace std;

int main(int argc, char** argv){

    // Benchmark paramters to loop over
    double NumberOfGridSizes = 5;
    int    NumberOfVersions  = 1;     // Loop Distributed CPU, Loop Distributed GPU
    int    NumberOfRepeats   = 10;

    double minGridSize = 1;
    double maxGridSize = 25;



    // Allocate vector to store the benchmark result in
    std::vector<int> timer;
    timer.reserve(NumberOfGridSizes * NumberOfVersions * NumberOfRepeats);

    // Allocate iteration vectors
    std::vector<int> nGrid;
    nGrid.reserve(NumberOfGridSizes * NumberOfVersions * NumberOfRepeats);

    std::vector<int> repeat;
    repeat.reserve(NumberOfGridSizes * NumberOfVersions * NumberOfRepeats);

    std::vector<int> version;
    version.reserve(NumberOfGridSizes * NumberOfVersions * NumberOfRepeats);


    // Make a random permutation of the vectors
    std::vector<int> index;
    for (int n = 0; n < NumberOfVersions*NumberOfRepeats*NumberOfGridSizes; n++) {
        index.push_back(n);
    }
    std::random_shuffle( index.begin(), index.end() );

    // Fill the iteration vectors
    for (int n = 0; n < NumberOfVersions; n++) {
        for (int k = 0; k < NumberOfRepeats; k++) {
            for (int i = 0; i < NumberOfGridSizes; i++) {
                int ind = index[n * NumberOfRepeats * NumberOfGridSizes + k * NumberOfGridSizes + i];
                nGrid[ind]   = static_cast<int>(minGridSize + i * (maxGridSize - minGridSize) / NumberOfGridSizes);
                repeat[ind]  = k;
                version[ind] = n;
            }
        }
    }

    // Parameters
    int beta  	 = 100;
    double delta = 1.0/10.0;
    double eta   = 1e4;
    double tau   = 0.5;
    double B_0   = pow(10,4);
    double P_0   = pow(10,5);
    double T     = 5;


    ofstream f_benchmark;
    string pathName = "3D_Example_benchmarking";
    string fileName = "benchmark_results";
    string path = "benchmarking";


    // Check if the output file exists
	time_t theTime = time(NULL);
	struct tm *aTime = localtime(&theTime);

	string streamPath;
	streamPath = "\tSaving data to file: "+path+"/"+fileName+"_"+std::to_string(aTime->tm_hour)+"_"+std::to_string(aTime->tm_min)+".csv";
    cout << streamPath << endl << endl;

	// Open the file stream
    f_benchmark.open(streamPath, fstream::trunc);

//  My python statistics script cannot handle headlines.
//  f_benchmark << "Function;GridSize;Seconds\n";

    for (int n = 0; n < NumberOfVersions*NumberOfRepeats*NumberOfGridSizes; n++) {

        // Load simulation module
        Colonies3D s(B_0, P_0);
        string finalPath = pathName+"_repeat_"+std::to_string(repeat[n])+"_version_"+std::to_string(version[n])+"_nGrid_"+std::to_string(nGrid[n]);
        s.SetPath(finalPath);

        s.SetRngSeed(5);

        s.SetGridSize(nGrid[n]);
        s.PhageBurstSize(beta);
        s.PhageDecayRate(delta);
        s.PhageAdsorptionRate(eta);
        s.PhageLatencyTime(tau);
        s.SetSamples(10);

        // s.ExportAll();

        // Get start time
        time_t  tic;
        time(&tic);

        if (version[n] == 0) {
            f_benchmark << "LoopDistributedCPU;";
            s.Run_LoopDistributed_CPU(T);
        } else if (version[n] == 1) {
            f_benchmark << "LoopDistributedGPU;";
            s.Run_LoopDistributed_GPU(T);
        }

        // Get stop time
        time_t  toc;
        time(&toc);
        float seconds = difftime(toc, tic);

        f_benchmark << nGrid[n];
        f_benchmark << ";";
        f_benchmark << seconds << "\n";
    }

    f_benchmark.flush();
    f_benchmark.close();
    return 0;
}
