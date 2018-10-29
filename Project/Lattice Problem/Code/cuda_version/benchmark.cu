#include "colonies3D.cu"
#include "cstdlib"

using namespace std;

void shuffleArray(int *arr, int size, mt19937 rng){
    for(int i = 0; i < size; i++){
        uniform_int_distribution <int> distr(0, size);
        int r_i = distr(rng);
        int tmp = arr[i];
        arr[i] = arr[r_i];
        arr[r_i] = tmp;
    
    }
}


int main(int argc, char** argv){
     // Default parameters 
    
    int minGridSize = 6;
    int stepSize    = 4;
    int steps       = 4;
    int repeats     = 5;
    int beta  	 = 100;
    double delta = 1.0/10.0;
    double eta   = 1e4;
    double tau   = 0.5;
    double B_0 = pow(10,4);
    double P_0 = pow(10,5);
    double T   = 1;
    
    
    mt19937 rng;
    static random_device rd;
    rng.seed(rd());
    
    int arr_steps[steps];
    for(int i = 0; i < steps; i++){
        arr_steps[i] = minGridSize + i*stepSize;
    }
    shuffleArray(arr_steps, steps, rng);
    

    
    ofstream f_benchmark;
    string pathName = "3D_Example_Benchmarking";
    string fileName = "benchmark_results";
    string path = "benchmark";
    
    // Debug info
	cout << "\tSaving data to file: " << path << "/" << fileName << ".txt" << "\n";
    
    // Check if the output file exists
	time_t theTime = time(NULL);
	struct tm *aTime = localtime(&theTime);

	string streamPath;
	streamPath = path+"/"+fileName+"_"+std::to_string(aTime->tm_hour)+"_"+std::to_string(aTime->tm_min)+".csv";
    cout << streamPath;

	// Open the file stream
    f_benchmark.open(streamPath, fstream::trunc);    

//  My python statistics script cannot handle headlines.
//  f_benchmark << "Function;GridSize;Seconds\n";
   
    for(int i = 0; i < repeats; i++){
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < steps; i++){
           
                // Load simulation module
                Colonies3D s(B_0, P_0);

                s.SetPath(pathName);

                s.SetRngSeed(5);

                s.SetGridSize(arr_steps[i]);
                s.PhageBurstSize(beta);
                s.PhageDecayRate(delta);
                s.PhageAdsorptionRate(eta);
                s.PhageLatencyTime(tau);
                s.SetSamples(10);

                s.ExportAll();

                // Start the simulation
                
                // Get start time
                time_t  tic;
                time(&tic);
                
                if (k == 0) {
                    f_benchmark << "LoopDistributedCPU;";
                    s.Run_LoopDistributed_CPU(T);  // Uses Armadillo
                } else if (k == 1) {
//                    f_benchmark << "LoopDistributedCPU_cuRand;";
//                    s.Run_LoopDistributed_CPU_cuRand(T);  // Uses Arrays only
 //               } else if (k == 2) {
                    f_benchmark << "LoopDistributedGPU;";
                    s.Run_LoopDistributed_GPU(T);
                }
                // Get stop time
                time_t  toc;
                time(&toc);
                f_benchmark << arr_steps[i];
                f_benchmark << ";";
                float seconds = difftime(toc, tic);
                f_benchmark << seconds << "\n";
            }
        }
    }
    f_benchmark.flush();
    f_benchmark.close();
    return 0;
}
