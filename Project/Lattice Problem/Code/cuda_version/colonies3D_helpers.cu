#include "colonies3D.hpp"

#include <iostream>         // Input and output
#include <iomanip>          // Input and output formatting
#include <fstream>          // File streams

#include <random>           // Random numbers
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

using namespace std;


// Returns poisson dist. number with mean l
double RandP(double l, int i, int j, int k) {

    // Set limit on distribution
    poisson_distribution <long long> distr(l);

    return distr(arr_rng[i*nGridXY*nGridZ + j*nGridZ + k]);
}

// Returns poisson dist. number with mean l, flat array
double RandP(double l, int i) {

    // Set limit on distribution
    poisson_distribution <long long> distr(l);

    return distr(arr_rng[i]);
}

// Returns poisson dist. number with mean l
double RandP(double l) {

    // Set limit on distribution
    poisson_distribution <long long> distr(l);

    return distr(rng);
}

// Returns the number of events ocurring for given n and p
double ComputeEvents(double n, double p, int flag, int i, int j, int k) {

    // Trivial cases
    if (p == 1) return n;
    if (p == 0) return 0.0;
    if (n < 1)  return 0.0;

    double N = RandP(n*p, i, j, k);

    return round(N);
}

// Returns the number of events ocurring for given n and p, flat array
double ComputeEvents(double n, double p, int flag, int i) {

    // Trivial cases
    if (p == 1) return n;
    if (p == 0) return 0.0;
    if (n < 1)  return 0.0;

    double N = RandP(n*p, i);

    return round(N);
}
