#include "colonies3D.hpp"

using namespace arma;
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


// Controls the evaluation of the simulation
int Colonies3D::Run_Original(double T_end) {

    this->T_end = T_end;

    // Get start time
    time_t  tic;
    time(&tic);

    // Generate a path
    path = GeneratePath();

    // Initilize the simulation matrices
    Initialize();

    // Export data
    ExportData(T,"original");

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

            for (uword k = 0; k < nGridZ; k++ ) {
                if (exit) break;

                for (uword j = 0; j < nGridXY; j++ ) {
                    if (exit) break;

                    for (uword i = 0; i < nGridXY; i++) {
                        if (exit) break;

                        // Ensure nC is updated
                        if (Occ(i, j, k) < nC(i, j, k)) nC(i,j,k) = Occ(i, j, k);

                        // Skip empty sites
                        if ((Occ(i, j, k) < 1) and (P(i, j, k) < 1)) continue;

                        // Record the maximum observed density
                        if (Occ(i, j, k) > maxOccupancy) maxOccupancy = Occ(i, j, k);

                        // Compute the growth modifier
                        double growthModifier = nutrient(i, j, k) / (nutrient(i, j, k) + K);

                        // Compute beta
                        double Beta = beta;
                        if (reducedBeta) {
                            Beta *= growthModifier;
                        }

                        double p = 0;
                        double N = 0;
                        double M = 0;

                        // Birth //////////////////////////////////////////////////////////////////////
                        p = g*growthModifier*dT;
                        if (nutrient(i, j, k) < 1) p = 0;

                        if ((p > 0.1) and (!Warn_g)) {
                            cout << "\tWarning: Birth Probability Large!" << "\n";
                            f_log  << "Warning: Birth Probability Large!" << "\n";
                            Warn_g = true;
                        }

                        N = ComputeEvents(B(i, j, k), p, 1);

                        // Ensure there is enough nutrient
                        if ( N > nutrient(i, j, k) ) {
                            if (!Warn_fastGrowth) {
                                cout << "\tWarning: Colonies growing too fast!" << "\n";
                                f_log  << "Warning: Colonies growing too fast!" << "\n";
                                Warn_fastGrowth = true;
                            }

                            N = round( nutrient(i, j, k) );
                        }

                        // Update count
                        B_new(i, j, k) += N;
                        nutrient(i, j, k) = max(0.0, nutrient(i, j, k) - N);

                        // Increase Infections ////////////////////////////////////////////////////////
                        if (r > 0.0) {
                            p = r*growthModifier*dT;
                            if ((p > 0.25) and (!Warn_r)) {
                                cout << "\tWarning: Infection Increase Probability Large!" << "\n";
                                f_log  << "Warning: Infection Increase Probability Large!" << "\n";
                                Warn_r = true;
                            }
                            N = ComputeEvents(I9(i, j, k), p, 2);  // Bursting events

                            // Update count
                            I9(i, j, k)    = max(0.0, I9(i, j, k) - N);
                            Occ(i, j, k)   = max(0.0, Occ(i, j, k) - N);
                            P_new(i, j, k) += round( (1 - alpha) * Beta * N);   // Phages which escape the colony
                            M = round(alpha * Beta * N);                        // Phages which reinfect the colony

                            // Non-bursting events
                            N = ComputeEvents(I8(i, j, k), p, 2); I8(i, j, k) = max(0.0, I8(i, j, k) - N); I9(i, j, k) += N;
                            N = ComputeEvents(I7(i, j, k), p, 2); I7(i, j, k) = max(0.0, I7(i, j, k) - N); I8(i, j, k) += N;
                            N = ComputeEvents(I6(i, j, k), p, 2); I6(i, j, k) = max(0.0, I6(i, j, k) - N); I7(i, j, k) += N;
                            N = ComputeEvents(I5(i, j, k), p, 2); I5(i, j, k) = max(0.0, I5(i, j, k) - N); I6(i, j, k) += N;
                            N = ComputeEvents(I4(i, j, k), p, 2); I4(i, j, k) = max(0.0, I4(i, j, k) - N); I5(i, j, k) += N;
                            N = ComputeEvents(I3(i, j, k), p, 2); I3(i, j, k) = max(0.0, I3(i, j, k) - N); I4(i, j, k) += N;
                            N = ComputeEvents(I2(i, j, k), p, 2); I2(i, j, k) = max(0.0, I2(i, j, k) - N); I3(i, j, k) += N;
                            N = ComputeEvents(I1(i, j, k), p, 2); I1(i, j, k) = max(0.0, I1(i, j, k) - N); I2(i, j, k) += N;
                            N = ComputeEvents(I0(i, j, k), p, 2); I0(i, j, k) = max(0.0, I0(i, j, k) - N); I1(i, j, k) += N;
                        }

                        // Infectons //////////////////////////////////////////////////////////////////
                        if ((Occ(i, j, k) >= 1) and (P(i, j, k) >= 1)) {
                            double s;   // The factor which modifies the adsorption rate
                            double n;   // The number of targets the phage has

                            if (clustering) {   // Check if clustering is enabled
                                s = pow(Occ(i, j, k) / nC(i, j, k), 1.0 / 3.0);
                                n = nC(i, j, k);
                            } else {            // Else use mean field computation
                                s = 1.0;
                                n = Occ(i, j, k);
                            }

                            // Compute the number of hits
                            if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
                                N = P(i, j, k);
                            } else {
                                p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
                                N = ComputeEvents(P(i, j, k), p, 4);     // Number of targets hit
                            }

                            // If bacteria were hit, update events
                            if ((N + M) >= 1) {

                                P(i, j, k)    = max(0.0, P(i, j, k) - N);     // Update count

                                double S;
                                if (shielding) {
                                    // Absorbing medium model
                                    double d = pow(Occ(i, j, k) / nC(i, j, k), 1.0 / 3.0) - pow(B(i, j, k) / nC(i, j, k), 1.0 / 3.0);
                                    S = exp(-zeta*d); // Probability of hitting succebtible target

                                } else {
                                    // Well mixed model
                                    S = B(i, j, k) / Occ(i, j, k);
                                }

                                p = max(0.0, min(B(i, j, k) / Occ(i, j, k), S)); // Probability of hitting succebtible target
                                N = ComputeEvents(N + M, p, 4);                  // Number of targets hit

                                if (N > B(i, j, k)) N = B(i, j, k);              // If more bacteria than present are set to be infeced, round down

                                // Update the counts
                                B(i, j, k)      = max(0.0, B(i, j, k) - N);
                                if (r > 0.0) {
                                    I0_new(i, j, k) += N;
                                } else {
                                    P_new(i, k, k) += N * (1 - alpha) * Beta;
                                }
                            }
                        }

                        // Phage Decay ////////////////////////////////////////////////////////////////
                        p = delta*dT;
                        if ((p > 0.1) and (!Warn_delta)) {
                            cout << "\tWarning: Decay Probability Large!" << "\n";
                            f_log  << "Warning: Decay Probability Large!" << "\n";
                            Warn_delta = true;
                        }
                        N = ComputeEvents(P(i, j, k), p, 5);

                        // Update count
                        P(i, j, k)    = max(0.0, P(i, j, k) - N);


                        // Movement ///////////////////////////////////////////////////////////////////
                        if (nGridXY > 1) {

                            // Update positions
                            uword ip, jp, kp, im, jm, km;

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
                            ComputeDiffusion(B(i, j, k), lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b,1);
                            B_new(i, j, k) += n_0; B_new(ip, j, k) += n_u; B_new(im, j, k) += n_d; B_new(i, jp, k) += n_r; B_new(i, jm, k) += n_l; B_new(i, j, kp) += n_f; B_new(i, j, km) += n_b;

                            if (r > 0.0) {
                                ComputeDiffusion(I0(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I0_new(i, j, k) += n_0; I0_new(ip, j, k) += n_u; I0_new(im, j, k) += n_d; I0_new(i, jp, k) += n_r; I0_new(i, jm, k) += n_l; I0_new(i, j, kp) += n_f; I0_new(i, j, km) += n_b;

                                ComputeDiffusion(I1(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I1_new(i, j, k) += n_0; I1_new(ip, j, k) += n_u; I1_new(im, j, k) += n_d; I1_new(i, jp, k) += n_r; I1_new(i, jm, k) += n_l; I1_new(i, j, kp) += n_f; I1_new(i, j, km) += n_b;

                                ComputeDiffusion(I2(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I2_new(i, j, k) += n_0; I2_new(ip, j, k) += n_u; I2_new(im, j, k) += n_d; I2_new(i, jp, k) += n_r; I2_new(i, jm, k) += n_l; I2_new(i, j, kp) += n_f; I2_new(i, j, km) += n_b;

                                ComputeDiffusion(I3(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I3_new(i, j, k) += n_0; I3_new(ip, j, k) += n_u; I3_new(im, j, k) += n_d; I3_new(i, jp, k) += n_r; I3_new(i, jm, k) += n_l; I3_new(i, j, kp) += n_f; I3_new(i, j, km) += n_b;

                                ComputeDiffusion(I4(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I4_new(i, j, k) += n_0; I4_new(ip, j, k) += n_u; I4_new(im, j, k) += n_d; I4_new(i, jp, k) += n_r; I4_new(i, jm, k) += n_l; I4_new(i, j, kp) += n_f; I4_new(i, j, km) += n_b;

                                ComputeDiffusion(I5(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I5_new(i, j, k) += n_0; I5_new(ip, j, k) += n_u; I5_new(im, j, k) += n_d; I5_new(i, jp, k) += n_r; I5_new(i, jm, k) += n_l; I5_new(i, j, kp) += n_f; I5_new(i, j, km) += n_b;

                                ComputeDiffusion(I6(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I6_new(i, j, k) += n_0; I6_new(ip, j, k) += n_u; I6_new(im, j, k) += n_d; I6_new(i, jp, k) += n_r; I6_new(i, jm, k) += n_l; I6_new(i, j, kp) += n_f; I6_new(i, j, km) += n_b;

                                ComputeDiffusion(I7(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I7_new(i, j, k) += n_0; I7_new(ip, j, k) += n_u; I7_new(im, j, k) += n_d; I7_new(i, jp, k) += n_r; I7_new(i, jm, k) += n_l; I7_new(i, j, kp) += n_f; I7_new(i, j, km) += n_b;

                                ComputeDiffusion(I8(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I8_new(i, j, k) += n_0; I8_new(ip, j, k) += n_u; I8_new(im, j, k) += n_d; I8_new(i, jp, k) += n_r; I8_new(i, jm, k) += n_l; I8_new(i, j, kp) += n_f; I8_new(i, j, km) += n_b;

                                ComputeDiffusion(I9(i, j, k), lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I9_new(i, j, k) += n_0; I9_new(ip, j, k) += n_u; I9_new(im, j, k) += n_d; I9_new(i, jp, k) += n_r; I9_new(i, jm, k) += n_l; I9_new(i, j, kp) += n_f; I9_new(i, j, km) += n_b;
                            }

                            // PHAGES
                            ComputeDiffusion(P(i, j, k), lambdaP, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 3);
                            P_new(i, j, k) += n_0; P_new(ip, j, k) += n_u; P_new(im, j, k) += n_d; P_new(i, jp, k) += n_r; P_new(i, jm, k) += n_l; P_new(i, j, kp) += n_f; P_new(i, j, km) += n_b;



                        } else {
                            // CELLS
                            B_new += B;

                            if (r > 0.0) {
                                I0_new += I0;
                                I1_new += I1;
                                I2_new += I2;
                                I3_new += I3;
                                I4_new += I4;
                                I5_new += I5;
                                I6_new += I6;
                                I7_new += I7;
                                I8_new += I8;
                                I9_new += I9;
                            }

                            // PHAGES
                            P_new += P;
                        }
                    }
                }
            }

            // Update arrays
            B.swap(B_new);      B_new.zeros();
            I0.swap(I0_new);    I0_new.zeros();
            I1.swap(I1_new);    I1_new.zeros();
            I2.swap(I2_new);    I2_new.zeros();
            I3.swap(I3_new);    I3_new.zeros();
            I4.swap(I4_new);    I4_new.zeros();
            I5.swap(I5_new);    I5_new.zeros();
            I6.swap(I6_new);    I6_new.zeros();
            I7.swap(I7_new);    I7_new.zeros();
            I8.swap(I8_new);    I8_new.zeros();
            I9.swap(I9_new);    I9_new.zeros();
            P.swap(P_new);      P_new.zeros();

            // Update occupancy
            Occ = B + I0 + I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9;

            // NUTRIENT DIFFUSION
            // Create copy of nutrient to store the diffusion update
            cube nn = nutrient;
            assert(2 * D_n * dT / pow( L / (double)nGridXY, 2) <= 1);
            assert(2 * D_n * dT / pow( H / (double)nGridZ, 2) <= 1);

            // Compute the X & Y diffusion
            for (uword k = 0; k < nGridZ; k++) {
                nutrient.slice(k) += D_n * dT / pow(L / (double)nGridXY, 2) * ( (lapXY * nn.slice(k).t()).t() + lapXY * nn.slice(k) );
            }

            // Compute the Z diffusion
            for (uword i = 0; i < nGridXY; i++) {
                mat Q = D_n * dT / pow(H / (double)nGridZ, 2) * (lapZ * static_cast<mat>(nn.tube( span(i), span::all )).t()).t();

                for (uword k = 0; k < Q.n_cols; k++) {
                    for (uword j = 0; j < Q.n_rows; j++) {
                        nutrient(i, j, k) += Q(j,k);
                        nutrient(i, j, k) = max(0.0, nutrient(i, j, k));
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
        if ((fastExit) and (accu(B) < 1)) {
            P += (1-alpha)*beta * (I0 + I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9);
            I0.zeros();
            I1.zeros();
            I2.zeros();
            I3.zeros();
            I4.zeros();
            I5.zeros();
            I6.zeros();
            I7.zeros();
            I8.zeros();
            I9.zeros();
            exit = true;
        }

        // 2) There are no more alive cells
        // -> Stop simulation

        if ((fastExit) and (accu(Occ) < 1)) {
            exit = true;
        }

        // 3) The food is on average less than one per gridpoint
        // and the maximal nutrient at any point in space is less than 1

        if (fastExit) {
            if  ((accu(nutrient) < nGridZ*pow(nGridXY,2)) and (nutrient.max() < 0.5)) {
                exit = true;
            }
        }

        // Store the state
        ExportData(T,"original");

        // Check for nutrient stability
        assert(accu(nutrient) >= 0);
        assert(accu(nutrient) <= n_0 * L * L * H);
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
    f_out.open(GetPath() + "/Completed.txt",fstream::trunc);
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

    if (true) {
    std::ofstream f_timing;
    // cout << "\n";
    f_timing << "\t"       << setw(3) << difftime(toc, tic) << " s of total time" << "\n";

    f_timing.flush();
    f_timing.close();
    // cout << "\t----------------------------------------------------"<< "\n" << "\n" << "\n";
    }

    if (exit) {
        return 1;
    } else {
        return 0;
    }
}

int Colonies3D::Run_NoMatrixMatrixMultiplication_with_arma(double T_end) {

    this->T_end = T_end;

    // Get start time
    time_t  tic;
    time(&tic);

    // Generate a path
    path = GeneratePath();

    // Initilize the simulation matrices
    Initialize();

    // Export data
    ExportData(T,"noMMM_with_arma");

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

            for (uword k = 0; k < nGridZ; k++ ) {
                if (exit) break;

                for (uword j = 0; j < nGridXY; j++ ) {
                    if (exit) break;

                    for (uword i = 0; i < nGridXY; i++) {
                        if (exit) break;

                        // Ensure nC is updated
                        if (Occ(i, j, k) < nC(i, j, k)) nC(i,j,k) = Occ(i, j, k);

                        // Skip empty sites
                        if ((Occ(i, j, k) < 1) and (P(i, j, k) < 1)) continue;

                        // Record the maximum observed density
                        if (Occ(i, j, k) > maxOccupancy) maxOccupancy = Occ(i, j, k);

                        // Compute the growth modifier
                        double growthModifier = nutrient(i, j, k) / (nutrient(i, j, k) + K);

                        // Compute beta
                        double Beta = beta;
                        if (reducedBeta) {
                            Beta *= growthModifier;
                        }

                        double p = 0;
                        double N = 0;
                        double M = 0;

                        // Birth //////////////////////////////////////////////////////////////////////
                        p = g*growthModifier*dT;
                        if (nutrient(i, j, k) < 1) p = 0;

                        if ((p > 0.1) and (!Warn_g)) {
                            cout << "\tWarning: Birth Probability Large!" << "\n";
                            f_log  << "Warning: Birth Probability Large!" << "\n";
                            Warn_g = true;
                        }

                        N = ComputeEvents(B(i, j, k), p, 1);

                        // Ensure there is enough nutrient
                        if ( N > nutrient(i, j, k) ) {
                            if (!Warn_fastGrowth) {
                                cout << "\tWarning: Colonies growing too fast!" << "\n";
                                f_log  << "Warning: Colonies growing too fast!" << "\n";
                                Warn_fastGrowth = true;
                            }

                            N = round( nutrient(i, j, k) );
                        }

                        // Update count
                        B_new(i, j, k) += N;
                        nutrient(i, j, k) = max(0.0, nutrient(i, j, k) - N);

                        // Increase Infections ////////////////////////////////////////////////////////
                        if (r > 0.0) {
                            p = r*growthModifier*dT;
                            if ((p > 0.25) and (!Warn_r)) {
                                cout << "\tWarning: Infection Increase Probability Large!" << "\n";
                                f_log  << "Warning: Infection Increase Probability Large!" << "\n";
                                Warn_r = true;
                            }
                            N = ComputeEvents(I9(i, j, k), p, 2);  // Bursting events

                            // Update count
                            I9(i, j, k)    = max(0.0, I9(i, j, k) - N);
                            Occ(i, j, k)   = max(0.0, Occ(i, j, k) - N);
                            P_new(i, j, k) += round( (1 - alpha) * Beta * N);   // Phages which escape the colony
                            M = round(alpha * Beta * N);                        // Phages which reinfect the colony

                            // Non-bursting events
                            N = ComputeEvents(I8(i, j, k), p, 2); I8(i, j, k) = max(0.0, I8(i, j, k) - N); I9(i, j, k) += N;
                            N = ComputeEvents(I7(i, j, k), p, 2); I7(i, j, k) = max(0.0, I7(i, j, k) - N); I8(i, j, k) += N;
                            N = ComputeEvents(I6(i, j, k), p, 2); I6(i, j, k) = max(0.0, I6(i, j, k) - N); I7(i, j, k) += N;
                            N = ComputeEvents(I5(i, j, k), p, 2); I5(i, j, k) = max(0.0, I5(i, j, k) - N); I6(i, j, k) += N;
                            N = ComputeEvents(I4(i, j, k), p, 2); I4(i, j, k) = max(0.0, I4(i, j, k) - N); I5(i, j, k) += N;
                            N = ComputeEvents(I3(i, j, k), p, 2); I3(i, j, k) = max(0.0, I3(i, j, k) - N); I4(i, j, k) += N;
                            N = ComputeEvents(I2(i, j, k), p, 2); I2(i, j, k) = max(0.0, I2(i, j, k) - N); I3(i, j, k) += N;
                            N = ComputeEvents(I1(i, j, k), p, 2); I1(i, j, k) = max(0.0, I1(i, j, k) - N); I2(i, j, k) += N;
                            N = ComputeEvents(I0(i, j, k), p, 2); I0(i, j, k) = max(0.0, I0(i, j, k) - N); I1(i, j, k) += N;
                        }

                        // Infectons //////////////////////////////////////////////////////////////////
                        if ((Occ(i, j, k) >= 1) and (P(i, j, k) >= 1)) {
                            double s;   // The factor which modifies the adsorption rate
                            double n;   // The number of targets the phage has

                            if (clustering) {   // Check if clustering is enabled
                                s = pow(Occ(i, j, k) / nC(i, j, k), 1.0 / 3.0);
                                n = nC(i, j, k);
                            } else {            // Else use mean field computation
                                s = 1.0;
                                n = Occ(i, j, k);
                            }

                            // Compute the number of hits
                            if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
                                N = P(i, j, k);
                            } else {
                                p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
                                N = ComputeEvents(P(i, j, k), p, 4);     // Number of targets hit
                            }

                            // If bacteria were hit, update events
                            if ((N + M) >= 1) {

                                P(i, j, k)    = max(0.0, P(i, j, k) - N);     // Update count

                                double S;
                                if (shielding) {
                                    // Absorbing medium model
                                    double d = pow(Occ(i, j, k) / nC(i, j, k), 1.0 / 3.0) - pow(B(i, j, k) / nC(i, j, k), 1.0 / 3.0);
                                    S = exp(-zeta*d); // Probability of hitting succebtible target

                                } else {
                                    // Well mixed model
                                    S = B(i, j, k) / Occ(i, j, k);
                                }

                                p = max(0.0, min(B(i, j, k) / Occ(i, j, k), S)); // Probability of hitting succebtible target
                                N = ComputeEvents(N + M, p, 4);                  // Number of targets hit

                                if (N > B(i, j, k)) N = B(i, j, k);              // If more bacteria than present are set to be infeced, round down

                                // Update the counts
                                B(i, j, k)      = max(0.0, B(i, j, k) - N);
                                if (r > 0.0) {
                                    I0_new(i, j, k) += N;
                                } else {
                                    P_new(i, k, k) += N * (1 - alpha) * Beta;
                                }
                            }
                        }

                        // Phage Decay ////////////////////////////////////////////////////////////////
                        p = delta*dT;
                        if ((p > 0.1) and (!Warn_delta)) {
                            cout << "\tWarning: Decay Probability Large!" << "\n";
                            f_log  << "Warning: Decay Probability Large!" << "\n";
                            Warn_delta = true;
                        }
                        N = ComputeEvents(P(i, j, k), p, 5);

                        // Update count
                        P(i, j, k)    = max(0.0, P(i, j, k) - N);


                        // Movement ///////////////////////////////////////////////////////////////////
                        if (nGridXY > 1) {

                            // Update positions
                            uword ip, jp, kp, im, jm, km;

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
                            ComputeDiffusion(B(i, j, k), lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b,1);
                            B_new(i, j, k) += n_0; B_new(ip, j, k) += n_u; B_new(im, j, k) += n_d; B_new(i, jp, k) += n_r; B_new(i, jm, k) += n_l; B_new(i, j, kp) += n_f; B_new(i, j, km) += n_b;

                            if (r > 0.0) {
                                ComputeDiffusion(I0(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I0_new(i, j, k) += n_0; I0_new(ip, j, k) += n_u; I0_new(im, j, k) += n_d; I0_new(i, jp, k) += n_r; I0_new(i, jm, k) += n_l; I0_new(i, j, kp) += n_f; I0_new(i, j, km) += n_b;

                                ComputeDiffusion(I1(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I1_new(i, j, k) += n_0; I1_new(ip, j, k) += n_u; I1_new(im, j, k) += n_d; I1_new(i, jp, k) += n_r; I1_new(i, jm, k) += n_l; I1_new(i, j, kp) += n_f; I1_new(i, j, km) += n_b;

                                ComputeDiffusion(I2(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I2_new(i, j, k) += n_0; I2_new(ip, j, k) += n_u; I2_new(im, j, k) += n_d; I2_new(i, jp, k) += n_r; I2_new(i, jm, k) += n_l; I2_new(i, j, kp) += n_f; I2_new(i, j, km) += n_b;

                                ComputeDiffusion(I3(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I3_new(i, j, k) += n_0; I3_new(ip, j, k) += n_u; I3_new(im, j, k) += n_d; I3_new(i, jp, k) += n_r; I3_new(i, jm, k) += n_l; I3_new(i, j, kp) += n_f; I3_new(i, j, km) += n_b;

                                ComputeDiffusion(I4(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I4_new(i, j, k) += n_0; I4_new(ip, j, k) += n_u; I4_new(im, j, k) += n_d; I4_new(i, jp, k) += n_r; I4_new(i, jm, k) += n_l; I4_new(i, j, kp) += n_f; I4_new(i, j, km) += n_b;

                                ComputeDiffusion(I5(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I5_new(i, j, k) += n_0; I5_new(ip, j, k) += n_u; I5_new(im, j, k) += n_d; I5_new(i, jp, k) += n_r; I5_new(i, jm, k) += n_l; I5_new(i, j, kp) += n_f; I5_new(i, j, km) += n_b;

                                ComputeDiffusion(I6(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I6_new(i, j, k) += n_0; I6_new(ip, j, k) += n_u; I6_new(im, j, k) += n_d; I6_new(i, jp, k) += n_r; I6_new(i, jm, k) += n_l; I6_new(i, j, kp) += n_f; I6_new(i, j, km) += n_b;

                                ComputeDiffusion(I7(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I7_new(i, j, k) += n_0; I7_new(ip, j, k) += n_u; I7_new(im, j, k) += n_d; I7_new(i, jp, k) += n_r; I7_new(i, jm, k) += n_l; I7_new(i, j, kp) += n_f; I7_new(i, j, km) += n_b;

                                ComputeDiffusion(I8(i, j, k),  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I8_new(i, j, k) += n_0; I8_new(ip, j, k) += n_u; I8_new(im, j, k) += n_d; I8_new(i, jp, k) += n_r; I8_new(i, jm, k) += n_l; I8_new(i, j, kp) += n_f; I8_new(i, j, km) += n_b;

                                ComputeDiffusion(I9(i, j, k), lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                I9_new(i, j, k) += n_0; I9_new(ip, j, k) += n_u; I9_new(im, j, k) += n_d; I9_new(i, jp, k) += n_r; I9_new(i, jm, k) += n_l; I9_new(i, j, kp) += n_f; I9_new(i, j, km) += n_b;
                            }

                            // PHAGES
                            ComputeDiffusion(P(i, j, k), lambdaP, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 3);
                            P_new(i, j, k) += n_0; P_new(ip, j, k) += n_u; P_new(im, j, k) += n_d; P_new(i, jp, k) += n_r; P_new(i, jm, k) += n_l; P_new(i, j, kp) += n_f; P_new(i, j, km) += n_b;



                        } else {
                            // CELLS
                            B_new += B;

                            if (r > 0.0) {
                                I0_new += I0;
                                I1_new += I1;
                                I2_new += I2;
                                I3_new += I3;
                                I4_new += I4;
                                I5_new += I5;
                                I6_new += I6;
                                I7_new += I7;
                                I8_new += I8;
                                I9_new += I9;
                            }

                            // PHAGES
                            P_new += P;
                        }
                    }
                }
            }

            // Update arrays
            B.swap(B_new);      B_new.zeros();
            I0.swap(I0_new);    I0_new.zeros();
            I1.swap(I1_new);    I1_new.zeros();
            I2.swap(I2_new);    I2_new.zeros();
            I3.swap(I3_new);    I3_new.zeros();
            I4.swap(I4_new);    I4_new.zeros();
            I5.swap(I5_new);    I5_new.zeros();
            I6.swap(I6_new);    I6_new.zeros();
            I7.swap(I7_new);    I7_new.zeros();
            I8.swap(I8_new);    I8_new.zeros();
            I9.swap(I9_new);    I9_new.zeros();
            P.swap(P_new);      P_new.zeros();

            // Update occupancy
            Occ = B + I0 + I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9;

            // NUTRIENT DIFFUSION
            double alphaXY = D_n * dT / pow(L / (double)nGridXY, 2);
            double alphaZ  = D_n * dT / pow(H / (double)nGridZ, 2);
            for (uword k = 0; k < nGridZ; k++ ) {
                for (uword j = 0; j < nGridXY; j++ ) {
                    for (uword i = 0; i < nGridXY; i++) {

                        // Update positions
                        uword ip, jp, kp, im, jm, km;

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

                        double tmp = nutrient(i, j, k);
                        nutrient_new(i, j, k)  += tmp - 6 * alphaXY * tmp;
                        nutrient_new(ip, j, k) += alphaXY * tmp;
                        nutrient_new(im, j, k) += alphaXY * tmp;
                        nutrient_new(i, jp, k) += alphaXY * tmp;
                        nutrient_new(i, jm, k) += alphaXY * tmp;
                        nutrient_new(i, j, kp) += alphaZ  * tmp;
                        nutrient_new(i, j, km) += alphaZ  * tmp;
                    }
                }
            }
            nutrient.swap(nutrient_new);
            nutrient_new.zeros();


            if ((maxOccupancy > L * L * H / (nGridXY * nGridXY * nGridZ)) and (!Warn_density)) {
                cout << "\tWarning: Maximum Density Large!" << "\n";
                f_log  << "Warning: Maximum Density Large!" << "\n";
                Warn_density = true;
            }
        }

        // Fast exit conditions
        // 1) There are no more sucebtible cells
        // -> Convert all infected cells to phages and stop simulation
        if ((fastExit) and (accu(B) < 1)) {
            P += (1-alpha)*beta * (I0 + I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9);
            I0.zeros();
            I1.zeros();
            I2.zeros();
            I3.zeros();
            I4.zeros();
            I5.zeros();
            I6.zeros();
            I7.zeros();
            I8.zeros();
            I9.zeros();
            exit = true;
        }

        // 2) There are no more alive cells
        // -> Stop simulation

        if ((fastExit) and (accu(Occ) < 1)) {
            exit = true;
        }

        // 3) The food is on average less than one per gridpoint
        // and the maximal nutrient at any point in space is less than 1

        if (fastExit) {
            if  ((accu(nutrient) < nGridZ*pow(nGridXY,2)) and (nutrient.max() < 0.5)) {
                exit = true;
            }
        }

        // Store the state
        ExportData(T,"noMMM_with_arma");

        // Check for nutrient stability
        assert(accu(nutrient) >= 0);
        assert(accu(nutrient) <= n_0 * L * L * H);
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
    f_out.open(GetPath() + "/Completed.txt",fstream::trunc);
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

    if (true) {
    std::ofstream f_timing;
    // cout << "\n";
    f_timing << "\t"       << setw(3) << difftime(toc, tic) << " s of total time" << "\n";

    f_timing.flush();
    f_timing.close();
    // cout << "\t----------------------------------------------------"<< "\n" << "\n" << "\n";
    }

    if (exit) {
        return 1;
    } else {
        return 0;
    }
}

int Colonies3D::Run_NoMatrixMatrixMultiplication(double T_end) {

    this->T_end = T_end;

    // Get start time
    time_t  tic;
    time(&tic);

    // Generate a path
    path = GeneratePath();

    // Initilize the simulation matrices
    Initialize();

    // Export data
    ExportData_arr(T,"noMMM");

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

            for (int i = 0; i < nGridXY; i++) {
                if (exit) break;

                for (int j = 0; j < nGridXY; j++ ) {
                    if (exit) break;

                    for (int k = 0; k < nGridZ; k++ ) {
                        if (exit) break;

                        // rng.seed(n*pow(nGridXY,2)*nStepsPerSample*nSamplings + t*pow(nGridXY,2)*nStepsPerSample +  i*pow(nGridXY,2) + j*nGridXY + k);

                        // Ensure nC is updated
                        if (arr_Occ[i][j][k] < arr_nC[i][j][k]) arr_nC[i][j][k] = arr_Occ[i][j][k];

                        // Skip empty sites
                        if ((arr_Occ[i][j][k] < 1) and (arr_P[i][j][k] < 1)) continue;

                        // Record the maximum observed density
                        if (arr_Occ[i][j][k] > maxOccupancy) maxOccupancy = arr_Occ[i][j][k];

                        // Compute the growth modifier
                        double growthModifier = arr_nutrient[i][j][k] / (arr_nutrient[i][j][k] + K);

                        // Compute beta
                        double Beta = beta;
                        if (reducedBeta) {
                            Beta *= growthModifier;
                        }

                        double p = 0;
                        double N = 0;
                        double M = 0;

                        // Birth //////////////////////////////////////////////////////////////////////
                        p = g*growthModifier*dT;
                        if (arr_nutrient[i][j][k] < 1) p = 0;

                        if ((p > 0.1) and (!Warn_g)) {
                            cout << "\tWarning: Birth Probability Large!" << "\n";
                            f_log  << "Warning: Birth Probability Large!" << "\n";
                            Warn_g = true;
                        }

                        N = ComputeEvents(arr_B[i][j][k], p, 1);

                        // Ensure there is enough nutrient
                        if ( N > arr_nutrient[i][j][k] ) {
                            if (!Warn_fastGrowth) {
                                cout << "\tWarning: Colonies growing too fast!" << "\n";
                                f_log  << "Warning: Colonies growing too fast!" << "\n";
                                Warn_fastGrowth = true;
                            }

                            N = round( arr_nutrient[i][j][k] );
                        }

                        // Update count
                        arr_B_new[i][j][k] += N;
                        arr_nutrient[i][j][k] = max(0.0, arr_nutrient[i][j][k] - N);

                        // Increase Infections ////////////////////////////////////////////////////////
                        if (r > 0.0) {
                            p = r*growthModifier*dT;
                            if ((p > 0.25) and (!Warn_r)) {
                                cout << "\tWarning: Infection Increase Probability Large!" << "\n";
                                f_log  << "Warning: Infection Increase Probability Large!" << "\n";
                                Warn_r = true;
                            }
                            N = ComputeEvents(arr_I9[i][j][k], p, 2);  // Bursting events

                            // Update count
                            arr_I9[i][j][k]    = max(0.0, arr_I9[i][j][k] - N);
                            arr_Occ[i][j][k]   = max(0.0, arr_Occ[i][j][k] - N);
                            arr_P_new[i][j][k] += round( (1 - alpha) * Beta * N);   // Phages which escape the colony
                            M = round(alpha * Beta * N);                        // Phages which reinfect the colony

                            // Non-bursting events
                            N = ComputeEvents(arr_I8[i][j][k], p, 2);
                            arr_I8[i][j][k] = max(0.0, arr_I8[i][j][k] - N);
                            arr_I9[i][j][k] += N;

                            N = ComputeEvents(arr_I7[i][j][k], p, 2);
                            arr_I7[i][j][k] = max(0.0, arr_I7[i][j][k] - N);
                            arr_I8[i][j][k] += N;

                            N = ComputeEvents(arr_I6[i][j][k], p, 2);
                            arr_I6[i][j][k] = max(0.0, arr_I6[i][j][k] - N);
                            arr_I7[i][j][k] += N;

                            N = ComputeEvents(arr_I5[i][j][k], p, 2);
                            arr_I5[i][j][k] = max(0.0, arr_I5[i][j][k] - N);
                            arr_I6[i][j][k] += N;

                            N = ComputeEvents(arr_I4[i][j][k], p, 2);
                            arr_I4[i][j][k] = max(0.0, arr_I4[i][j][k] - N);
                            arr_I5[i][j][k] += N;

                            N = ComputeEvents(arr_I3[i][j][k], p, 2);
                            arr_I3[i][j][k] = max(0.0, arr_I3[i][j][k] - N);
                            arr_I4[i][j][k] += N;

                            N = ComputeEvents(arr_I2[i][j][k], p, 2);
                            arr_I2[i][j][k] = max(0.0, arr_I2[i][j][k] - N);
                            arr_I3[i][j][k] += N;

                            N = ComputeEvents(arr_I1[i][j][k], p, 2);
                            arr_I1[i][j][k] = max(0.0, arr_I1[i][j][k] - N);
                            arr_I2[i][j][k] += N;

                            N = ComputeEvents(arr_I0[i][j][k], p, 2);
                            arr_I0[i][j][k] = max(0.0, arr_I0[i][j][k] - N);
                            arr_I1[i][j][k] += N;

                        }

                        // Infectons //////////////////////////////////////////////////////////////////
                        if ((arr_Occ[i][j][k] >= 1) and (arr_P[i][j][k] >= 1)) {
                            double s;   // The factor which modifies the adsorption rate
                            double n;   // The number of targets the phage has

                            if (clustering) {   // Check if clustering is enabled
                                s = pow(arr_Occ[i][j][k] / arr_nC[i][j][k], 1.0 / 3.0);
                                n = arr_nC[i][j][k];
                            } else {            // Else use mean field computation
                                s = 1.0;
                                n = arr_Occ[i][j][k];
                            }

                            // Compute the number of hits
                            if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
                                N = arr_P[i][j][k];
                            } else {
                                p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
                                N = ComputeEvents(arr_P[i][j][k], p, 4);     // Number of targets hit
                            }

                            // If bacteria were hit, update events
                            if ((N + M) >= 1) {

                                arr_P[i][j][k]    = max(0.0, arr_P[i][j][k] - N);     // Update count

                                double S;
                                if (shielding) {
                                    // Absorbing medium model
                                    double d = pow(arr_Occ[i][j][k] / arr_nC[i][j][k], 1.0 / 3.0) - pow(arr_B[i][j][k] / arr_nC[i][j][k], 1.0 / 3.0);
                                    S = exp(-zeta*d); // Probability of hitting succebtible target

                                } else {
                                    // Well mixed model
                                    S = arr_B[i][j][k] / arr_Occ[i][j][k];
                                }

                                p = max(0.0, min(arr_B[i][j][k] / arr_Occ[i][j][k], S)); // Probability of hitting succebtible target
                                N = ComputeEvents(N + M, p, 4);                  // Number of targets hit

                                if (N > arr_B[i][j][k]) N = arr_B[i][j][k];              // If more bacteria than present are set to be infeced, round down

                                // Update the counts
                                arr_B[i][j][k]      = max(0.0, arr_B[i][j][k] - N);
                                if (r > 0.0) {
                                    arr_I0_new[i][j][k] += N;
                                } else {
                                    arr_P_new[i][j][k] += N * (1 - alpha) * Beta;
                                }
                            }
                        }

                        // Phage Decay ////////////////////////////////////////////////////////////////
                        p = delta*dT;
                        if ((p > 0.1) and (!Warn_delta)) {
                            cout << "\tWarning: Decay Probability Large!" << "\n";
                            f_log  << "Warning: Decay Probability Large!" << "\n";
                            Warn_delta = true;
                        }
                        N = ComputeEvents(arr_P[i][j][k], p, 5);

                        // Update count
                        arr_P[i][j][k]    = max(0.0, arr_P[i][j][k] - N);


                        // Movement ///////////////////////////////////////////////////////////////////
                        if (nGridXY > 1) {

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

                            // Update counts
                            double n_0; // No movement
                            double n_u; // Up
                            double n_d; // Down
                            double n_l; // Left
                            double n_r; // Right
                            double n_f; // Front
                            double n_b; // Back

                            // CELLS
                            ComputeDiffusion(arr_B[i][j][k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b,1);
                            arr_B_new[i][j][k] += n_0; arr_B_new[ip][j][k] += n_u; arr_B_new[im][j][k] += n_d; arr_B_new[i][jp][k] += n_r; arr_B_new[i][jm][k] += n_l; arr_B_new[i][j][kp] += n_f; arr_B_new[i][j][km] += n_b;

                            if (r > 0.0) {
                                ComputeDiffusion(arr_I0[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I0_new[i][j][k] += n_0; arr_I0_new[ip][j][k] += n_u; arr_I0_new[im][j][k] += n_d; arr_I0_new[i][jp][k] += n_r; arr_I0_new[i][jm][k] += n_l; arr_I0_new[i][j][kp] += n_f; arr_I0_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I1[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I1_new[i][j][k] += n_0; arr_I1_new[ip][j][k] += n_u; arr_I1_new[im][j][k] += n_d; arr_I1_new[i][jp][k] += n_r; arr_I1_new[i][jm][k] += n_l; arr_I1_new[i][j][kp] += n_f; arr_I1_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I2[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I2_new[i][j][k] += n_0; arr_I2_new[ip][j][k] += n_u; arr_I2_new[im][j][k] += n_d; arr_I2_new[i][jp][k] += n_r; arr_I2_new[i][jm][k] += n_l; arr_I2_new[i][j][kp] += n_f; arr_I2_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I3[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I3_new[i][j][k] += n_0; arr_I3_new[ip][j][k] += n_u; arr_I3_new[im][j][k] += n_d; arr_I3_new[i][jp][k] += n_r; arr_I3_new[i][jm][k] += n_l; arr_I3_new[i][j][kp] += n_f; arr_I3_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I4[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I4_new[i][j][k] += n_0; arr_I4_new[ip][j][k] += n_u; arr_I4_new[im][j][k] += n_d; arr_I4_new[i][jp][k] += n_r; arr_I4_new[i][jm][k] += n_l; arr_I4_new[i][j][kp] += n_f; arr_I4_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I5[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I5_new[i][j][k] += n_0; arr_I5_new[ip][j][k] += n_u; arr_I5_new[im][j][k] += n_d; arr_I5_new[i][jp][k] += n_r; arr_I5_new[i][jm][k] += n_l; arr_I5_new[i][j][kp] += n_f; arr_I5_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I6[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I6_new[i][j][k] += n_0; arr_I6_new[ip][j][k] += n_u; arr_I6_new[im][j][k] += n_d; arr_I6_new[i][jp][k] += n_r; arr_I6_new[i][jm][k] += n_l; arr_I6_new[i][j][kp] += n_f; arr_I6_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I7[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I7_new[i][j][k] += n_0; arr_I7_new[ip][j][k] += n_u; arr_I7_new[im][j][k] += n_d; arr_I7_new[i][jp][k] += n_r; arr_I7_new[i][jm][k] += n_l; arr_I7_new[i][j][kp] += n_f; arr_I7_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I8[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I8_new[i][j][k] += n_0; arr_I8_new[ip][j][k] += n_u; arr_I8_new[im][j][k] += n_d; arr_I8_new[i][jp][k] += n_r; arr_I8_new[i][jm][k] += n_l; arr_I8_new[i][j][kp] += n_f; arr_I8_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I9[i][j][k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I9_new[i][j][k] += n_0; arr_I9_new[ip][j][k] += n_u; arr_I9_new[im][j][k] += n_d; arr_I9_new[i][jp][k] += n_r; arr_I9_new[i][jm][k] += n_l; arr_I9_new[i][j][kp] += n_f; arr_I9_new[i][j][km] += n_b;
                            }

                            // PHAGES
                            ComputeDiffusion(arr_P[i][j][k], lambdaP, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 3);
                            arr_P_new[i][j][k] += n_0; arr_P_new[ip][j][k] += n_u; arr_P_new[im][j][k] += n_d; arr_P_new[i][jp][k] += n_r; arr_P_new[i][jm][k] += n_l; arr_P_new[i][j][kp] += n_f; arr_P_new[i][j][km] += n_b;



                        } else {
                            // CELLS
                            arr_B_new[i][j][k] += arr_B[i][j][k];

                            if (r > 0.0) {
                                arr_I0_new[i][j][k] += arr_I0[i][j][k];
                                arr_I1_new[i][j][k] += arr_I1[i][j][k];
                                arr_I2_new[i][j][k] += arr_I2[i][j][k];
                                arr_I3_new[i][j][k] += arr_I3[i][j][k];
                                arr_I4_new[i][j][k] += arr_I4[i][j][k];
                                arr_I5_new[i][j][k] += arr_I5[i][j][k];
                                arr_I6_new[i][j][k] += arr_I6[i][j][k];
                                arr_I7_new[i][j][k] += arr_I7[i][j][k];
                                arr_I8_new[i][j][k] += arr_I8[i][j][k];
                                arr_I9_new[i][j][k] += arr_I9[i][j][k];
                            }

                            // PHAGES
                            arr_P_new[i][j][k] += arr_P[i][j][k];
                        }
                    }
                }
            }

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
                        arr_B_new[i][j][k]  = 0.0;
                        arr_I0_new[i][j][k] = 0.0;
                        arr_I1_new[i][j][k] = 0.0;
                        arr_I2_new[i][j][k] = 0.0;
                        arr_I3_new[i][j][k] = 0.0;
                        arr_I4_new[i][j][k] = 0.0;
                        arr_I5_new[i][j][k] = 0.0;
                        arr_I6_new[i][j][k] = 0.0;
                        arr_I7_new[i][j][k] = 0.0;
                        arr_I8_new[i][j][k] = 0.0;
                        arr_I9_new[i][j][k] = 0.0;
                        arr_P_new[i][j][k]  = 0.0;
                    }
                }
            }


            // Update occupancy
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_Occ[i][j][k] = arr_B[i][j][k] + arr_I0[i][j][k] + arr_I1[i][j][k] + arr_I2[i][j][k] + arr_I3[i][j][k] + arr_I4[i][j][k] + arr_I5[i][j][k] + arr_I6[i][j][k] + arr_I7[i][j][k] + arr_I8[i][j][k] + arr_I9[i][j][k];
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

                        double tmp = arr_nutrient[i][j][k];
                        arr_nutrient_new[i][j][k]  += tmp - 6 * alphaXY * tmp;
                        arr_nutrient_new[ip][j][k] += alphaXY * tmp;
                        arr_nutrient_new[im][j][k] += alphaXY * tmp;
                        arr_nutrient_new[i][jp][k] += alphaXY * tmp;
                        arr_nutrient_new[i][jm][k] += alphaXY * tmp;
                        arr_nutrient_new[i][j][kp] += alphaZ  * tmp;
                        arr_nutrient_new[i][j][km] += alphaZ  * tmp;
                    }
                }
            }

            std::swap(arr_nutrient, arr_nutrient_new);

            // Zero the _new arrays
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_nutrient_new[i][j][k]  = 0.0;
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
                    accuB += arr_B[i][j][k];
                }
            }
        }
        if ((fastExit) and (accuB < 1)) {
            // Update the P array
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_P[i][j][k] += (1-alpha)*beta * (arr_I0[i][j][k] + arr_I1[i][j][k] + arr_I2[i][j][k] + arr_I3[i][j][k] + arr_I4[i][j][k] + arr_I5[i][j][k] + arr_I6[i][j][k] + arr_I7[i][j][k] + arr_I8[i][j][k] + arr_I9[i][j][k]);
                    }
                }
            }


            // Zero the I arrays
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_I0[i][j][k] = 0.0;
                        arr_I1[i][j][k] = 0.0;
                        arr_I2[i][j][k] = 0.0;
                        arr_I3[i][j][k] = 0.0;
                        arr_I4[i][j][k] = 0.0;
                        arr_I5[i][j][k] = 0.0;
                        arr_I6[i][j][k] = 0.0;
                        arr_I7[i][j][k] = 0.0;
                        arr_I8[i][j][k] = 0.0;
                        arr_I9[i][j][k] = 0.0;
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
                    accuOcc += arr_Occ[i][j][k];
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
                    double tmpN = arr_nutrient[i][j][k];
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
        ExportData_arr(T,"noMMM");

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

			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						double p = 0; // privatize
						double N = 0; // privatize
						double M = 0; // privatize

						/* BEGIN første Map-kernel */
						// Ensure nC is updated
						if (arr_Occ[i][j][k] < arr_nC[i][j][k]) arr_nC[i][j][k] = arr_Occ[i][j][k];

						// Skip empty sites
						if ((arr_Occ[i][j][k] < 1) and (arr_P[i][j][k] < 1)) continue;

						// Record the maximum observed density
						if (arr_Occ[i][j][k] > maxOccupancy) maxOccupancy = arr_Occ[i][j][k];

						// Compute the growth modifier
						double growthModifier = arr_nutrient[i][j][k] / (arr_nutrient[i][j][k] + K);

						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= growthModifier;
						}

						/* END første Map-kernel */

					}
				}
			}

			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						// Birth //////////////////////////////////////////////////////////////////////

						double p = 0; // privatize
						double N = 0; // privatize


						// Compute the growth modifier
						double growthModifier = arr_nutrient[i][j][k] / (arr_nutrient[i][j][k] + K);
///////////// should the growth modifier have been an array instead?

						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= growthModifier;
						}



						p = g * growthModifier*dT;				// MO flyttet til kernel 2
						if (arr_nutrient[i][j][k] < 1) {		//
							p = 0;								//
						}										//

						if ((p > 0.1) and (!Warn_g)) {
                            cout << "\tWarning: Birth Probability Large!" << "\n";
                            f_log  << "Warning: Birth Probability Large!" << "\n";
                            Warn_g = true;
                        }

                        /* BEGIN anden Map-kernel */
                        N = ComputeEvents(arr_B[i][j][k], p, 1);
                        // Ensure there is enough nutrient
                        if ( N > arr_nutrient[i][j][k] ) {
                            if (!Warn_fastGrowth) {
                                cout << "\tWarning: Colonies growing too fast!" << "\n";
                                f_log  << "Warning: Colonies growing too fast!" << "\n";
                                Warn_fastGrowth = true;
                            }

                            N = round( arr_nutrient[i][j][k] );
                        }

                        // Update count
                        arr_B_new[i][j][k] += N;
                        arr_nutrient[i][j][k] = max(0.0, arr_nutrient[i][j][k] - N);
                        /* END anden Map-kernel */

					}
				}
			}

			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						double p = 0; // privatize
						double N = 0; // privatize
						double M = 0; // privatize

						// Compute the growth modifier
						double growthModifier = arr_nutrient[i][j][k] / (arr_nutrient[i][j][k] + K);
///////////// should the growth modifier have been an array instead?
						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= growthModifier;
						}

                        // Increase Infections ////////////////////////////////////////////////////////
                        if (r > 0.0) {
                            /* BEGIN tredje Map-kernel */




                            p = r*growthModifier*dT;
                            if ((p > 0.25) and (!Warn_r)) {
                                cout << "\tWarning: Infection Increase Probability Large!" << "\n";
                                f_log  << "Warning: Infection Increase Probability Large!" << "\n";
                                Warn_r = true;
                            }
                            N = ComputeEvents(arr_I9[i][j][k], p, 2);  // Bursting events

                            // Update count
                            arr_I9[i][j][k]    = max(0.0, arr_I9[i][j][k] - N);
                            arr_Occ[i][j][k]   = max(0.0, arr_Occ[i][j][k] - N);
                            arr_P_new[i][j][k] += round( (1 - alpha) * Beta * N);   // Phages which escape the colony
                            M = round(alpha * Beta * N);                        // Phages which reinfect the colony

                            // Non-bursting events
                            N = ComputeEvents(arr_I8[i][j][k], p, 2);
                            arr_I8[i][j][k] = max(0.0, arr_I8[i][j][k] - N);
                            arr_I9[i][j][k] += N;

                            N = ComputeEvents(arr_I7[i][j][k], p, 2);
                            arr_I7[i][j][k] = max(0.0, arr_I7[i][j][k] - N);
                            arr_I8[i][j][k] += N;

                            N = ComputeEvents(arr_I6[i][j][k], p, 2);
                            arr_I6[i][j][k] = max(0.0, arr_I6[i][j][k] - N);
                            arr_I7[i][j][k] += N;

                            N = ComputeEvents(arr_I5[i][j][k], p, 2);
                            arr_I5[i][j][k] = max(0.0, arr_I5[i][j][k] - N);
                            arr_I6[i][j][k] += N;

                            N = ComputeEvents(arr_I4[i][j][k], p, 2);
                            arr_I4[i][j][k] = max(0.0, arr_I4[i][j][k] - N);
                            arr_I5[i][j][k] += N;

                            N = ComputeEvents(arr_I3[i][j][k], p, 2);
                            arr_I3[i][j][k] = max(0.0, arr_I3[i][j][k] - N);
                            arr_I4[i][j][k] += N;

                            N = ComputeEvents(arr_I2[i][j][k], p, 2);
                            arr_I2[i][j][k] = max(0.0, arr_I2[i][j][k] - N);
                            arr_I3[i][j][k] += N;

                            N = ComputeEvents(arr_I1[i][j][k], p, 2);
                            arr_I1[i][j][k] = max(0.0, arr_I1[i][j][k] - N);
                            arr_I2[i][j][k] += N;

                            N = ComputeEvents(arr_I0[i][j][k], p, 2);
                            arr_I0[i][j][k] = max(0.0, arr_I0[i][j][k] - N);
                            arr_I1[i][j][k] += N;

                            /* END tredje Map-kernel */

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

						double p = 0; // privatize
						double N = 0; // privatize
						double M = 0; // privatize

						// Compute the growth modifier
						double growthModifier = arr_nutrient[i][j][k] / (arr_nutrient[i][j][k] + K);
///////////// should the growth modifier have been an array instead?
						// Compute beta
						double Beta = beta;
						if (reducedBeta) {
							Beta *= growthModifier;
						}

                        // PRIVATIZE BOTH OF THESE
                        double s;   // The factor which modifies the adsorption rate
                        double n;   // The number of targets the phage has
                        // Infectons


                        // KERNEL THIS
                        if ((arr_Occ[i][j][k] >= 1) and (arr_P[i][j][k] >= 1)) {
                            if (clustering) {   // Check if clustering is enabled
                                s = pow(arr_Occ[i][j][k] / arr_nC[i][j][k], 1.0 / 3.0);
                                n = arr_nC[i][j][k];
                            } else {            // Else use mean field computation
                                s = 1.0;
                                n = arr_Occ[i][j][k];
                            }
                        }

                        if ((arr_Occ[i][j][k] >= 1) and (arr_P[i][j][k] >= 1)) {
                            // Compute the number of hits
                            if (eta * s * dT >= 1) { // In the diffusion limited case every phage hits a target
                                N = arr_P[i][j][k];
                            } else {
                                p = 1 - pow(1 - eta * s * dT, n);        // Probability hitting any target
                                N = ComputeEvents(arr_P[i][j][k], p, 4);     // Number of targets hit
                            }

                            if (N + M >= 1) {
                                // If bacteria were hit, update events
                                arr_P[i][j][k] = max(0.0, arr_P[i][j][k] - N);     // Update count

                                double S;
                                if (shielding) {
                                    // Absorbing medium model
                                    double d = pow(arr_Occ[i][j][k] / arr_nC[i][j][k], 1.0 / 3.0) -
                                               pow(arr_B[i][j][k] / arr_nC[i][j][k], 1.0 / 3.0);
                                    S = exp(-zeta * d); // Probability of hitting succebtible target

                                } else {
                                    // Well mixed model
                                    S = arr_B[i][j][k] / arr_Occ[i][j][k];
                                }

                                p = max(0.0, min(arr_B[i][j][k] / arr_Occ[i][j][k],
                                                 S)); // Probability of hitting succebtible target
                                N = ComputeEvents(N + M, p, 4);                  // Number of targets hit

                                if (N > arr_B[i][j][k])
                                    N = arr_B[i][j][k];              // If more bacteria than present are set to be infeced, round down

                                // Update the counts
                                arr_B[i][j][k] = max(0.0, arr_B[i][j][k] - N);
                                if (r > 0.0) {
                                    arr_I0_new[i][j][k] += N;
                                } else {
                                    arr_P_new[i][j][k] += N * (1 - alpha) * Beta;
                                }
                            }
                        }

                        // Phage Decay ////////////////////////////////////////////////////////////////

					}
				}
			}

			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

						double p = 0; // privatize
						double N = 0; // privatize

                        // KERNEL BEGIN
                        p = delta*dT;
                        if ((p > 0.1) and (!Warn_delta)) {
                            cout << "\tWarning: Decay Probability Large!" << "\n";
                            f_log  << "Warning: Decay Probability Large!" << "\n";
                            Warn_delta = true;
                        }
                        N = ComputeEvents(arr_P[i][j][k], p, 5);

                        // Update count
                        arr_P[i][j][k]    = max(0.0, arr_P[i][j][k] - N);
                        // KERNEL END

					}
				}
			}

			for (int i = 0; i < nGridXY; i++) {
				if (exit) break;

				for (int j = 0; j < nGridXY; j++) {
					if (exit) break;

					for (int k = 0; k < nGridZ; k++) {
						if (exit) break;

                        // Movement ///////////////////////////////////////////////////////////////////
                        if (nGridXY > 1) {
                            // KERNEL BEGIN
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

                            // Update counts
                            double n_0; // No movement
                            double n_u; // Up
                            double n_d; // Down
                            double n_l; // Left
                            double n_r; // Right
                            double n_f; // Front
                            double n_b; // Back

                            // CELLS
                            ComputeDiffusion(arr_B[i][j][k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b,1);
                            arr_B_new[i][j][k] += n_0; arr_B_new[ip][j][k] += n_u; arr_B_new[im][j][k] += n_d; arr_B_new[i][jp][k] += n_r; arr_B_new[i][jm][k] += n_l; arr_B_new[i][j][kp] += n_f; arr_B_new[i][j][km] += n_b;

                            if (r > 0.0) {
                                ComputeDiffusion(arr_I0[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I0_new[i][j][k] += n_0; arr_I0_new[ip][j][k] += n_u; arr_I0_new[im][j][k] += n_d; arr_I0_new[i][jp][k] += n_r; arr_I0_new[i][jm][k] += n_l; arr_I0_new[i][j][kp] += n_f; arr_I0_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I1[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I1_new[i][j][k] += n_0; arr_I1_new[ip][j][k] += n_u; arr_I1_new[im][j][k] += n_d; arr_I1_new[i][jp][k] += n_r; arr_I1_new[i][jm][k] += n_l; arr_I1_new[i][j][kp] += n_f; arr_I1_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I2[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I2_new[i][j][k] += n_0; arr_I2_new[ip][j][k] += n_u; arr_I2_new[im][j][k] += n_d; arr_I2_new[i][jp][k] += n_r; arr_I2_new[i][jm][k] += n_l; arr_I2_new[i][j][kp] += n_f; arr_I2_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I3[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I3_new[i][j][k] += n_0; arr_I3_new[ip][j][k] += n_u; arr_I3_new[im][j][k] += n_d; arr_I3_new[i][jp][k] += n_r; arr_I3_new[i][jm][k] += n_l; arr_I3_new[i][j][kp] += n_f; arr_I3_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I4[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I4_new[i][j][k] += n_0; arr_I4_new[ip][j][k] += n_u; arr_I4_new[im][j][k] += n_d; arr_I4_new[i][jp][k] += n_r; arr_I4_new[i][jm][k] += n_l; arr_I4_new[i][j][kp] += n_f; arr_I4_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I5[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I5_new[i][j][k] += n_0; arr_I5_new[ip][j][k] += n_u; arr_I5_new[im][j][k] += n_d; arr_I5_new[i][jp][k] += n_r; arr_I5_new[i][jm][k] += n_l; arr_I5_new[i][j][kp] += n_f; arr_I5_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I6[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I6_new[i][j][k] += n_0; arr_I6_new[ip][j][k] += n_u; arr_I6_new[im][j][k] += n_d; arr_I6_new[i][jp][k] += n_r; arr_I6_new[i][jm][k] += n_l; arr_I6_new[i][j][kp] += n_f; arr_I6_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I7[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I7_new[i][j][k] += n_0; arr_I7_new[ip][j][k] += n_u; arr_I7_new[im][j][k] += n_d; arr_I7_new[i][jp][k] += n_r; arr_I7_new[i][jm][k] += n_l; arr_I7_new[i][j][kp] += n_f; arr_I7_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I8[i][j][k],  lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I8_new[i][j][k] += n_0; arr_I8_new[ip][j][k] += n_u; arr_I8_new[im][j][k] += n_d; arr_I8_new[i][jp][k] += n_r; arr_I8_new[i][jm][k] += n_l; arr_I8_new[i][j][kp] += n_f; arr_I8_new[i][j][km] += n_b;

                                ComputeDiffusion(arr_I9[i][j][k], lambdaB, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 2);
                                arr_I9_new[i][j][k] += n_0; arr_I9_new[ip][j][k] += n_u; arr_I9_new[im][j][k] += n_d; arr_I9_new[i][jp][k] += n_r; arr_I9_new[i][jm][k] += n_l; arr_I9_new[i][j][kp] += n_f; arr_I9_new[i][j][km] += n_b;
                            }

                            // PHAGES
                            ComputeDiffusion(arr_P[i][j][k], lambdaP, &n_0, &n_u, &n_d, &n_l, &n_r, &n_f, &n_b, 3);
                            arr_P_new[i][j][k] += n_0; arr_P_new[ip][j][k] += n_u; arr_P_new[im][j][k] += n_d; arr_P_new[i][jp][k] += n_r; arr_P_new[i][jm][k] += n_l; arr_P_new[i][j][kp] += n_f; arr_P_new[i][j][km] += n_b;

                            // KERNEL END



                        } else {
                            // KERNEL BEGIN
                            // CELLS
                            arr_B_new[i][j][k] += arr_B[i][j][k];

                            if (r > 0.0) {
                                arr_I0_new[i][j][k] += arr_I0[i][j][k];
                                arr_I1_new[i][j][k] += arr_I1[i][j][k];
                                arr_I2_new[i][j][k] += arr_I2[i][j][k];
                                arr_I3_new[i][j][k] += arr_I3[i][j][k];
                                arr_I4_new[i][j][k] += arr_I4[i][j][k];
                                arr_I5_new[i][j][k] += arr_I5[i][j][k];
                                arr_I6_new[i][j][k] += arr_I6[i][j][k];
                                arr_I7_new[i][j][k] += arr_I7[i][j][k];
                                arr_I8_new[i][j][k] += arr_I8[i][j][k];
                                arr_I9_new[i][j][k] += arr_I9[i][j][k];
                            }

                            // PHAGES
                            arr_P_new[i][j][k] += arr_P[i][j][k];
                            // KERNEL END
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
                        arr_B_new[i][j][k]  = 0.0;
                        arr_I0_new[i][j][k] = 0.0;
                        arr_I1_new[i][j][k] = 0.0;
                        arr_I2_new[i][j][k] = 0.0;
                        arr_I3_new[i][j][k] = 0.0;
                        arr_I4_new[i][j][k] = 0.0;
                        arr_I5_new[i][j][k] = 0.0;
                        arr_I6_new[i][j][k] = 0.0;
                        arr_I7_new[i][j][k] = 0.0;
                        arr_I8_new[i][j][k] = 0.0;
                        arr_I9_new[i][j][k] = 0.0;
                        arr_P_new[i][j][k]  = 0.0;
                    }
                }
            }


            // Update occupancy
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_Occ[i][j][k] = arr_B[i][j][k] + arr_I0[i][j][k] + arr_I1[i][j][k] + arr_I2[i][j][k] + arr_I3[i][j][k] + arr_I4[i][j][k] + arr_I5[i][j][k] + arr_I6[i][j][k] + arr_I7[i][j][k] + arr_I8[i][j][k] + arr_I9[i][j][k];
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

                        double tmp = arr_nutrient[i][j][k];
                        arr_nutrient_new[i][j][k]  += tmp - 6 * alphaXY * tmp;
                        arr_nutrient_new[ip][j][k] += alphaXY * tmp;
                        arr_nutrient_new[im][j][k] += alphaXY * tmp;
                        arr_nutrient_new[i][jp][k] += alphaXY * tmp;
                        arr_nutrient_new[i][jm][k] += alphaXY * tmp;
                        arr_nutrient_new[i][j][kp] += alphaZ  * tmp;
                        arr_nutrient_new[i][j][km] += alphaZ  * tmp;
                    }
                }
            }

            std::swap(arr_nutrient, arr_nutrient_new);

            // Zero the _new arrays
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_nutrient_new[i][j][k]  = 0.0;
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
                    accuB += arr_B[i][j][k];
                }
            }
        }
        if ((fastExit) and (accuB < 1)) {
            // Update the P array
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_P[i][j][k] += (1-alpha)*beta * (arr_I0[i][j][k] + arr_I1[i][j][k] + arr_I2[i][j][k] + arr_I3[i][j][k] + arr_I4[i][j][k] + arr_I5[i][j][k] + arr_I6[i][j][k] + arr_I7[i][j][k] + arr_I8[i][j][k] + arr_I9[i][j][k]);
                    }
                }
            }


            // Zero the I arrays
            for (int i = 0; i < nGridXY; i++) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int k = 0; k < nGridZ; k++ ) {
                        arr_I0[i][j][k] = 0.0;
                        arr_I1[i][j][k] = 0.0;
                        arr_I2[i][j][k] = 0.0;
                        arr_I3[i][j][k] = 0.0;
                        arr_I4[i][j][k] = 0.0;
                        arr_I5[i][j][k] = 0.0;
                        arr_I6[i][j][k] = 0.0;
                        arr_I7[i][j][k] = 0.0;
                        arr_I8[i][j][k] = 0.0;
                        arr_I9[i][j][k] = 0.0;
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
                    accuOcc += arr_Occ[i][j][k];
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
                    double tmpN = arr_nutrient[i][j][k];
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
    B.zeros(nGridXY, nGridXY, nGridZ);
    I0.zeros(nGridXY, nGridXY, nGridZ);
    I1.zeros(nGridXY, nGridXY, nGridZ);
    I2.zeros(nGridXY, nGridXY, nGridZ);
    I3.zeros(nGridXY, nGridXY, nGridZ);
    I4.zeros(nGridXY, nGridXY, nGridZ);
    I5.zeros(nGridXY, nGridXY, nGridZ);
    I6.zeros(nGridXY, nGridXY, nGridZ);
    I7.zeros(nGridXY, nGridXY, nGridZ);
    I8.zeros(nGridXY, nGridXY, nGridZ);
    I9.zeros(nGridXY, nGridXY, nGridZ);
    P.zeros(nGridXY, nGridXY, nGridZ);
    nC.zeros(nGridXY, nGridXY, nGridZ);

    B_new.zeros(nGridXY, nGridXY, nGridZ);
    I0_new.zeros(nGridXY, nGridXY, nGridZ);
    I1_new.zeros(nGridXY, nGridXY, nGridZ);
    I2_new.zeros(nGridXY, nGridXY, nGridZ);
    I3_new.zeros(nGridXY, nGridXY, nGridZ);
    I4_new.zeros(nGridXY, nGridXY, nGridZ);
    I5_new.zeros(nGridXY, nGridXY, nGridZ);
    I6_new.zeros(nGridXY, nGridXY, nGridZ);
    I7_new.zeros(nGridXY, nGridXY, nGridZ);
    I8_new.zeros(nGridXY, nGridXY, nGridZ);
    I9_new.zeros(nGridXY, nGridXY, nGridZ);
    P_new.zeros(nGridXY, nGridXY, nGridZ);

    nutrient.zeros(nGridXY, nGridXY, nGridZ);
    nutrient_new.zeros(nGridXY, nGridXY, nGridZ);

    Occ.zeros(nGridXY, nGridXY, nGridZ);



    // Compute the step size
    double dXY = L / nGridXY;
    double dZ  = H / nGridZ;
    double dV  = dXY * dXY * dZ;

    // Allocate arrays
    arr_B   = new double**[nGridXY];
    arr_I0  = new double**[nGridXY];
    arr_I1  = new double**[nGridXY];
    arr_I2  = new double**[nGridXY];
    arr_I3  = new double**[nGridXY];
    arr_I4  = new double**[nGridXY];
    arr_I5  = new double**[nGridXY];
    arr_I6  = new double**[nGridXY];
    arr_I7  = new double**[nGridXY];
    arr_I8  = new double**[nGridXY];
    arr_I9  = new double**[nGridXY];
    arr_P   = new double**[nGridXY];
    arr_nC  = new double**[nGridXY];

    arr_B_new   = new double**[nGridXY];
    arr_I0_new  = new double**[nGridXY];
    arr_I1_new  = new double**[nGridXY];
    arr_I2_new  = new double**[nGridXY];
    arr_I3_new  = new double**[nGridXY];
    arr_I4_new  = new double**[nGridXY];
    arr_I5_new  = new double**[nGridXY];
    arr_I6_new  = new double**[nGridXY];
    arr_I7_new  = new double**[nGridXY];
    arr_I8_new  = new double**[nGridXY];
    arr_I9_new  = new double**[nGridXY];
    arr_P_new   = new double**[nGridXY];

    arr_nutrient = new double**[nGridXY];
    arr_Occ      = new double**[nGridXY];
    arr_nutrient_new = new double**[nGridXY];


    for (int i = 0; i < nGridXY; i++) {

        arr_B[i]   = new double*[nGridXY];
        arr_I0[i]  = new double*[nGridXY];
        arr_I1[i]  = new double*[nGridXY];
        arr_I2[i]  = new double*[nGridXY];
        arr_I3[i]  = new double*[nGridXY];
        arr_I4[i]  = new double*[nGridXY];
        arr_I5[i]  = new double*[nGridXY];
        arr_I6[i]  = new double*[nGridXY];
        arr_I7[i]  = new double*[nGridXY];
        arr_I8[i]  = new double*[nGridXY];
        arr_I9[i]  = new double*[nGridXY];
        arr_P[i]   = new double*[nGridXY];
        arr_nC[i]  = new double*[nGridXY];

        arr_B_new[i]    = new double*[nGridXY];
        arr_I0_new[i]   = new double*[nGridXY];
        arr_I1_new[i]   = new double*[nGridXY];
        arr_I2_new[i]   = new double*[nGridXY];
        arr_I3_new[i]   = new double*[nGridXY];
        arr_I4_new[i]   = new double*[nGridXY];
        arr_I5_new[i]   = new double*[nGridXY];
        arr_I6_new[i]   = new double*[nGridXY];
        arr_I7_new[i]   = new double*[nGridXY];
        arr_I8_new[i]   = new double*[nGridXY];
        arr_I9_new[i]   = new double*[nGridXY];
        arr_P_new[i]    = new double*[nGridXY];

        arr_nutrient[i] = new double*[nGridXY];
        arr_Occ[i]      = new double*[nGridXY];
        arr_nutrient_new[i] = new double*[nGridXY];

        for (int j = 0; j < nGridXY; j++) {

            arr_B[i][j]   = new double[nGridZ];
            arr_I0[i][j]  = new double[nGridZ];
            arr_I1[i][j]  = new double[nGridZ];
            arr_I2[i][j]  = new double[nGridZ];
            arr_I3[i][j]  = new double[nGridZ];
            arr_I4[i][j]  = new double[nGridZ];
            arr_I5[i][j]  = new double[nGridZ];
            arr_I6[i][j]  = new double[nGridZ];
            arr_I7[i][j]  = new double[nGridZ];
            arr_I8[i][j]  = new double[nGridZ];
            arr_I9[i][j]  = new double[nGridZ];
            arr_P[i][j]   = new double[nGridZ];
            arr_nC[i][j]  = new double[nGridZ];

            arr_B_new[i][j]     = new double[nGridZ];
            arr_I0_new[i][j]    = new double[nGridZ];
            arr_I1_new[i][j]    = new double[nGridZ];
            arr_I2_new[i][j]    = new double[nGridZ];
            arr_I3_new[i][j]    = new double[nGridZ];
            arr_I4_new[i][j]    = new double[nGridZ];
            arr_I5_new[i][j]    = new double[nGridZ];
            arr_I6_new[i][j]    = new double[nGridZ];
            arr_I7_new[i][j]    = new double[nGridZ];
            arr_I8_new[i][j]    = new double[nGridZ];
            arr_I9_new[i][j]    = new double[nGridZ];
            arr_P_new[i][j]     = new double[nGridZ];

            arr_nutrient[i][j]  = new double[nGridZ];
            arr_Occ[i][j]       = new double[nGridZ];
            arr_nutrient_new[i][j]  = new double[nGridZ];


           for (int k = 0; k < nGridZ; k++) {
                arr_B[i][j][k]  = 0.0;
                arr_I0[i][j][k] = 0.0;
                arr_I1[i][j][k] = 0.0;
                arr_I2[i][j][k] = 0.0;
                arr_I3[i][j][k] = 0.0;
                arr_I4[i][j][k] = 0.0;
                arr_I5[i][j][k] = 0.0;
                arr_I6[i][j][k] = 0.0;
                arr_I7[i][j][k] = 0.0;
                arr_I8[i][j][k] = 0.0;
                arr_I9[i][j][k] = 0.0;
                arr_P[i][j][k]  = 0.0;
                arr_nC[i][j][k] = 0.0;

                arr_B_new[i][j][k]  = 0.0;
                arr_I0_new[i][j][k] = 0.0;
                arr_I1_new[i][j][k] = 0.0;
                arr_I2_new[i][j][k] = 0.0;
                arr_I3_new[i][j][k] = 0.0;
                arr_I4_new[i][j][k] = 0.0;
                arr_I5_new[i][j][k] = 0.0;
                arr_I6_new[i][j][k] = 0.0;
                arr_I7_new[i][j][k] = 0.0;
                arr_I8_new[i][j][k] = 0.0;
                arr_I9_new[i][j][k] = 0.0;
                arr_P_new[i][j][k]  = 0.0;

                arr_nutrient[i][j][k] = n_0 / 1e12 * dV;
                arr_Occ[i][j][k]  = 0.0;
                arr_nutrient_new[i][j][k] = 0.0;
            }
        }
    }

    // Initialize nutrient
    nutrient.fill( n_0 / 1e12 * dV ); // (n_0 / 1e12) is the nutrient per µm^Z

    lapXY.zeros(nGridXY, nGridXY);
    lapZ.zeros(nGridZ, nGridZ);

    if (nGridXY >= 3) {

        // Fill the laplace operator
        for (int i = 1; i < nGridXY - 1; i++ ) {
            lapXY(i, i)      = -2;
            lapXY(i, i + 1)  =  1;
            lapXY(i, i - 1)  =  1;
        }

        // Boundary conditions for laplace operator
        lapXY(0, 0)           = -2;
        lapXY(nGridXY - 1, 0) =  1;
        lapXY(0, 1)           =  1;

        lapXY(nGridXY - 1, nGridXY - 1) = -2;
        lapXY(0, nGridXY - 1)           =  1;
        lapXY(nGridXY - 1, nGridXY - 2) =  1;

    } else if (nGridXY > 1) {
        lapXY(0, 0) = -1;
        lapXY(0, 1) =  1;
        lapXY(1, 0) =  1;
        lapXY(1, 1) = -1;

    }

    if (nGridZ >= 3) {

        // Fill the laplace operator
        for (int i = 1; i < nGridZ - 1; i++ ) {
            lapZ(i, i)      = -2;
            lapZ(i, i + 1)  =  1;
            lapZ(i, i - 1)  =  1;
        }

        if (experimentalConditions) { // Apply reflective boundary conditions
            lapZ(0, 0) = -1;
            lapZ(0, 1) =  1;

            lapZ(nGridZ - 1, nGridZ - 1)  = -1;
            lapZ(nGridZ - 1, nGridZ - 2)  =  1;

        } else {
            // Boundary conditions for laplace operator
            lapZ(0, 0)          = -2;
            lapZ(nGridZ - 1, 0) =  1;
            lapZ(0, 1)          =  1;

            lapZ(nGridZ - 1, nGridZ - 1) = -2;
            lapZ(0, nGridZ - 1)          =  1;
            lapZ(nGridZ - 1, nGridZ - 2) =  1;
        }

    } else if (nGridZ > 1) {
        lapZ(0, 0) = -1;
        lapZ(0, 1) =  1;
        lapZ(1, 0) =  1;
        lapZ(1, 1) = -1;

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
                    nC(i, j, k) = BB;
                    arr_nC[i][j][k] = BB;

                    // Add the bacteria
                    B(i, j, k) = BB;
                    arr_B[i][j][k] = BB;
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
        B(i, j, k)++;
        nC(i, j, k)++;

        arr_B[i][j][k]++;
        arr_nC[i][j][k]++;

        numB++;


    }

    // Correct for overspawning
    while (numB > nBacteria) {
        int i = RandI(nGridXY - 1);
        int j = RandI(nGridXY - 1);
        int k = RandI(nGridZ  - 1);

        if (B(i, j, k) < 1) continue;

        B(i, j, k)--;
        numB--;
        nC(i, j, k)--;

        // if (arr_B[i][j][k] < 1) continue;

        arr_B[i][j][k]--;
        numB--;
        arr_nC[i][j][k]--;
    }

    // Count the initial occupancy
    // uvec nz = find(B);
    // initialOccupancy = nz.n_elem;

    int initialOccupancy = 0;
    for (int k = 0; k < nGridZ; k++ ) {
        for (int j = 0; j < nGridXY; j++ ) {
            for (int i = 0; i < nGridXY; i++) {
                if (arr_B[i][j][k] > 0.0) {
                    initialOccupancy++;
                }
            }
        }
    }


    // Determine the occupancy
    Occ = B + I0 + I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9;

    // Determine the occupancy
    for (int k = 0; k < nGridZ; k++ ) {
        for (int j = 0; j < nGridXY; j++ ) {
            for (int i = 0; i < nGridXY; i++) {
                arr_Occ[i][j][k] = arr_B[i][j][k] + arr_I0[i][j][k] + arr_I1[i][j][k] + arr_I2[i][j][k] + arr_I3[i][j][k] + arr_I4[i][j][k] + arr_I5[i][j][k] + arr_I6[i][j][k] + arr_I7[i][j][k] + arr_I8[i][j][k] + arr_I9[i][j][k];
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
                P(i, j, k)++;
                arr_P[i][j][k]++;
                numP++;
            }
        } else {
            for (int k = 0; k < nGridZ; k++ ) {
                for (int j = 0; j < nGridXY; j++ ) {
                    for (int i = 0; i < nGridXY; i++) {
                        double PP = RandP(nPhages / (nGridXY * nGridXY * nGridZ));

                        if (PP < 1) continue;
                        P(i, j, k) = PP;
                        arr_P[i][j][k] = PP;
                        numP += PP;
                    }
                }
            }
            // Correct for overspawning
            while (numP > nPhages) {
                int i = RandI(nGridXY - 1);
                int j = RandI(nGridXY - 1);
                int k = RandI(nGridZ - 1);

                if (P(i, j, k) > 0) {
                    P(i, j, k)--;
                    arr_P[i][j][k]--;
                    numP--;
                }
            }
            // Correct for underspawning
            while (numP < nPhages) {
                int i = RandI(nGridXY - 1);
                int j = RandI(nGridXY - 1);
                int k = RandI(nGridZ - 1);

                P(i, j, k)++;
                arr_P[i][j][k]++;
                numP++;
            }
        }

    } else { // Apply scenario specific settings

        // Determine the number of phages to spawn
        double nPhages = (double)round(L * L * H * P_0 / 1e12);
        double nGridXY = this->nGridXY;

        double numP = 0;
        if (nPhages <= nGridXY * nGridXY) {
            for (double n = 0; n < nPhages; n++) {
                P(RandI(nGridXY - 1), RandI(nGridXY - 1), nGridZ - 1)++;
                numP++;
            }
        } else {
            for (int j = 0; j < nGridXY; j++ ) {
                for (int i = 0; i < nGridXY; i++ ) {
                        P(i, j, nGridZ - 1) = RandP(nPhages / (nGridXY * nGridXY * nGridZ));
                        numP += P(i, j, nGridZ - 1);
                }
            }
            // Correct for overspawning
            while (numP > nPhages) {
                int i = RandI(nGridXY - 1);
                int j = RandI(nGridXY - 1);

                if (P(i, j, nGridZ - 1) > 0) {
                    P(i, j, nGridZ - 1)--;
                    numP--;
                }
            }
            // Correct for underspawning
            while (numP < nPhages) {
                int i = RandI(nGridXY - 1);
                int j = RandI(nGridXY - 1);

                P(i, j, nGridZ - 1)++;
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
double Colonies3D::ComputeEvents(double n, double p, int flag) {

    // Trivial cases
    if (p == 1) return n;
    if (p == 0) return 0.0;
    if (n < 1)  return 0.0;

    double N = RandP(n*p);

    return round(N);
}

// Computes how many particles has moved to neighbouing points
void Colonies3D::ComputeDiffusion(double n, double lambda, double* n_0, double* n_u, double* n_d, double* n_l, double* n_r, double* n_f, double* n_b, int flag) {

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

        for (int k = 0; k < round(n); k++) {

            double r = rand(rng);
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
        double N = RandP(3*lambda*n); // Factor of 3 comes from 3D

        *n_u = RandP(N/6);
        *n_d = RandP(N/6);
        *n_l = RandP(N/6);
        *n_r = RandP(N/6);
        *n_f = RandP(N/6);
        *n_b = RandP(N/6);
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

// Returns random double between 0 and n
double Colonies3D::Rand(double n) {

    // Set limit on distribution
    uniform_real_distribution <double> distr(0, n);

    return distr(rng);
}

// Returns random normal dist. number with mean m and variance s^2
double Colonies3D::RandN(double m, double s) {

    // Set limit on distribution
    normal_distribution <double> distr(m, s);

    return distr(rng);
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
void Colonies3D::ExportData(double t, std::string filename_suffix){

    // Verify the file stream is open
    string fileName = "PopulationSize_"+filename_suffix;
    OpenFileStream(f_N, fileName);

    // Writes the time, number of cells, number of infected cells, number of phages
    f_N << fixed    << setprecision(2);
    f_N << setw(6)  << t       << "\t";
    f_N << setw(12) << round(accu(B))    << "\t";
    f_N << setw(12) << round(accu(I0) + accu(I1) + accu(I2) + accu(I3) + accu(I4) + accu(I5) + accu(I6) + accu(I7) + accu(I8) + accu(I9))    << "\t";
    f_N << setw(12) << round(accu(P))    << "\t";

    uvec nz = find(B);
    f_N << setw(12) << static_cast<double>(nz.n_elem) / initialOccupancy << "\t";
    f_N << setw(12) << n_0 / 1e12 * pow(L, 2) * H - accu(nutrient) << "\t";
    f_N << setw(12) << accu(nC) << endl;

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

                    f_B << setw(6) << B(x,y,z) << "\t";
                    f_P << setw(6) << P(x,y,z) << "\t";
                    double nI = round(I0(x,y,z) + I1(x,y,z) + I2(x,y,z) + I3(x,y,z) + I4(x,y,z) + I5(x,y,z) + I6(x,y,z) + I7(x,y,z) + I8(x,y,z) + I9(x,y,z));
                    f_I << setw(6) << nI       << "\t";
                    f_n << setw(6) << nutrient(x,y,z) << "\t";
                }

                // Write last line ("\n" instead of tab)
                f_B << setw(6) << round(B(x,nGridXY - 1,z)) << "\n";
                f_P << setw(6) << round(P(x,nGridXY - 1,z)) << "\n";
                double nI = round(I0(x,nGridXY - 1,z) + I1(x,nGridXY - 1,z) + I2(x,nGridXY - 1,z) + I3(x,nGridXY - 1,z) + I4(x,nGridXY - 1,z) + I5(x,nGridXY - 1,z) + I6(x,nGridXY - 1,z) + I7(x,nGridXY - 1,z) + I8(x,nGridXY - 1,z) + I9(x,nGridXY - 1,z));
                f_I << setw(6) << nI                        << "\n";
                f_n << setw(6) << round(nutrient(x,nGridXY - 1,z)) << "\n";
            }
        }
    }

}

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
                accuB += arr_B[i][j][k];
                accuI += arr_I0[i][j][k] + arr_I1[i][j][k] + arr_I2[i][j][k] + arr_I3[i][j][k] + arr_I4[i][j][k] + arr_I5[i][j][k] + arr_I6[i][j][k] + arr_I7[i][j][k] + arr_I8[i][j][k] + arr_I9[i][j][k];
                accuP += arr_P[i][j][k];
                accuNutrient += arr_nutrient[i][j][k];
                accuClusters += arr_nC[i][j][k];
            }
        }
    }

    // Writes the time, number of cells, number of infected cells, number of phages
    f_N << fixed    << setprecision(2);
    f_N << setw(6)  << t       << "\t";
    f_N << setw(12) << round(accuB)    << "\t";
    f_N << setw(12) << round(accuI)    << "\t";
    f_N << setw(12) << round(accuP)    << "\t";

    uvec nz = find(B);
    f_N << setw(12) << static_cast<double>(nz.n_elem) / initialOccupancy << "\t";
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

                    f_B << setw(6) << arr_B[x][y][z] << "\t";
                    f_P << setw(6) << arr_P[x][y][z] << "\t";
                    double nI = round(arr_I0[x][y][z] + arr_I1[x][y][z] + arr_I2[x][y][z] + arr_I3[x][y][z] + arr_I4[x][y][z] + arr_I5[x][y][z] + arr_I6[x][y][z] + arr_I7[x][y][z] + arr_I8[x][y][z] + arr_I9[x][y][z]);
                    f_I << setw(6) << nI       << "\t";
                    f_n << setw(6) << arr_nutrient[x][y][z] << "\t";
                }

                // Write last line ("\n" instead of tab)
                f_B << setw(6) << round(arr_B[x][nGridXY - 1][z]) << "\n";
                f_P << setw(6) << round(arr_P[x][nGridXY - 1][z]) << "\n";
                double nI = round(arr_I0[x][nGridXY - 1][z] + arr_I1[x][nGridXY - 1][z] + arr_I2[x][nGridXY - 1][z] + arr_I3[x][nGridXY - 1][z] + arr_I4[x][nGridXY - 1][z] + arr_I5[x][nGridXY - 1][z] + arr_I6[x][nGridXY - 1][z] + arr_I7[x][nGridXY - 1][z] + arr_I8[x][nGridXY - 1][z] + arr_I9[x][nGridXY - 1][z]);
                f_I << setw(6) << nI                        << "\n";
                f_n << setw(6) << round(arr_nutrient[x][nGridXY - 1][z]) << "\n";
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
        struct stat info;
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
}
