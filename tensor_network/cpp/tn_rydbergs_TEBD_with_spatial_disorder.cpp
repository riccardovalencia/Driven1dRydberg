#include <itensor/all.h>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>
#include <ctime>
#include <math.h>       /* exp */
#include <fstream>	//output file
#include <sstream>	//for ostringstream
#include <iomanip>
#include "../../library_cpp/1d_rydberg_array.h"
#include <filesystem>

using namespace std;
using namespace itensor;
namespace fs = std::filesystem;

// Closed dynamics of a chain of N spin-1/2 with dipole-dipole interactions 1/r^\alpha with alpha=6, kept up to next-nearest neighbour.
// The Hamiltonian simulated is in Eq. 1 in  https://arxiv.org/abs/2309.12392
// The code allows to reproduce results in Fig. S1 in https://arxiv.org/abs/2309.12392

// The spins are along a chain of periodicity of two sites: they are at alternating distances r1 and r2 (possibly equal if desired).
// r1 = 1 in our units, so that V1=1. Instead V2 is given as an input, from which r2 is exctracted. 
// sigmax encodes a disorder (in units of r1) on the ideal position along the x direction.
// sigmay = sigmax and sigmaz = 5*sigmax . This choice is experimentally motivated and can be modified.
// Given a certain disorder realization of the atomic positions, the actual distances are computed, together with the actual
// interactions V(r) between atoms. 

// GOAL
// We aim to investigate the impact of finite temperature T in the ideal transport case. We want to check that 
// propagation is still mainly towards east (namely, we do not have undesired resonances).
// We include thermal fluctuations adding either a noise on the interactions or positions. The issue with sampling for the interactions
// is that they are not independent. There is a constraint (see https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.118.063606)

// Dynamics is performed via a 3-sites Time Evolving Block Decimation algorithm.
// Successfully tested against Exact Diagonalization: YES

int main(int argc , char* argv[]){
	

    string main_dir = "data/";
    // Hamiltonian parameters
    int N           = atoi(argv[1]);
    int M           = atoi(argv[2]) ;   // number of consecutives excitations
    double V2       = atof(argv[3]);
    double Omega    = atof(argv[4]);
    double T        = atof(argv[5]);
    double dt       = atof(argv[6]);         // time step
    int maxDim      = atoi(argv[7]);       // max bond dimension
    string name_state = argv[8];
    double sigmax     = atof(argv[9]);
    int index_seed    = atoi(argv[10]);

    double sigmay = sigmax;
    double sigmaz = 5*sigmax; // I am using Hannes' trap where the transversal direction has a trap 5 times weaker

    cerr << sigmax << endl;
    double cut_off = 1E-14; // cut_off TEBD
    int steps_measure = 10;
    int steps_save_state = int(1/dt);
    int total_steps = int(T / dt);

    double V1 = 1.;
    double Delta1 = -V1; // antiblockade (detuning even sites)
    double Delta2 = -V2; // antiblockade (detuning odd sites)
    
    // print variables
    cerr << N << "\n";
    cerr << M << "\n";
    cerr << V2 << "\n";
    cerr << Omega << "\n";
    cerr << T << "\n";
    cerr << dt << "\n";
    cerr << maxDim << "\n";

    
    // observables to measure
    vector<string> name_obs; 
    name_obs.push_back("fidelity");
    name_obs.push_back("nkink");
    name_obs.push_back("entropy");
    name_obs.push_back("MaxD");   
    name_obs.push_back("nj");
    name_obs.push_back("entropy_j");
    name_obs.push_back("correlation");


    // initial state (1: rydberg - 0: ground)
    
    vector<int> initial_state;

    if(name_state == "kink")
    {
        for(int j = 1 ; j<= N ; j++)
        {
            if(j <= M) initial_state.push_back(1);
            else initial_state.push_back(0);
        }
    }
    else if(name_state == "periodic")
    {
    for(int j = 1 ; j<= N ; j++)
        {   
            if((j-1) % M == 0) initial_state.push_back(1);
            else initial_state.push_back(0);
        }
    }
    else
    {
        cerr << "Name : " << name_state << " currently not implemented.\n";
        exit(0);
    }
    
    
    for(auto j : initial_state) cerr << j << " ";

    // initialization of state

    SiteSet sites = SpinHalf(N,{"ConserveQNs=",false});

    MPS psi = initial_computational_state(sites,initial_state);

    cerr << "Magnetization along x" << endl;
    vector<double> mj = measure_magnetization(&psi,sites,"x");
    for(double m : mj) cerr << " " << m ;
    cerr << "\nMagnetization along z" << endl;
    mj = measure_magnetization(&psi,sites,"z");
    for(double m : mj) cerr << " " << (1-m)/2. ;
    cerr << "\n\n\n";

    // noise due to finite temperature 

    default_random_engine generator;
    generator.seed(index_seed);
    normal_distribution<double> noise_x(0,sigmax);
    normal_distribution<double> noise_y(0,sigmay);
    normal_distribution<double> noise_z(0,sigmaz);

    vector<vector<double> > rj;
    for(int j = 1 ; j<= N ; j++) rj.push_back({0.,0.,0.});

    // ideal distances
    double d1 = pow(1/V1, 1./6);
    double d2 = pow(1/V2, 1./6);
    double d = 0.;

    // ideal positions
    for(int j = 0 ; j < N ; j++)
    {
        rj[j] = {d,0.,0.};
        if(j%2==0) d += d1;
        else d += d2;
    }

    // add noise


    for(vector<double> a : rj) cerr << a[0] << " " << a[1] << " " << a[1] << "\n";
    cerr << "Add noise\n";
    for(int j = 0 ; j < N ; j++)
    {
        double dr_x = noise_x(generator);
        double dr_y = noise_y(generator);
        double dr_z = noise_z(generator);

        rj[j][0] += dr_x;
        rj[j][1] += dr_y;
        rj[j][2] += dr_z;
        
    }

    for(vector<double> a : rj) cerr << a[0] << " " << a[1] << " " << a[1] << "\n";
    cerr << "\n";


    // building gates of Hamiltonian

    vector<double> Deltaj;
    vector<double> Omegaj;
    vector<double> Vj;

    for(int j : range1(N))
    {
        Omegaj.push_back(Omega);
        if(j % 2 == 0) Deltaj.push_back(Delta1);
        else Deltaj.push_back(Delta2);
    }


    // return actual potentials

    Vj = compute_potential(rj,6.);

    for(double v : Vj) cerr << v << " ";
    cerr << "\n\n";
    exit(0);

    // vector<MyBondGate> gate = gates_rydberg_up_to_VNNN(sites , Deltaj, Omegaj, Vj, dt);
    
    vector<MyBondGate> gate = gates_rydberg_up_to_VNNN(sites , Deltaj, Omegaj, Vj, dt);

    // ---------------------------------
    MPS psi_t0 = psi;
    vector<double> overlap_t;

    cerr << setprecision(10);

    string file_root = tinyformat::format("%sTN_N%d_M%d_V1_%.2f_V2_%.2f_Om_%.3f_D%d_state%s_sigmax%.5f_index%d",main_dir,N,M,V1,V2,Omega,maxDim,name_state,sigmax,index_seed);

    string file_obs    = tinyformat::format("%s.txt",file_root);
    ofstream save_file( file_obs) ;
    save_file << "# t";
    for(string name : name_obs) save_file << " . " + name;
    save_file << endl;

    file_obs    = tinyformat::format("%s_entanglement.txt",file_root);
    ofstream save_file_ent( file_obs) ;
    save_file_ent << "# t";
    for(int j : range1(1,N)) save_file_ent << " . " << j;
    save_file_ent << endl;

    file_obs    = tinyformat::format("%s_nj.txt",file_root);
    ofstream save_file_nj( file_obs) ;
    save_file_nj << "# t";
    for(int j : range1(1,N+1)) save_file_nj << " . " << j;
    save_file_nj << endl;

    file_obs    = tinyformat::format("%s_C1j.txt",file_root);
    ofstream save_file_Cij( file_obs) ;
    save_file_Cij << "# t";
    for(int j : range1(1,N+1)) save_file_Cij << " . " << j;
    save_file_Cij << endl;

    save_file     << setprecision(12);
    save_file_ent << setprecision(12);
    save_file_nj  << setprecision(12);
    save_file_Cij << setprecision(12);
    
    file_obs    = tinyformat::format("data/TN_time.txt");
    ofstream save_time( file_obs) ;


    writeToFile(tinyformat::format("%s_sites",file_root),sites); 

    for(int k=0 ; k<total_steps ; k++)
    {
        double t = (k+1)*dt;
        
        for (MyBondGate g : gate)
        {

            vector<int> jn = g.jn();

            int j = jn[0];
            ITensor AA = g.gate();
            psi.position(j);
            // psi.normalize();

            for(int q : jn) AA *= psi(q);  
            AA.mapPrime(1,0);

            if(jn.size() == 1)
            {
                psi.set(j,AA);
            }

            else if(jn.size() == 2)
            {
                auto [U,S,V] = svd(AA,inds(psi(j)),{"Cutoff=",cut_off,"MaxDim=",maxDim});
                psi.set(j,U);
                psi.set(j+1,S*V);
            }

            else if(jn.size() == 3)
            {
                auto [U,S,V] = svd(AA,inds(psi(j)),{"Cutoff=",cut_off,"MaxDim=",maxDim});
                Index l =  commonIndex(U,S);
                psi.set(j,U);

                auto [U1,S1,V1] = svd(S*V,{inds(psi(j+1)),l},{"Cutoff=",cut_off,"MaxDim=",maxDim});
                psi.set(j+1,U1);
                psi.set(j+2,S1*V1);
            }

            else
            {
            cerr << jn.size() <<"-TEBD not implemented yet!\n";
            exit(0);
            }
            
        }

        psi.position(1);
        psi.normalize();
        
        if (k % steps_measure == 0)
        {
            save_file << t ;
            save_time << t << "\n";

            for(string name : name_obs)
            {
                if(name=="fidelity")
                {
                    double fidelity = abs(innerC(psi_t0,psi));
                    fidelity *= fidelity;
                    save_file << " " << fidelity;
                }
                if(name=="Sx")
                {
                    mj = measure_magnetization(&psi,sites,"x");
                    double Sx = 0.;
                    for(long unsigned int idx=1; idx < mj.size() ; idx++) Sx += mj[idx];
                    save_file << " " << Sx/N ;
                }

                if(name=="Sz")
                {
                    mj = measure_magnetization(&psi,sites,"z");
                    double Sz = 0.;
                    for(long unsigned int idx=1; idx < mj.size() ; idx++) Sz += mj[idx];
                    save_file << " " << Sz/N ;
                }


                if(name=="nj")
                {
                    mj = measure_magnetization(&psi,sites,"z");
                    save_file_nj << t;
                    for(double m : mj) save_file_nj << " " << (1-m)/2.;
                    save_file_nj << "\n";
                }
                if(name=="Na")
                {
                    mj = measure_magnetization(&psi,sites,"x");
                    save_file << " " << mj[0];
                }
                if(name=="entropy")
                {
		            double EE = entanglement_entropy( &psi , int(N/2));		
		            save_file << " " << EE;
	            }

                if(name=="entropy_j")
                {
                    save_file_ent << t;
                    for(int k : range(1,N))
                    {
                        save_file_ent << " "  << entanglement_entropy( &psi ,k);		
                    }
                    save_file_ent << "\n";
	            }
                
                if(name=="nkink")
                {
                    double nkink = measure_kink(&psi,sites);
                    save_file << " " << nkink;
                }

                if(name == "correlation")
                {
                    vector<double> Cij = measure_correlations(&psi, sites, 1, true);
                    save_file_Cij << t;

                    for(double c : Cij)
                    {
                        save_file_Cij << " "  << c;		
                    }
                    save_file_Cij << "\n";
                }

                if(name=="MaxD")
                {
                    save_file << " " << maxLinkDim(psi);
                }
            }
            save_file << "\n";
            save_file.flush();
            save_file_ent.flush();
            save_file_nj.flush();
            save_time.flush();
            save_file_Cij.flush();
            // save_file << endl;
            // double overlap = abs(innerC(psi_t0,psi));
            // save_file << t  overlap.real() << endl; 
            // save_imag_F << overlap.imag() << endl;
            // save_time   << t << endl;
            cerr << t << " " << maxLinkDim(psi) << endl;
        }
        
    
        // if ( (k+1) % steps_save_state == 0)
        // {
        //     writeToFile(tinyformat::format("%s_psi_t%.3f",file_root,t),psi); 
        // }
    }       

    save_time.close();
    save_file_ent.close();
    save_file_nj.close();
    save_file_Cij.close();
    save_file.close();
    return 0;

}
