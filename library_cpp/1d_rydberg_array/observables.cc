#include "observables.h"
#include <itensor/all.h>
#include <math.h>       
#include <iostream>

// ----------------------------------------------------------
// Given a mized spin-boson or spin-1/2 system it measures:
// - occupation number for bosons
// - magnetization along direction (x,y,z) for spin-1/2 
 
vector<double>
measure_magnetization(MPS* psi, const SiteSet sites , string direction)
{

    int N = length(sites);
    vector<double> mj;

    for(int j=1 ; j<=N ; j++)
    {
        Index sj = sites(j);
        Index sjp = prime(sites(j));

        ITensor S_j  = ITensor(sj ,sjp );
		
	
		if(hasTags(sj,"Site,Boson"))
		{
			for(int d=1; d <= dim(sj) ; d++) S_j.set(sj(d),sjp(d),d-1.);
        }
				
		if(hasTags(sj,"Site,S=1/2"))
		{
            if (direction == "x")
            {
                S_j.set(sj(1),sjp(2),1.);
			    S_j.set(sj(2),sjp(1),1.);	
                // S_j = 2*op(sites,"Sx",j);
                
            }
            else if(direction == "y")
            {
                S_j.set(sj(1),sjp(2),-1*Cplx_i);
			    S_j.set(sj(2),sjp(1), 1*Cplx_i);
                // S_j = 2*op(sites,"Sy",j);

            }
            else if(direction == "z")
            {
                S_j.set(sj(1),sjp(1),1.);
                S_j.set(sj(2),sjp(2),-1.);
                // S_j = 2*op(sites,"Sz",j);
            }
            else
            {
                cerr << "Direction choses is neither 'x' , 'y' or 'z'" << endl;
                return mj;
            }
		}
        (*psi).position(j);
        ITensor ket = (*psi)(j);
		ITensor bra = dag(prime((*psi)(j),"Site"));
		
		complex<double> exp_Sj = eltC(bra * S_j * ket);

		mj.push_back(real(exp_Sj));
        
    }

    return mj;
}


// ----------------------------------------------------------
// Compute entanglement entropy along the bond [site,site+1]

double
entanglement_entropy( MPS* psi , int site)
	{
	(*psi).position(site); 
	ITensor wf = (*psi)(site) * (*psi)(site+1);
	ITensor U  = (*psi)(site);
	ITensor S,V;
	auto spectrum = svd(wf,U,S,V);
	
	double SvN = 0.;
	for(auto p : spectrum.eigs())
		{
		if(p > 1E-12) SvN += -p*log2(p);
		}
	return SvN;
	}


// ----------------------------------------------------------
// Compute number of kinks (|\up_z \dw_z>) on a state psi

double 
measure_kink( MPS* psi, const SiteSet sites)
{
    int N = length(sites);

    double kink = 0.;

    if(N==1)
    {
        cerr << "Cannot measure number of kinks in a single-site system.\n";
        cerr << "Returnin 0.\n";
        return kink;
    }

    for(int j=1 ; j<N ; j++)
    {
        (*psi).position(j);
        
        ITensor N_1 = (op(sites,"Id",j)   - 2*op(sites,"Sz",j))    /2.;
        ITensor N_2 = (op(sites,"Id",j+1) + 2*op(sites,"Sz",j+1))  /2.;
        
        ITensor ket = (*psi)(j)*(*psi)(j+1);
		ITensor bra = dag(prime((*psi)(j),"Site"))*dag(prime((*psi)(j+1),"Site"));
		
        complex<double> n_j = eltC(bra * N_1 * N_2 * ket);
        kink += n_j.real();
    }

    return kink;
}


// ----------------------------------------------------------
// Compute correlation functions <N_start N_(start+i)> (both connected and disconnected).
// Where N = (1-2*S^z)/2 = |down_z> <down_z|

vector<double>
measure_correlations(MPS* psi, const SiteSet sites, const int start, const bool connected)
{
    int N = length(sites);
    vector<double> C;

    // reference site
    ITensor Ns = (op(sites,"Id",start) - 2*op(sites,"Sz",start))  /2.;

    vector<double> nj;
    if(connected)
    {
        vector<double> mz = measure_magnetization( psi,sites,"z");
        for(double m : mz) nj.push_back((1-m)/2.);
    }

    for(int j=1 ; j<= N; j++)
    {
        double Cjs;
        ITensor M;
        
        ITensor N1, N2;
        int jmax = max(j,start);
        int jmin = min(j,start);

        if(jmin == j)
        {
            N1 = (op(sites,"Id",j) - 2*op(sites,"Sz",j))  /2.;
            N2 = Ns;
        }
        else
        {
            N1 = Ns;
            N2 = (op(sites,"Id",j) - 2*op(sites,"Sz",j))  /2.;
        }

        (*psi).position(jmin);
        ITensor ket = (*psi)(jmin);


        if(j==start)
        {
		    ITensor bra = dag(prime((*psi)(j),"Site"));
            Cjs = eltC(ket*N1*bra).real(); //nb N^2 = N (it is a projector)
        }


        else
        {    
            Index ir = commonIndex( (*psi)(jmin) , (*psi)(jmin + 1) ,"Link");
			M = ket * N1 * dag( prime( prime( ket , "Site") , ir ) );

            for(int q = jmin + 1 ; q < jmax ; q++)
            {
                M *= (*psi)(q);
                M *= dag(prime( (*psi)(q) , "Link"));
            }

            Index il = commonIndex( (*psi)( jmax-1 ), (*psi)(jmax), "Link");
            M *= (*psi)(jmax);
            M *= N2;
            M *= dag( prime( prime((*psi)( jmax ), il) , "Site") );
            Cjs = eltC(M).real();
        }
			
		
        if(connected)
        {
            Cjs = Cjs - nj[jmin-1] * nj[jmax-1];
        }

        C.push_back(Cjs);

    }

    return C;
}
