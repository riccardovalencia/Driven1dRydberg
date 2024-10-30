#include "initial_state.h"
#include <itensor/all.h>
#include <math.h>       
#include <iostream>
#include <vector>


// ----------------------------------------------------------
// Given a mixed spin_boson SiteSet, initialize product state |n_photon > \otimes |\theta,\phi>^(N-1)
// where |n_photon> is a Fock state
//       |theta,\phi> is a spin-coherent state of a spin-1/2 pointing on the Bloch sphere 
// 
// TO DO : do a function ITensor spin_coherent_state(IndexSet,theta,phi) which returns a spin coherent state
//       : do a function ITensor fock_state(IndexSet,n) which return |n>

MPS
initialize_spin_boson_state(const SiteSet sites , const int n_photon , double theta, double phi)
{

    MPS psi = randomMPS(sites);
    int N   = length(sites);

    Index sj ,rj, lj;

    // first site
	sj = sites(1);
	rj = commonIndex(psi(1),psi(2));
	ITensor wf = ITensor(sj,rj);
	
    if(hasTags(sj,"Site,Boson"))
    {
        for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1), 0);
        wf.set(sj(n_photon+1),rj(1),1);
        for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1), 0);
    }
				
    else if(hasTags(sj,"Site,S=1/2"))
    {
        cerr << "Inserting spin coherent state" << endl;
        wf.set(sj(1),rj(1),cos(theta/2.));
        // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
        wf.set(sj(2),rj(1),sin(theta/2.));
    }
    
    else{
        cerr << "SiteSet not recognize : " << sj << endl;
        cerr << "Return a random initial state" << endl;
        return psi; 
    }

	psi.set(1,wf);
	
    cerr << "Inserted spin coherent state" << endl;

    for(int j=2 ; j < N ; j++)
    {
        sj = sites(j);
        lj = commonIndex(psi(j-1),psi(j));
		rj = commonIndex(psi(j),psi(j+1));
		wf = ITensor(sj,lj,rj);

        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1),lj(1), 0);
            wf.set(sj(n_photon+1),rj(1),lj(1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1),lj(1), 0);
        }
                    
        else if(hasTags(sj,"Site,S=1/2"))
        {
            wf.set(sj(1),lj(1),rj(1),cos(theta/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),lj(1),rj(1),sin(theta/2.));
        }

		psi.set(j,wf); 
    }

    sj = sites(N);
    lj = commonIndex(psi(N-1),psi(N));
    wf = ITensor(sj,lj);

    wf.set(sj(1),lj(1),cos(theta/2.));
    // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
    wf.set(sj(2),lj(1),sin(theta/2.));


    if(hasTags(sj,"Site,Boson"))
    {
        for( int d=1; d <= n_photon; d++) wf.set(sj(d),lj(1), 0);
        wf.set(sj(n_photon+1),lj(1),1);
        for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),lj(1), 0);
    }
                    
    else if(hasTags(sj,"Site,S=1/2"))
    {
        wf.set(sj(1),lj(1),cos(theta/2.));
        // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
        wf.set(sj(2),lj(1),sin(theta/2.));
    }


    psi.set(N,wf); 

    return psi;
}


// ----------------------------------------------------------
// Override previous function - it can accept vectors as theta and phi
// Given a mixed spin_boson SiteSet, initialize product state |n_photon > \otimes |\theta,\phi>^(N-1)
// where |n_photon> is a Fock state
//       |theta,\phi> is a spin-coherent state of a spin-1/2 pointing on the Bloch sphere 
// 
// TO DO : do a function ITensor spin_coherent_state(IndexSet,theta,phi) which returns a spin coherent state
//       : do a function ITensor fock_state(IndexSet,n) which return |n>
MPS
initialize_spin_boson_state(const SiteSet sites , const int n_photon , const vector<double> theta, const vector<double> phi)
{

    MPS psi = randomMPS(sites);
    int N   = length(sites);

    if(N==1)
    {
        Index sj = sites(1);
        ITensor wf = ITensor(sj);
        
        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d), 0);
            wf.set(sj(n_photon+1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d), 0);
        }
                    
        else if(hasTags(sj,"Site,S=1/2"))
        {
            cerr << "Inserting spin coherent state" << endl;
            wf.set(sj(1),cos(theta[0]/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),sin(theta[0]/2.));
        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
    }

    if(N>1)
    {

        Index sj ,rj, lj;

        // first site
        sj = sites(1);
        rj = commonIndex(psi(1),psi(2));
        ITensor wf = ITensor(sj,rj);
        
        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1), 0);
            wf.set(sj(n_photon+1),rj(1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1), 0);
        }
                    
        else if(hasTags(sj,"Site,S=1/2"))
        {
            cerr << "Inserting spin coherent state" << endl;
            wf.set(sj(1),rj(1),cos(theta[0]/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),rj(1),sin(theta[0]/2.));
        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
        
        cerr << "Inserted spin coherent state" << endl;

        for(int j=2 ; j < N ; j++)
        {
            sj = sites(j);
            lj = commonIndex(psi(j-1),psi(j));
            rj = commonIndex(psi(j),psi(j+1));
            wf = ITensor(sj,lj,rj);

            if(hasTags(sj,"Site,Boson"))
            {
                for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1),lj(1), 0);
                wf.set(sj(n_photon+1),rj(1),lj(1),1);
                for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1),lj(1), 0);
            }
                        
            else if(hasTags(sj,"Site,S=1/2"))
            {
                wf.set(sj(1),lj(1),rj(1),cos(theta[j-1]/2.));
                // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
                wf.set(sj(2),lj(1),rj(1),sin(theta[j-1]/2.));
            }

            psi.set(j,wf); 
        }

        sj = sites(N);
        lj = commonIndex(psi(N-1),psi(N));
        wf = ITensor(sj,lj);

        // wf.set(sj(1),lj(1),cos(theta/2.));
        // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
        // wf.set(sj(2),lj(1),sin(theta/2.));


        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d),lj(1), 0);
            wf.set(sj(n_photon+1),lj(1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),lj(1), 0);
        }
                        
        else if(hasTags(sj,"Site,S=1/2"))
        {
            wf.set(sj(1),lj(1),cos(theta[N-1]/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),lj(1),sin(theta[N-1]/2.));
        }


        psi.set(N,wf); 
    }

    return psi;
}



// ----------------------------------------------------------
// Initialize product states in the computation basis
// |psi> = |0/1> |0/1> ...
// where |0> = |up_z> and |1> = |down_z>
MPS
initial_computational_state(const SiteSet sites , const vector<int> initial_state)
{

    MPS psi = randomMPS(sites);
    int N   = length(sites);

    if(N != initial_state.size())
    {
        cerr << "Lenght of initial state is different from N!\nReturning random MPS.\n";
        return psi;
    }

    // for(int q : initial_state) cerr << q << " ";
    // exit(0);


    if(N==1)
    {
        Index sj = sites(1);
        ITensor wf = ITensor(sj);
        
        
        
        if(hasTags(sj,"Site,S=1/2"))
        {
            cerr << "Inserting spin coherent state" << endl;
            if (initial_state[0]==0)
            {
                wf.set(sj(1),1);
                wf.set(sj(2),0);
            }
            else
            {
                wf.set(sj(1),0);
                wf.set(sj(2),1);
            }
            
        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
    }

    if(N>1)
    {

        Index sj ,rj, lj;

        // first site
        sj = sites(1);
        rj = commonIndex(psi(1),psi(2));
        ITensor wf = ITensor(sj,rj);
        
        if(hasTags(sj,"Site,S=1/2"))
        {
            if (initial_state[0]==0)
            {
                wf.set(sj(1),rj(1),1);
                wf.set(sj(2),rj(1),0);
            }
            else
            {
                wf.set(sj(1),rj(1),0);
                wf.set(sj(2),rj(1),1);
            }

        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
        
        for(int j=2 ; j < N ; j++)
        {
            sj = sites(j);
            lj = commonIndex(psi(j-1),psi(j));
            rj = commonIndex(psi(j),psi(j+1));
            wf = ITensor(sj,lj,rj);

            if(hasTags(sj,"Site,S=1/2"))
            {
                if (initial_state[j-1]==0)
                {
                    cerr << "Inserted spin down\n";
                    wf.set(sj(1),lj(1),rj(1),1);
                    wf.set(sj(2),lj(1),rj(1),0);
                }
                else
                {
                    cerr << "Inserted spin up\n";
                    wf.set(sj(1),lj(1),rj(1),0);
                    wf.set(sj(2),lj(1),rj(1),1);
                }

            }
        
            else
            {
                cerr << "SiteSet not recognize : " << sj << endl;
                cerr << "Return a random initial state" << endl;
                return psi; 
            }
                
            psi.set(j,wf); 
        }

        sj = sites(N);
        lj = commonIndex(psi(N-1),psi(N));
        wf = ITensor(sj,lj);

                        
        if(hasTags(sj,"Site,S=1/2"))
        {
            if (initial_state[N-1]==0)
            {
                wf.set(sj(1),lj(1),1);
                wf.set(sj(2),lj(1),0);
            }
            else
            {
                wf.set(sj(1),lj(1),0);
                wf.set(sj(2),lj(1),1);
            }
        }

        psi.set(N,wf); 
    }

    return psi;
}


