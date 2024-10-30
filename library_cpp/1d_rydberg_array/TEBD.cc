#include "TEBD.h"
#include "MyClasses.h"
#include <itensor/all.h>
#include <iostream>
#include <math.h>       
#include <complex>
using namespace std;
using namespace itensor;


// -----------------------------------------------------------------
// Gates of the PXP Hamiltonian
// H = omega \sum_j P_j X_{j-1} P_{j+1}
// where P_j = (1+Z_j)/2

vector<MyBondGate>
gates_pxp(const SiteSet sites , const double omega, const double dt)
{

	int N = length(sites);

	
	vector<MyBondGate> gates;

	// first layer (acts on sites [1,2,3] , [4,5,6] , ... )
	for(int j=1 ; j <= N-2 ; j+=3)
	{
		ITensor P1 = (op(sites,"Id",j) + 2*op(sites,"Sz",j))/2;
		ITensor X2 = 2*op(sites,"Sx",j+1);
		ITensor P3 = (op(sites,"Id",j+2) + 2*op(sites,"Sz",j+2))/2;
		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,omega*P1*X2*P3);
		gates.push_back(g);
	}

	// second layer (acts on sites [2,3,4] , [5,6,7] , ... )
	for(int j=2 ; j <= N-2 ; j+=3)
	{
		ITensor P1 = (op(sites,"Id",j) + 2*op(sites,"Sz",j))/2;
		ITensor X2 = 2*op(sites,"Sx",j+1);
		ITensor P3 = (op(sites,"Id",j+2) + 2*op(sites,"Sz",j+2))/2;
		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,omega*P1*X2*P3);
		gates.push_back(g);
	}

	// third layer (acts on sites [3,4,5] , [6,7,8] , ... )
	for(int j=3 ; j <= N-2 ; j+=3)
	{
		ITensor P1 = (op(sites,"Id",j) + 2*op(sites,"Sz",j))/2;
		ITensor X2 = 2*op(sites,"Sx",j+1);
		ITensor P3 = (op(sites,"Id",j+2) + 2*op(sites,"Sz",j+2))/2;
		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,omega*P1*X2*P3);
		gates.push_back(g);
	}


	// if(open_system == true) cerr << "TO DO : Include non-hermitian part" << endl;

	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}

// -----------------------------------------------------------------
// Rydberg Hamiltonian - we keep up to nearest neighbor interactions
// H = \sum_j Delta_j Z_j + \sum_j Omega_j X_j + \sum_j Vj Nj N{j+1}
vector<MyBondGate>
gates_rydberg_up_to_VNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt)
{

	int N = length(sites);

	vector<MyBondGate> gates;


	for(int j=1 ; j <= N-1 ; j+=1)
	{
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;
		for(int q=j ; q<=j+1; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q) );
		}

		// for(ITensor A : Nj) PrintData(A);

		
		double V = Vj[j-1];
		double Omega1 = Omegaj[j-1];
		double Omega2 = Omegaj[j];
		double Delta1 = Deltaj[j-1];
		double Delta2 = Deltaj[j];

		if(j<N-1)
		{
			Omega2 /= 2.;
		 	Delta2 /= 2.;
		}

		if(j>1)
		{
			Omega1 /= 2.;
			Delta1 /= 2.;
		}



		ITensor H_om, H_N, H_NN; 
		H_om = Omega1 * Xj[0] * Ij[1] + Omega2 * Ij[0] * Xj[1];
		H_N  = Delta1 * Nj[0] * Ij[1] + Delta2 * Ij[0] * Nj[1];
		H_NN = V * Nj[0] * Nj[1];

		ITensor H = H_NN + H_N + H_om;
	


		vector<int> jn = {j,j+1};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}
	

	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}


// -----------------------------------------------------------------
// Rydberg Hamiltonian - we keep up to next-nearest neighbor interactions
// 1. We split H = H_1 + H_2 + H_3, so that [H_i,H_j] \neq 0 while the elements within each H_i commute.
// 2. We prepare the gates for H_j, and then we put them inside a time-evolving operator U_j (of time step dt/2) via SVDs. Namely: we construct the gates and then the resulting MPO
// 3. Either we return the vector [U_1,U_2,U_3,U_3,U_2,U_1]. Or we multiply the MPOs in order to have a single one.

// PLUS: it does not split the single-site terms separately. You earn ~30% in computation time
// H = \sum_j Delta_j Z_j + \sum_j Omega_j X_j + \sum_j Vj Nj N{j+1} + \sum_j Vj Nj N_{j+2}
// Note: the next-nearest neighour interaction is exctracted by the nearest neighbour ones (which could be not homogenous in space)

vector<MyBondGate>
gates_rydberg_up_to_VNNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt)
{

	int N = length(sites);

	vector<MyBondGate> gates;
	vector<double> omega;
	vector<double> delta;

	// first layer (acts on sites [1,2,3] , [4,5,6] , ... )
	for(int j=1 ; j <= N-2 ; j+=3)
	{
		int js = j;
		int jf = js + 2;
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;

		if(js==1)
			{
			omega = {Omegaj[j-1] , Omegaj[j]/2. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1] , Deltaj[j]/2. , Deltaj[j+1]/3.};
		}
		else if(js==2)
		{
			omega = {Omegaj[j-1]/2. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/2. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}
		else if(jf==N-1)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/2.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/2.};
		}
		else if(jf==N)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/2. , Omegaj[j+1]};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/2. , Deltaj[j+1]};
		}
		else
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}

		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q)) ; 
		}

		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);

		if(j<N-2) V23 /= 2.;		
		if(j>1)   V12 /= 2.;

		ITensor H1, H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;


		H1  =  omega[0] * Xj[0] * Ij[1] * Ij[2];
		H1  += omega[1] * Ij[0] * Xj[1] * Ij[2];
		H1  += omega[2] * Ij[0] * Ij[1] * Xj[2];

		H1  += delta[0] * Nj[0] * Ij[1] * Ij[2];
		H1  += delta[1] * Ij[0] * Nj[1] * Ij[2];
		H1  += delta[2] * Ij[0] * Ij[1] * Nj[2];


		ITensor H = H_NN + H1;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}

	// second layer (acts on sites [2,3,4] , [5,6,7] , ... )
	for(int j=2 ; j <= N-2 ; j+=3)
	{
		int js = j;
		int jf = js + 2;
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;

		if(js==1)
			{
			omega = {Omegaj[j-1] , Omegaj[j]/2. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1] , Deltaj[j]/2. , Deltaj[j+1]/3.};
		}
		else if(js==2)
		{
			omega = {Omegaj[j-1]/2. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/2. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}
		else if(jf==N-1)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/2.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/2.};
		}
		else if(jf==N)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/2. , Omegaj[j+1]};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/2. , Deltaj[j+1]};
		}
		else
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}

		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q)) ; 

		}

		
		
		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);

		if(j < N-2) V23 /= 2.;
		if(j > 1)   V12 /= 2.;

		ITensor H1, H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;


		H1  =  omega[0] * Xj[0] * Ij[1] * Ij[2];
		H1  += omega[1] * Ij[0] * Xj[1] * Ij[2];
		H1  += omega[2] * Ij[0] * Ij[1] * Xj[2];

		H1  += delta[0] * Nj[0] * Ij[1] * Ij[2];
		H1  += delta[1] * Ij[0] * Nj[1] * Ij[2];
		H1  += delta[2] * Ij[0] * Ij[1] * Nj[2];

		ITensor H = H_NN + H1;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}

	// third layer (acts on sites [3,4,5] , [6,7,8] , ... )
	for(int j=3 ; j <= N-2 ; j+=3)
	{
		int js = j;
		int jf = js + 2;
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;

		if(js==1)
			{
			omega = {Omegaj[j-1] , Omegaj[j]/2. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1] , Deltaj[j]/2. , Deltaj[j+1]/3.};
		}

		else if(js==2)
		{
			omega = {Omegaj[j-1]/2. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/2. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}
		else if(jf==N-1)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/2.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/2.};
		}
		else if(jf==N)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/2. , Omegaj[j+1]};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/2. , Deltaj[j+1]};
		}
		else
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}


		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q)) ; 
		}
		
		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);

		if(j<N-2) V23 /= 2.;
		if(j>1)   V12 /= 2.;

		ITensor H1, H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;


		H1  =  omega[0] * Xj[0] * Ij[1] * Ij[2];
		H1  += omega[1] * Ij[0] * Xj[1] * Ij[2];
		H1  += omega[2] * Ij[0] * Ij[1] * Xj[2];

		H1  += delta[0] * Nj[0] * Ij[1] * Ij[2];
		H1  += delta[1] * Ij[0] * Nj[1] * Ij[2];
		H1  += delta[2] * Ij[0] * Ij[1] * Nj[2];

		ITensor H = H_NN + H1;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}


	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}

