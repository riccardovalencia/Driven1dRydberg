#include "parameters.h"
#include <itensor/all.h>
#include <math.h>       
#include <iostream>

// given spatial configurations, it returs the potentials 1/rj^alpha
 
vector<double>
compute_potential(const vector< vector<double> > rj , const double alpha)
{

    vector<double> V;
    int N = rj.size();

    for(int j = 0 ; j < N - 1 ; j++)
    {
        double drx = rj[j][0] - rj[j+1][0];
        double dry = rj[j][1] - rj[j+1][1];
        double drz = rj[j][2] - rj[j+1][2];

        double d = sqrt(drx*drx + dry*dry + drz*drz );

        V.push_back(pow(1/d,alpha) );        

    }

    return V ; 
}

