#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <itensor/all.h>

using namespace std;
using namespace itensor;

// given spatial configurations, it returs the potentials

vector<double>
compute_potential(const vector< vector<double> > rj , const double alpha);


#endif
