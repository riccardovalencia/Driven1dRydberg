#ifndef OBSERVABLES_H
#define OBSERVABLES_H

#include <itensor/all.h>

using namespace std;
using namespace itensor;

 
vector<double>
measure_magnetization(MPS* psi, const SiteSet sites , string direction);

double
entanglement_entropy( MPS* , int  );

double 
measure_kink( MPS* psi, const SiteSet sites);

vector<double>
measure_correlations(MPS* psi, const SiteSet sites, const int start, const bool connected);


#endif
