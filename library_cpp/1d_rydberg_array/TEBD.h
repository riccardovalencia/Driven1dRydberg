#ifndef TEBD_H
#define TEBD_H

#include <itensor/all.h>
#include "MyClasses.h"
using namespace std;
using namespace itensor;



vector<MyBondGate>
gates_pxp(const SiteSet sites , const double omega, const double dt);

vector<MyBondGate>
gates_rydberg_up_to_VNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt);

vector<MyBondGate>
gates_rydberg_up_to_VNNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt);

vector<MyBondGate>
gates_rydberg_up_to_VNNN_deprecated(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt);


#endif
