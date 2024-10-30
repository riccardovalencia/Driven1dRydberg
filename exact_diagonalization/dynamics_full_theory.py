import sys
sys.path.append("../library_python")
import quimb as qu
import quimb.tensor as qtn
import numpy as np
from quimb import *
import os
from pathlib import Path
import Hamiltonian, initial_state, parameters, ED_observables, input, time_evolution
import json
from sympy.utilities.iterables import multiset_permutations

# Dynamics of a 1d Rydberg array subjected to two set of drive fields having different frequencies
# Dynamics is performed in the rotating frame with respect to the fequency of one of the two drive fields

# Computationally, we solve the dynamics under the time-dependent Hamiltonian
# H = \sum_j (Hj + Hj(t)) where
# Hj = Omega1/2 X_{j} + Delta_j N_j + V_{j,j+1} N_j N_{j+1} + V_{j,j+2} N_j N_{j+2}
# Hj(t) = Omega2/2 (exp^{-i alpha t} S^+ + h.c.)
# where alpha is the detuning between the laser with Rabi frequency Omega1
# If Omega2 = 0, meaning Hj(t)=0, dynamics is performed using standard methods for time-independent Hamiltonians.

# folder where to save data
save_folder = "./"
# sparse matrices (True) or not (False)
sparse = True

#------------------------------------------------------
# useful operators
Id = qu(np.eye(2),qtype='dop')
X = qu(np.array([[0,1.],[1.,0.]])/2.,qtype='dop')
N = qu(np.diag([0,1]),qtype='dop')

# --------------- INPUT ---------------
args = input.input_rydbergs()
# system size
L   = args['L']      
# ideal interaction on odd bonds (it is the energy scale, i.e. V1 = 1)
V1  = args['V1']
# ideal interaction on even bonds 
V2  = args['V2']
# Rabi frequency set of drive fields acting on odd sites
Omega1   = args['Omega']
# Rabi frequency set of drive fields acting on even sites
Omega2   = args['Omega2']
# Detuning of drive fields from perfect antiblockade condition
epsilon = args['eps']
# std. deviation of distribution of positions from equilibrium 
sigma = np.array([float(y) for y in args['sigma']])   
# controls the seed for generating white Gaussian noise
index_seed = args['index_seed']
# total time of evolution
T       = args['T']
# number of measurements
n_steps = args['number_steps']
# power-law interaction (needed for exctracting actual positions of atoms due to thermal fluctuations)
alpha = 6.
# initial state - density profile (1 = down_z ; 0 = up_z)
state = [0,0,1,1,0,0,0,0,0,0]

# --------------- END INPUT ---------------
Delta1 = -V1 + epsilon # detuning even site 
Delta2 = -V2 + epsilon # detuning odd site 

ts  = np.linspace(0, T, n_steps)


L = len(state)
dims = [2] * L
#------------------------------------
# array containing positions of the atoms. We set the atoms along the x-axis
rj = np.zeros(shape=(L,3))

# Compute ideal atomic configuration
d1 = parameters.distance(V1,alpha)
d2 = parameters.distance(V2,alpha)
d = 0
for k in range(L):
    rj[k] = np.array([d,0,0])
    if k%2==0:
        d += d1
    else:
        d += d2

# impact of thermal fluctuations
np.random.seed(index_seed)
for k in range(3):
    rj[:,k] += np.random.normal(loc=0.0,scale=sigma[k],size=L)

#------------------------------------
# parameters and building Hamiltonian
Delta = []
for j in range(L):
    if j % 2 == 0:
        Delta += [Delta2]
    else:
        Delta += [Delta1]

V = parameters.V_power_law(rj,alpha)
Omega_1 = [Omega1 for _ in range(L)]
Omega_2 = [Omega2 for _ in range(L)]

# Observables

# density profile 
n_t = []
# correlation function
C_t = []
# entanglement entropy profile
ent_t = []
# fidelity
F_t = []
# equal time correlation function
Cij_t = []
# unequal time autocorrelation function
C_auto_t = []
# number of kinks (|10> strings)
n_kink_t = []
# imbalance on the east of first excitation
Imbalance_east_t = []
# imbalance on the west of first excitation
Imbalance_wast_t = []
# occupation initially occupied
n_initially_occupied_t = []

psi , [x_exc,n_exc] = initial_state.computational_spin_state(state,sparse)

if Omega2 == 0:
    print("Since Omega2 = 0 -> dynamics performed via exact diagonalization of a time-independent Hamiltonian")
    prefix = "ED"
    H ,  _ , _ = Hamiltonian.H_rydbergs_single_drive(L,V,Delta,Omega_1,sparse,VNNN=True,VNNNN=False)
    if L > 12:
        evo = Evolution(psi, H, method='integrate', progbar=False)
    else:
        evo = Evolution(psi, H, method='solve', progbar=False)

else:
    print("Since Omega2 /= 0 -> dynamics performed by solving time-dependent problem")
    prefix = "ED_timedep"
    H = time_evolution.MyTimeDepRyd(L,V,Delta,Omega_1,Omega_2,sparse)
    evo = Evolution(psi, H, method='integrate', progbar=False)

for idx, psi_t in enumerate(evo.at_times(ts)):

    Im_east = 0
    Im_west = 0

    n = []
    C_auto = []
    for j in range(L):
        n += [np.real(ED_observables.local_obs(psi_t,N,j,dims,sparse))]
        if j < x_exc[0]:
            Im_west += n[-1]
        if j > x_exc[0]:
            Im_east += n[-1]
        if state[j] == 1:
            C_auto += [n[-1]]
        
    n_kink = 0
    for j in range(L-1):
        O = pkron(N & (Id-N), dims=dims,inds=[j,j+1],sparse=sparse)
        n_kink += np.real(expec(O,psi_t))

    Cij = []
    for j in range(L):
        Cij += [np.abs(ED_observables.correlations(psi_t,N,j,0,dims,sparse) )]

    ent = []
    for j in range(1,L):
        ent += [entropy_subsys(psi_t, dims=[2] * L, sysa=range(j))]

    F_t += [fidelity(psi_t,psi)**2]
    n_t   += [n]
    Cij_t += [Cij]
    ent_t += [ent]
    Imbalance_east_t += [Im_east]
    Imbalance_wast_t += [Im_west]
    n_kink_t += [n_kink]
    C_auto_t += [sum(C_auto[1:])/(n_exc-1)]
    n_initially_occupied_t += [C_auto]


# perform time average of autocorrelation function
C_auto_av_t = []
dt = ts[1:] - ts[:-1]
for j in range(1,len(C_auto_t)):  
    C_auto_av_t     += [2*np.sum(C_auto_t[:j]*dt[:j])/np.sum(dt[:j])-1]
C_auto_av_t = np.array(C_auto_av_t)

# save data
string_state = ''
for q in state:
    if q == 1:
        string_state += '1'
    else:
        string_state += '0' 

folder = f'{save_folder}{string_state}'

if not os.path.isdir(folder):
    Path(folder).mkdir() 

folder = f'{folder}/{prefix}_rydberg_L{L}_V1_{V1:.1f}_V2_{V2:.1f}_Om1_{Omega1:.2f}_Om2_{Omega2:.2f}_eps{epsilon:.3f}_sigmax{sigma[0]:.3f}_sigmay{sigma[1]:.3f}_sigmaz{sigma[2]:.3f}'


if not os.path.isdir(folder):
    Path(folder).mkdir()    

with open(f'{folder}/input.json','w') as f:
    json.dump(args,f)
    

np.save(f'{folder}/Im_east_t_index{index_seed}.npy',Imbalance_east_t)
np.save(f'{folder}/Im_west_t_index{index_seed}.npy',Imbalance_wast_t)
np.save(f'{folder}/timedep_n_t_index{index_seed}.npy',n_t)
np.save(f'{folder}/timedep_ent_t_index{index_seed}.npy',ent_t)
np.save(f'{folder}/timedep_C2j_t_index{index_seed}.npy',Cij_t)
np.save(f'{folder}/timedep_F_t_index{index_seed}.npy',F_t)
np.save(f'{folder}/timedep_time_index{index_seed}.npy',ts)
np.save(f'{folder}/timedep_n_kink_t_index{index_seed}.npy',n_kink_t)
np.save(f"{folder}/timedep_C_auto_t_index{index_seed}.npy",C_auto_t)
np.save(f"{folder}/timedep_C_auto_av_t_index{index_seed}.npy",C_auto_av_t)
np.save(f"{folder}/timedep_n_initially_occupied_t.npy",n_initially_occupied_t)