
import sys
sys.path.append("../library_python")
import Hamiltonian , initial_state, ED_observables
import numpy as np
from quimb import *
import os
from pathlib import Path

# Simulating dynamics under the time-independent Hamiltonian
# H = \sum_j Hj where
# Hj = Omega1 N_j X_{j+1} (1-N_{j+2}) 
#    + Omega2 N_j X_{j+1} N_{j+2} 
#    + eps N_j 
#    + VNN N_j N_{j+1} + VNNN N_j N_{j+2}

# path where to save data
save_folder = './'
sparse = True

# --------------- INPUT ---------------
# initial state - density profile (1 = down_z ; 0 = up_z)
state = [1,1,0,1,1,1,0,1,1]
# drive field Rabi frequencies 
Omega1 = 0.    
Omega2 = 1.
# detuning of the drive fields from perfect antiblockade condition (eps = 0 corresponds to perfect antiblockade)
eps = 0
# VNN -> nearest neighbor interaction. Ideal antiblockade condition implies VNN = 0 (in interaction picture)
# VNNN -> next-nearest neighbor interaction. Still there in the ideal antiblockade condition
VNN = 0.
VNNN = 0.

# boundary conditions
# n0 -> fixes the first excitation = 1 (it is a conserved quantity)
# nf = 0, controls  Omega1 N_{L-1} X_L (1-nf) + Omega2 N_{L-1} X_L nf
n0 = 1 # I want to show the East sector fixed
nf = 1 # I want the last site to be active
# total time
T = 30
# number of steps saved
nsteps = 300

# --------------- END INPUT ---------------


psi , [x_exc,n_exc] = initial_state.computational_spin_state(state,sparse)
L = len(state)
dims = [2] * L
ts = np.linspace(0,T,num=nsteps)

# number operator
N = qu(np.diag([0,1]),qtype='dop',sparse=sparse)
# Hamiltonian
H = Hamiltonian.H_east_tunable(L,Omega1,Omega2,eps,VNN,VNNN,n0,nf,sparse)

# observables
# density profile 
n_t = []
# correlation function
C_t = []
# entanglement entropy profile
ent_t = []
# fidelity
F_t = []
# unequal time autocorrelation function
C_auto_t = []




if L <= 12:
    evo = Evolution(psi, H, method='solve', progbar=True)

else:
    evo = Evolution(psi, H, method='integrate', progbar=True)



for idx, psi_t in enumerate(evo.at_times(ts)):
    n = []
    Cij = []
    C_auto = 0
    for j in range(L):
        n += [np.real(ED_observables.local_obs(psi_t,N,j,dims,sparse))]
        if state[j] == 1:# and j != x_exc[0]:
            C_auto += n[-1]
        
    for j in range(L):
        Cij += [np.abs(ED_observables.correlations(psi_t,N,j,0,dims,sparse) )]

    ent = []
    for j in range(1,L):
        ent += [entropy_subsys(psi_t, dims=[2] * L, sysa=range(j))]

    F_t += [fidelity(psi_t,psi)**2]
    n_t   += [n]
    C_t += [Cij]
    ent_t += [ent]
    C_auto_t += [C_auto/(n_exc)]


n_t     = np.array(n_t)
ent_t   = np.array(ent_t)
C_t     = np.array(C_t)
F_t     = np.array(F_t)
C_auto_t = np.array(C_auto_t)


C_auto_av_t = []
dt = ts[1:] - ts[:-1]

# perform time average of autocorrelation function
for j in range(1,len(C_auto_t)):  
    C_auto_av_t     += [2*np.sum(C_auto_t[:j]*dt[:j])/np.sum(dt[:j])-1]
C_auto_av_t = np.array(C_auto_av_t)


# saving data
string_state =''
for q in state:
    if q == 0:
        string_state += '0'
    else:
        string_state += '1'

save_folder = f"{save_folder}{string_state}"
if not os.path.isdir(save_folder):
    Path(save_folder).mkdir()
save_folder = f"{save_folder}/QXQ_VNNN{VNNN:.2f}"
if not os.path.isdir(save_folder):
    Path(save_folder).mkdir()


np.save(f"{save_folder}/time.npy",ts)
np.save(f"{save_folder}/ent_t.npy",ent_t)
np.save(f"{save_folder}/n_t.npy",n_t)
np.save(f"{save_folder}/C_t.npy",C_t)
np.save(f"{save_folder}/F_t.npy",F_t)
np.save(f"{save_folder}/C_auto_t.npy",C_auto_t)
np.save(f"{save_folder}/C_auto_av_t.npy",C_auto_av_t)
