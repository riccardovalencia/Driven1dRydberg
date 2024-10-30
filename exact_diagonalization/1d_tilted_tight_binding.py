import sys
sys.path.append("../library_python")
import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
from quimb import *
import math
from scipy.fft import fft, fftfreq
import Hamiltonian, initial_state, parameters, ED_observables
import os
from pathlib import Path


# The effective theory \sum_j n_j ( \Omega x_{j+1} (1-n_{j+2}) + V_NNN n_{j+2})
# can be simplified exploting the conservation of kinks, and the fact that the states 
# with M kinks are built as product states of the ones of 1 kink (due to the strong constraint).
# Specifically, for a single-kink, the Hamiltonian reduces to
# H = Omega sum_k (|k><k+1| + h.c.)  [kinetic term]
#   + eps  sum_k k |k><k|            [potential term due to detuning from perfect antiblockade]
#   + VNNN sum_k (k-2) |k><k|        [potential term]

# Here, we study the dynamics of a state with a well defined single kink (it is not a superposition of different kinks)
# We want to observe the rise of unidirectional Bloch oscillations due to V_NNN, which is the term leading to a linear potential
# along the chain (in its absence, we would have free propagating particles)

# The basis adopted |q>, in terms of the usual basis on the lattice is given by |q> = |1>^q |0>^(N-q)
# We order the states from |q=1> |q=2> ... |q=N> (I am keeping the first site as the one fixing the East symmetry. I could gain an additional
# site, but since the computational cost scales linearly with the system size it is not a big deal)


# Path where to save the data
save_folder = './'

# system size
L = 30
# Rabi frequency external drive - controls the kinetic term in the effective theory
Omega = 1
# nearest-neighbor interaction - leads to confinment of kinks
VNNN = 0
# detuning from perfect antiblockade condition (eps = 0 -> perfect antiblockade)
eps = 0
# initial state given by a kink of M excitations (i.e. |psi(t=0)> = |1>^M |0>^{L-M})
M = int(L/2)
# total time
T = 30
ts = np.linspace(0,T,num=100)



# -------------------------------------------------------------------


psi = np.zeros(shape=L)
psi[M-1] = 1

H   = qu(Hamiltonian.Heff_single_kink(L,Omega,VNNN,eps),qtype='dop')
psi = qu(psi,qtype='ket',normalized=True)

evo = Evolution(psi, H, progbar=True,method='solve')
# density profile
n_t   = []
# fidelity
F_t   = []
# entanglement entropy profile
ent_t = []
# probability distribution of wavefunctions
psi_t_abs = []
# equal-time density-density correlation function
C_t   = []
for psi_t in evo.at_times(ts):
    n = []
    for j in range(L):
        n += [np.sum(np.abs([psi_t[j:]])**2  )]

    ent_ = []    
    rho_t = psi_t @ psi_t.H
    pop_t = np.diag(rho_t)
    for j in range(L-1):
        # reduced density matrix (from kink basis to lattice basis)
        rho_ = rho_t[:j+1,:j+1].copy()
        rho_[-1,-1] += np.sum(pop_t[j+1:])
        lam , _ = eigh(rho_)
        index_pos = lam > 1E-12
        lam = lam[index_pos]
        S = -np.sum(lam * np.log2(lam))
        ent_ += [S]

    C = []
    for j in range(L):
        n0nj = np.sum(np.abs([psi_t[max(M-1,j):]])**2)
        n0  =  np.sum(np.abs([psi_t[M-1:]])**2)
        nj  =  np.sum(np.abs([psi_t[j:]])**2)
        C += [n0nj - n0*nj]
    C_t += [C]


    ent_t += [ent_]
    n_t   += [n]
    psi_t_abs += [np.abs(psi_t)**2]

    F_t += [fidelity(psi_t,psi)**2]
    
ent_t = np.array(ent_t)
n_t   = np.array(n_t)
psi_t_abs = np.array(psi_t_abs)
F_t = np.array(F_t)
C_t = np.array(C_t)

save_folder = f"{save_folder}QXP_N{L}_M{M}_VNNN{VNNN:.2f}"

if not os.path.isdir(save_folder):
    Path(save_folder).mkdir()

np.save(f"{save_folder}/time.npy",ts)
np.save(f"{save_folder}/ent_t.npy",ent_t)
np.save(f"{save_folder}/n_t.npy",n_t)
np.save(f"{save_folder}/C_t.npy",C_t)
np.save(f"{save_folder}/F_t.npy",F_t)
