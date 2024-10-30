import sys
sys.path.append("../library_python")
import quimb.tensor as qtn
import numpy as np
from quimb import *
from pathlib import Path
import os, json
import parameters, input

save_folder = './'

# Allows simulation of dynamics of 1d Rybderg arrays up to NEAREST-NEIGHBOR interactions (truncated next-order corrections)
# It uses a 2-sites TEBD algorithm. If it is desired to keep up to NEXT-NEAREST-NEIGHBOR interactions, it is necessary to use the .cpp code.

# Neglecting higher order interactions works at short times (~1/VNNN), with VNNN next-nearest interaction
# We aim to investigate the impact of finite temperature T in the ideal transport case. We want to check that 
# propagation is still mainly towards east (namely, we do not have undesired resonances).
# We include thermal fluctuations adding either a noise on the interactions or positions. The issue with sampling for the interactions
# is that they are not independent. There is a constraint (see https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.118.063606)


# useful operators
I = qu(np.eye(2),qtype='dop')
X = qu(np.array([[0,1.],[1.,0.]])/2.,qtype='dop')
N = qu(np.diag([0,1]),qtype='dop')

#------------------------------------
# get input
args = input.input_rydbergs()
L = args['L']
V1  = args['V1']
V2  = args['V2']
Omega = args['Omega']
epsilon = args['eps']
T = args['T'] / np.abs(Omega)
n_steps = args['number_steps']

sigma = np.array([float(y) for y in args['sigma']])        
index_seed = args['index_seed']
alpha = 6. # power-law interaction
Delta1 = -V1 + epsilon # detuning even site 
Delta2 = -V2 + epsilon # detuning odd site 

#------------------------------------
# Initial state
binary = '0' * (L//3-1) + '1' 
binary += '0' * (L-len(binary))
# position of first excitation
for j, bit in enumerate(binary):
    if bit == '1':
        x_exc = j
        break
psi_t0 = qtn.MPS_computational_state(binary)

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

# Build 2-sites gate
H1 = {i: Delta[i] * N + Omega * X for i in range(L)}
H2 = {(i,i+1): V[i] * (N & N)        for i in range(L-1)}
H = qtn.tensor_1d_tebd.LocalHam1D(L=L, H2=H2, H1=H1)

tebd = qtn.TEBD(psi_t0, H)
tebd.split_opts['cutoff'] = 1e-12
tol = 1e-3 # tolerance of tebd in the Trotter decomposition
ts  = np.linspace(0, T, n_steps)

#------------------------------------
# Observables

n_west_t = []
I_t      = []
n_t = []
S_t = []
for psit in tebd.at_times(ts, tol=1e-3):

    nj = []
    Se_b = []
    nj += [0.5-psit.magnetization(0) ]
    for j in range(1, L):
        nj += [0.5-psit.magnetization(j, cur_orthog=j-1)]
        Se_b += [psit.entropy(j, cur_orthog=j)]
        
    n_west_t += [max(nj[:x_exc])]
    I_t      += [sum(np.real(nj[x_exc:]))]
    n_t      += [nj]
    S_t += [Se_b]


n_west_t = np.array(n_west_t)
I_t      = np.array(I_t)
n_t      = np.array(n_t)
S_t = np.array(S_t)

folder = f'{save_folder}rydberg_L{L}_V1{V1:.1f}_V2{V2:.1f}_Omega{Omega:.2f}_eps{epsilon:.2f}_sigmax{sigma[0]:.3f}'

if not os.path.isdir(folder):
    Path(folder).mkdir()    

folder = f'{folder}/{binary}'

if not os.path.isdir(folder):
    Path(folder).mkdir()  

with open(f'{folder}/input.json','w') as f:
    json.dump(args,f)

np.save(f'{folder}/n_west_t_index{index_seed}.npy',n_west_t)
np.save(f'{folder}/I_t_index{index_seed}.npy',I_t)
np.save(f'{folder}/n_t_index{index_seed}.npy',n_t)
np.save(f'{folder}/S_t_index{index_seed}.npy',n_t)
