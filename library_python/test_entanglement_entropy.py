from quimb import *
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/ricval/Documenti/localized_protected_quantum_order/src/library")
import Hamiltonian as Ham


L = 2
sparse = False

dims = [2] * L

psi_j_up   = qu(np.array([1,0]),qtype='ket',sparse=sparse).real.astype(float)
psi_j_down = qu(np.array([0,1]),qtype='ket',sparse=sparse).real.astype(float)

psi_up = psi_j_up
psi_down = psi_j_down
for j in range(1,L):
    psi_up   = psi_up   & psi_j_up
    psi_down = psi_down & psi_j_down

psi = (psi_up + psi_down) / np.sqrt(2)
rho_reduced = partial_trace(psi,dims,keep=[j for j in range(L//2)])
print(rho_reduced)
print(entropy(rho_reduced))

a = entropy_subsys(psi, dims, [1])
print(a)