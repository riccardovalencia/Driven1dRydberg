from quimb import *
import numpy as np
from scipy.sparse.linalg import expm
import Hamiltonian

def imag_time_evolution(H,L,target):

    # dims = [2] * L 

    # X = pauli('X')
    # Z = pauli('Z')
    # I = np.eye(2)

    dbeta = 0.01


    energy, eigenstates = eigh(H)
    # h = np.diag(energy)
    U = eigenstates

    # inifinite temperature state
    # print(energy[0])
    # print(energy[-1])
    # print(np.sum(energy))
    for beta in np.linspace(-100,100,num=int(200/dbeta)):
        # print(beta)
        # rho_beta = expm(-beta*H)
        # rho_beta = U @ @ np.conj(np.transpose(U))
        # rho_beta = np.diag(np.exp(-beta* energy)) * energy
        E = np.sum(energy * np.exp(-beta * energy)) / np.sum(np.exp(-beta * energy))
        # E = np.trace(rho_beta @ H)/np.trace(rho_beta)
        # print(f'{E} {target}')
        if E < target:
            # print(f'Procedure converged at 1/T = {beta}')
            rho_beta = U @ np.diag(np.exp(-beta* energy)) @ np.conj(np.transpose(U))
            return rho_beta/np.trace(rho_beta)
    
    # print('Procedure not converged... Returning nan')
    return float('nan')


# as above, but it returns the eigenvalues of the Hamiltonian and the set of temperatures
# satisfying the condition regarding the energy window

def imag_time_evolution_beta(H,L,target,deltaE=0):


    dbeta = 0.01
    energy, eigenstates = eigh(H)

    beta_list = []
    for beta in np.linspace(-100,100,num=int(200/dbeta)):
        E = np.sum(energy * np.exp(-beta * energy)) / np.sum(np.exp(-beta * energy))
        if E < target + deltaE:
            beta_list += [beta]
            if E <  target - deltaE:
                return beta_list, energy, eigenstates
    
    print('Procedure not converged... Returning nan')
    return float('nan')




class MyTimeDepRyd:
    
    def __init__(self, L, V, Delta, Omega1, Omega2, sparse):
        self.h0 , _ , _ = Hamiltonian.H_rydbergs_single_drive(L,V,Delta,Omega1,sparse,VNNN=True,VNNNN=False)

        self.alpha = np.array(V+[V[-2]]) # drive field frequencies 
        Sm = qu([[0,1],[0,0]],qtype='dop',sparse=sparse)
        Sp = qu([[0,0],[1,0]],qtype='dop',sparse=sparse)
        dims = [2]*L
        self.sp = []
        self.sm = []
        for j in range(L):
            self.sp += [ikron(Sp,dims=dims,inds=[j])]
            self.sm += [ikron(Sm,dims=dims,inds=[j])]
        
        self.sp = np.array(self.sp)*Omega2/2.
        self.sm = np.array(self.sm)*Omega2/2.

    def __call__(self, t):
        return self.h0 + np.sum(np.exp(-1j*self.alpha * t) * self.sp + np.exp(+1j*self.alpha * t) * self.sm)


