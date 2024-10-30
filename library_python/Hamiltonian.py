from quimb import *
import numpy as np


# ------------------------------------------------
# Hamiltonian of the east model
# H = \sum_j Hj, where
# Hj = -J/2 n_j x_{j+1} + n_j/2

def H_east(L,s,sparse=False):

    J = np.exp(-s)

    dims = [2] * L 


    X = pauli('X',sparse=sparse)
    Z = pauli('Z',sparse=sparse)
    I = qu(np.eye(2),qtype='dop',sparse=sparse).real.astype(float)
    N = qu(np.diag([0,1]),qtype='dop',sparse=sparse).real.astype(float)


    # Hew = 0
    # eigenvalue 
    symmetry_sector = -1
    xs = J * X - I
    Hew = -0.5 * ikron(xs, dims , inds=[0]) 

    for j in range(L-1):
        Hew += -0.5 * pkron(N  & xs, dims , inds=[j,j+1]) 
        # Hew += -0.5 * pkron(xs & N , dims , inds=[j,j+1]) 
    Hew += -0.5 * ikron(N,dims,inds=[L-1]) * ( J * symmetry_sector - 1)
    return Hew


def H_rydbergs_single_drive(L,Vij,Deltaj,Omegaj,sparse=False,VNNN=False,VNNNN=False):

    # operators
    dims = [2] * L 
    X = pauli('X',sparse=sparse)
    Z = pauli('Z',sparse=sparse)
    I = qu(np.eye(2),qtype='dop',sparse=sparse)
    N = qu(np.diag([0,1]),qtype='dop',sparse=sparse)

    H = 0
    # local fields
    for j in range(L):
        H += Deltaj[j] * ikron(N,dims,inds=[j])
        H += Omegaj[j]/2. * ikron(X,dims,inds=[j])

    # interaction
    for j in range(L-1):
        H += Vij[j] * pkron(N  & N, dims , inds=[j,j+1]) 

    if VNNN:
        print(f'Including VNNN')
        # Compute r1 + r2 distance from nearest neighbour interactions
        VNNN = []
        for j in range(L-2):
            r1 = 1/Vij[j]**(1./6)
            r2 = 1/Vij[j+1]**(1./6)
            VNNN += [1/(r1+r2)**6.]
            
            H+= VNNN[-1] * pkron(N & N , dims , inds=[j,j+2])

        
        print(f'Min ration VNN/VNNN : {min(Vij)/max(VNNN):.3f}')
        print(f'Ratio VNNN/Rabi-frequency : {min(VNNN)/np.abs(Omegaj[0]):.3f}')

    if VNNNN:
        print(f'Including VNNNN')
    

        VNNNN = []

        for j in range(L-3):
            r = 0
            for k in range(j,j+3):
                r +=  1/Vij[k]**(1./6)
            VNNNN += [1/r**6.]
            H+= VNNNN[-1] * pkron(N & N , dims , inds=[j,j+3])
            
        print(f'Min ration VNN/VNNNN : {min(Vij)/max(VNNNN):.3f}')
        print(f'Ratio VNNN/Rabi-frequency : {min(VNNNN)/np.abs(Omegaj[0]):.3f}')


    return H , VNNN, VNNNN


def compute_VNNN(Vij):
    r1 = 1/Vij[0]**(1./6)
    r2 = 1/Vij[1]**(1./6)
    return 1/(r1+r2)**6.



# Hamiltonian of the effective east model

def H_east_with_NN(L,Omega,eps,VNNN,n0=0,sparse=False,VNN=0):


    dims = [2] * L 

    X = pauli('X',sparse=sparse)
    Z = pauli('Z',sparse=sparse)
    I = qu(np.eye(2),qtype='dop',sparse=sparse).real.astype(float)
    N = qu(np.diag([0,1]),qtype='dop',sparse=sparse).real.astype(float)

    # print(Omega)
    # exit(0)
    H = n0 * Omega[0] * ikron(X,dims=dims,inds=[0])/2

    for j in range(L):
        H += eps * ikron(N,dims=dims,inds=[j])
    
    for j in range(L-1):
        H += Omega[j]/2 * pkron( N & X , dims=dims,inds=[j,j+1])
    
    for j in range(L-2):
        H += VNNN[j]  * pkron( N & N , dims=dims,inds=[j,j+2])

    if len(VNN) != 0:
        for j in range(L-1):
            H+= VNN[j] * pkron( N & N , dims=dims,inds=[j,j+1])
      

    return H


# Hamiltonian of the effective East model obtained with up to two external drive fields
# H = \sum_j Hj where
# Hj = Omega1 N_j X_{j+1} (1-N_{j+2}) 
#    + Omega2 N_j X_{j+1} N_{j+2} 
#    + eps N_j 
#    + VNN N_j N_{j+1} + VNNN N_j N_{j+2}

def H_east_tunable(L,Omega1,Omega2,eps,VNN,VNNN,n0=0,nf=1,sparse=False):

    dims = [2] * L 

    X = pauli('X',sparse=sparse)/2
    Z = pauli('Z',sparse=sparse)
    I = qu(np.eye(2),qtype='dop',sparse=sparse).real.astype(float)
    N = qu(np.diag([0,1]),qtype='dop',sparse=sparse).real.astype(float)


    H  = n0 * Omega2 * ikron( X & N,dims=dims,inds=[0,1])
    H += n0 * Omega1 * pkron( X & (I-N),dims=dims,inds=[0,1]) 
    
    for j in range(L-2):
        H += Omega2 * pkron( N & X & N , dims=dims,inds=[j,j+1,j+2])
        H += Omega1 * pkron( N & X & (I-N) , dims=dims,inds=[j,j+1,j+2])

    H += Omega2 * pkron( N & X , dims=dims,inds=[L-2,L-1]) * nf
    H += Omega1 * pkron( N & X , dims=dims,inds=[L-2,L-1]) * (1-nf)
    
    for j in range(L):
        H += eps * ikron(N,dims=dims,inds=[j])
    

    for j in range(L-2):
        H += VNNN  * pkron( N & N , dims=dims,inds=[j,j+2])

    for j in range(L-1):
        H+= VNN * pkron( N & N , dims=dims,inds=[j,j+1])
      
    return H


# The effective theory \sum_j n_j ( \Omega x_{j+1} (1-n_{j+2}) + V_NNN n_{j+2})
# can be simplified exploting the conservation of kinks, and the fact that the states 
# with M kinks are built as product states of the ones of 1 kink (due to the strong constraint).
# Specifically, for a single-kink, the Hamiltonian reduces to
# H = Omega sum_k (|k><k+1| + h.c.)  [kinetic term]
#   + eps  sum_k k |k><k|            [potential term due to detuning from perfect antiblockade]
#   + VNNN sum_k (k-2) |k><k|        [potential term]

def Heff_single_kink(L,Omega,VNNN,eps=0):
    
    # diagonal terms (linear potential)
    h_d = VNNN * np.diag([0,0] + [(k-2) for k in range(3,L+1)])
    h_d += eps * np.diag([k for k in range(L+1)])
    # kinetic term
    h_kin = np.diag([Omega for _ in range(L-1)], k =1) + np.diag([Omega for _ in range(L-1)], k = -1)

    # single-kink Hamiltonian
    H = h_d + h_kin

    return H