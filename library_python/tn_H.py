import quimb.tensor as qtn
import numpy as np


def projector_on_zero_mpo (N) : 
    P0 = np.zeros((2,2))
    P0[0,0] = 1.0
    arrays_MPO = []
    for i in range(N):
        if i == 0 or i == N-1 :
            arrays_MPO.append(P0.reshape(1,2,2))
        else :
            arrays_MPO.append(P0.reshape(1,1,2,2))
    return qtn.MatrixProductOperator(arrays_MPO)


def make_minus_qeast_mpo (s=0, L=1 , args=0) :

    if args != 0:
        if "s" in args and "L" in args:
            s = args['s']
            L = args['L']


    print(f'Building MPO East model with (L,s) = ({L}, {s:.2f})')

    '''Create MPO East model'''
    nopr = 3
    d = 2
    D = 3
    
    n1 = np.array([[0.,0.], 
                   [0.,1.]])
    x = np.array([[0.,1.], 
                    [1.,0.]])
    
    
    Z = np.zeros((nopr, d, d))
    Z[0, :, :] = np.eye(d)
    Z[1, :, :] = - n1
    Z[2, :, :] = np.exp(-s) * x - np.eye(2) 
    
    C0 = np.zeros((D,nopr))
    C0[0,0] = 1.0
    C0[1,1] = 1.0
    C0[2,2] = - 1.0
    C0 = np.einsum('Dk,kxy->Dxy', C0, Z)
    
    CL = np.zeros((D,nopr))
    CL[1,2] = 1.0
    CL[2,0] = 1.0
    CL = np.einsum('Dk,kxy->Dxy', CL, Z)
    
    C = np.zeros((D, D, nopr))
    C[0,0,0] = 1.0
    C[0,1,1] = 1.0
    C[1,2,2] = 1.0
    C[2,2,0] = 1.0
    C = np.einsum('DXk,kxy->DXxy', C, Z)
    
    arrays_MPO = [-1*C0]
    for i in range(L-2) :
        arrays_MPO.append(C)
    arrays_MPO.append(CL)
    
    return qtn.MatrixProductOperator(arrays_MPO)

def make_qeast_mpo (s=0, L=1 , args=0) :

    if args != 0:
        if "s" in args and "L" in args:
            s = args['s']
            L = args['L']


    print(f'Building MPO East model with (L,s) = ({L}, {s:.2f})')

    '''Create MPO East model'''
    nopr = 3
    d = 2
    D = 3
    
    n1 = np.array([[0.,0.], 
                   [0.,1.]])
    x = np.array([[0.,1.], 
                    [1.,0.]])
    
    
    Z = np.zeros((nopr, d, d))
    Z[0, :, :] = np.eye(d)
    Z[1, :, :] = - n1
    Z[2, :, :] = np.exp(-s) * x - np.eye(2) 
    
    C0 = np.zeros((D,nopr))
    C0[0,0] = 1.0
    C0[1,1] = 1.0
    C0[2,2] = - 1.0
    C0 = np.einsum('Dk,kxy->Dxy', C0, Z)
    
    CL = np.zeros((D,nopr))
    CL[1,2] = 1.0
    CL[2,0] = 1.0
    CL = np.einsum('Dk,kxy->Dxy', CL, Z)
    
    C = np.zeros((D, D, nopr))
    C[0,0,0] = 1.0
    C[0,1,1] = 1.0
    C[1,2,2] = 1.0
    C[2,2,0] = 1.0
    C = np.einsum('DXk,kxy->DXxy', C, Z)
    
    arrays_MPO = [C0]
    for i in range(L-2) :
        arrays_MPO.append(C)
    arrays_MPO.append(CL)
    
    return qtn.MatrixProductOperator(arrays_MPO)


def make_FA_mpo (s=0, L=1 , l1=1, l2=1, args=0) :

    if args != 0:
        if "s" in args and "L" in args:
            s = args['s']
            L = args['L']
            # print(s)
            # print(L)


    '''Create MPO FA model'''
    nopr = 3
    d = 2
    D = 4
    
    n1 = np.array([[0.,0.], 
                   [0.,1.]])
    # x = 0.5* np.array([[0.,1.], 
                    # [1.,0.]])
    x = np.array([[0.,1.], 
                [1.,0.]])
    a = - np.exp(-s) * x + np.eye(2)
    
    Z = np.zeros((nopr, d, d))
    Z[0, :, :] = np.eye(d)
    Z[1, :, :] = n1
    Z[2, :, :] = a
    
    C0 = np.zeros((D,nopr))
    C0[0,0] = 1.0
    C0[1,1] = l1
    C0[2,2] = 1.0
    C0 = np.einsum('Dk,kxy->Dxy', C0, Z)
    
    CL = np.zeros((D,nopr))
    CL[1,2] = 1.0
    CL[2,1] = l2
    CL[3,0] = 1.0
    CL = np.einsum('Dk,kxy->Dxy', CL, Z)
    
    C = np.zeros((D, D, nopr))
    C[0,0,0] = 1.0
    C[0,1,1] = l1
    C[0,2,2] = 1.0
    C[1,3,2] = 1.0
    C[2,3,1] = l2
    C[3,3,0] = 1.0
    C = np.einsum('DXk,kxy->DXxy', C, Z)
    
    arrays_MPO = [C0]
    for i in range(L-2) :
        arrays_MPO.append(C)
    arrays_MPO.append(CL)
    
    return qtn.MatrixProductOperator(arrays_MPO)




def make_FA_mpo_bosons (s=0, U = 0, mu=0, L=1 , d=2, args=0) :

    if args != 0:
        if "s" in args: 
            s = args['s']
        if "L" in args:
            L = args['L']
        if "U" in args:
            U = args['U']
        if "Lambda" in args:
            d = args['Lambda'] + 1
        if "mu" in args:
            mu = args['mu']
        
            
            

    '''Create MPO FA model'''
    nopr = 3
    D = 4
    
    a = [np.sqrt(j+1) for j in range(d-1)]
    n1 = np.diag([j for j in range(d)])
    x  = np.diag(a,k=1) + np.diag(a,k=-1)
    a = - np.exp(-s) * x + np.eye(d) + U * n1

    Z = np.zeros((nopr, d, d))
    Z[0, :, :] = np.eye(d)
    Z[1, :, :] = n1
    Z[2, :, :] = a
    
    C0 = np.zeros((D,nopr))
    C0[0,0] = 1.0
    C0[1,1] = 1.0
    C0[2,2] = 1.0
    C0 = np.einsum('Dk,kxy->Dxy', C0, Z)
    
    CL = np.zeros((D,nopr))
    CL[1,2] = 1.0
    CL[2,1] = 1.0
    CL[3,0] = 1.0
    CL = np.einsum('Dk,kxy->Dxy', CL, Z)
    
    C = np.zeros((D, D, nopr))
    C[0,0,0] = 1.0
    C[0,1,1] = 1.0
    C[0,2,2] = 1.0
    C[1,3,2] = 1.0
    C[2,3,1] = 1.0
    C[3,3,0] = 1.0
    C = np.einsum('DXk,kxy->DXxy', C, Z)
    
    arrays_MPO = [0.5*C0]
    for i in range(L-2) :
        arrays_MPO.append(C)
    arrays_MPO.append(CL)
    
    return qtn.MatrixProductOperator(arrays_MPO)


# def get_inverted_mps (psi) :
#     arrays_mps = []
#     for p in range(psi.L)[::-1] :
#         if p == psi.L - 1 :
#             arrays_mps.append(psi[p].data)
#         elif p == 0 :
#             arrays_mps.append(psi[p].data)
#         else :
#             arrays_mps.append(psi[p].data.transpose((1,0,2)))

#     return qtn.MatrixProductState(arrays_mps) 

def get_inverted_mps (psi) :
    arrays_mps = []
    for p in psi.sites[::-1] :
        if p == psi.sites[-1] :
            arrays_mps.append(psi[p].data)
        elif p == 0 :
            arrays_mps.append(psi[p].data)
        else :
            arrays_mps.append(psi[p].data.transpose((1,0,2)))

    return qtn.MatrixProductState(arrays_mps) 


def FA_wo_zero (N, s) :
    '''Exclude vacuum for the first ground state'''
    mpoFA = make_FA_mpo (s, N)
    proj0 = projector_on_zero_mpo (N)
    return mpoFA.add_MPO(10. * N * proj0, inplace=True)

def magnetization_per_site (psi) :
    mag = []
    for i in psi.sites :
        mag.append(abs(psi.magnetization(i)-0.5)) 

    return mag


def variance_operator ( psi , O ) :
    
    O_prime = O.copy()
    psi_d = psi.H.copy()
    
    inds     = O.outer_inds()
    ket_inds = inds[ ::2]
    bra_inds = inds[1::2]

    Map = {}
    for j in O.sites:
        Map[ket_inds[j]] = bra_inds[j]

    psi_d.reindex(Map,inplace=True)    

    exp_O  = psi & O & psi_d ^ ...
    Map1 = {}
    Map2 = {}
    for j in O.sites:
        Map1[bra_inds[j]] = f'bprime{j}'
        Map2[ket_inds[j]] = bra_inds[j]

    O_prime.reindex(Map1,inplace = True)
    O_prime.reindex(Map2,inplace = True)

    psi_d.reindex(Map1,inplace = True)


    exp_O2 = psi_d & O & O_prime & psi ^ ...
    return exp_O2 - exp_O**2




# MPO of Rydberg atoms


def make_rydbergs_mpo (L,Vij,Deltaj,Omegaj, d=2) :
    
    '''Create MPO FA model'''
    nopr = 3
    D = 3
    
    n = np.array([[0.,0.], 
                   [0.,1.]])
    x = 0.5* np.array([[0.,1.], 
                [1.,0.]])
    
    Z = np.zeros((nopr, d, d))
    Z[0, :, :] = np.eye(d)
    Z[1, :, :] = n
    Z[2, :, :] = x
    
    # first site
    C0 = np.zeros((D,nopr))
    C0[0,1] = Deltaj[0]
    C0[0,2] = Omegaj[0]
    C0[1,0] = 1.0
    C0[2,1] = Vij[0]
    C0 = np.einsum('Dk,kxy->Dxy', C0, Z)
    
    # last site
    CL = np.zeros((D,nopr))
    CL[0,0] = 1.0
    CL[1,1] = Deltaj[-1]
    CL[1,2] = Omegaj[-1]
    CL[2,1] = 1.0
    CL = np.einsum('Dk,kxy->Dxy', CL, Z)
    
    # even site
    Ce = np.zeros((D, D, nopr))
    Ce[0,0,0] = 1.0
    Ce[1,1,0] = 1.0
    Ce[1,0,1] = Deltaj[1]
    Ce[1,0,2] = Omegaj[1]
    Ce[1,2,1] = Vij[1] 
    Ce[2,0,1] = 1.0
    Ce = np.einsum('DXk,kxy->DXxy', Ce, Z)

    # odd site
    Co = np.zeros((D, D, nopr))
    Co[0,0,0] = 1.0
    Co[1,1,0] = 1.0
    Co[1,0,1] = Deltaj[0]
    Co[1,0,2] = Omegaj[0]
    Co[1,2,1] = Vij[0] 
    Co[2,0,1] = 1.0
    Co = np.einsum('DXk,kxy->DXxy', Co, Z)
    
    arrays_MPO = [C0]
    for i in range(L-2) :
        if i % 2 == 0:
            arrays_MPO.append(Ce)
        else:
            arrays_MPO.append(Co)
    arrays_MPO.append(CL)
    
    return qtn.MatrixProductOperator(arrays_MPO)

