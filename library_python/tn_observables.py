import quimb.tensor as qtn
import numpy as np

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


def measure_O_each_site (psi , O) :
    obs = []
    for i in psi.sites:
        obs.append( psi.gate(O,i).H @ psi ) 
    return obs
    

def expectation_value(psi, O):
    psiH = psi.H
    psi.align_(O, psiH)
    return (psiH & O & psi) ^ ...


def zero_edge_mode(L,s):


    '''Create MPO zero edge mode'''
    nopr = 4
    d = 2
    D = 2
    
    n = np.array([[0.,0.], 
                   [0.,1.]])
    x = np.array([[0.,1.], 
                    [1.,0.]])
    
    z = np.array([[1.,0.], 
                    [0.,-1.]])
    # store the operators necessary
    Z = np.zeros((nopr, d, d))
    Z[0, :, :] = np.eye(d)
    Z[1, :, :] = n
    Z[2, :, :] = -2 * np.exp(-s) * x
    Z[3, :, :] = z
    
    # MPO - 0th site
    C0 = np.zeros((D,nopr))
    C0[0,3] = 1.0
    C0[1,2] = 1.0
    C0 = np.einsum('Dk,kxy->Dxy', C0, Z)
    
    # MPO - last site
    CL = np.zeros((D,nopr))
    CL[0,0] = 1.0
    CL = np.einsum('Dk,kxy->Dxy', CL, Z)
    
    # MPO - 1st site
    C1 = np.zeros((D, D, nopr))
    C1[0,0,0] = 1.0
    C1[1,0,1] = 1.0
    C1 = np.einsum('DXk,kxy->DXxy', C1, Z)

    # MPO - bulk
    Cj = np.zeros((D, D, nopr))
    Cj[0,0,0] = 1.0
    Cj = np.einsum('DXk,kxy->DXxy', Cj, Z)
    
    # assembling
    arrays_MPO = [C0]
    arrays_MPO.append(C1)
    for i in range(1,L-2) :
        arrays_MPO.append(Cj)
    arrays_MPO.append(CL)
    
    return qtn.MatrixProductOperator(arrays_MPO)



def zero_edge_mode_v2(L,s,site=1):

    '''Create MPO zero edge mode'''
    nopr = 4
    d = 2
    D = 2
    
    n = np.array([[0.,0.], 
                   [0.,1.]])
    x = np.array([[0.,1.], 
                    [1.,0.]])
    
    z = np.array([[1.,0.], 
                    [0.,-1.]])
    # store the operators necessary
    Z = np.zeros((nopr, d, d))
    Z[0, :, :] = np.eye(d)
    Z[1, :, :] = n
    Z[2, :, :] = - np.exp(-s) * x
    Z[3, :, :] = z

    # MPO - bulk
    Cj = np.zeros((D, D, nopr))
    Cj[0,0,0] = 1.0
    Cj = np.einsum('DXk,kxy->DXxy', Cj, Z)

    # MPO - last site
    CL = np.zeros((D,nopr))
    CL[0,0] = 1.0
    CL = np.einsum('Dk,kxy->Dxy', CL, Z)

    if site == 1:

        # MPO - 0th site
        C0 = np.zeros((D,nopr))
        C0[0,3] = 1.0
        C0[1,2] = 2.0
        C0 = np.einsum('Dk,kxy->Dxy', C0, Z)

        # MPO - 1st site
        C1 = np.zeros((D, D, nopr))
        C1[0,0,0] = 1.0
        C1[1,0,1] = 1.0
        C1 = np.einsum('DXk,kxy->DXxy', C1, Z)

        arrays_MPO = [C0]
        arrays_MPO.append(C1)
        for i in range(1,L-2) :
            arrays_MPO.append(Cj)
        arrays_MPO.append(CL)
    
    if site == 2:
        C0 = np.zeros((D, nopr))
        C0[0,1] = 1.0
        C0[1,0] = 1.0
        C0 = np.einsum('Dk,kxy->Dxy', C0, Z)

        C1 = np.zeros((D, D, nopr))
        C1[0,0,2] = 1.0
        C1[1,0,3] = 1.0
        C1[1,1,2] = 1.0
        C1 = np.einsum('DXk,kxy->DXxy', C1, Z)

        C2 = np.zeros((D, D, nopr))
        C2[0,0,0] = 1.0
        C2[1,0,1] = 1.0
        C2 = np.einsum('DXk,kxy->DXxy', C2, Z)

        arrays_MPO = [C0]
        arrays_MPO.append(C1)
        arrays_MPO.append(C2)
        for _ in range(3,L-1) :
            arrays_MPO.append(Cj)
        arrays_MPO.append(CL)



    if site > 2:
        C0 = np.zeros((D,nopr))
        C0[0,0] = 1.0
        C0 = np.einsum('Dk,kxy->Dxy', C0, Z)

        C1 = np.zeros((D, D, nopr))
        C1[0,0,1] = 1.0
        C1[0,1,0] = 1.0
        C1 = np.einsum('DXk,kxy->DXxy', C1, Z)

        C2 = np.zeros((D, D, nopr))
        C2[0,0,2] = 1.0
        C2[1,0,3] = 1.0
        C2[1,1,2] = 1.0
        C2 = np.einsum('DXk,kxy->DXxy', C2, Z)

        C3 = np.zeros((D, D, nopr))
        C3[0,0,0] = 1.0
        C3[1,0,1] = 1.0
        C3 = np.einsum('DXk,kxy->DXxy', C3, Z)

        arrays_MPO = [C0]
        for _ in range(1,site-2):
            arrays_MPO.append(Cj)
        arrays_MPO.append(C1)
        arrays_MPO.append(C2)
        arrays_MPO.append(C3)

        for _ in range(site+1,L-1) :
            arrays_MPO.append(Cj)
        arrays_MPO.append(CL)

    
    return qtn.MatrixProductOperator(arrays_MPO)