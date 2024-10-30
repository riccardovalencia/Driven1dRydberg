import quimb.tensor as qtn
import numpy as np

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



# the right index of j+1 has to match the dimension of the left index on site j
# the physical index is always the same (it is the third index)
# encoding of indiced (left,right,physical)

# we have to merge the two mps - I have to insert a left index of dimension psi_1[-1].shape[0]
# li -> left index
# ri -> right index
# d  -> physical index
# to the first matrix of psi_2 (i.e. psi_2[0])


def concatenate_MPS(psi_1,psi_2):

    mps_merged = []

    # inserting mps psi_1

    for j in psi_1.sites[:-1]:
        mps_merged.append(psi_1[j].data)

    # inserting right index of dimension 1 to first mps which does not have one
    
    #  qtn.tensor_core.bonds_size(psi_1[-2], psi_1[-1]) , 1 , psi_1[-1]
    idx = psi_1.site_ind(psi_1.L-1)
    li, ri, d = psi_1.bond_size(psi_1.L-2,psi_1.L-1) , 1 , psi_1.ind_size(ind=idx)

    # li, ri, d = psi_1[-1].shape[0] , 1 , psi_1[0].shape[1]
    phi = np.zeros((li,ri,d),dtype=complex)
    for k in range(ri):
        phi[:,k,:] = psi_1[-1].data
    mps_merged.append(phi)

    # inserting mps psi_2

    if psi_2.L > 1:

        # inserting left index of dimension 1 to first mps which does not have one

        # li, ri, d = 1 , psi_2[0].shape[0] , psi_2[0].shape[1]
        idx = psi_2.site_ind(0)
        li , ri , d = 1 , psi_2.bond_size(0,1) , psi_2.ind_size(ind=idx)
           
        phi = np.zeros((li,ri,d),dtype=complex)
        for k in range(li):
            phi[k,:,:] = psi_2[0].data
        mps_merged.append(phi)

        for j in psi_2.sites[1:]:
            mps_merged.append(psi_2[j].data)

    else:

        # inserting left index of dimension 1 to first mps which does not have one
        # li , d = 1 , psi_2[0].shape[1]
        idx = psi_2.site_ind(0)
        li , d = 1 , psi_2.ind_size(ind=idx)

        phi = np.zeros((li,d),dtype=complex)
        for k in range(li):
            phi[k,:] = psi_2[0].data
        mps_merged.append(phi)

    # converting into a new MPS

    psi = qtn.MatrixProductState(mps_merged)
    psi.normalize()

    return psi