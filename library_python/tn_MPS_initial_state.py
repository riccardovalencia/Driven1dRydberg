import quimb.tensor as qtn
import numpy as np
import tn_H
import tn_observables
from scipy.linalg import eigh
import sys
sys.path.append("library")
import Hamiltonian as Ham

import matplotlib.pyplot as plt



# state given by |1>^n1|\xi>|00000>
# where |1>^n1 |\xi> is obtained runnig DMRG-X on |1>^n1|00000> with the Hamiltonian H_name

def exp_dressed_kink(H_name,args, n1 , n0 ):

    # We initialize a binary state of {n1+n0} sites
    binary = '1' * n1 + '0' * n0
    psi_start = qtn.MPS_computational_state(binary)
    # We build the MPO of the Hamiltonian H_name
    args_dmrgx = args.copy()
    args_dmrgx['L'] = n1 + n0
    H = H_name(args=args_dmrgx)
    # We perform DMRG-X
    maxD = 10
    dmrgX = qtn.DMRGX(H, psi_start, bond_dims=[2*maxD]*args_dmrgx['L'], cutoffs=1e-12)
    dmrgX.solve(tol=1e-9, verbosity=1)
    
    return dmrgX._k   
    

def mps_logic_product_state(n,bit='1'):
    binary = bit * n
    return qtn.MPS_computational_state(binary)




def mps_kth_eigenstate(H,L,d,k=0):
    _ , eigenstates = eigh(H)
    dims = [d]*L
    return qtn.MatrixProductState.from_dense(eigenstates[:,k],dims=[d]*L)



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
    
    li, ri, d = psi_1[-1].shape[0] , 1 , psi_1[0].shape[1]
    phi = np.zeros((li,ri,d),dtype=complex)
    for k in range(ri):
        phi[:,k,:] = psi_1[-1].data
    mps_merged.append(phi)

    # inserting mps psi_2

    if psi_2.L > 1:

        # inserting left index of dimension 1 to first mps which does not have one

        li, ri, d = 1 , psi_2[0].shape[0] , psi_2[0].shape[1]
        phi = np.zeros((li,ri,d),dtype=complex)
        for k in range(li):
            phi[k,:,:] = psi_2[0].data
        mps_merged.append(phi)

        for j in psi_2.sites[1:]:
            mps_merged.append(psi_2[j].data)

    else:

        # inserting left index of dimension 1 to first mps which does not have one
        li , d = 1 , psi_2[0].shape[1]
        phi = np.zeros((li,d),dtype=complex)
        for k in range(li):
            phi[k,:] = psi_2[0].data
        mps_merged.append(phi)

    # converting into a new MPS

    psi = qtn.MatrixProductState(mps_merged)
    psi /= np.sqrt((psi.H @ psi))

    return psi


##################
# codes for the different initial states of interest

############################################################################################################

# exponentially localized tails concatenated one after the other separated by an empty site
# |1>^n1 |\xi> |0> |1>^n1 |\xi> ...
# |\xi> has length given by n_0 (it is the vacuum before dmrg-x)
# initial state over which perform DMRG-X and obtain a localized tail

n_1 = 5
n_0 = 10
tails_number = 2
s = 1.5

args = {}
args['L'] = n_1 + n_0
args['s'] = s


psi_exp = exp_dressed_kink(tn_H.make_FA_mpo,args, n_1 , n_0 )
psi_vacuum = mps_logic_product_state(bit='0',n=1)

psi = concatenate_MPS(psi_exp,psi_vacuum)
for _ in range(1,tails_number):
    psi = concatenate_MPS(psi,psi_exp)
    psi = concatenate_MPS(psi,psi_vacuum)


mag = tn_observables.magnetization_per_site(psi)

fig, ax = plt.subplots(1,1)
ax.set_yscale('log')

ax.plot(mag)

plt.show()


############################################################################################################
# symmetrized set of domain walls 
# |\Psi> = |DW> + |-DW>
# where |-psi> -> inverted state |psi> (perform reflection wrt center of the chain)
# |DW> = |1>^n1 |0>^n0 |1>^n1 \dots .. |0>^n0

n_1 = 5
n_0 = 5
number_DW = 2

psi_1 = mps_logic_product_state(bit='1',n=n_1)
psi_0 = mps_logic_product_state(bit='0',n=n_0)

DW = concatenate_MPS(psi_1,psi_0)
psi = DW.copy(deep=True)
for _ in range(number_DW):
    psi = concatenate_MPS(psi,DW)

psi_inv = get_inverted_mps(psi)
psi = psi.add_MPS(psi_inv,inplace=False)

mag = tn_observables.magnetization_per_site(psi)
mag_inv = tn_observables.magnetization_per_site(psi_inv)

fig, ax = plt.subplots(1,1)
# ax.set_yscale('log')

ax.plot(mag)
# ax.plot(mag_inv)

plt.show()





# psi = concatenate_MPS(psi_exp,psi_vacuum)
# for _ in range(2):
#     psi = concatenate_MPS(psi,psi_exp)
#     psi = concatenate_MPS(psi,psi_vacuum)


# print(psi)
# mag = tn_observables.magnetization_per_site(psi)

fig, ax = plt.subplots(1,1)
ax.set_yscale('log')

# ax.plot(mag)






def random_domain_walls(L, n1 , n0, number_domains=1,seed=0):

    binary = ''
    np.random.seed(seed)    
    for _ in range(number_domains):
        psi_random = np.random.randint(2, size=n1)
        for item in psi_random:
            binary += f'{item}'
        binary += '0' * n0

    if len(binary) < L:
        binary += '0' * (L -len(binary))
    else:
        binary = binary[:L]
    
    print(f'Initializing product state : {binary}')

    return qtn.MPS_computational_state(binary)










# DEPRECATED
# def exp_dressed_domain_walls(H_name,args, L, n1 , n0 ):

#     print(f'1. We initialize a binary state of {n1+n0} sites:')
#     binary = '1' * n1 + '0' * n0

#     # if n1 < L/2:
#     #     binary = '1' * n1 + '0' * (int(L/2-n1))
#     # else:
#     #     binary = '1' * (int(L/4)) + '0' * (int(L/4))

#     print(f'{binary}')
#     psi_start = qtn.MPS_computational_state(binary)
#     print(f'2. We build the MPO of the Hamiltonian {H_name}.')
#     args_dmrgx = args.copy()
#     args_dmrgx['L'] = n1 + n0
#     H = H_name(args=args_dmrgx)
#     print(f'3. DMRG-X')
#     maxD = 10
#     dmrgX = qtn.DMRGX(H, psi_start, bond_dims=[2*maxD]*L, cutoffs=1e-12)
#     dmrgX.solve(tol=1e-9, verbosity=0)
#     psi_block = dmrgX._k   
#     print(f'4. Building state given by the concatenation of variationally optimized state found via DMRG-x.')

#     number_blocks = int(L/psi_block.L)
#     number_blocks = 1
#     array_MPS = []
#     vacuum_state = [1.,0.]

#     for _ in range(number_blocks):
#         for j in psi_block.sites:
#             if j < psi_block.L - 1 and psi_block.L < L:
#                 array_MPS.append(psi_block[j].data)
#             else:
#                 psi_j = np.zeros((psi_block[-1].shape[0],)+(1,psi_block[-1].shape[1]))
#                 psi_j[:,0,:] = psi_block[-1].data
#                 array_MPS.append(psi_j)

    


#     length_MPS = len(array_MPS)

#     if len(array_MPS) < L:
#         for j in range(L-length_MPS):
#             if j < L- length_MPS-1:
#                 psi_j = np.zeros((array_MPS[-1].shape[1],1,array_MPS[-1].shape[2]))
#                 for k in range(psi_j.shape[0]):
#                     for l in range(psi_j.shape[1]):
#                         psi_j[k,l,:] = vacuum_state
#             else:
#                 psi_j = np.zeros((array_MPS[-1].shape[1],2))
#                 psi_j[0,:] = vacuum_state
#             array_MPS.append(psi_j)
            

#     else:
#         array_MPS = array_MPS[:L]
    
#     return qtn.MatrixProductState(array_MPS)



# to concatenate remember:



n1 = 5
n0 = 10

s = 1.5

args = {}
args['L'] = n1 + n0
args['s'] = s


psi_exp = exp_dressed_kink(tn_H.make_FA_mpo,args, n1 , n0 )
psi_vacuum = vacuum(1)

# psi = concatenate_MPS(psi_exp,psi_vacuum)
# for _ in range(2):
#     psi = concatenate_MPS(psi,psi_exp)
#     psi = concatenate_MPS(psi,psi_vacuum)


# print(psi)
# mag = tn_observables.magnetization_per_site(psi)

fig, ax = plt.subplots(1,1)
ax.set_yscale('log')

# ax.plot(mag)
L = 5
H = Ham.H_east_west(L=L,s=1.5,sparse=False)

psi_k = mpsform_eigenstates_k(H,L=L,d=2,k=1)

index = psi_k[1].inds
k = psi_k.site_ind_id
print(psi_k[1].inds)
psi_0 = psi_k[0]
print(psi_0)
print(psi_0['k0'])
# print(psi_k[0]['k0'].data)
# print(psi_k[0].select(tags='k0').data)
exit(0)
# print(qtn.tensor_core.get_tags(psi_k))
# print(psi_k.site_ind_id)
# p = psi_k.site_ind_id[0]
# print(psi_k[-1][f'{p}{psi_k.L-1}'])

# exit(0)
# print(psi.bond_name)
# for j in psi.sites:
#     print(f'{p}{j} : {psi[j].shape}')
# exit(0)
psi = concatenate_MPS(psi_k,psi_vacuum)

for _ in range(2):
    psi = concatenate_MPS(psi,psi_k)
    psi = concatenate_MPS(psi,psi_vacuum)


mag = tn_observables.magnetization_per_site(psi)


ax.plot(mag)


plt.show()
exit(0)

# def concatenate_MPS


L = 30
n1 = 7
n0 = 5

# psi = domain_wall(L,n1,n0,3)

L = 20
s = 1.5
n1 = 5
n0 = 5



p = qtn.MPS_rand_state(3, 3)
# print(p[0].data)
# print(p[1].data)
# print(p[2].data)

# print(p[2])
# psi = np.zeros((1,)+p[2].shape)
# psi[0,:,:] = p[2].data

# print(p)
# exit(0)


# psi = np.zeros(p[1].shape)
# psi[0,0,:] = [0.,1.]
# psi_array = []
# psi_array.append(p[0].data)
# psi_array.append(p[1].data)
# psi_array.append(psi)
# psi_array.append(p[2].data)

# Psi = qtn.MatrixProductState(psi_array)

# print(psi)
# print(Psi)
# print(p)
# exit(0)
# # print(p[3].data)
# # print(p[4].data)
# print(p.L)
# exit(0)

psi_1 = exp_dressed_domain_walls(tn_H.make_FA_mpo,args, L, n1 , n0 )

# psi_2 = exp_dressed_domain_walls(tn_H.make_FA_mpo,args, L, n1 , n0 )

psi = concatenate_MPS(psi_1,psi_1)
psi /= np.sqrt((psi.H @ psi))
psi = concatenate_MPS(psi,psi)
psi /= np.sqrt((psi.H @ psi))

# array_MPS = []
# vacuum_state = [[1.,0.]]

# array_MPS = np.zeros(L,dtype=object)
# array_MPS[0] = np.array(vacuum_state)
# for j in range(1,L-1):
#     array_MPS[j] = np.array([vacuum_state])

# array_MPS[-1] = np.array(vacuum_state)
# psi = qtn.MatrixProductState(array_MPS)


# print(psi.N)

# psi = random_domain_wall(L, n1 , n0, seed=0, number_domains=3)

# def initialize_state(psi,args):

#     if args['initial_state'] == 'domain_wall':

#         psi = initialize_domain_wall(args['L'],args[''])

#         args.pop('population')
#         theta  = args['theta'] * math.pi
#         phi    = args['phi']   * math.pi
#         if args['d'] == 2:
#             spin_d_coherent_state = spin_coherent_state_two_levels

#         elif args['d'] == 3:
#             spin_d_coherent_state = spin_coherent_state_three_levels

#         else:
#             print(f"Spin {args['d']} not implemented. Exit")
#             return -1

#         for j in range(args['L']):
#             psi_j = initialize_spin_coherent_state(args['L'],j,theta,phi,args['kink'],spin_d_coherent_state)
#             psi.update_single_site(psi_j,j)
    
#     else:
#         args.pop('theta')
#         args.pop('phi')


#     if args['initial_state'] == 'coherent_state':

#         if not 'population' in args:
#             print("Missing 'population' in args. Exit")
#             exit -1

#         n_av = np.array([float(y) for y in args['population']])
#         for k in range(args['d']):
#             args[f'n{k}'] = n_av[k]
#         args.pop('population')

    

#         n_av /= np.sum(n_av)
#         beta  = np.sqrt(n_av)

#         for j in range(args['L']):

#             if args['kink'] == 0:
#                 psi_j = beta

#             if args['kink'] == 1:
#                 if j < args['L']/2:
#                     psi_j = beta
#                 else:
#                     psi_j = np.flip(beta)

#             psi.update_single_site(psi_j,j)

    
#     if args['initial_state'] == 'coherent_state_rolled':

#         if not 'population' in args:
#             print("Missing 'population' in args. Exit")
#             exit -1

#         n_av = np.array([float(y) for y in args['population']])
#         for k in range(args['d']):
#             args[f'n{k}'] = n_av[k]
#         args.pop('population')

#         n_av /= np.sum(n_av)
#         beta  = np.sqrt(n_av)

#         for j in range(args['L']):

#             if args['kink'] == 0:
#                 psi_j = beta

#             if args['kink'] == 1:
#                 if j < args['L']/2:
#                     psi_j = beta
#                 else:
#                     psi_j = np.roll(beta,1)

#             psi.update_single_site(psi_j,j)
            


#     return 1