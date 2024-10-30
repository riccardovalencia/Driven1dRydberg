import quimb.tensor as qtn
from quimb import *


def spin_states(args):

    if args['initial_state'] == 'kink':
        psi =  '1' * args['n'] + '0' * (args['L']-args['n'])
        return qtn.MPS_computational_state(psi)

    else:
        print(f"Initial state {args['initial_state']} still not initialized")
        return -1
    

def computational_spin_state(string,sparse=False):

    psi1 = qu([0,1], qtype='ket', sparse=sparse)
    psi0 = qu([1,0], qtype='ket', sparse=sparse)

    n_exc = 0
    x_exc = []
    if string[0] == 1:
        psi = psi1
        n_exc += 1
        x_exc += [0]
    else:
        psi = psi0

    for idx , n in enumerate(string[1:]):
        if n==1:
            psi = psi & psi1
            n_exc += 1
            x_exc += [idx+1]
        else:
            psi = psi & psi0
    
    return psi , [x_exc, n_exc] 