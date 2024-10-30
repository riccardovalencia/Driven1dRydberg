from quimb import *
import numpy as np

def local_obs(psi,O,site,dims,sparse=False):
    Oj = ikron(O,dims,inds=site,sparse=sparse)
    return expec(psi,Oj)


def correlations(psi,O,i,j,dims,sparse):

    Oi = ikron(O,dims,inds=i,sparse=sparse)
    Oj = ikron(O,dims,inds=j,sparse=sparse)

    if i != j:
        OiOj = pkron(O&O,dims,inds = [i,j],sparse=sparse)
    else:
        OiOj = ikron(O@O,dims,inds = i,sparse=sparse)


    return expec(psi,OiOj) - expec(psi,Oi) * expec(psi,Oj)
