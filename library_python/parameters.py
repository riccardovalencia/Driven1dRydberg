import numpy as np
from numpy.linalg import norm

# return distance given strenght of potential and its power-law decay

def distance(V,alpha):
    return (1/V)**(1/alpha)


# return potential given distance and power-law decay
def V_power_law(d : float,alpha):
    return 1/(d**alpha)

# def V_power_law(r1 : np.array, r2 : np.array,alpha):
#     d = norm(r1-r2)
#     return 1/(d**alpha)

def V_power_law(rj : np.array ,alpha):
    V = []
    N = len(rj)
    for k in range(N-1):
        d = norm(rj[k]-rj[k+1])
        V += [1/(d**alpha)]
    return V
