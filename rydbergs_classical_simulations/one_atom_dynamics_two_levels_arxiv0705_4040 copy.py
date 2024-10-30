import numpy as np
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt

# Setup - two Rydbergs atoms interacting via van-der vars interactions.
# We initialize in a certain configuration and we make them evolve.
# Time evolution given by a master equation with fixed rates.
# The vector of probabilities is P(\vec{\Sigma}) = [ P(00) , P(01) , P(10) , P(11) ]
# d P(\vec{\Sigma}) / dt = M P(\vec{\Sigma}),
# with M a 4 x 4 matrix
# units are in MHz
# times are in microseconds



first_set = {
    'Omega' : 4.,
    'omega' : 0.2,
    'Gamma' : 6.,
}

second_set = {
    'Omega' : 22.1,
    'omega' : 0.8,
    'Gamma' : 6.,
}

Omega = first_set['Omega']
omega = first_set['omega']
Gamma = first_set['Gamma']

def gamma(Delta,sigma):
    Omega2 = Omega**2
    omega2 = omega**2
    Delta2 = Delta**2
    Gamma2 = Gamma**2

    gamma_ = ( 2 * Gamma * Omega2) / ( Gamma2 + 4 * Delta2)
    return gamma_ , 0


def beta_function(t, y, M):
    # print(t)
    return M @ y

P0 = 1.
P1 = 0.
P_t0 = np.array([P0, P1])
P_t0 /= np.sum(P_t0)

T = 2
n_measures = 3

fig, ax = plt.subplots(1,1)

Pt = []
Pinfty = []
Delta_bare_list = np.linspace(-100,100,num=1000)
# Delta_bare_list = [-10,0,10]
# Delta_bare_list = Delta_bare_list[1:]
for Delta_bare in Delta_bare_list:
    print(Delta_bare)

    gamma0 , rho_ee_infty = gamma(Delta_bare,0)
    gamma1 , _ = gamma(Delta_bare,1)

    M1 = [-gamma0 ,   gamma1 ]
    M2 = [ gamma0  , -gamma1 ]
    M = np.array([M1,M2])
    print(M)
    sol =  solve_ivp(beta_function, t_span=[0, T], y0=P_t0,args=(M,),t_eval=[T])
    print(f'Status : {sol.status}')
    t = sol.t
    Pt.append(sol.y[1])
    Pinfty.append(rho_ee_infty)
    
# # print(Pt)
ax.plot(Delta_bare_list,Pt,linestyle='-')
# ax.plot(Delta_bare_list,Pinfty,linestyle='--',color='black')
# # rho_ee_infty = Omega**2 * (Omega**2 + omega**2) / ( (Omega**2 + omega**2)**2 + 4 * Delta_bare**2 * (Gamma**2 + 2 * Omega**2) )
# # ax.hlines(y=rho_ee_infty,xmin=0,xmax=time[-1],linestyle='--',color='black')
plt.show() 