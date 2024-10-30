import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.linalg import det
import matplotlib.pyplot as plt

# Setup - two Rydbergs atoms interacting via van-der vars interactions.
# We initialize in a certain configuration and we make them evolve.
# Time evolution given by a master equation with fixed rates.
# The vector of probabilities is P(\vec{\Sigma}) = [ P(00) , P(01) , P(10) , P(11) ]
# d P(\vec{\Sigma}) / dt = M P(\vec{\Sigma}),
# with M a 4 x 4 matrix
# units are in MHz
# times are in microseconds

def cm2inch(value):
    return value/2.54

plt.rc('xtick' , labelsize=10)    # fontsize of the tick labels
plt.rc('ytick' , labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize =10)    # legend fontsize
plt.rc('axes'  , titlesize=10)     # fontsize of the axes set_title
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })
width_images = 7


# a.u -> MHz
conversion_energy = 1E-6 * (1/4.43982167204992E-24)

# a.u -> cm
conversion_length = 1 / 188972612.54535

# Delta_bare_list = np.linspace(-150,30,num=1000)
Delta_bare_list = [0]
Delta_bare_list = [-100,-50,-25,-10][::-1]

eta = 3.13 

# distance of the atoms in cm converted into atomic units
r = 5 * 1E-4 / conversion_length


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

# n0 = 48
delta0_n0   = -0.0378
mu0_n0      =  843800
n0 = 48 - eta

def Delta(sigma, U):
    return Delta_bare + sigma * U


# def gamma(Delta,sigma):

#     gamma_up = ( Gamma * (omega / Omega)**2 ) / ( 2 * ( 1 - 4 * (Delta / Omega)**2 )**2 )
#     rho_ee_infty = Omega**2 * (Omega**2 + omega**2) / ( (Omega**2 + omega**2)**2 + 4 * Delta**2 * (Gamma**2 + 2 * Omega**2) )
#     print(rho_ee_infty)
#     exit(0)
#     if sigma == 0:
#         return gamma_up
#     else:
#         return gamma_up * ( 1 - rho_ee_infty ) / rho_ee_infty


def gamma(Delta,sigma):
    Omega2 = Omega**2
    omega2 = omega**2
    Delta2 = Delta**2
    Gamma2 = Gamma**2
    a0 = (Omega2 + omega2 ) * ( ( omega2 - 2 * Omega2 )**2 + 2 * Gamma2 * (Omega2 + omega2) )
    a2 = 8 * (Gamma2**2 - 4 * (Omega2**2)) + 4 * omega2 * ( Gamma2 + 4 * Omega2 ) + 8 * (omega2**2)
    a4 = 32 * (Gamma2 + 2 * Omega2)

    rho_ee_infty = Omega2 * (Omega2 + omega2) / ( (Omega2 + omega2)**2 + 4 * Delta2 * (Gamma2 + 2 * Omega2) )
    gamma_up = 2 * Gamma * omega2 * Omega2 * (Omega2 + omega2)
    gamma_up /= ( a0 + a2 * Delta2 + a4 * (Delta2**2) )
    print(f'rho : {rho_ee_infty}')
    if sigma == 0:
        print(f'Sigma = {sigma}')
        return gamma_up 
    else:
        print( ( 1 - rho_ee_infty ) / rho_ee_infty  )
        return gamma_up * ( 1 - rho_ee_infty ) / rho_ee_infty 


def Uij(nstar):
    delta0 = delta0_n0 * (( n0 / nstar )**3)
    mu     = mu0_n0 * (( nstar / n0 )**4)
    V = mu / (r**3)
    return 0.5 * ( delta0 + np.sqrt( delta0**2 + 4 * (V**2) ) )



def beta_function(t, state, M):
    return M @ state

P00 = 1.
P01 = 0.
P10 = 0.
P11 = 0.
P0 = np.array([P00, P01, P10, P11])
P0 = P0 / np.sum(P0)

T = 1000
t_eval = [T]
fig, ax = plt.subplots(1,1,figsize=(cm2inch(width_images), cm2inch(7.6)))

n_list = [j for j in range(40,80)]
# n_list = np.linspace(40,80,num=1000)
# n_list = [100,200]
Pt_delta = []
Pexc_delta = []
for Delta_bare in Delta_bare_list:
    Pt = []
    Pt_exc = []
    U_list = [ ]
    Pttot = []

    for n in n_list:
        nstar = n - eta


        U = Uij(nstar)
        U *= conversion_energy
        U_list.append(U)

        Delta0 = Delta(0,U)
        Delta1 = Delta(1,U)

        print(Delta0)
        print(Delta1)

        # the rates depends on the state of the spin and its surrounding, so we have to distinguish
        # four different states, corresponding to configurations 00, 01, 10, 11

        gamma00 = gamma(Delta0,0)
        gamma01 = gamma(Delta0,1)
        gamma10 = gamma(Delta1,0)
        gamma11 = gamma(Delta1,1)


        M1 = [-(2*gamma00) ,  gamma01  , gamma01   , 0]
        M2 = [ gamma00  , -(gamma01+gamma10) , 0        , gamma11]
        M3 = [ gamma00 ,  0        , -(gamma01+gamma10) , gamma11]
        M4 = [0        ,  gamma10   , gamma10   , - 2*gamma11]

        # gamma00 = 2 * gamma0
        # gamma01 = gamma0 + gamma1
        # gamma10 = gamma01
        # gamma11 = 2 * gamma1


        # M1 = [-gamma00 ,  gamma1  , gamma1   , 0]
        # M2 = [ gamma0  , -gamma01 , 0        , gamma1]
        # M3 = [ gamma0  ,  0        , -gamma10 , gamma1]
        # M4 = [0        ,  gamma0   , gamma0   , - gamma11]


        M = np.array([M1,M2,M3,M4])
        print(M)
        # print(np.sum(M,axis=0))
        # print(det(M))
        # exit(0)
        # M = np.transpose(M)
        # print(M)

        # print(P0)
        sol =  solve_ivp(beta_function, t_span=[0, T], y0=P0,args=(M,),t_eval=t_eval)
        print(f'Status : {sol.status}')
        t = sol.t
        # print(sol.y[1:][-1])
        Pt.append(sol.y[-1])
        # print()
        # print(Pt)
        Pt_exc.append(np.sum(np.transpose(sol.y)[-1][1:]))
        Pttot.append(np.sum(np.transpose(sol.y)[-1]))
        # exit(0)
    Pt_delta.append(Pt)
    # print(U_list)
    # print(Pttot)
    # print(Pt_exc)
    Pexc_delta.append(Pt_exc)
    # ax.plot(n_list,U_list)
    Pt = np.transpose(np.array(Pt))
    for idx_t, t in enumerate(t_eval):
        # ax.plot(n_list,Pt[idx_t],label=f'$t = {t}\mu s$')
        ax.plot(n_list,Pt[idx_t],label=f'$\Delta = {Delta_bare}$')

    # ax[0].plot(n_list,Pt_exc,label=f'$\Delta = {Delta_bare}$')

# ax.set_title(f'$p(rr)$ $\Delta$(MHz)$={Delta_bare}$')
ax.set_title(f'$p(rr)$ $T={T}\mu s$')

ax.set_xlabel('$n$ (principal quantum number)')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
save_folder = '/home/ricval/Documenti/east_rydbergs/notes/'
# fig.savefig(f'{save_folder}two_rydberg_atom_mysim_Omega{Omega:.2f}_omega{omega:.2f}_Gamma{Gamma:.2f}_Delta{Delta_bare}.pdf')
fig.savefig(f'{save_folder}two_rydberg_atom_mysim_Omega{Omega:.2f}_omega{omega:.2f}_Gamma{Gamma:.2f}_T{T}.pdf')

# ax.plot(Delta_bare_list,Pt_delta)
# ax.plot(Delta_bare_list,Pexc_delta)
# ax.set_yscale('log')
plt.show()