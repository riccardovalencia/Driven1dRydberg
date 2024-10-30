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

    
linewidth = 1

width_images = 7


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
    a0 = (Omega2 + omega2 ) * ( ( omega2 - 2 * Omega2 )**2 + 2 * Gamma2 * (Omega2 + omega2) )
    a2 = 8 * (Gamma2**2 - 4 * (Omega2**2)) + 4 * omega2 * ( Gamma2 + 4 * Omega2 ) + 8 * (omega2**2)
    a4 = 32 * (Gamma2 + 2 * Omega2)

    # gamma_up = ( Gamma * (omega / Omega)**2 ) / ( 2 * ( 1 - 4 * (Delta / Omega)**2 )**22 )
    rho_ee_infty = Omega2 * (Omega2 + omega2) / ( (Omega2 + omega2)**2 + 4 * Delta2 * (Gamma2 + 2 * Omega2) )
    # rho_ee_infty = 1 / ( 1 + 8 * (Delta/Omega)**2 )
    
    gamma_up = 2 * Gamma * omega2 * Omega2 * (Omega2 + omega2)
    gamma_up /= ( a0 + a2 * Delta2 + a4 * (Delta2**2) )

    if sigma == 0:
        return gamma_up , rho_ee_infty
    else:
        return gamma_up * ( 1 - rho_ee_infty ) / rho_ee_infty , rho_ee_infty


def beta_function(t, y, M):
    # print(t)
    return M @ y

P0 = 1.
P1 = 0.
P_t0 = np.array([P0, P1])
P_t0 /= np.sum(P_t0)

T = 1000
t_eval = [0.3,1.,2.,10,T]
n_measures = 3

fig, ax = plt.subplots(1,1,figsize=(cm2inch(width_images), cm2inch(7.6)))

Pt = []
Pinfty = []
Delta_bare_list = np.linspace(-30,30,num=1000)
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
    sol =  solve_ivp(beta_function, t_span=[0, T], y0=P_t0,args=(M,),t_eval=t_eval,method='RK23')
    print(f'Status : {sol.status}')
    t = sol.t
    Pt.append(sol.y[1])
    Pinfty.append(rho_ee_infty)
    
# # print(Pt)
Pt = np.transpose(np.array(Pt))
for idx_t, t in enumerate(t_eval):
    ax.plot(Delta_bare_list,Pt[idx_t],label=f'$t = {t}\mu s$')
ax.set_xlabel('$\Delta$')
ax.set_title('$p_r$')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
save_folder = '/home/ricval/Documenti/east_rydbergs/notes/'
fig.savefig(f'{save_folder}single_rydberg_atom_mysim_Omega{Omega:.2f}_omega{omega:.2f}_Gamma{Gamma:.2f}.pdf')
# ax.plot(Delta_bare_list,Pinfty,linestyle='--',color='black')
# # rho_ee_infty = Omega**2 * (Omega**2 + omega**2) / ( (Omega**2 + omega**2)**2 + 4 * Delta_bare**2 * (Gamma**2 + 2 * Omega**2) )
# # ax.hlines(y=rho_ee_infty,xmin=0,xmax=time[-1],linestyle='--',color='black')
plt.show() 