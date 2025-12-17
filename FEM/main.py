import mesh as Mesh
import stiffness as Stiffness
import mass as Mass
import load as Load
import basisFunctions as BF
import utilities as util
import numpy as np
import matplotlib.pyplot as plt
import time

'''
Case 1
nx = 4
'''
def FEM_solve(nx_in, dt_in, t_end_in):
    #Building Mesh
    print('Loading Mesh...')
    T = Mesh.Mesh1D(1,nx_in)

    #Constructing Basis Function Coefficients
    print('Constructing Basis Function Coefficients...')
    bf = BF.basis_fn(T)

    #Build Stiffness Matrix
    print('Builing Stiffness Matrix K...')
    K, k_l = Stiffness.Stiffness(T,bf,1)

    #Build Mass Matrix
    print('Builing Mass Matrix M...')
    M, m_l = Mass.Mass(T,bf)

    #Build F
    print('Building Load Vector F...')
    F, dir_val = Load.load_vector(T, bf)

    def solve(T_in,M_in,K_in,dt,t_end):
        #Solving for nodal points
        print('Timestepping for nodal values')

        x = T_in["Nodes"][:]
        len_x = len(x)
        U0 = np.sin(np.pi*x)
        dx = T_in["Nodes"][1] - T_in["Nodes"][0]

        num_t = int(t_end/dt) + 1 
        t = np.linspace(0.0, t_end, num_t)
        dt = t[1] - t[0]
        len_t = len(t)

        u = np.zeros((len_t,len_x))
        u[0,:] = U0
        u0 = u[0,1:-1].copy()
        v0 = np.zeros_like(u0)
        a0 = np.linalg.solve(M_in, -K_in @ u0)
        u[1,1:-1] = u0 + dt*v0 + 0.5*dt**2*a0

        u[:,0] = 0
        u[:,-1] = 0

        M_inv = np.linalg.inv(M_in)
        for n in range(1,len_t-1):
            a = -K_in @ u[n,1:-1] 
            b = M_inv @ a
            u[n+1,1:-1] = dt**2*b + 2*u[n,1:-1] - u[n-1,1:-1]
        
        return dx, x, t, u
    dx_out, x_out, t_out, u_out = solve(T,M,K,dt_in,t_end_in)

    return dx_out, x_out, t_out, u_out

def u_exact_fn(t,x):
    len_t = len(t)
    len_x = len(x)
    u_exact = np.zeros((len_t,len_x))

    for i in range(len_t):
        for j in range(len_x):
            u_exact[i,j] = np.cos(np.pi * t[i])*np.sin(np.pi * x[j])
    
    return u_exact

def error_plot(hs,Es,axis=0):
    logx = np.log10(hs)
    logy = np.log10(Es)

    slope, intercept = np.polyfit(logx, logy, 1)

    print(f'Slope of dx convergence plot = {slope}')
    x_fit = np.linspace(hs[0], hs[-1], 10)
    y_fit = 10**(intercept) * x_fit**slope

    plt.loglog(hs, Es, color = 'black', marker='o', linewidth = 2)
    plt.loglog(x_fit, y_fit, '--', color = 'red', label = f'Slope = {round(slope,3)}')
    plt.ylabel('L2 Error', fontsize = 28)
    if axis == 0:
        plt.xlabel(r'$\Delta h$', fontsize = 28)
    if axis == 1:
        plt.xlabel(r'$\Delta t$', fontsize = 28)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 24)
    plt.grid()
    plt.show()

def plot_solution_snapshots(u, x, t_end=1.0, times=[0.0, 0.25, 0.5, 0.75]):
    nt, nx = u.shape
    t = np.linspace(0.0, t_end, nt)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    for ax, t_plot in zip(axs, times):
        idx = np.argmin(np.abs(t - t_plot))

        ax.plot(x, u[idx], color='black', lw=2, label='FEM solution')
        ax.scatter(x, u[idx], color='red', s=25, zorder=3, label='Nodes')

        ax.set_title(rf"$t = {t[idx]:.2f}$ s", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    axs[2].set_xlabel("x", fontsize=12)
    axs[3].set_xlabel("x", fontsize=12)
    axs[0].set_ylabel("u(x,t)", fontsize=12)
    axs[2].set_ylabel("u(x,t)", fontsize=12)

    axs[0].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

dx_r1, x_r1, t_r1, U_r1 = FEM_solve(10, 1e-5, 1)
uexact = u_exact_fn(t_r1,x_r1)
ani = util.animate_solution(U_r1, x_r1, uexact, x_r1, dt=1e-4, fps=60, speed=0.5)
plot_solution_snapshots(U_r1, x_r1, t_end=1.0, times=[0.0, 0.25, 0.5, 0.75])

##############################
### Error Analysis (Space) ###
##############################
ne_test = [10, 100, 500, 1000]

dt_ne = 1e-5
dh = []

error = []
times_ne = []
for item in ne_test:
    start_time = time.perf_counter()
    dx, x, t, U = FEM_solve(item, 1e-5, 0.5)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    uexact = u_exact_fn(t,x)

    dh.append(dx)
    times_ne.append(elapsed_time)
    e = U[-1,:] - uexact[-1,:]
    error_val = np.sqrt(dx) * np.linalg.norm(e[1:-1], 2)
    error.append(error_val)
    print('finished')

error_plot(dh,error,axis=0)


#######################
### Simulation Time ###
#######################

plt.figure(figsize = (12,8))
plt.plot(dh, times_ne, color = 'black', linewidth = 3, marker = 'o')
plt.xlabel(r'$\Delta h$', fontsize = 18)
plt.ylabel('runtime (s)', fontsize = 18)
plt.grid()
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xscale('log')
plt.show()






