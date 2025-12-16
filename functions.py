'''
Functions for Math715 Final Project
'''
#Imports
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.special import roots_legendre, eval_legendre
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def gll_points_weights(P):
    N = P + 1
    if P == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    dLp = Legendre.basis(P).deriv()
    xi_int = np.sort(dLp.roots())

    xi = np.zeros(N)
    xi[0], xi[-1] = -1.0, 1.0
    xi[1:-1] = xi_int

    Lp = eval_legendre(P, xi)
    w = 2.0 / (P * (P + 1) * (Lp**2))
    return xi, w

def Lagrange_basis_functions(P):
    N = P + 1

    #Weights and Roots
    if P == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    
    zeta_gll, w_gll = gll_points_weights(P)

    #Basis Functions
    zeta = np.linspace(-1,1,100)
    len_zeta = len(zeta)
    
    bf_vec = np.zeros((N,len_zeta))

    for j in range(N):
        prod = 1
        for i in range(N):
            if i != j:
                A = (zeta-zeta_gll[i])/(zeta_gll[j]-zeta_gll[i])
                prod = prod*A
        bf_vec[j,:] = prod

    return zeta_gll, w_gll, zeta, bf_vec

def Lagrange_Derivative_Matrix(P,roots):
    N = P+1
    D = np.zeros((N,N))

    for j in range(N):
        for i in range(N):
            D_sum = 0
            D_mult = 1
            if j == i:
                for k in range(N):
                    if k != j:
                        D_sum = D_sum + 1/(roots[j]-roots[k])
                D[j,i] = D_sum
            else:
                for k in range(N):
                    if k !=i and k != j:
                        D_mult = D_mult*(roots[i]-roots[k])/(roots[j]-roots[k])
                D[j,i] = 1/(roots[j]-roots[i])*D_mult

    return D

def Stiffness_Matrix(P,D,weights,dh):
    N = P + 1
    K = np.zeros((N,N))

    for j in range(N):
        for i in range(N):
            k_sum = 0
            for k in range(N):
                k_sum = k_sum + weights[k]*D[j,k]*D[i,k]*(2/dh)**2*(dh/2)
            K[j,i] = k_sum
    return K

def Global_Sum(P, M, ne, free_nodes):
    N = P + 1
    ndof = ne * (N - 1) + 1

    M_full = np.zeros((ndof, ndof))

    for e in range(ne):
        for a in range(N):
            A = e * (N - 1) + a
            for b in range(N):
                B = e * (N - 1) + b
                M_full[A, B] += M[a, b]

    free_nodes = np.asarray(free_nodes, dtype=int)
    M_reduced = M_full[np.ix_(free_nodes, free_nodes)]

    return M_full, M_reduced


def Mesh_1D_DD(ne, L, roots):
    P = len(roots) - 1
    dx = L / ne
    elements = np.array([[i*dx, (i+1)*dx] for i in range(ne)])

    nodes_total = []
    for e in range(ne):
        xL, xR = elements[e]
        x_e = 0.5*(roots + 1.0)*dx + xL

        if e == 0:
            nodes_total.extend(x_e)
        else:
            nodes_total.extend(x_e[1:])

    nodes_total = np.array(nodes_total)

    ndof = ne * P + 1
    free_nodes = np.arange(1, ndof - 1)

    return dx, np.linspace(0, L, ne+1), elements, nodes_total, free_nodes

def load_vec(K_global,u_l,u_r):
    K_l = K_global[1:-1,0]
    K_r = K_global[1:-1,-1]

    K_load = np.column_stack((K_l, K_r))
    u_b = np.array([u_l, u_r])

    F = -K_load@u_b

    return F

def Mass_Matrix(P,weights,dh):
    N = P + 1
    M = np.zeros((N,N))

    for j in range(N):
        for i in range(N):
            if i == j:
                M[j,i] = weights[j]*(dh/2)
            else:
                M[j,i] = 0

    return M

def animate_solution(u, x, u_exact, x_exact, dt=1.0, fps=60, speed=1.0):
    nt, nx = u.shape
    t_end = (nt - 1) * dt

    # How many frames we *can* show during playback
    n_show = max(2, int(np.ceil((t_end / speed) * fps)))

    # Pick evenly-spaced frame indices to show
    frame_idx = np.linspace(0, nt - 1, n_show).astype(int)

    # Interval in ms to make total duration match (t_end / speed)
    interval = 1000 * (t_end / speed) / n_show

    fig, ax = plt.subplots()
    line,  = ax.plot(x, u[0], lw=2, color='black', label='Numerical')
    line2, = ax.plot(x_exact, u_exact[0], lw=2, color='red', linestyle='--', label='Exact')

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u(x,t)", fontsize=12)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.min(u), np.max(u))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    def update(k):
        n = frame_idx[k]
        line.set_ydata(u[n])
        line2.set_ydata(u_exact[n])
        ax.set_title(f"t = {n*dt:.4f} s")
        return line, line2

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_idx),
        interval=interval,
        blit=False,
        repeat=False
    )

    plt.show()
    return ani

def plot_solution_snapshots(u, x, t_end=1.0, times=[0.0, 0.25, 0.5, 0.75]):
    nt, nx = u.shape
    t = np.linspace(0.0, t_end, nt)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    for ax, t_plot in zip(axs, times):
        idx = np.argmin(np.abs(t - t_plot))

        ax.plot(x, u[idx], color='black', lw=2, label='SEM solution')
        ax.scatter(x, u[idx], color='red', s=25, zorder=3, label='GLL nodes')

        ax.set_title(rf"$t = {t[idx]:.2f}$ s", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    axs[2].set_xlabel("x", fontsize=12)
    axs[3].set_xlabel("x", fontsize=12)
    axs[0].set_ylabel("u(x,t)", fontsize=12)
    axs[2].set_ylabel("u(x,t)", fontsize=12)

    axs[0].legend(fontsize=10)

    plt.tight_layout()
    plt.show()
