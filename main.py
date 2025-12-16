import functions as fn
import matplotlib.pyplot as plt
import numpy as np

#Parameters
P = 5 #Polynomail Order
ne = 4 #Number of Elements
L = 1 #Length of domain
u_l = 0 #Left Dirichlet BC
u_r = 0 #Right Dirichlet BC
t_end = 1
dt = 1e-5

def wave_solution(P,ne,L,u_l,u_r,t_end,dt):
    #Constructing Basis Functions
    roots, weights, zeta_c, bf = fn.Lagrange_basis_functions(P)

    #Constructing Mesh
    dx, x, elements, nodes_total, free_nodes = fn.Mesh_1D_DD(ne,L,roots)

    #Constructing Derivative Matrix
    D_matrix = fn.Lagrange_Derivative_Matrix(P,roots)

    #Constructing Stiffness Matrix
    K_matrix = fn.Stiffness_Matrix(P,D_matrix,weights,dx)
    K_global, K_global_free_nodes = fn.Global_Sum(P,K_matrix,ne,free_nodes)

    #Constructing Mass Matrix
    M_matrix = fn.Mass_Matrix(P,weights,dx)
    M_global, M_global_free_nodes = fn.Global_Sum(P,M_matrix,ne,free_nodes)

    #Constructing Load Vector
    F = fn.load_vec(K_global,u_l,u_r)

    #Timestepping
    num_t = int(np.round(t_end / dt)) + 1
    dt = t_end / (num_t - 1)
    t = np.arange(num_t) * dt
    len_t = len(t)
    
    Nx = len(nodes_total)
    u = np.zeros((num_t,Nx))
    u_init = np.sin(np.pi*nodes_total)
    u[0,:] = u_init

    u0 = u[0,1:-1].copy()
    v0 = np.zeros_like(u0)
    a0 = np.linalg.solve(M_global_free_nodes, -K_global_free_nodes @ u0)
    u[1,1:-1] = u0 + dt*v0 + 0.5*dt**2*a0

    u[:,0] = u_l
    u[:,-1] = u_r

    M_inv = 1/np.diag(M_global_free_nodes)
    for n in range(1,len_t-1):
        a = -K_global_free_nodes @ u[n,1:-1] 
        b = M_inv * a
        u[n+1,1:-1] = dt**2*b + 2*u[n,1:-1] - u[n-1,1:-1]

    #Exact
    u_exact = np.zeros((num_t,Nx))

    for i in range(len_t):
        for j in range(Nx):
            u_exact[i,j] = np.cos(np.pi * t[i])*np.sin(np.pi * nodes_total[j])
    
    return weights, dx, nodes_total, u, u_exact

w, dx, x, u, u_exact = wave_solution(P,ne,L,u_l,u_r,t_end,dt)

ani = fn.animate_solution(u, x, u_exact, x, dt=dt, fps=60, speed=0.5)
fn.plot_solution_snapshots(u, x, t_end=1.0, times=[0.0, 0.25, 0.5, 0.75])

#ani.save("solution.mp4", writer="ffmpeg", fps=30)

