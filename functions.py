'''
Functions for Math715 Final Project
'''
#Imports
import numpy as np
from scipy.special import roots_legendre, eval_legendre

def Lagrange_basis_functions(P):
    N = P + 1

    #Weights and Roots
    if P == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    
    zeta_interior, _ = roots_legendre(P - 1)

    zeta_gll = np.zeros(N)
    zeta_gll[0]  = -1.0
    zeta_gll[-1] =  1.0
    zeta_gll[1:-1] = zeta_interior

    L_P_at_zeta = eval_legendre(P, zeta_gll)

    w_gll = 2.0 / (P * (P + 1) * (L_P_at_zeta ** 2))

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