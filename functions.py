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