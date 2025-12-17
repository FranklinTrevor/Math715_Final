import numpy as np
'''
Stiffness Matrix
'''

def Stiffness(T, basisFunction, k):
    F = T["FNodePtrs"]
    Nf = len(F)
    K = np.zeros((Nf, Nf))

    bfn = basisFunction
    len_E = len(T["Elements"])
    k_l = np.zeros((len_E, 2, 2))

    for i in range(len_E):
        n1, n2 = T["Elements"][i]
        n = [n1, n2]

        k1 = int(n[0] - 1)
        k2 = int(n[1] - 1)
        x1 = T["Nodes"][k1]
        x2 = T["Nodes"][k2]

        A = x2 - x1

        b0 = bfn[i, 1, 0]
        b1 = bfn[i, 1, 1]

        k_l[i, 0, 0] = A * k * (b0 * b0)
        k_l[i, 1, 1] = A * k * (b1 * b1) 
        k_l[i, 0, 1] = A * k * (b0 * b1)
        k_l[i, 1, 0] = k_l[i, 0, 1]

        for jj in range(2):
            for kk in range(jj, 2):
                Na = n[jj]
                Nb = n[kk]
                if (Na in F) and (Nb in F):
                    Ni_idx = np.where(F == Na)[0]
                    Nj_idx = np.where(F == Nb)[0]
                    val = k_l[i, jj, kk]
                    K[Ni_idx, Nj_idx] += val
                    if Nj_idx != Ni_idx:
                        K[Nj_idx, Ni_idx] += val  

    return K, k_l





