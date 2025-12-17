import numpy as np
import matplotlib.pyplot as plt
'''
Mass Matrix
'''

def Mass(T, basisFunction):
    F = T["FNodePtrs"]
    Nf = len(F)
    M = np.zeros((Nf, Nf))

    bfn = basisFunction
    len_E = len(T["Elements"])
    m_l = np.zeros((len_E,2,2))

    for i in range(len_E):
        n1, n2 = T["Elements"][i]
        n = [n1, n2]

        k1 = int(n[0] - 1)
        k2 = int(n[1] - 1)
        x1 = T["Nodes"][k1]
        x2 = T["Nodes"][k2]

        a1 = bfn[i, 0, 0]
        b1 = bfn[i, 1, 0]
        a2 = bfn[i, 0, 1]
        b2 = bfn[i, 1, 1]

        val_a = (a1*a2)*x2 + ((1/2)*a1*b2)*x2**2 + ((1/2)*a2*b1)*x2**2 + ((1/3)*b1*b2)*x2**3
        val_b = (a1*a2)*x1 + ((1/2)*a1*b2)*x1**2 + ((1/2)*a2*b1)*x1**2 + ((1/3)*b1*b2)*x1**3
        m_l[i,0,1] = val_a-val_b
        m_l[i,1,0] = m_l[i,0,1]

        val_a = (a1*a1)*x2 + ((1/2)*a1*b1)*x2**2 + ((1/2)*a1*b1)*x2**2 + ((1/3)*b1*b1)*x2**3
        val_b = (a1*a1)*x1 + ((1/2)*a1*b1)*x1**2 + ((1/2)*a1*b1)*x1**2 + ((1/3)*b1*b1)*x1**3
        m_l[i,0,0] = val_a - val_b

        val_a = (a2*a2)*x2 + ((1/2)*a2*b2)*x2**2 + ((1/2)*a2*b2)*x2**2 + ((1/3)*b2*b2)*x2**3
        val_b = (a2*a2)*x1 + ((1/2)*a2*b2)*x1**2 + ((1/2)*a2*b2)*x1**2 + ((1/3)*b2*b2)*x1**3
        m_l[i,1,1] = val_a - val_b

        for jj in range(2):
            for kk in range(jj, 2):
                Na = n[jj]
                Nb = n[kk]
                if (Na in F) and (Nb in F):
                    Ni_idx = np.where(F == Na)[0]
                    Nj_idx = np.where(F == Nb)[0]
                    val = m_l[i, jj, kk]
                    M[Ni_idx, Nj_idx] += val
                    if Nj_idx != Ni_idx:
                        M[Nj_idx, Ni_idx] += val  

    return M, m_l