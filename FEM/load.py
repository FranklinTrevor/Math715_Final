import numpy as np
'''
Load Vector
'''

def load_vector(T,basisFunctions):
    Nf = len(T["FNodePtrs"])
    F = np.zeros(Nf)
    len_E = len(T["Elements"])

    #Computing value at nodes
    nodes_len = len(T["Nodes"])
    dir_val = np.zeros((nodes_len))
    dir_val[0] = 0
    dir_val[-1] = 1

    #Solving for F
    bfn = basisFunctions
    for i in range(len_E):
        n1, n2 = T["Elements"][i]
        n = [n1, n2]

        k1 = int(n[0] - 1)
        k2 = int(n[1] - 1)
        x1 = T["Nodes"][k1]
        x2 = T["Nodes"][k2]

        A = x2 - x1
        
        n1_basis = np.array([bfn[i, 1, 0]])
        n2_basis = np.array([bfn[i, 1, 1]])
        basis = [n1_basis, n2_basis]

        omega1 = dir_val[int(n[0] - 1)]
        omega2 = dir_val[int(n[1] - 1)]

        gradG = omega1*n1_basis + omega2*n2_basis

        for local_idx, node_id in enumerate(n):
            if node_id in T["FNodePtrs"]:
                Fi_loc = - A * float(np.dot(gradG, basis[local_idx]))
                Ni = np.where(T["FNodePtrs"] == node_id)[0]
                F[Ni] += Fi_loc

    return F, dir_val



