'''
Data Frame For 1D FEM Mesh
'''
import numpy as np

def Mesh1D(Lx,Nx):
    T = {} #Initializing T Dictionary

    #T.Nodes
    loc_x = np.linspace(0,Lx,Nx+1)
    num_x = len(loc_x)

    T["Nodes"] = loc_x

    #T.CNodePtrs
    node_indicies = np.zeros(Nx+1)
    len_ni = len(node_indicies)
    num = 1
    for i in range(len_ni):
        node_indicies[i] = num
        num = num + 1

    cnodeptrs = [node_indicies[0], node_indicies[-1]]

    #T.FNodePtrs
    fnodeptrs = node_indicies[1:-1]

    T["FNodePtrs"] = fnodeptrs

    #T.Elements

    elements = np.zeros((Nx,2))
    len_elems = len(elements)

    for i in range(len_elems):
        elements[i,0] = i+1
        elements[i,1] = i+2

    T["Elements"] = elements
    
    return T

