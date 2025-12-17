import numpy as np

def basis_fn(T):

    numElements = len(T["Elements"])

    basisFunctions = np.zeros((numElements,2,2))
    for i in range(numElements):
        n1, n2 = T["Elements"][i]

        n = [n1, n2]

        k1 = int(n[0]-1)
        k2 = int(n[1]-1)

        x1 = T["Nodes"][k1]
        x2 = T["Nodes"][k2]

        coeff_array = np.array([[1, x1],
                                [1, x2]])
        
        coeff_array_inv = np.linalg.inv(coeff_array)
        
        basisFunctions[i,:,:] = coeff_array_inv @ np.array([[1, 0],
                                                            [0, 1]])
        
    return basisFunctions
    





 