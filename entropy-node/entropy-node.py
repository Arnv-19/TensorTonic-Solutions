import numpy as np

def entropy_node(Y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    Y = np.asarray(Y,dtype = int)
    unique_vals, counts = np.unique(Y, return_counts=True)
    probabilities = counts / counts.sum()
    H_max = np.log2(len(unique_vals))
    if Y.size == 0:
        return 0.0
    probabilities = probabilities[probabilities > 0]
    H = -np.sum(probabilities*np.log2(probabilities))
        
       
    return float(H)