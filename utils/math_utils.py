import numpy as np

def skew(a):
    """
    Returns the skew-symmetric matrix of a 3D vector a.
    """
    v = np.asarray(a).flatten()
    if v.size != 3:
        raise ValueError("Vector must be of dimension 3, either (3,) or (3,1)")
    
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])