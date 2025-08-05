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

def extendedTranslation(a):
    v = np.asarray(a).flatten()

    if v.size != 3:
        raise ValueError("Vector must be of dimension 3, either (3,) or (3,1)")
    
    ET = np.block([
        [      np.eye(3),  -skew(v)],
        [np.zeros((3,3)), np.eye(3)]
    ])

    return ET

def SkewMatrix6D(a):
    v = np.asarray(a).flatten()

    if v.size != 6:
        raise ValueError("Vector must be of dimension 6, either (6,) or (6,1)")
    
    Om = np.block([
        [ skew(v[3:6]),  skew(v[:3])],
        [np.zeros((3,3)), skew(v[3:6])]
    ])

    return Om

def PlukerOperator(Ti: np.ndarray):
    
    # Plucker Operator X
    X = np.block([[            Ti[:3,:3].T, -Ti[:3,:3].T@skew(Ti[:3,3].T) ],
                  [ np.zeros((3,3)),                          Ti[:3,:3].T ]
                ])

    return X
    
