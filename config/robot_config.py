import numpy as np

# Generalized Coordinates
q = np.array([0, 0, np.pi/2, 0, 0, 0])
dq = np.array([0, 0, 0, 0, 0, 0])

# 6D-Unitary Director Vector
lam = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1]
])

# Distance vectors between frames
d = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])