import numpy as np

def R2RPY(R: np.ndarray) -> np.ndarray:
    # ZYX (roll-pitch-yaw)

    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])

def Jth_RPY(th):
    phi, theta, psi = th  # roll, pitch, yaw
    
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    
    Jth = np.array([
        [c_theta * c_psi, -s_psi,         0],
        [c_theta * s_psi,  c_psi,         0],
        [-s_theta,            0,          1]
    ])
    return Jth