import numpy as np
from utils.math_utils import skew

class Kinematics:
    def __init__(self, lam: np.ndarray, d: np.ndarray) -> None:
        """
        Initialize the object ForwardKinematics with 'lam' matrix and 'd' matrix

        Args:
            lam (np.ndarray): Matrix of director vectors , expected size (m x 6).
            d (np.ndarray): Matrix of distance vectors , expected size (m x 3).
        """
        if lam.shape[1] != 6:
            raise ValueError(f" 'lam' must have 6 columns, it has {lam.shape[1]}")
        if d.shape[1] != 3:
            raise ValueError(f" 'd' must have 6 columns, it has {d.shape[1]}")
        if lam.shape[0] != d.shape[0]:
            raise ValueError("lam y d must have the same number of rows")

        self.lam = lam
        self.d = d

    def computeFK(self, k: int, q: np.ndarray) -> np.ndarray:
        """
        Computes the forward kinematics from frame 0 to frame k.

        Args:
            k (int): Index of the frame up to which the forward kinematics is computed.
            q (np.ndarray): Generalized coordinates vector (e.g., joint angles).

        Returns:
            np.ndarray: Resulting 4x4 homogeneous transformation matrix.
        """
        # Condition for k: it must not be greater than n (otherwise it doesn't make sense)
        # System order or robot's DoFs
        n = len(q)
        if k < 0 or k > n:
            raise ValueError("'k' must be between 0 and n inclusive, " f"but received k={k}, n={n}")

        # Extraction of director vectors for translation and rotation
        LamT = self.lam[:, 0:3] # Traslational direction 
        LamR = self.lam[:, 3:6] # Rotational direction 

        # Initialize homogeneous transformation as identity matrix
        T = np.eye(4)

        for i in range(k):
            S = skew(LamR[i,:3])
            R = np.eye(3) + S * np.sin(q[i]) + S @ S * (1 - np.cos(q[i]))
            di = self.d[i,:3].reshape(3, 1)
            Ti = np.block([
                [R, di + LamT[i].reshape(3, 1) * q[i]],
                [np.zeros((1, 3)), 1]
            ])
            T = T @ Ti

        return T