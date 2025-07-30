import numpy as np
from utils.math_utils import skew

class ForwardKinematics:
    @staticmethod
    def compute(lam, d, k, q):
        """
        Computes the forward kinematics from frame 0 to frame k
        """
        n = len(q)
        if k < 0 or k > n:
            raise ValueError("k must be between 0 and n inclusive")

        LamT = lam[:, 0:3]
        LamR = lam[:, 3:6]

        T = np.eye(4)

        for i in range(k):
            S = skew(LamR[i])
            R = np.eye(3) + S*np.sin(q[i]) + S@S*(1 - np.cos(q[i]))
            di = d[i].reshape(3,1)
            Ti = np.block([
                [R, di + LamT[i].reshape(3,1)*q[i]],
                [np.zeros((1,3)), [1]]
            ])
            T = T @ Ti
        
        return T