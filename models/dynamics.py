import numpy as np
from utils.operators import extendedTranslation, SkewMatrix6D, PlukerOperator
from typing import Tuple
from models.kinematics import Kinematics


class Dynamics:
    def __init__(self, kinematics: Kinematics, m: np.ndarray, rc: np.ndarray, Icm: np.ndarray) -> None:
        self.kinematics = kinematics
        self.m = m
        self.rc = rc
        self.Icm = Icm

    def compute_ForwardDynamics(self, q: np.ndarray, dq: np.ndarray, tau:np.ndarray) -> np.ndarray:
        n = len(q)
        # --- Initialization ---
        # Dynamics
        H = np.zeros((n,n)) # Inertia Matrix
        h = np.zeros((n,1)) # Non-linear terms vector
        g = np.array([0, 0, -9.81]) # 3D gravity vector
        G = np.block([
                    [g.reshape(3,1)],
                    [np.zeros((3,1))]
        ])  # 3D gravity vector

        # Kinematics
        Jp = np.zeros((n,n)) # Geometric Jacobian (parent)
        a_bar_p = -G # Residual acceleration
        # ----------------------

        for i in range(n):
            J = self.kinematics.compute_geometricJacobian_i(i, q, Jp)
            Jp = J

            V = J @ dq.reshape(n,1) # Twist at local frame
            ET = extendedTranslation(self.rc[i,:])
            Vc = ET @ V # Twist at CoM
            Om_q = SkewMatrix6D(self.kinematics.compute_jointVelocity(i,dq[i]))

            Ti = self.kinematics.compute_homogeneousTransformation(q,i)
            X = PlukerOperator(Ti)
            a_bar = X @ a_bar_p - Om_q.T @ V
            acm = ET @ a_bar
            a_bar_p = a_bar

            Jcm = ET @ J

            # Mass Matrix
            Mi = np.block([
                [np.eye(3)*self.m[i], np.zeros((3,3))],
                [    np.zeros((3,3)), np.diag(self.Icm[i,:3])]
            ])

            # Compute Inertia Matrix (H)
            H = H + Jcm.T @ Mi @ Jcm
            
            # Compute non-linear terms (h)
            Om_V = SkewMatrix6D(Vc)
            h = h + Jcm.T @ (Mi @ acm - Om_V @ Mi @ Vc)

        # Forward Dynamics (H@dqq = tau - h)
        ddq = np.linalg.solve(H,tau-h)
        return ddq

