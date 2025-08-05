import numpy as np
from utils.operators import extended_translation, skew_matrix_6D, pluker_operator
from typing import Tuple
from models.kinematics import Kinematics


class Dynamics:
    def __init__(self, kinematics: Kinematics, m: np.ndarray, rc: np.ndarray, Icm: np.ndarray) -> None:
        self.kinematics = kinematics
        self.m = m
        self.rc = rc
        self.Icm = Icm

    def compute_forward_dynamics(self, q: np.ndarray, dq: np.ndarray, tau:np.ndarray) -> np.ndarray:
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
            J = self.kinematics.compute_geometric_jacobian_i(i, q, Jp)
            Jp = J

            V = J @ dq.reshape(n,1) # Twist at local frame
            ET = extended_translation(self.rc[i,:])
            Vc = ET @ V # Twist at CoM
            Om_q = skew_matrix_6D(self.kinematics.compute_joint_velocity(i,dq[i]))

            Ti = self.kinematics.compute_homogeneous_transformation(q,i)
            X = pluker_operator(Ti)
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
            Om_V = skew_matrix_6D(Vc)
            h = h + Jcm.T @ (Mi @ acm - Om_V @ Mi @ Vc)

        # Forward Dynamics (H@dqq = tau - h)
        ddq = np.linalg.solve(H,tau-h)
        return ddq

