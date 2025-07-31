from kinematics.forward import ForwardKinematics
from models.robot_base import RobotBase
import numpy as np

class six_dof_robot(RobotBase):
    def __init__(self, lam: np.ndarray, d: np.ndarray):
        super().__init__(lam, d)
        self.forward_kinematics = ForwardKinematics(lam, d)

    def fk(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.forward_kinematics.compute(k, q)