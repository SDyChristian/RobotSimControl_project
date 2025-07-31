from kinematics.kinematics import Kinematics
from models.robot_base import RobotBase
import numpy as np

class six_dof_robot(RobotBase):
    def __init__(self, lam: np.ndarray, d: np.ndarray):
        super().__init__(lam, d)
        self.forward_kinematics = Kinematics(lam, d)

    def forwardKin(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.forward_kinematics.computeFK(k, q)