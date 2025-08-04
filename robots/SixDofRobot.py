from models.kinematics import Kinematics
from models.dynamics import Dynamics
from robots.robot_base import RobotBase
import numpy as np

class six_dof_robot(RobotBase):
    def __init__(self, 
                 lam: np.ndarray, 
                 d: np.ndarray,
                 m: np.ndarray = None,
                 rc: np.ndarray = None,
                 Icm: np.ndarray = None):
        
        super().__init__(lam, d)

        self.kinematics = Kinematics(lam, d)
        if m is not None and rc is not None and Icm is not None:
            self.dynamics = Dynamics(lam, d)
        else:
            self.dynamics = None

    def forwardKin(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.kinematics.computeFK(k, q)
    
    def geoJac(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.kinematics.compute_geometricJacobian(k, q)
    
    def AnaJac(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.kinematics.compute_AnalyticJacobian(k, q)
    
    def ModelMatrices(self, q: np.ndarray, dq: np.ndarray):
        return self.dynamics.compute_ModelMatrices(k, q)