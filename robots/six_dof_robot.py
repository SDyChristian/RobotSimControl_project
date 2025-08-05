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
            self.dynamics = Dynamics(self.kinematics, m, rc, Icm)
        else:
            self.dynamics = None

    def forward_kin(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.kinematics.compute_fk(k, q)
    
    def geo_jac(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.kinematics.compute_geometric_jacobian_recursive(k, q)
    
    def ana_jac(self, q: np.ndarray, k: int = None):
        if k is None:
            k = len(q)
        return self.kinematics.compute_analytic_jacobian(k, q)
    
    def forward_dyn(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray):
        return self.dynamics.compute_forward_dynamics(q, dq, tau)