import numpy as np
from typing import Tuple
from models.kinematics import Kinematics


class Dynamics:
    def __init__(self, kinematics: Kinematics, m: np.ndarray, rc: np.ndarray, Icm: np.ndarray) -> None:
        self.kinematics = kinematics
        self.m = m
        self.rc = rc
        self.Icm = Icm

    def ComputeModelMatrices(self, q: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Matrix Initialization
        H = np.zeros((6,6))
        C = np.zeros((6,6))
        g = np.zeros((6,1))
        
        return H, C, g

