from abc import ABC, abstractmethod
import numpy as np

class RobotBase(ABC):
    def __init__(self, lam: np.ndarray, d: np.ndarray):
        self.lam = lam
        self.d = d

    @abstractmethod
    def forwardKin(self, q: np.ndarray, k: int = None):
        """Compute forward kinematics."""
        pass

    @abstractmethod
    def geoJac(self, q: np.ndarray, k: int = None):
        """Compute forward kinematics."""
        pass

    @abstractmethod
    def AnaJac(self, q: np.ndarray, k: int = None):
        """Compute forward kinematics."""
        pass