from kinematics.forward import ForwardKinematics

class RobotModel:
    def __init__(self, lam, d):
        self.lam = lam
        self.d = d

    def fk(self, q, k=None):
        if k is None:
            k = len(q)
        return ForwardKinematics.compute(self.lam, self.d, k, q)