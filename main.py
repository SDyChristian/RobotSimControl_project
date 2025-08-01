from config.robot_config import lam, d
from models.SixDofRobot import six_dof_robot
import numpy as np

# Generalized Coordinates
q = np.array([0, 0, 0, 0, 0, 0])
dq = np.array([0, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    robot_arm = six_dof_robot(lam, d)
    k = 6
    T = robot_arm.forwardKin(q,k)
    Jg = robot_arm.geoJac(q,dq,k)

    print("Forward Kinematics result:")
    print(T)
    print("Geometric Jacobian result:")
    print(Jg)