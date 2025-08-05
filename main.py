from config.robot_config import lam, d, m, rc, Icm
from robots.SixDofRobot import six_dof_robot
import numpy as np
from utils.AttitudeConversion import R2RPY, Jth_RPY

# Generalized Coordinates
q = np.array([0, 0, 0, 0, 0, 0])
dq = np.array([0, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    robot_arm = six_dof_robot(lam, d, m, rc, Icm)
    k = 6
    T = robot_arm.forwardKin(q, k)
    Jg = robot_arm.geoJac(q, k)
    Ja = robot_arm.AnaJac(q, k)

    ddq = robot_arm.forwardDyn(q, dq, np.zeros((len(q),1)))

    print("Forward Kinematics result:")
    print(Jg)
    print("Geometric Jacobian result:")
    print(Jg)
    print("Analytic Jacobian result:")
    print(Ja)
    print("Froward Dynamics result:")
    print(ddq)

