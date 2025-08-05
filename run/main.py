from config.robot_config import lam, d, m, rc, Icm
from robots.six_dof_robot import six_dof_robot
import numpy as np
from utils.attitude_conversion import R2RPY, Jth_RPY

# Generalized Coordinates
q = np.array([0, 0, 0, 0, 0, 0])
dq = np.array([0, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    robot_arm = six_dof_robot(lam, d, m, rc, Icm)
    k = 6
    T = robot_arm.forward_kin(q, k)
    Jg = robot_arm.geo_jac(q, k)
    Ja = robot_arm.ana_jac(q, k)

    ddq = robot_arm.forward_dyn(q, dq, np.zeros((len(q),1)))

    print("Forward Kinematics result:")
    print(Jg)
    print("Geometric Jacobian result:")
    print(Jg)
    print("Analytic Jacobian result:")
    print(Ja)
    print("Froward Dynamics result:")
    print(ddq)

    