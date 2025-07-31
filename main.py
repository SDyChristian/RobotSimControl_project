from config.robot_config import lam, d, q
from models.SixDofRobot import six_dof_robot

if __name__ == "__main__":
    robot = six_dof_robot(lam, d)
    k = 6
    T = robot.fk(q,k)

    print("Forward Kinematics result:")
    print(T)