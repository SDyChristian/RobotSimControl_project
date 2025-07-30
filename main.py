from config.robot_config import lam, d, q
from models.robot_model import RobotModel

if __name__ == "__main__":
    robot = RobotModel(lam, d)
    T = robot.fk(q)

    print("Forward Kinematics result:")
    print(T)