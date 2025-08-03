from config.robot_config import lam, d
from models.SixDofRobot import six_dof_robot
from utils.AttitudeConversion import R2RPY
from scipy.integrate import solve_ivp
import numpy as np


robot_arm = six_dof_robot(lam, d)
k = 6

##### SIMULATION #####

# Kinematic control
def KinControl(t, q):
    k = 6
    # Get forward kinematics until end efector
    T = robot_arm.forwardKin(q,k)
    # Extract robot pose
    x = np.block([
        [ T[:3,3].reshape(3,1) ],
        [ R2RPY(T[:3,:3]).reshape(3,1) ]
    ])
    # Set desired end effector position and velocity
    xd = np.array([[0.1*np.cos(t)],
                   [0.1*np.sin(t)],
                   [1.5],
                   [0.1],
                   [0.1],
                   [0]
                  ])
    dxd = np.array([[-0.1*np.sin(t)],
                    [0.1*np.cos(t)],
                    [0],
                    [0],
                    [0],
                    [0]])
    
    # --- Kinematic Control ---
    K = np.eye(6)
    e = x-xd # Compute Error
    Ja = robot_arm.AnaJac(q,k) # Compute Analytic Jacobian
    dqd = np.linalg.pinv(Ja)@(dxd-K@e)
    print(np.linalg.norm(e))
    return dqd.flatten()

# Initial conditions
# Generalized Coordinates
q0 = np.array([0, 0, 0, 0, 0, 0])
T_sim = (0,10)
T_eval = np.linspace(T_sim[0], T_sim[1], 1000)

# Solve 
sol = solve_ivp(KinControl, T_sim, q0, t_eval=T_eval)
