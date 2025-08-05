from config.robot_config import lam, d, m, rc, Icm
from robots.SixDofRobot import six_dof_robot
from utils.AttitudeConversion import R2RPY
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


robot_arm = six_dof_robot(lam, d, m, rc, Icm)
k = 6

##### SIMULATION #####

# Kinematic control
def RobotStateSpace(t, x):
    q = x[:6]
    dq = x[6:12]
    n = len(q)
    tau = np.zeros((n,1))
    ddq = robot_arm.forwardDyn(q, dq, tau)

    dx = np.block([
            [dq.reshape(n,1)],
            [ddq]
    ])
    return dx.flatten()

# Initial conditions
# Generalized Coordinates
q0 = np.array([0, 0, 0, 0, 0, 0])
dq0 = np.array([0, 0, 0, 0, 0, 0])
x0 = np.concatenate((q0, dq0))
T_sim = (0,20)
T_eval = np.linspace(T_sim[0], T_sim[1], 5000)

# Solve 
sol = solve_ivp(RobotStateSpace, T_sim, x0, t_eval=T_eval)

# Position
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0, :], label='q_1')
plt.plot(sol.t, sol.y[1, :], label='q_2')
plt.plot(sol.t, sol.y[2, :], label='q_3')
plt.plot(sol.t, sol.y[3, :], label='q_4')
plt.plot(sol.t, sol.y[4, :], label='q_5')
plt.plot(sol.t, sol.y[5, :], label='q_6')

plt.title("Robot Dynamics")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid()
plt.legend()

plt.show()

