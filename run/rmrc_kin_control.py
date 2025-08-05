from config.robot_config import lam, d
from robots.six_dof_robot import six_dof_robot
from utils.attitude_conversion import R2RPY
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


robot_arm = six_dof_robot(lam, d)
k = 6

##### SIMULATION #####

# Kinematic control
def kin_control(t, q):
    k = 6
    # Get forward kinematics until end efector
    T = robot_arm.forward_kin(q,k)
    # Extract robot pose
    x = np.block([
        [ T[:3,3].reshape(3,1) ],
        [ R2RPY(T[:3,:3]).reshape(3,1) ]
    ])
    # Set desired end effector position and velocity
    xd = np.array([[0.1*np.cos(t)],
                   [0.1*np.sin(t)],
                   [0.1],
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
    Ja = robot_arm.ana_jac(q,k) # Compute Analytic Jacobian
    dqd = np.linalg.pinv(Ja)@(dxd-K@e)

    return dqd.flatten()

# Initial conditions
# Generalized Coordinates
q0 = np.array([0, 0, 0, 0, np.pi/4, np.pi/4])
T_sim = (0,20)
T_eval = np.linspace(T_sim[0], T_sim[1], 5000)

# Solve 
sol = solve_ivp(kin_control, T_sim, q0, t_eval=T_eval)

# Plot
pose_d = []
for time in sol.t:
    xd = np.array([[0.1*np.cos(time)],
                   [0.1*np.sin(time)],
                   [0.1],
                   [0.1],
                   [0.1],
                   [0]
                  ])
    pose_d.append(xd.flatten())

pose_d = np.array(pose_d)  # (N, 6)

pose = []
for q in sol.y.T:
    T = robot_arm.forward_kin(q, k)
    x = np.block([
        [T[:3, 3].reshape(3, 1)],
        [R2RPY(T[:3, :3]).reshape(3, 1)]
    ])
    pose.append(x.flatten())

pose = np.array(pose)  # (N, 6)

# Position
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(sol.t, pose[:, 0], label='x')
plt.plot(sol.t, pose_d[:, 0], label='x_d', linestyle='--')
plt.plot(sol.t, pose[:, 1], label='y')
plt.plot(sol.t, pose_d[:, 1], label='y_d', linestyle='--')
plt.plot(sol.t, pose[:, 2], label='z')
plt.plot(sol.t, pose_d[:, 2], label='z_d', linestyle='--')
plt.title("End Effector Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid()
plt.legend()

# Orientation
plt.subplot(1, 2, 2)
plt.plot(sol.t, pose[:, 3], label=r'$\phi$')
plt.plot(sol.t, pose_d[:, 3], label=r'$\phi_d$', linestyle='--')
plt.plot(sol.t, pose[:, 4], label=r'$\theta$')
plt.plot(sol.t, pose_d[:, 4], label=r'$\theta_d$', linestyle='--')
plt.plot(sol.t, pose[:, 5], label=r'$\psi$')
plt.plot(sol.t, pose_d[:, 5], label=r'$\psi_d$', linestyle='--')
plt.title(r"End Effector Orientation (Roll-Pitch-Yaw) -> ($\phi$-$\theta$-$\psi$)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()