import numpy as np
from utils.operators import skew
from utils.AttitudeConversion import Jth_RPY, R2RPY

class Kinematics:
    def __init__(self, lam: np.ndarray, d: np.ndarray) -> None:
        """
        Initialize the object ForwardKinematics with 'lam' matrix and 'd' matrix

        Args:
            lam (np.ndarray): Matrix of director vectors , expected size (m x 6).
            d (np.ndarray): Matrix of distance vectors , expected size (m x 3).
        """
        if lam.shape[1] != 6:
            raise ValueError(f" 'lam' must have 6 columns, it has {lam.shape[1]}")
        if d.shape[1] != 3:
            raise ValueError(f" 'd' must have 6 columns, it has {d.shape[1]}")
        if lam.shape[0] != d.shape[0]:
            raise ValueError("lam y d must have the same number of rows")

        self.lam = lam
        self.d = d

    def compute_FK(self, k: int, q: np.ndarray) -> np.ndarray:
        """
        Computes the forward kinematics from frame 0 to frame k.

        Args:
            k (int): Index of the frame up to which the forward kinematics is computed.
            q (np.ndarray): Generalized coordinates vector (e.g., joint angles).

        Returns:
            np.ndarray: Resulting 4x4 homogeneous transformation matrix.
        """
        # Condition for k: it must not be greater than n (otherwise it doesn't make sense)
        # System order or robot's DoFs
        n = len(q)
        if k < 0 or k > n:
            raise ValueError("'k' must be between 0 and n inclusive, " f"but received k={k}, n={n}")

        # Initialize homogeneous transformation as identity matrix
        T = np.eye(4)

        for i in range(k):
            Ti = self.compute_homogeneousTransformation(q,i)
            T = T @ Ti

        return T
    
    def compute_homogeneousTransformation(self, q: np.ndarray, i:int) -> np.ndarray:

        if i < 0 or i >= len(q):
            raise IndexError(f"Index i={i} is out of range for q of length {len(q)}.")

        # Extraction of director vectors for translation and rotation
        LamT = self.lam[:, 0:3] # Traslational direction 
        LamR = self.lam[:, 3:6] # Rotational direction 
        S = skew(LamR[i,:3])
        R0i = np.eye(3) + S * np.sin(q[i]) + S @ S * (1 - np.cos(q[i]))
        d0i = self.d[i,:3].reshape(3, 1)
        Ti = np.block([
                [               R0i, d0i + LamT[i].reshape(3, 1) * q[i] ],
                [np.zeros((1, 3)),                                 1 ]
            ])
        return Ti
    
    def compute_geometricJacobian_recursive(self, k: int, q: np.ndarray ) -> np.ndarray:
        """
        Computes the Geometric Jacobian from frame 0 to frame k.

        Args:
            k (int): Index of the frame up to which the Geometric Jacobian is computed.
            q (np.ndarray): Generalized coordinates vector (e.g., joint angles).
        Returns:
            np.ndarray: Resulting 6x6 Geometric Jacobian matrix.
        """

        Jg_p = np.zeros((6,len(q)))

        for i in range(k):
            Jg_k = self.compute_geometricJacobian_i(i, q, Jg_p)
            Jg_p = Jg_k
            
        return Jg_k
    
    def compute_geometricJacobian_i(self, i:int, q: np.ndarray, Jg_p: np.ndarray ) -> np.ndarray:
        # Extraction of director vectors for translation and rotation
        LamT = self.lam[:, 0:3] # Traslational direction 
        LamR = self.lam[:, 3:6] # Rotational direction 

        # Compute FK form frame i-1 to i 
        S = skew(LamR[i,:3])
        Ri = np.eye(3) + S * np.sin(q[i]) + S @ S * (1 - np.cos(q[i]))
        di = self.d[i,:3].reshape(3, 1)

        Ti = np.block([
                [              Ri, di + LamT[i].reshape(3, 1) * q[i] ],
                [np.zeros((1, 3)),                                 1 ]
            ])

        # Plucker Operator X
        X = np.block([[            Ti[:3,:3].T, -Ti[:3,:3].T@skew(Ti[:3,3].T) ],
                      [ np.zeros((3,3)),                          Ti[:3,:3].T ]
                      ])
            
        # Compute Geometric Jacobian
        LAM = np.zeros((6,len(q)))
        LAM[:,i] = self.lam[i,:]
        Jg_k = X@Jg_p + LAM

        return Jg_k
    
    def compute_AnalyticJacobian(self, k: int, q: np.ndarray) -> np.ndarray:

        # Get Geometric Jacobian
        Jg1 = self.compute_geometricJacobian_recursive(6,q)

        # Get Forward Kinematics
        T = self.compute_FK(k, q)
        # Get RPY angles
        th = R2RPY(T[:3,:3])
        # Get Attitude Operator
        Jth0 = Jth_RPY(th)

        Re = np.block([
                    [T[:3,:3], np.zeros((3,3))],
                    [np.zeros((3,3)), T[:3,:3]]
                     ])

        # --- Compute Jx of \nu = Jx dx ---
        Jx0 = np.block([
            [ np.eye(3), np.zeros((3,3))],
            [np.zeros((3,3)), Jth0]
        ])
        # --- ---

        # Compute Analytic Jacobian Ja
        Ja = np.linalg.inv(Jx0)@Re@Jg1 # = Jg0 = Re@Jg1

        return Ja
    
    def compute_jointVelocity(self, i: int, dq_i: float ) -> np.ndarray:
        V_qi = self.lam[i,0:6]*dq_i
        return V_qi
