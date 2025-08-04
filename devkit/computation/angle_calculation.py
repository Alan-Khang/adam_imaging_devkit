import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def rotationMatrixToEulerAngles(R):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    """
    assert (R.shape == (3, 3))
    
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])

def compute_angle(rmat):
    """Computes the angle from the rotation vector."""
    euler_angles = rotationMatrixToEulerAngles(rmat)
    angle = np.rad2deg(euler_angles)[1]
    angle = -1 * angle  # Adjust angle to match the expected direction
    return angle

def calculate_mid_pose_from_two_sides(tvec1, tvec2, rvec1, rvec2):
    """
    Calculate the mid pose from two sides of the ArUco markers.
    This function assumes that the two markers are on opposite sides of the robot.
    """
    t_mid = (tvec1 + tvec2) / 2

    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    # Simple average of the matrices (not strictly valid but reasonable for nearby rotations)
    R_mid = (R1 + R2) / 2.0

    # Re-orthonormalize the result (optional but better for accuracy)
    U, _, Vt = np.linalg.svd(R_mid)
    R_mid = U @ Vt

    return R_mid, t_mid

def compute_pitch_angle2(rmat):
    """
    Other pitch angle calculation range from -90 to 90 degrees which is 
    not suitable for our application. This function computes the pitch angle
    in the range of -180 to 180 degrees and then converts it to our 
    application which is 0 degree when the transporter look at the 
    rear of the guilder.
    """

    degrees = R.from_matrix(rmat).as_euler('yxz', degrees=True)
    pitch_angle = degrees[0]
    sign = int(np.sign(pitch_angle))
    angle = abs(pitch_angle)
    pitch_angle = (180. - angle) * -sign
    return pitch_angle
