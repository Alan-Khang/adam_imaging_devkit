import numpy as np
from scipy.spatial.transform import Rotation as R

def get_corners(center, whl, pitch_angle, whl_factor: float = 1.0) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, h, l = whl * whl_factor

    # 3D bounding box corners. (Convention: x to the right, y down, z points forward.)
    x_corners = w / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    y_corners = h / 2 * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    z_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    rot_mat = R.from_euler('y', pitch_angle, degrees=True).as_matrix()
    corners = np.dot(rot_mat, corners)

    # Translate
    x, y, z = center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners
