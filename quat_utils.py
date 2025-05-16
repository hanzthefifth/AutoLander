import numpy as np
from scipy.spatial.transform import Rotation as R

# Convert roll, pitch, yaw (Euler angles) to quaternion (w, x, y, z)
def quat_from_rpy(roll, pitch, yaw):
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    q = rotation.as_quat()  # (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])  # reorder to (w, x, y, z)

# Convert quaternion (w, x, y, z) to roll, pitch, yaw
def quat_to_rpy(quat):
    rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # (x, y, z, w)
    return rotation.as_euler('xyz')

# Convert quaternion to 3x3 rotation matrix
def quat_to_matrix(quat):
    rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return rotation.as_matrix()

# Return the inverse of the given quaternion (w, x, y, z)
def quat_invert(quat):
    rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    inv = rotation.inv().as_quat()
    return np.array([inv[3], inv[0], inv[1], inv[2]])
