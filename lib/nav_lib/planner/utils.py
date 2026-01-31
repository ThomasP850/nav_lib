import numpy as np
import open3d as o3d

import rospy

from nav_lib.pointcloud.pc_mem import PointCloudMemory

class RobotParams:
    def __init__(self, max_speed: float, max_yaw_rate: float,
                 max_accel: float, max_yaw_rate_delta: float,
                 length: float, width: float, safety_radius: float):
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_accel = max_accel
        self.max_yaw_rate_delta = max_yaw_rate_delta
        self.length = length
        self.width = width
        self.safety_radius = safety_radius

def calc_dynamic_window(v, w, params: RobotParams, dt: float) -> np.array:
    """
    Calculate the dynamic window based on robot parameters and time step.

    Parameters:
    v            : Current linear velocity (m/s).
    w            : Current yaw rate (rad/s).
    robot_params : RobotParams object containing robot constraints.
    dt           : Time interval (s).
    v_res        : Velocity resolution (m/s).
    w_res        : Yaw rate resolution (rad/s).

    Returns:
        Dynamic window as a numpy array: [min_v, max_v, min_w, max_w]
    """
    
    max_linear_delta = params.max_accel * dt
    max_yaw_delta = params.max_yaw_rate_delta * dt

    return np.array([
        np.clip(v - max_linear_delta, 0, params.max_speed),
        np.clip(v + max_linear_delta, 0, params.max_speed),
        np.clip(w - max_yaw_delta, -params.max_yaw_rate, params.max_yaw_rate),
        np.clip(w + max_yaw_delta, -params.max_yaw_rate, params.max_yaw_rate)
    ])

def calc_nearest_obstacle_cost(trajectory: np.ndarray, obstacles: np.ndarray, params: RobotParams) -> float:
    """
    Calculate the cost based on the nearest obstacle to the predicted trajectory.

    This uses a rectangular robot model with a safety radius.

    Parameters:
        trajectory : Predicted trajectory as a numpy array of shape (N, 3).
        obstacles  : Obstacle positions as a numpy array of shape (M, 2).

    Returns:
        Nearest obstacle cost (float). Higher cost for closer obstacles.
    """

    if obstacles.shape[0] == 0:
        # rospy.loginfo("No obstacles detected; returning zero cost.")
        return 0.0

    dx = obstacles[:, 0][None, :] - trajectory[:, 0][:, None]
    dy = obstacles[:, 1][None, :] - trajectory[:, 1][:, None]

    c = np.cos(trajectory[:, 2])
    s = np.sin(trajectory[:, 2])

    # Rotate points into robot frame for each pose
    f = dx * c[:, None] + dy * s[:, None]
    l = -dx * s[:, None] + dy * c[:, None]

    df = np.maximum(np.abs(f) - params.length * 0.5, 0.0)
    dl = np.maximum(np.abs(l) - params.width * 0.5, 0.0)

    d2 = df * df + dl * dl
    min_d2 = float(d2.min())

    # rospy.loginfo("Minimum squared distance to obstacle: %f", min_d2)

    if min_d2 <= 0.0:
        return float('inf')

    if min_d2 <= params.safety_radius**2:
        return 1.0 / np.sqrt(min_d2)

    return 0.0