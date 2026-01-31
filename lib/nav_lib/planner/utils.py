import numpy as np

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
    
    if len(obstacles) == 0:
        return 0.0

    min_dist = float('inf')

    for x, y, yaw in trajectory:
        relative = obstacles - np.array([x, y])
        c = np.cos(yaw)
        s = np.sin(yaw)

        # Points in robot frame
        f = relative[:, 0] * c + relative[:, 1] * s
        l = -relative[:, 0] * s + relative[:, 1] * c

        # Clamped to robot boundaries
        f_clamped = np.clip(f, -params.length / 2, params.length / 2)
        l_clamped = np.clip(l, -params.width / 2, params.width / 2)

        d_forward = f - f_clamped
        d_lateral = l - l_clamped


        min_dist = min(min_dist, np.min(np.hypot(d_forward, d_lateral)))

        if min_dist == 0:
            break


    if min_dist <= 0:
        return float('inf')
    elif min_dist <= params.safety_radius:
        return 1.0 / min_dist
    
    return 0.0