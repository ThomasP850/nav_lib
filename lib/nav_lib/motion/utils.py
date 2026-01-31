import numpy as np

def predict_trajectory(initial_state: np.array, time_horizon=5.0, dt=0.1) -> np.ndarray:
    """
    Predict the trajectory of the robot over a specified time horizon.

    Parameters:
        initial_state : [x, y, yaw, v, w]
        time_horizon  : Total time to predict (s)
        dt            : Time interval for each prediction step (s)

    Returns:
        Trajectory as (N+1, 3) array [[x0,y0,yaw0], ..., [xN,yN,yawN]]
    """
    x_0, y_0, yaw_0, v, w = initial_state

    # Number of whole steps
    num_steps = int(time_horizon // dt)
    # Time samples including t=0
    t = np.linspace(0.0, num_steps * dt, num_steps + 1, dtype=float)

    yaw = yaw_0 + w * t

    if abs(w) < 1e-6:
        # Straight-line motion
        c_0 = np.cos(yaw_0)
        s_0 = np.sin(yaw_0)
        x = x_0 + v * t * c_0
        y = y_0 + v * t * s_0
    else:
        # Circular arc
        r = v / w
        x = x_0 + r * (np.sin(yaw) - np.sin(yaw_0))
        y = y_0 - r * (np.cos(yaw) - np.cos(yaw_0))

    return np.column_stack((x, y, yaw))
