import numpy as np

def velocity_motion_model(x: np.array, dt: float) -> np.array:
    """
    Compute the new state of a robot given its current state and motion commands
    using a velocity motion model.

    Parameters:
        x   : Current state of the robot [x, y, yaw, v, w]
        dt  : Time interval (s).

    Returns:
        New state of the robot [new_x_position, new_y_position, new_orientation].
    """
    
    new_state = x.copy()

    if abs(x[4]) < 1e-6:
        # Simple linear model
        new_state[0] += x[3] * np.cos(x[2]) * dt
        new_state[1] += x[3] * np.sin(x[2]) * dt

        return new_state
    
    turn_radius = x[3] / x[4]
    new_yaw = x[2] + x[4] * dt

    new_state[0] += turn_radius * (np.sin(new_yaw) - np.sin(x[2]))
    new_state[1] -= turn_radius * (np.cos(new_yaw) - np.cos(x[2]))

    new_state[2] = new_yaw

    return new_state


def predict_trajectory(initial_state: np.array, time_horizon=5.0, dt=0.1) -> np.ndarray:
    """
    Predict the trajectory of the robot over a specified time horizon
    given initial state and motion commands.

    Parameters:
        initial_state : Initial state of the robot [x, y, yaw, v, w].
        v             : Linear velocity command (m/s).
        w             : Yaw rate command (rad/s).
        time_horizon  : Total time to predict (s).
        dt            : Time interval for each prediction step (s).

    Returns:
        Predicted trajectory as a numpy array of shape (N, 3),
        where N is the number of time steps and each row is [x, y, yaw].
    """
    
    state = initial_state.copy()
    num_steps = int(time_horizon // dt)
    trajectory = np.zeros((num_steps + 1, 3))
    trajectory[0] = state[0:3]

    for t in range(1, num_steps + 1):
        state = velocity_motion_model(state, dt)
        trajectory[t] = state[0:3]

    return trajectory