from .planner import *
from ..motion import utils as motion_utils

import numpy as np

class ImprovedDwaPlanner(Planner):
    def __init__(
            self,
            robot_params: RobotParams,
            obstacle_weight=4.0,
            goal_weight=3.0,
            dt=0.1,
            resolution=(0.02, 0.05),
            time_horizon=3.0,
            publish_cost_data=False
        ):

        super().__init__(robot_params, dt, resolution, time_horizon, publish_cost_data)

        self.obstacle_weight = obstacle_weight
        self.goal_weight = goal_weight

    def plan(self, state: np.array, goal: np.array, obstacles: np.ndarray) -> tuple:
        best_result = (0.0, 0.0)
        best_cost = float('inf')

        dynamic_window = calc_dynamic_window(
            state[3],
            state[4],
            self.robot_params,
            self.dt
        )

        if self.publish_cost_data:
            new_cost_data = []

        for v in np.arange(dynamic_window[0], dynamic_window[1], self.resolution[0]):
            for w in np.arange(dynamic_window[2], dynamic_window[3], self.resolution[1]):
                # Simulate the robot's motion
                trajectory = motion_utils.predict_trajectory(
                    np.array([state[0], state[1], state[2], v, w]),
                    time_horizon=self.time_horizon,
                    dt=self.dt
                )

                obstacle_cost = self.obstacle_weight * calc_nearest_obstacle_cost(trajectory, obstacles, self.robot_params)
                goal_cost = self.goal_weight * ImprovedDwaPlanner._calcGoalCost(trajectory[-2], trajectory[-1], goal)

                total_cost = obstacle_cost + goal_cost

                if self.publish_cost_data:
                    new_cost_data.append((trajectory[-1][0], trajectory[-1][1], obstacle_cost, goal_cost, 0.0, total_cost))

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_result = (v, w)


        return best_result
    
    def _calcGoalCost(prev_state: np.array, state: np.array, goal: np.array) -> float:
        prev_distance = np.hypot(goal[0] - prev_state[0], goal[1] - prev_state[1])
        distance = np.hypot(goal[0] - state[0], goal[1] - state[1])

        return distance - prev_distance
    