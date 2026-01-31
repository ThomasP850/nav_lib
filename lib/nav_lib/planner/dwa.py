from nav_lib.planner.planner import *

import nav_lib.motion.utils as motion_utils

import numpy as np

class DwaPlanner(Planner):
    def __init__(
            self,
            robot_params: RobotParams,
            obstacle_weight=4.0,
            goal_weight=3.0,
            speed_weight=9.0, 
            dt=0.1,
            resolution=(0.02, 0.05),
            time_horizon=3.0,
            publish_cost_data=False
        ):

        super().__init__(robot_params, dt, resolution, time_horizon, publish_cost_data)

        self.obstacle_weight = obstacle_weight
        self.goal_weight = goal_weight
        self.speed_weight = speed_weight

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
                goal_cost = self.goal_weight * DwaPlanner._calcGoalCost(trajectory[-1], goal)
                speed_cost  = self.speed_weight * (self.robot_params.max_speed - trajectory[-1][3])

                total_cost = obstacle_cost + goal_cost + speed_cost

                if self.publish_cost_data:
                    new_cost_data.append((trajectory[-1][0], trajectory[-1][1], obstacle_cost, goal_cost, speed_cost, total_cost))

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_result = (v, w)


        return best_result
    
    def _calcGoalCost(state: np.array, goal: np.array) -> float:
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]

        angle = np.arctan2(dy, dx) - state[2]

        if angle > np.pi:
            angle -= 2.0 * np.pi
        elif angle < -np.pi:
            angle += 2.0 * np.pi

        return np.abs(angle)
    
