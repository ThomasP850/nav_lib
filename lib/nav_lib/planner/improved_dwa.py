from nav_lib.planner.planner import *

import nav_lib.motion.utils as motion_utils
from nav_lib.pointcloud.pc_mem import PointCloudMemory

import numpy as np
import rospy
import time

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

    def plan(self, state: np.array, goal: np.array, obstacle_points: np.ndarray) -> tuple:
        start_time = time.perf_counter()

        best_result = (0.0, 0.0)
        best_cost = float('inf')

        dynamic_window = calc_dynamic_window(
            state[3],
            state[4],
            self.robot_params,
            self.dt
        )

        t_1 = time.perf_counter()
        # rospy.logdebug("[ImprovedDwaPlanner] Dynamic window calculation time: %.6fs", t_1 - start_time)

        if self.publish_cost_data:
            new_cost_data = []

        v_window = np.arange(dynamic_window[0], dynamic_window[1] + 1e-9, self.resolution[0])
        w_window = np.arange(dynamic_window[2], dynamic_window[3] + 1e-9, self.resolution[1])

        cur_state = state[:]

        total_time = 0.0
        total_pred_time = 0.0
        total_obs_time = 0.0
        total_goal_time = 0.0

        for v in v_window:
            for w in w_window:
                t_n = time.perf_counter()
                cur_state[3] = v
                cur_state[4] = w

                # Simulate the robot's motion
                trajectory = motion_utils.predict_trajectory(
                    cur_state,
                    time_horizon=self.time_horizon,
                    dt=self.dt
                )

                t_pred = time.perf_counter()
                total_pred_time += t_pred - t_n
                total_time += t_pred - t_n

                obstacle_cost = self.obstacle_weight * calc_nearest_obstacle_cost(trajectory, obstacle_points, self.robot_params)

                t_obs = time.perf_counter()
                total_obs_time += t_obs - t_pred
                total_time += t_obs - t_pred

                goal_cost = self.goal_weight * ImprovedDwaPlanner._calcGoalCost(state, trajectory[-1], goal)

                t_goal = time.perf_counter()
                total_goal_time += t_goal - t_obs
                total_time += t_goal - t_obs

                total_cost = obstacle_cost + goal_cost

                if self.publish_cost_data:
                    new_cost_data.append((trajectory[-1][0], trajectory[-1][1], obstacle_cost, goal_cost, 0.0, total_cost))

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_result = (v, w)

                total_time += time.perf_counter() - t_goal

        # rospy.loginfo("[ImprovedDwaPlanner] Planning time breakdown: Total: %.6fs, Prediction: %.6fs, Obstacle Cost: %.6fs, Goal Cost: %.6fs",
        #                total_time, total_pred_time, total_obs_time, total_goal_time)

        if self.publish_cost_data:
            self.cost_data = new_cost_data
        result = best_result
        return result

    def _calcGoalCost(prev_state: np.array, state: np.array, goal: np.array) -> float:
        prev_distance = np.hypot(goal[0] - prev_state[0], goal[1] - prev_state[1])
        distance = np.hypot(goal[0] - state[0], goal[1] - state[1])

        return distance - prev_distance

