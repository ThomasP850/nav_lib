from nav_lib.planner.utils import *

import numpy as np

class Planner(object):
    def __init__(self, robot_params: RobotParams, dt=0.1, resolution=(0.02, 0.05), time_horizon=3.0,publish_cost_data=False):
        self.robot_params = robot_params
        self.dt = dt
        self.resolution = resolution
        self.time_horizon = time_horizon
        
        self.publish_cost_data = publish_cost_data

        self.goal = None
        self.cost_data = None

    def plan(self, state: np.array, goal: np.array, obstacles: np.ndarray) -> tuple:
        pass