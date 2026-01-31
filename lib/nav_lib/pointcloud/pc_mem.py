import numpy as np
import open3d as o3d

from sensor_msgs.msg import LaserScan

class PointCloudMemory:
    def __init__(self, blind_radius: float = 0.3, retention_radius: float = 0.7, voxel_size: float = 0.04, downsample_interval: int = 5):
        """
        Initialize the PointCloudMemory.

        Args:
            blind_radius (float): Radius around the robot to retain points.
            retention_radius (float): Maximum radius to retain points.
            voxel_size (float): Size of the voxel for downsampling.
            downsample_interval (int): Number of updates between downsampling.
            
        """

        self.pc = o3d.geometry.PointCloud()
        self.retention_radius = retention_radius
        self.blind_radius = blind_radius
        self.voxel_size = voxel_size
        self.downsample_interval = downsample_interval
        self.update_count = 0

    def update(self, new_points: np.ndarray, robot_position: np.ndarray):
        """
        Update the point cloud memory with new points and retain points within the retention radius.
        Args:
            new_points (np.ndarray): New point cloud data as an (N, 3) array.
            robot_position (np.ndarray): Current robot position as a (3,) array.
        """

        past_points = np.asarray(self.pc.points)

        # Remove past points outside of the blind radius, they will be refreshed from the new scan.
        dists = np.linalg.norm(past_points[:, :2] - robot_position[:2], axis=1)
        past_points = past_points[dists <= self.blind_radius]


        # Filter new points outside of the retention radius
        new_dists = np.linalg.norm(new_points - robot_position[:2], axis=1)
        new_points = new_points[new_dists <= self.retention_radius]

        # add z axis to new points
        new_points = np.hstack((new_points, np.zeros((new_points.shape[0], 1))))

        total_points = np.vstack((past_points, new_points))
        self.pc.points = o3d.utility.Vector3dVector(total_points)

        # Downsample every downsample_interval updates
        self.update_count += 1
        if self.update_count % self.downsample_interval == 0:
            self.pc = self.pc.voxel_down_sample(self.voxel_size)

        # cached kd-tree is dirty and must be rebuilt
        self.kd_tree = None

    def size(self) -> int:
        """
        Get the number of points in the point cloud memory.

        Returns:
            int: Number of points.
        """

        return len(self.pc.points)

    def clear(self):
        """
        Clear the point cloud memory.
        """

        self.pc.clear()
    
    def get_kd_tree(self) -> o3d.geometry.KDTreeFlann:
        """
        Get a KD-Tree for the current point cloud.

        Returns:
            o3d.geometry.KDTreeFlann: KD-Tree of the current point cloud.
        """
        if self.kd_tree is None:
            self.kd_tree = o3d.geometry.KDTreeFlann(self.pc)
        
        return self.kd_tree
    
    def get_points(self) -> np.ndarray:
        """
        Get the points in the point cloud memory as a numpy array.

        Returns:
            np.ndarray: Points as an (N, 3) array.
        """

        return np.asarray(self.pc.points)
    

def get_points_from_laser(laser_msg: LaserScan, robot_position: np.ndarray) -> np.ndarray:
    """
    Convert a LaserScan message to a point cloud numpy array.

    Args:
        laser_msg (LaserScan): The input LaserScan message.
        robot_position (np.ndarray): The current position of the robot in the map frame. [x, y, yaw]

    Returns:
        np.ndarray: An (N, 2) array of points in the map frame.
    """

    angles = np.arange(laser_msg.angle_min, laser_msg.angle_max, laser_msg.angle_increment)[:len(laser_msg.ranges)]
    ranges = np.array(laser_msg.ranges)

    valid_mask = np.isfinite(ranges) & (ranges >= laser_msg.range_min) & (ranges <= laser_msg.range_max)
    angles = angles[valid_mask]
    ranges = ranges[valid_mask]

    c = np.cos(robot_position[2])
    s = np.sin(robot_position[2])

    c_laser = np.cos(angles)
    s_laser = np.sin(angles)

    x_laser = robot_position[0] + ranges * (c * c_laser - s * s_laser)
    y_laser = robot_position[1] + ranges * (s * c_laser + c * s_laser)

    points_laser = np.column_stack((x_laser, y_laser))

    return points_laser