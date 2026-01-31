"""
Point cloud memory: maintains a local obstacle map from laser scans
with blind zone, retention radius, and optional voxel downsampling.
"""
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def get_points_from_laser(scan_msg, robot_pos: np.ndarray) -> np.ndarray:
    """
    Convert a LaserScan message to global x,y,z points (z=0) in the map frame.

    Parameters:
        scan_msg : sensor_msgs.msg.LaserScan
        robot_pos : np.array [x, y, yaw] in map frame

    Returns:
        np.ndarray of shape (N, 3) with columns [x, y, z]
    """
    ranges = np.array(scan_msg.ranges, dtype=np.float64)
    angle_min = scan_msg.angle_min
    angle_increment = scan_msg.angle_increment
    valid = np.isfinite(ranges) & (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
    indices = np.where(valid)[0]
    if len(indices) == 0:
        return np.zeros((0, 3))

    angles = angle_min + indices * angle_increment
    r = ranges[indices]
    # Robot frame: x forward, y left (typical laser)
    rx = r * np.cos(angles)
    ry = r * np.sin(angles)
    x, y, yaw = robot_pos[0], robot_pos[1], robot_pos[2]
    c, s = np.cos(yaw), np.sin(yaw)
    gx = x + c * rx - s * ry
    gy = y + s * rx + c * ry
    return np.column_stack([gx, gy, np.zeros_like(gx)])


class PointCloudMemory:
    """
    Maintains a point cloud of obstacles from laser updates with blind radius,
    retention radius, and optional voxel downsampling.
    """

    def __init__(
        self,
        blind_radius: float = 0.3,
        retention_radius: float = 3.0,
        voxel_size: float = 0.04,
        downsample_interval: int = 5,
    ):
        self.blind_radius = blind_radius
        self.retention_radius = retention_radius
        self.voxel_size = voxel_size
        self.downsample_interval = downsample_interval
        self._update_count = 0
        if HAS_OPEN3D:
            self.pc = o3d.geometry.PointCloud()
        else:
            self.pc = _SimplePointCloud()

    def update(self, points: np.ndarray, robot_pos: np.ndarray) -> None:
        """
        Update the point cloud with new points (Nx3 in map frame).
        Drops points inside blind_radius and beyond retention_radius;
        downsamples every downsample_interval updates.
        """
        if points is None or len(points) == 0:
            self._prune(robot_pos)
            self._maybe_downsample()
            return

        xy = points[:, :2]
        r_robot = np.array([robot_pos[0], robot_pos[1]])
        dist = np.linalg.norm(xy - r_robot, axis=1)
        mask = (dist >= self.blind_radius) & (dist <= self.retention_radius)
        points = points[mask]
        if len(points) == 0:
            self._prune(robot_pos)
            self._maybe_downsample()
            return

        if HAS_OPEN3D:
            new_pc = o3d.geometry.PointCloud()
            new_pc.points = o3d.utility.Vector3dVector(points)
            if self.pc.has_points() and len(self.pc.points) > 0:
                combined = self.pc + new_pc
                self.pc.points = combined.points
            else:
                self.pc.points = o3d.utility.Vector3dVector(points)
        else:
            self.pc.add_points(points)

        self._prune(robot_pos)
        self._update_count += 1
        self._maybe_downsample()

    def _prune(self, robot_pos: np.ndarray) -> None:
        """Remove points outside retention_radius from robot."""
        if self.size() == 0:
            return
        pts = self._get_xy()
        r = np.array([robot_pos[0], robot_pos[1]])
        dist = np.linalg.norm(pts - r, axis=1)
        keep = dist <= self.retention_radius
        if HAS_OPEN3D:
            all_pts = np.asarray(self.pc.points)
            self.pc.points = o3d.utility.Vector3dVector(all_pts[keep])
        else:
            self.pc.keep_mask(keep)

    def _maybe_downsample(self) -> None:
        if not HAS_OPEN3D or self.voxel_size <= 0 or self._update_count % self.downsample_interval != 0:
            return
        if self.size() == 0:
            return
        self.pc = self.pc.voxel_down_sample(self.voxel_size)

    def _get_xy(self) -> np.ndarray:
        if self.size() == 0:
            return np.zeros((0, 2))
        if HAS_OPEN3D:
            pts = np.asarray(self.pc.points)
        else:
            pts = np.asarray(self.pc.points)
        return pts[:, :2]

    def size(self) -> int:
        if HAS_OPEN3D:
            return len(self.pc.points)
        return len(self.pc.points)

    def get_points(self) -> np.ndarray:
        """
        Return obstacle points as (N, 2) array [x, y] in map frame for the DWA planner.
        """
        xy = self._get_xy()
        return xy if xy.shape[0] > 0 else np.zeros((0, 2))


class _SimplePointCloud:
    """Minimal in-memory point cloud when open3d is not available."""

    def __init__(self):
        self._points = np.zeros((0, 3))

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value if isinstance(value, np.ndarray) else np.array(value)

    def add_points(self, pts: np.ndarray) -> None:
        self._points = np.vstack([self._points, pts]) if self._points.size else np.asarray(pts)

    def keep_mask(self, keep: np.ndarray) -> None:
        self._points = self._points[keep]
