"""Data models for the camera and user specification."""

from dataclasses import dataclass


class DatasetSpec:
    """
    Data model for specifications of an image dataset.
    """
    def __init__(self, overlap: float, sidelap: float, height: float,
                 scan_dimension_x: float, scan_dimension_y: float,
                 exposure_time_ms: float, camera_angle: float = 90.0):
        self.overlap = overlap
        self.sidelap = sidelap
        self.height = height
        self.scan_dimension_x = scan_dimension_x
        self.scan_dimension_y = scan_dimension_y
        self.exposure_time_ms = exposure_time_ms
        self.camera_angle = camera_angle  # angle from horizontal X-axis in degrees (90 = nadir)

    def __repr__(self):
        return (f"DatasetSpec(overlap={self.overlap}, sidelap={self.sidelap}, "
                f"height={self.height}m, scan=({self.scan_dimension_x}x{self.scan_dimension_y}), "
                f"exposure={self.exposure_time_ms}ms, camera_angle={self.camera_angle}°)")


@dataclass
class Camera:
    """
    Data model for a simple pinhole camera.
    References:
    - https://github.com/colmap/colmap/blob/3f75f71310fdec803ab06be84a16cee5032d8e0d/src/colmap/sensor/models.h#L220
    - https://en.wikipedia.org/wiki/Pinhole_camera_model
    """
    fx: float                # focal length along x axis (in pixels)
    fy: float                # focal length along y axis (in pixels)
    sensor_size_x_mm: float  # sensor width in mm
    sensor_size_y_mm: float  # sensor height in mm
    num_pixels_x: int        # image width in pixels
    num_pixels_y: int        # image height in pixels


@dataclass
class Waypoint:
    """
    Waypoints are positions where the drone should fly to and capture a photo.

    Attributes:
        x: horizontal position in meters.
        y: lateral position in meters.
        z: altitude / height above ground in meters.
        speed: maximum speed during photo capture (m/s).
    """
    x: float
    y: float
    z: float
    speed: float