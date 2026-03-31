"""Data models for the camera and user specification."""

class DatasetSpec:
    """
    Data model for specifications of an image dataset.
    """
    def __init__(self, overlap: float, sidelap: float, height: float,
                 scan_dimension_x: float, scan_dimension_y: float,
                 exposure_time_ms: float):
        self.overlap = overlap
        self.sidelap = sidelap
        self.height = height
        self.scan_dimension_x = scan_dimension_x
        self.scan_dimension_y = scan_dimension_y
        self.exposure_time_ms = exposure_time_ms

    def __repr__(self):
        return (f"DatasetSpec(overlap={self.overlap}, sidelap={self.sidelap}, "
                f"height={self.height}m, scan=({self.scan_dimension_x}x{self.scan_dimension_y}), "
                f"exposure={self.exposure_time_ms}ms)")

class Camera:
    """
    Data model for a simple pinhole camera.
    """
    pass

class Waypoint:
    """
    Waypoints are positions where the drone should fly to and capture a photo.
    """
    pass