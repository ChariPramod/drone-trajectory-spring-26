"""Utility functions for the camera model.
"""
import math

from src.data_model import Camera


def compute_focal_length_in_mm(camera: Camera) -> tuple[float, float]:
    """Computes the focal length in mm for the given camera

    Args:
        camera: the camera model.

    Returns:
        (fx, fy) in mm
    """
    pixel_to_mm_x = camera.sensor_size_x_mm / camera.num_pixels_x
    pixel_to_mm_y = camera.sensor_size_y_mm / camera.num_pixels_y

    return camera.fx * pixel_to_mm_x, camera.fy * pixel_to_mm_y


def project_world_point_to_image(camera: Camera, world_point: tuple[float, float, float]) -> tuple[float, float]:
    """Project a 3D world point into the image coordinates.

    Args:
        camera: the camera model
        world_point: the 3D world point

    Returns:
        (x, y) image coordinates on the film corresponding to world_point (in pixels).
    """
    X, Y, Z = world_point
    x = camera.fx * X / Z
    y = camera.fy * Y / Z
    return (x, y)


def compute_image_footprint_on_surface(
    camera: Camera, distance_from_surface: float
) -> tuple[float, float]:
    """Compute the footprint of the image captured by the camera at a given distance from the surface.

    Args:
        camera: the camera model.
        distance_from_surface: distance from the surface (in m).

    Returns:
        (footprint_x, footprint_y) in meters.
    """
    half_x = camera.num_pixels_x / 2
    half_y = camera.num_pixels_y / 2

    footprint_x = 2 * (half_x * distance_from_surface / camera.fx)
    footprint_y = 2 * (half_y * distance_from_surface / camera.fy)

    return (footprint_x, footprint_y)


def compute_ground_sampling_distance(
    camera: Camera, distance_from_surface: float
) -> float:
    """Compute the ground sampling distance (GSD) at a given distance from the surface.

    Args:
        camera: the camera model.
        distance_from_surface: distance from the surface (in m).

    Returns:
        The GSD in meters (smaller among x and y directions).
    """
    gsd_x = distance_from_surface / camera.fx
    gsd_y = distance_from_surface / camera.fy

    return min(gsd_x, gsd_y)


def compute_image_footprint_non_nadir(
    camera: Camera, distance_from_surface: float, camera_angle_deg: float
) -> tuple[float, float]:
    """Compute the footprint of the image on the ground when the camera is tilted.

    The camera_angle is measured from the horizontal X-axis:
    - 90° = nadir (straight down)
    - <90° = tilted forward

    For non-nadir angles, the footprint along the tilt direction (x) becomes
    a trapezoid. We compute the total footprint width on the ground.
    The perpendicular direction (y) scales by 1/cos(tilt_from_vertical).

    Args:
        camera: the camera model.
        distance_from_surface: height above ground (in m).
        camera_angle_deg: camera angle from horizontal X-axis (in degrees).

    Returns:
        (footprint_x, footprint_y) in meters.
    """
    h = distance_from_surface

    # Convert camera angle from horizontal to tilt from vertical
    # gamma = 0 means nadir, gamma > 0 means tilted
    gamma = math.radians(90.0 - camera_angle_deg)

    # Half field-of-view angles
    half_fov_x = math.atan(camera.num_pixels_x / (2 * camera.fx))
    half_fov_y = math.atan(camera.num_pixels_y / (2 * camera.fy))

    # Ground intersection angles from vertical
    angle_near = gamma - half_fov_x
    angle_far = gamma + half_fov_x

    # Check that the far edge ray hits the ground (angle < 90° from vertical)
    if angle_far >= math.pi / 2:
        raise ValueError(
            f"Camera angle {camera_angle_deg}° is too far from nadir: "
            f"far edge ray does not intersect the ground."
        )

    # Ground footprint along tilt direction (x)
    footprint_x = h * abs(math.tan(angle_far) - math.tan(angle_near))

    # Ground footprint perpendicular to tilt (y)
    # At center of footprint, the slant range is h / cos(gamma)
    # The y-footprint scales by the slant range
    footprint_y = camera.num_pixels_y * h / (camera.fy * math.cos(gamma))

    return (footprint_x, footprint_y)