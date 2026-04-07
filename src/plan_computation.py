import typing as T
import math
import numpy as np

from src.data_model import Camera, DatasetSpec, Waypoint
from src.camera_utils import (
    compute_image_footprint_on_surface,
    compute_image_footprint_non_nadir,
    compute_ground_sampling_distance,
)


def compute_distance_between_images(
    camera: Camera, dataset_spec: DatasetSpec
) -> tuple[float, float]:
    """Compute the distance between images in the horizontal and vertical directions for specified overlap and sidelap.

    Args:
        camera: Camera model used for image capture.
        dataset_spec: user specification for the dataset.

    Returns:
        The horizontal and vertical distance between images (in meters).
    """
    # Use non-nadir footprint if camera_angle is not 90 (nadir)
    if hasattr(dataset_spec, 'camera_angle') and dataset_spec.camera_angle != 90.0:
        footprint_x, footprint_y = compute_image_footprint_non_nadir(
            camera, dataset_spec.height, dataset_spec.camera_angle
        )
    else:
        footprint_x, footprint_y = compute_image_footprint_on_surface(
            camera, dataset_spec.height
        )

    distance_x = footprint_x * (1 - dataset_spec.overlap)
    distance_y = footprint_y * (1 - dataset_spec.sidelap)

    return (distance_x, distance_y)


def compute_speed_during_photo_capture(
    camera: Camera, dataset_spec: DatasetSpec, allowed_movement_px: float = 1
) -> float:
    """Compute the speed of drone during an active photo capture to prevent more than 1px of motion blur.

    The maximum ground movement allowed during exposure equals GSD * allowed_movement_px.
    Speed = max_ground_movement / exposure_time.

    Args:
        camera: Camera model used for image capture.
        dataset_spec: user specification for the dataset.
        allowed_movement_px: The maximum allowed movement in pixels. Defaults to 1 px.

    Returns:
        The speed at which the drone should move during photo capture (m/s).
    """
    # GSD tells us how much ground (in meters) one pixel covers
    gsd = compute_ground_sampling_distance(camera, dataset_spec.height)

    # Maximum allowed ground movement during a single exposure
    max_movement_m = gsd * allowed_movement_px

    # Convert exposure time from milliseconds to seconds
    exposure_time_s = dataset_spec.exposure_time_ms / 1000.0

    # Speed = distance / time
    speed = max_movement_m / exposure_time_s

    return speed


def generate_photo_plan_on_grid(
    camera: Camera, dataset_spec: DatasetSpec
) -> T.List[Waypoint]:
    """Generate the complete photo plan as a list of waypoints in a lawn-mower pattern.

    Steps:
    1. Compute the maximum distance between two images (horizontal & vertical).
    2. Determine number of images needed to cover the scan area in each direction.
       If the scan dimension is not an exact multiple of the distance, round up.
    3. Distribute waypoints evenly across the scan area.
    4. Follow a lawn-mower (boustrophedon) pattern: alternate direction in each row.
    5. Assign speed from compute_speed_during_photo_capture.

    Args:
        camera: Camera model used for image capture.
        dataset_spec: user specification for the dataset.

    Returns:
        Scan plan as a list of waypoints.
    """
    distance_x, distance_y = compute_distance_between_images(camera, dataset_spec)
    speed = compute_speed_during_photo_capture(camera, dataset_spec)

    # Number of images needed along each axis (at least 1 image)
    num_images_x = max(1, math.ceil(dataset_spec.scan_dimension_x / distance_x))
    num_images_y = max(1, math.ceil(dataset_spec.scan_dimension_y / distance_y))

    # Evenly space waypoints across the scan area
    # If only 1 image, place it at center; otherwise linspace from 0 to scan_dim
    if num_images_x == 1:
        x_positions = [dataset_spec.scan_dimension_x / 2.0]
    else:
        x_positions = np.linspace(0, dataset_spec.scan_dimension_x, num_images_x).tolist()

    if num_images_y == 1:
        y_positions = [dataset_spec.scan_dimension_y / 2.0]
    else:
        y_positions = np.linspace(0, dataset_spec.scan_dimension_y, num_images_y).tolist()

    waypoints = []
    for row_idx, y in enumerate(y_positions):
        # Lawn-mower pattern: reverse x direction on odd rows
        row_x_positions = x_positions if row_idx % 2 == 0 else list(reversed(x_positions))
        for x in row_x_positions:
            wp = Waypoint(
                x=x,
                y=y,
                z=dataset_spec.height,
                speed=speed,
            )
            waypoints.append(wp)

    return waypoints