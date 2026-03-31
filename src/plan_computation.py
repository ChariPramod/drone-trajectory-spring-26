import typing as T
import math

from src.data_model import Camera, DatasetSpec, Waypoint
from src.camera_utils import (
    compute_image_footprint_on_surface,
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

    Args:
        camera: Camera model used for image capture.
        dataset_spec: user specification for the dataset.
        allowed_movement_px: The maximum allowed movement in pixels. Defaults to 1 px.

    Returns:
        The speed at which the drone should move during photo capture.
    """
    raise NotImplementedError()


def generate_photo_plan_on_grid(
    camera: Camera, dataset_spec: DatasetSpec
) -> T.List[Waypoint]:
    """Generate the complete photo plan as a list of waypoints in a lawn-mower pattern.

    Args:
        camera: Camera model used for image capture.
        dataset_spec: user specification for the dataset.

    Returns:
        Scan plan as a list of waypoints.

    """
    raise NotImplementedError()