#!/usr/bin/env python3
"""
Drone Trajectory Planner — Complete Execution & Visualization Script
=====================================================================
Runs the entire main.ipynb logic (Weeks 1-7) and generates all
Week 7 visualizations as interactive HTML files.
"""

import sys
import os
import copy
import math
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotly.io as pio
pio.renderers.default = 'browser'

from src.camera_utils import (
    compute_image_footprint_on_surface,
    compute_image_footprint_non_nadir,
    compute_ground_sampling_distance,
    project_world_point_to_image,
)
from src.data_model import Camera, DatasetSpec, Waypoint
from src.plan_computation import (
    compute_distance_between_images,
    compute_speed_during_photo_capture,
    generate_photo_plan_on_grid,
)
from src.visualization import (
    plot_photo_plan,
    plot_3d_trajectory,
    plot_footprint_overlay,
    plot_parameter_sensitivity,
    plot_overlap_sidelap_heatmap,
    plot_scenario_comparison,
    plot_angle_analysis,
    plot_efficiency_gauge,
)

# ──────────────────────────────────────────────────────────────────────────────
# Output dir
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.html")
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"  ✅ Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 2: Data Models
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("WEEK 2: Data Models")
print("=" * 70)

overlap = 0.7
sidelap = 0.7
height = 30.48  # 100 ft
scan_dimension_x = 150
scan_dimension_y = 150
exposure_time_ms = 2  # 1/500 exposure time

dataset_spec = DatasetSpec(overlap, sidelap, height, scan_dimension_x, scan_dimension_y, exposure_time_ms)
print(f"Nominal specs: {dataset_spec}")

# Camera: Skydio VT300L – Wide
fx = 4938.56
fy = 4936.49
sensor_size_x_mm = 13.107
sensor_size_y_mm = 9.830
num_pixels_x = 8192
num_pixels_y = 6144

camera_x10 = Camera(fx, fy, sensor_size_x_mm, sensor_size_y_mm, num_pixels_x, num_pixels_y)
print(f"X10 camera model: {camera_x10}")

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 3: Camera Operations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WEEK 3: Camera Operations")
print("=" * 70)

point_3d = (25, -30, 50)
expected_xy = (2469.28, -2961.894)
xy = project_world_point_to_image(camera_x10, point_3d)
print(f"{point_3d} projected to {xy}")
assert np.allclose(xy, expected_xy, atol=1e-2), f"Projection mismatch: {xy} vs {expected_xy}"

footprint_at_100m = compute_image_footprint_on_surface(camera_x10, 100)
expected_footprint_at_100m = (165.88, 124.46)
print(f"Footprint at 100m = {footprint_at_100m}")
assert np.allclose(footprint_at_100m, expected_footprint_at_100m, atol=1e-2)

footprint_at_200m = compute_image_footprint_on_surface(camera_x10, 200)
expected_footprint_at_200m = (165.88 * 2, 124.46 * 2)
print(f"Footprint at 200m = {footprint_at_200m}")
assert np.allclose(footprint_at_200m, expected_footprint_at_200m, atol=1e-2)

gsd_at_100m = compute_ground_sampling_distance(camera_x10, 100)
expected_gsd_at_100m = 0.0202
print(f"GSD at 100m: {gsd_at_100m}")
assert np.allclose(gsd_at_100m, expected_gsd_at_100m, atol=1e-4)

print("✅ All Week 3 assertions passed!")

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 4: Distance Between Photos
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WEEK 4: Distance Between Photos")
print("=" * 70)

computed_distances = compute_distance_between_images(camera_x10, dataset_spec)
expected_distances = np.array([15.17, 11.38], dtype=np.float32)
print(f"Computed distance for X10 camera with nominal dataset specs: {computed_distances}")
assert np.allclose(computed_distances, expected_distances, atol=1e-2)

# Non-nadir
fp_nadir = compute_image_footprint_on_surface(camera_x10, dataset_spec.height)
fp_at_90 = compute_image_footprint_non_nadir(camera_x10, dataset_spec.height, 90.0)
print(f"Nadir footprint:      ({fp_nadir[0]:.4f}m, {fp_nadir[1]:.4f}m)")
print(f"Non-nadir at 90°:     ({fp_at_90[0]:.4f}m, {fp_at_90[1]:.4f}m)")

print("\nAngle comparison table:")
print(f"{'Angle':>6} | {'Footprint X':>12} | {'Footprint Y':>12} | {'Distance X':>12} | {'Distance Y':>12}")
print("-" * 72)
for angle in [90, 80, 70, 60, 50, 45]:
    spec = DatasetSpec(0.7, 0.7, 30.48, 150, 150, 2, camera_angle=angle)
    d = compute_distance_between_images(camera_x10, spec)
    fp = compute_image_footprint_non_nadir(camera_x10, 30.48, angle)
    print(f"{angle:>5}° | {fp[0]:>10.2f}m | {fp[1]:>10.2f}m | {d[0]:>10.2f}m | {d[1]:>10.2f}m")

print("✅ All Week 4 assertions passed!")

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 5: Maximum Speed
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WEEK 5: Maximum Speed for Blur-Free Photos")
print("=" * 70)

computed_speed = compute_speed_during_photo_capture(camera_x10, dataset_spec, allowed_movement_px=1)
expected_speed = 3.09
print(f"Computed speed during photo captures: {computed_speed:.2f}")
assert np.allclose(computed_speed, expected_speed, atol=1e-2)
print("✅ All Week 5 assertions passed!")

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 6: Full Flight Plans
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WEEK 6: Generate Full Flight Plans")
print("=" * 70)

computed_plan = generate_photo_plan_on_grid(camera_x10, dataset_spec)
print(f"Computed plan with {len(computed_plan)} waypoints")

for idx, waypoint in enumerate(computed_plan[:10]):
    print(f"  Idx {idx}: {waypoint}")
print("  ...")

# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 7: Visualizations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WEEK 7: Comprehensive Visualizations")
print("=" * 70)

# 1. Basic 2D Flight Plan
print("\n📊 Generating Visualization 1/8: 2D Flight Plan...")
fig1 = plot_photo_plan(computed_plan)
save_fig(fig1, "01_flight_plan_2d")

# 2. 3D Trajectory
print("📊 Generating Visualization 2/8: 3D Flight Trajectory...")
fig2 = plot_3d_trajectory(computed_plan, dataset_spec)
save_fig(fig2, "02_flight_trajectory_3d")

# 3. Camera Footprint Overlay
print("📊 Generating Visualization 3/8: Camera Footprint Overlay...")
fig3 = plot_footprint_overlay(camera_x10, dataset_spec, computed_plan)
save_fig(fig3, "03_footprint_overlay")

# 4. Parameter Sensitivity Dashboard
print("📊 Generating Visualization 4/8: Parameter Sensitivity...")
fig4 = plot_parameter_sensitivity(camera_x10, dataset_spec)
save_fig(fig4, "04_parameter_sensitivity")

# 5. Overlap/Sidelap Heatmap
print("📊 Generating Visualization 5/8: Overlap/Sidelap Heatmap...")
fig5 = plot_overlap_sidelap_heatmap(camera_x10, dataset_spec)
save_fig(fig5, "05_overlap_sidelap_heatmap")

# 6. Multi-Scenario Comparison
print("📊 Generating Visualization 6/8: Scenario Comparison...")
scenarios = [
    ("Nominal\n70%/70%", DatasetSpec(0.7, 0.7, 30.48, 150, 150, 2)),
    ("Low Overlap\n50%/50%", DatasetSpec(0.5, 0.5, 30.48, 150, 150, 2)),
    ("High Overlap\n85%/85%", DatasetSpec(0.85, 0.85, 30.48, 150, 150, 2)),
    ("High Alt\n60m", DatasetSpec(0.7, 0.7, 60, 150, 150, 2)),
    ("Low Alt\n15m", DatasetSpec(0.7, 0.7, 15, 150, 150, 2)),
    ("Slow Exp\n10ms", DatasetSpec(0.7, 0.7, 30.48, 150, 150, 10)),
]
fig6 = plot_scenario_comparison(camera_x10, scenarios)
save_fig(fig6, "06_scenario_comparison")

# 7. Non-Nadir Angle Analysis
print("📊 Generating Visualization 7/8: Camera Angle Analysis...")
fig7 = plot_angle_analysis(camera_x10, dataset_spec)
save_fig(fig7, "07_angle_analysis")

# 8. Efficiency Gauges
print("📊 Generating Visualization 8/8: Efficiency Metrics...")
fig8 = plot_efficiency_gauge(camera_x10, dataset_spec, computed_plan)
save_fig(fig8, "08_efficiency_gauges")

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment: Change exposure time (existing experiment from notebook)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📊 Bonus: Exposure time experiment...")
dataset_spec_slow = copy.deepcopy(dataset_spec)
dataset_spec_slow.exposure_time_ms = 1000
plan_slow = generate_photo_plan_on_grid(camera_x10, dataset_spec_slow)
fig_exp = plot_photo_plan(plan_slow, title='🛩️ Flight Plan — 1000ms Exposure Time')
save_fig(fig_exp, "09_experiment_slow_exposure")

print("\n" + "=" * 70)
print(f"✨ ALL DONE! {9} visualizations saved to: {OUT_DIR}/")
print("=" * 70)
print("\nOpen any .html file in your browser to view the interactive plots.")
