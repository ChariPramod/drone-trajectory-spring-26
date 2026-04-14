"""Utility to visualize photo plans — Week 7 comprehensive visualizations.

Includes:
- 2D flight path plot with waypoint markers
- 3D flight trajectory visualization
- Camera footprint overlay
- Parameter sensitivity analysis dashboards
- Multi-scenario comparison plots
"""

import typing as T
import math
import copy
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_model import Camera, DatasetSpec, Waypoint
from src.camera_utils import (
    compute_image_footprint_on_surface,
    compute_image_footprint_non_nadir,
    compute_ground_sampling_distance,
)
from src.plan_computation import (
    compute_distance_between_images,
    compute_speed_during_photo_capture,
    generate_photo_plan_on_grid,
)


# ──────────────────────────────────────────────────────────────────────────────
# Color palette & theme constants
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366f1",       # indigo
    "secondary": "#06b6d4",     # cyan
    "accent": "#f59e0b",        # amber
    "success": "#10b981",       # emerald
    "danger": "#ef4444",        # red
    "surface": "#0f172a",       # slate-900
    "surface_light": "#1e293b", # slate-800
    "text": "#e2e8f0",          # slate-200
    "muted": "#94a3b8",         # slate-400
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#e2e8f0"),
    margin=dict(l=60, r=40, t=80, b=60),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_size=12,
        font_family="Inter, monospace",
        bordercolor="#6366f1",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Basic 2D Photo Plan
# ──────────────────────────────────────────────────────────────────────────────
def plot_photo_plan(photo_plans: T.List[Waypoint], title: str = None) -> go.Figure:
    """Plot the photo plan on a 2D grid with rich annotations.

    Args:
        photo_plans: List of waypoints for the photo plan.
        title: Optional custom title.

    Returns:
        Plotly figure object.
    """
    x_coords = [wp.x for wp in photo_plans]
    y_coords = [wp.y for wp in photo_plans]
    speeds = [wp.speed for wp in photo_plans]
    indices = list(range(len(photo_plans)))

    fig = go.Figure()

    # Flight path line with gradient-like effect
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='lines',
        line=dict(color=COLORS["primary"], width=1.5, dash='dot'),
        name='Flight Path', opacity=0.6,
        hoverinfo='skip',
    ))

    # Waypoints colored by sequence index
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(
            size=7, color=indices,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Waypoint #', font=dict(size=11)),
                thickness=15, len=0.6,
                tickfont=dict(size=10),
            ),
            line=dict(width=0.3, color='white'),
        ),
        text=[f"<b>WP {i}</b><br>x={wp.x:.1f} m<br>y={wp.y:.1f} m<br>"
              f"z={wp.z:.1f} m<br>speed={wp.speed:.2f} m/s"
              for i, wp in enumerate(photo_plans)],
        hoverinfo='text',
        name='Waypoints',
    ))

    # Start marker
    fig.add_trace(go.Scatter(
        x=[x_coords[0]], y=[y_coords[0]],
        mode='markers+text',
        marker=dict(size=16, color=COLORS["success"], symbol='star',
                    line=dict(width=1.5, color='white')),
        text=['START'], textposition='top center',
        textfont=dict(size=11, color=COLORS["success"]),
        name='Start', showlegend=True,
    ))

    # End marker
    fig.add_trace(go.Scatter(
        x=[x_coords[-1]], y=[y_coords[-1]],
        mode='markers+text',
        marker=dict(size=14, color=COLORS["danger"], symbol='x',
                    line=dict(width=1.5, color='white')),
        text=['END'], textposition='top center',
        textfont=dict(size=11, color=COLORS["danger"]),
        name='End', showlegend=True,
    ))

    # Direction arrows (every 10th segment)
    for i in range(0, len(x_coords) - 1, max(1, len(x_coords) // 15)):
        fig.add_annotation(
            x=x_coords[i + 1], y=y_coords[i + 1],
            ax=x_coords[i], ay=y_coords[i],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=1.5, arrowcolor=COLORS["accent"],
            opacity=0.5,
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=title or '🛩️ Drone Photo Plan – Lawn-Mower Grid',
            font=dict(size=18),
        ),
        xaxis_title='X Position (m)',
        yaxis_title='Y Position (m)',
        xaxis=dict(scaleanchor='y', scaleratio=1, gridcolor='#334155',
                   zerolinecolor='#475569'),
        yaxis=dict(gridcolor='#334155', zerolinecolor='#475569'),
        width=900, height=750,
        legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155'),
    )

    # Stats annotation
    total_dist = sum(
        math.sqrt((x_coords[i+1]-x_coords[i])**2 + (y_coords[i+1]-y_coords[i])**2)
        for i in range(len(x_coords)-1)
    )
    avg_speed = sum(speeds) / len(speeds)
    stats_text = (
        f"<b>Plan Stats</b><br>"
        f"Waypoints: {len(photo_plans)}<br>"
        f"Total path: {total_dist:.1f} m<br>"
        f"Avg speed: {avg_speed:.2f} m/s<br>"
        f"Est. time: {total_dist/avg_speed:.1f} s"
    )
    fig.add_annotation(
        text=stats_text, xref="paper", yref="paper",
        x=1.0, y=0.0, xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=11, color=COLORS["text"]),
        bgcolor="rgba(30,41,59,0.9)",
        bordercolor=COLORS["primary"],
        borderwidth=1, borderpad=8,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. 3D Flight Trajectory
# ──────────────────────────────────────────────────────────────────────────────
def plot_3d_trajectory(photo_plans: T.List[Waypoint],
                       dataset_spec: DatasetSpec = None) -> go.Figure:
    """Interactive 3D visualization of the drone flight trajectory.

    Args:
        photo_plans: List of waypoints.
        dataset_spec: Optional spec to draw scan area boundary.

    Returns:
        Plotly figure.
    """
    x = [wp.x for wp in photo_plans]
    y = [wp.y for wp in photo_plans]
    z = [wp.z for wp in photo_plans]
    speeds = [wp.speed for wp in photo_plans]
    idx = list(range(len(photo_plans)))

    fig = go.Figure()

    # Ground plane grid
    if dataset_spec:
        sx, sy = dataset_spec.scan_dimension_x, dataset_spec.scan_dimension_y
        fig.add_trace(go.Mesh3d(
            x=[0, sx, sx, 0], y=[0, 0, sy, sy], z=[0, 0, 0, 0],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=COLORS["success"], opacity=0.12,
            name='Scan Area', showlegend=True,
            hoverinfo='skip',
        ))
        # Border lines
        corners_x = [0, sx, sx, 0, 0]
        corners_y = [0, 0, sy, sy, 0]
        corners_z = [0, 0, 0, 0, 0]
        fig.add_trace(go.Scatter3d(
            x=corners_x, y=corners_y, z=corners_z,
            mode='lines',
            line=dict(color=COLORS["success"], width=3),
            name='Scan Boundary',
            hoverinfo='skip',
        ))

    # Vertical drop-lines from waypoints to ground
    for i in range(0, len(photo_plans), max(1, len(photo_plans) // 20)):
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[i]], y=[y[i], y[i]], z=[0, z[i]],
            mode='lines',
            line=dict(color=COLORS["muted"], width=1, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))

    # Flight path
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=COLORS["primary"], width=3),
        name='Flight Path', opacity=0.7,
    ))

    # Waypoint markers
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4, color=idx,
            colorscale='Turbo',
            colorbar=dict(title='Waypoint #', thickness=12, len=0.5),
            line=dict(width=0.5, color='white'),
        ),
        text=[f"WP {i}<br>({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f})<br>"
              f"Speed: {wp.speed:.2f} m/s"
              for i, wp in enumerate(photo_plans)],
        hoverinfo='text',
        name='Waypoints',
    ))

    # Start / End
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers', marker=dict(size=10, color=COLORS["success"], symbol='diamond'),
        name='Start',
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers', marker=dict(size=10, color=COLORS["danger"], symbol='x'),
        name='End',
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='🚁 3D Flight Trajectory', font=dict(size=18)),
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Altitude (m)',
            xaxis=dict(backgroundcolor='#1e293b', gridcolor='#334155'),
            yaxis=dict(backgroundcolor='#1e293b', gridcolor='#334155'),
            zaxis=dict(backgroundcolor='#1e293b', gridcolor='#334155'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=950, height=750,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Camera Footprint Overlay
# ──────────────────────────────────────────────────────────────────────────────
def plot_footprint_overlay(camera: Camera, dataset_spec: DatasetSpec,
                           photo_plans: T.List[Waypoint],
                           max_footprints: int = 30) -> go.Figure:
    """Overlay camera footprints on the scan area to visualize coverage.

    Args:
        camera: Camera model.
        dataset_spec: Dataset specification.
        photo_plans: Generated waypoints.
        max_footprints: Max number of footprints to draw.

    Returns:
        Plotly figure.
    """
    fp_x, fp_y = compute_image_footprint_on_surface(camera, dataset_spec.height)
    half_x, half_y = fp_x / 2, fp_y / 2

    fig = go.Figure()

    # Scan area boundary
    sx, sy = dataset_spec.scan_dimension_x, dataset_spec.scan_dimension_y
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=sx, y1=sy,
        line=dict(color=COLORS["accent"], width=2, dash="dash"),
        fillcolor="rgba(245,158,11,0.05)",
    )

    # Footprint rectangles (sample evenly)
    step = max(1, len(photo_plans) // max_footprints)
    colorscale = px.colors.sample_colorscale('Viridis',
        [i / max(1, len(photo_plans) - 1) for i in range(0, len(photo_plans), step)])

    for ci, i in enumerate(range(0, len(photo_plans), step)):
        wp = photo_plans[i]
        c = colorscale[ci % len(colorscale)]
        fig.add_shape(
            type="rect",
            x0=wp.x - half_x, y0=wp.y - half_y,
            x1=wp.x + half_x, y1=wp.y + half_y,
            line=dict(color=c, width=1),
            fillcolor=c, opacity=0.12,
        )

    # Waypoints
    x_all = [wp.x for wp in photo_plans]
    y_all = [wp.y for wp in photo_plans]
    fig.add_trace(go.Scatter(
        x=x_all, y=y_all, mode='markers',
        marker=dict(size=4, color=COLORS["primary"], opacity=0.6),
        name='All Waypoints',
    ))

    # Highlighted footprints
    x_fp = [photo_plans[i].x for i in range(0, len(photo_plans), step)]
    y_fp = [photo_plans[i].y for i in range(0, len(photo_plans), step)]
    fig.add_trace(go.Scatter(
        x=x_fp, y=y_fp, mode='markers',
        marker=dict(size=6, color=COLORS["accent"],
                    symbol='circle', line=dict(width=1, color='white')),
        name='Shown Footprints',
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f'📷 Camera Footprint Overlay — {fp_x:.1f}×{fp_y:.1f} m per image',
            font=dict(size=16),
        ),
        xaxis_title='X (m)', yaxis_title='Y (m)',
        xaxis=dict(scaleanchor='y', scaleratio=1,
                   gridcolor='#334155', zerolinecolor='#475569'),
        yaxis=dict(gridcolor='#334155', zerolinecolor='#475569'),
        width=950, height=750,
        legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155'),
    )

    # Info box
    info = (
        f"<b>Footprint Info</b><br>"
        f"Single image: {fp_x:.2f} × {fp_y:.2f} m<br>"
        f"Scan area: {sx} × {sy} m<br>"
        f"Overlap: {dataset_spec.overlap*100:.0f}%<br>"
        f"Sidelap: {dataset_spec.sidelap*100:.0f}%<br>"
        f"Height: {dataset_spec.height:.1f} m"
    )
    fig.add_annotation(
        text=info, xref="paper", yref="paper",
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=11, color=COLORS["text"]),
        bgcolor="rgba(30,41,59,0.9)",
        bordercolor=COLORS["primary"],
        borderwidth=1, borderpad=8,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Parameter Sensitivity Dashboard
# ──────────────────────────────────────────────────────────────────────────────
def plot_parameter_sensitivity(camera: Camera, base_spec: DatasetSpec) -> go.Figure:
    """Dashboard showing how key outputs change with parameter variations.

    Produces a 2×2 subplot grid:
    - Top-left: Waypoints vs Overlap & Sidelap
    - Top-right: Max Speed vs Height
    - Bottom-left: Footprint Area vs Height
    - Bottom-right: GSD vs Height

    Args:
        camera: Camera model.
        base_spec: Baseline dataset spec.

    Returns:
        Plotly figure.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '🔄 Waypoints vs Overlap & Sidelap',
            '⚡ Max Speed vs Flight Height',
            '🗺️ Footprint Area vs Height',
            '🔬 GSD vs Height',
        ],
        horizontal_spacing=0.12, vertical_spacing=0.14,
    )

    # ── Panel 1: Waypoints vs Overlap/Sidelap ──
    overlaps = np.arange(0.3, 0.95, 0.05)
    sidelaps = [0.5, 0.6, 0.7, 0.8]
    colors_sl = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]]

    for j, sl in enumerate(sidelaps):
        wp_counts = []
        for ol in overlaps:
            spec = DatasetSpec(ol, sl, base_spec.height,
                               base_spec.scan_dimension_x, base_spec.scan_dimension_y,
                               base_spec.exposure_time_ms)
            plan = generate_photo_plan_on_grid(camera, spec)
            wp_counts.append(len(plan))
        fig.add_trace(go.Scatter(
            x=overlaps * 100, y=wp_counts,
            mode='lines+markers',
            marker=dict(size=5),
            line=dict(color=colors_sl[j], width=2),
            name=f'Sidelap {sl*100:.0f}%',
            legendgroup='panel1',
        ), row=1, col=1)

    # ── Panel 2: Max Speed vs Height ──
    heights = np.arange(10, 120, 5)
    exposures = [1, 2, 5, 10]
    colors_exp = ['#a78bfa', '#22d3ee', '#fb923c', '#f472b6']

    for j, exp in enumerate(exposures):
        sp = []
        for h in heights:
            spec = DatasetSpec(base_spec.overlap, base_spec.sidelap, h,
                               base_spec.scan_dimension_x, base_spec.scan_dimension_y,
                               exp)
            sp.append(compute_speed_during_photo_capture(camera, spec))
        fig.add_trace(go.Scatter(
            x=heights, y=sp,
            mode='lines',
            line=dict(color=colors_exp[j], width=2),
            name=f'Exp {exp} ms',
            legendgroup='panel2',
        ), row=1, col=2)

    # ── Panel 3: Footprint Area vs Height ──
    for h in heights:
        pass  # just collecting data below

    fp_areas = []
    fp_xs = []
    fp_ys = []
    for h in heights:
        fpx, fpy = compute_image_footprint_on_surface(camera, h)
        fp_areas.append(fpx * fpy)
        fp_xs.append(fpx)
        fp_ys.append(fpy)

    fig.add_trace(go.Scatter(
        x=heights, y=fp_areas,
        mode='lines',
        line=dict(color=COLORS["accent"], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(245,158,11,0.15)',
        name='Footprint Area',
        legendgroup='panel3',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=heights, y=fp_xs,
        mode='lines',
        line=dict(color=COLORS["primary"], width=1.5, dash='dash'),
        name='Footprint X',
        legendgroup='panel3',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=heights, y=fp_ys,
        mode='lines',
        line=dict(color=COLORS["secondary"], width=1.5, dash='dash'),
        name='Footprint Y',
        legendgroup='panel3',
    ), row=2, col=1)

    # ── Panel 4: GSD vs Height ──
    gsds = [compute_ground_sampling_distance(camera, h) for h in heights]
    fig.add_trace(go.Scatter(
        x=heights, y=[g * 100 for g in gsds],  # convert to cm
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(color=COLORS["danger"], width=2.5),
        name='GSD',
        legendgroup='panel4',
    ), row=2, col=2)

    # Add reference lines for common GSD thresholds
    for threshold, label in [(1.0, '1 cm/px'), (2.0, '2 cm/px'), (5.0, '5 cm/px')]:
        fig.add_hline(y=threshold, line_dash="dot",
                      line_color=COLORS["muted"], opacity=0.4,
                      annotation_text=label,
                      annotation_position="bottom right",
                      row=2, col=2)

    # Axis labels
    fig.update_xaxes(title_text='Overlap (%)', row=1, col=1, gridcolor='#334155')
    fig.update_yaxes(title_text='# Waypoints', row=1, col=1, gridcolor='#334155')
    fig.update_xaxes(title_text='Height (m)', row=1, col=2, gridcolor='#334155')
    fig.update_yaxes(title_text='Speed (m/s)', row=1, col=2, gridcolor='#334155')
    fig.update_xaxes(title_text='Height (m)', row=2, col=1, gridcolor='#334155')
    fig.update_yaxes(title_text='Area (m²) / Dim (m)', row=2, col=1, gridcolor='#334155')
    fig.update_xaxes(title_text='Height (m)', row=2, col=2, gridcolor='#334155')
    fig.update_yaxes(title_text='GSD (cm/px)', row=2, col=2, gridcolor='#334155')

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text='📊 Parameter Sensitivity Dashboard',
            font=dict(size=20), x=0.5,
        ),
        width=1100, height=850,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155',
            font=dict(size=10),
        ),
    )
    # Fix subplot title colors
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=COLORS["text"])

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Overlap/Sidelap Heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_overlap_sidelap_heatmap(camera: Camera, base_spec: DatasetSpec) -> go.Figure:
    """Heatmap of total waypoints and flight distance for overlap × sidelap grid.

    Args:
        camera: Camera model.
        base_spec: Baseline spec.

    Returns:
        Plotly figure with two side-by-side heatmaps.
    """
    overlaps = np.arange(0.3, 0.91, 0.05)
    sidelaps = np.arange(0.3, 0.91, 0.05)

    wp_matrix = np.zeros((len(sidelaps), len(overlaps)))
    dist_matrix = np.zeros((len(sidelaps), len(overlaps)))

    for i, sl in enumerate(sidelaps):
        for j, ol in enumerate(overlaps):
            spec = DatasetSpec(ol, sl, base_spec.height,
                               base_spec.scan_dimension_x, base_spec.scan_dimension_y,
                               base_spec.exposure_time_ms)
            plan = generate_photo_plan_on_grid(camera, spec)
            wp_matrix[i, j] = len(plan)
            # Total path distance
            total_d = sum(
                math.sqrt((plan[k+1].x - plan[k].x)**2 +
                          (plan[k+1].y - plan[k].y)**2)
                for k in range(len(plan) - 1)
            )
            dist_matrix[i, j] = total_d

    ol_labels = [f'{v*100:.0f}%' for v in overlaps]
    sl_labels = [f'{v*100:.0f}%' for v in sidelaps]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Total Waypoints', 'Total Flight Distance (m)'],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Heatmap(
        z=wp_matrix, x=ol_labels, y=sl_labels,
        colorscale='Viridis', colorbar=dict(title='Count', x=0.45, len=0.8),
        hovertemplate='Overlap: %{x}<br>Sidelap: %{y}<br>Waypoints: %{z}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=dist_matrix, x=ol_labels, y=sl_labels,
        colorscale='Inferno', colorbar=dict(title='Meters', x=1.0, len=0.8),
        hovertemplate='Overlap: %{x}<br>Sidelap: %{y}<br>Distance: %{z:.0f} m<extra></extra>',
    ), row=1, col=2)

    fig.update_xaxes(title_text='Overlap', row=1, col=1)
    fig.update_yaxes(title_text='Sidelap', row=1, col=1)
    fig.update_xaxes(title_text='Overlap', row=1, col=2)
    fig.update_yaxes(title_text='Sidelap', row=1, col=2)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text='🔥 Flight Plan Complexity — Overlap × Sidelap',
            font=dict(size=18), x=0.5,
        ),
        width=1100, height=550,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=COLORS["text"])

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Multi-Scenario Comparison
# ──────────────────────────────────────────────────────────────────────────────
def plot_scenario_comparison(camera: Camera, specs: T.List[T.Tuple[str, DatasetSpec]]) -> go.Figure:
    """Compare multiple flight plan scenarios side-by-side with key metrics.

    Args:
        camera: Camera model.
        specs: List of (label, DatasetSpec) tuples.

    Returns:
        Plotly figure.
    """
    labels = []
    waypoint_counts = []
    total_distances = []
    max_speeds = []
    gsds = []
    footprint_areas = []
    est_times = []

    for label, spec in specs:
        plan = generate_photo_plan_on_grid(camera, spec)
        speed = compute_speed_during_photo_capture(camera, spec)
        gsd = compute_ground_sampling_distance(camera, spec.height)
        fp_x, fp_y = compute_image_footprint_on_surface(camera, spec.height)

        total_d = sum(
            math.sqrt((plan[k+1].x - plan[k].x)**2 +
                      (plan[k+1].y - plan[k].y)**2)
            for k in range(len(plan) - 1)
        )

        labels.append(label)
        waypoint_counts.append(len(plan))
        total_distances.append(total_d)
        max_speeds.append(speed)
        gsds.append(gsd * 100)  # cm
        footprint_areas.append(fp_x * fp_y)
        est_times.append(total_d / speed if speed > 0 else 0)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            '📍 Waypoints', '📏 Flight Distance (m)', '⚡ Max Speed (m/s)',
            '🔬 GSD (cm/px)', '🗺️ Footprint Area (m²)', '⏱️ Est. Time (s)',
        ],
        vertical_spacing=0.18, horizontal_spacing=0.1,
    )

    bar_colors = px.colors.qualitative.Set2[:len(labels)]
    metrics = [waypoint_counts, total_distances, max_speeds,
               gsds, footprint_areas, est_times]
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

    for metric, (r, c) in zip(metrics, positions):
        fig.add_trace(go.Bar(
            x=labels, y=metric,
            marker_color=bar_colors,
            showlegend=False,
            text=[f'{v:.1f}' for v in metric],
            textposition='outside',
            textfont=dict(size=10, color=COLORS["text"]),
        ), row=r, col=c)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text='⚖️ Multi-Scenario Flight Plan Comparison',
            font=dict(size=18), x=0.5,
        ),
        width=1150, height=700,
        bargap=0.25,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=COLORS["text"])

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. Non-Nadir Angle Analysis
# ──────────────────────────────────────────────────────────────────────────────
def plot_angle_analysis(camera: Camera, base_spec: DatasetSpec) -> go.Figure:
    """Visualize how camera tilt angle affects footprint and flight plan.

    Args:
        camera: Camera model.
        base_spec: Baseline spec.

    Returns:
        Plotly figure.
    """
    angles = np.arange(45, 91, 1)
    fp_xs = []
    fp_ys = []
    fp_areas = []
    wps = []
    dists_x = []
    dists_y = []

    for angle in angles:
        try:
            fp_x, fp_y = compute_image_footprint_non_nadir(camera, base_spec.height, angle)
        except ValueError:
            fp_x, fp_y = np.nan, np.nan
        fp_xs.append(fp_x)
        fp_ys.append(fp_y)
        fp_areas.append(fp_x * fp_y if not np.isnan(fp_x) else np.nan)

        spec = DatasetSpec(base_spec.overlap, base_spec.sidelap, base_spec.height,
                           base_spec.scan_dimension_x, base_spec.scan_dimension_y,
                           base_spec.exposure_time_ms, camera_angle=angle)
        try:
            dx, dy = compute_distance_between_images(camera, spec)
        except:
            dx, dy = np.nan, np.nan
        dists_x.append(dx)
        dists_y.append(dy)

        try:
            plan = generate_photo_plan_on_grid(camera, spec)
            wps.append(len(plan))
        except:
            wps.append(np.nan)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '📐 Footprint Dimensions vs Angle',
            '📏 Inter-Image Distance vs Angle',
            '🗺️ Footprint Area vs Angle',
            '📍 Waypoints vs Angle',
        ],
        horizontal_spacing=0.12, vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(x=angles, y=fp_xs, mode='lines',
                             line=dict(color=COLORS["primary"], width=2.5),
                             name='Footprint X'), row=1, col=1)
    fig.add_trace(go.Scatter(x=angles, y=fp_ys, mode='lines',
                             line=dict(color=COLORS["secondary"], width=2.5),
                             name='Footprint Y'), row=1, col=1)

    fig.add_trace(go.Scatter(x=angles, y=dists_x, mode='lines',
                             line=dict(color=COLORS["accent"], width=2.5),
                             name='Dist X'), row=1, col=2)
    fig.add_trace(go.Scatter(x=angles, y=dists_y, mode='lines',
                             line=dict(color=COLORS["success"], width=2.5),
                             name='Dist Y'), row=1, col=2)

    fig.add_trace(go.Scatter(x=angles, y=fp_areas, mode='lines',
                             line=dict(color=COLORS["danger"], width=2.5),
                             fill='tozeroy',
                             fillcolor='rgba(239,68,68,0.1)',
                             name='Area'), row=2, col=1)

    fig.add_trace(go.Scatter(x=angles, y=wps, mode='lines+markers',
                             marker=dict(size=3),
                             line=dict(color=COLORS["primary"], width=2.5),
                             name='Waypoints'), row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text='Camera Angle (°)', row=r, col=c, gridcolor='#334155')

    fig.update_yaxes(title_text='Dimension (m)', row=1, col=1, gridcolor='#334155')
    fig.update_yaxes(title_text='Distance (m)', row=1, col=2, gridcolor='#334155')
    fig.update_yaxes(title_text='Area (m²)', row=2, col=1, gridcolor='#334155')
    fig.update_yaxes(title_text='# Waypoints', row=2, col=2, gridcolor='#334155')

    # Nadir reference line
    for r in [1, 2]:
        for c in [1, 2]:
            fig.add_vline(x=90, line_dash="dash", line_color=COLORS["muted"],
                          opacity=0.5, row=r, col=c)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text='📐 Camera Tilt Angle Impact Analysis',
            font=dict(size=18), x=0.5,
        ),
        width=1100, height=800,
        legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155'),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=COLORS["text"])

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8. Flight Efficiency Metrics
# ──────────────────────────────────────────────────────────────────────────────
def plot_efficiency_gauge(camera: Camera, dataset_spec: DatasetSpec,
                          photo_plans: T.List[Waypoint]) -> go.Figure:
    """Display key flight metrics as gauge indicators and a radar chart.

    Args:
        camera: Camera model.
        dataset_spec: Dataset spec.
        photo_plans: Generated plan.

    Returns:
        Plotly figure.
    """
    speed = compute_speed_during_photo_capture(camera, dataset_spec)
    gsd = compute_ground_sampling_distance(camera, dataset_spec.height) * 100  # cm
    fp_x, fp_y = compute_image_footprint_on_surface(camera, dataset_spec.height)

    x_all = [wp.x for wp in photo_plans]
    y_all = [wp.y for wp in photo_plans]
    total_dist = sum(
        math.sqrt((x_all[i+1]-x_all[i])**2 + (y_all[i+1]-y_all[i])**2)
        for i in range(len(x_all)-1)
    )
    scan_area = dataset_spec.scan_dimension_x * dataset_spec.scan_dimension_y
    est_time = total_dist / speed if speed > 0 else 0
    coverage_per_shot = fp_x * fp_y
    efficiency = min(100, scan_area / (len(photo_plans) * coverage_per_shot) * 100)

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}]*4],
        column_widths=[0.25, 0.25, 0.25, 0.25],
    )

    gauges = [
        ("Max Speed", speed, "m/s", 0, 20, [
            dict(range=[0, 5], color="#ef4444"),
            dict(range=[5, 12], color="#f59e0b"),
            dict(range=[12, 20], color="#10b981"),
        ]),
        ("GSD", gsd, "cm/px", 0, 5, [
            dict(range=[0, 1], color="#10b981"),
            dict(range=[1, 3], color="#f59e0b"),
            dict(range=[3, 5], color="#ef4444"),
        ]),
        ("Est. Time", est_time, "sec", 0, 300, [
            dict(range=[0, 60], color="#10b981"),
            dict(range=[60, 180], color="#f59e0b"),
            dict(range=[180, 300], color="#ef4444"),
        ]),
        ("Efficiency", efficiency, "%", 0, 100, [
            dict(range=[0, 40], color="#ef4444"),
            dict(range=[40, 70], color="#f59e0b"),
            dict(range=[70, 100], color="#10b981"),
        ]),
    ]

    for i, (title, val, suffix, lo, hi, steps) in enumerate(gauges):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(suffix=f" {suffix}", font=dict(size=18)),
            title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
            gauge=dict(
                axis=dict(range=[lo, hi], tickfont=dict(size=10)),
                bar=dict(color=COLORS["primary"]),
                bgcolor=COLORS["surface_light"],
                steps=steps,
                shape="angular",
            ),
        ), row=1, col=i+1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text='📈 Flight Efficiency Metrics',
            font=dict(size=18), x=0.5,
        ),
        width=1100, height=350,
    )

    return fig