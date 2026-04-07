"""Utility to visualize photo plans.
"""

import typing as T

import plotly.graph_objects as go

from src.data_model import Waypoint


def plot_photo_plan(photo_plans: T.List[Waypoint]) -> go.Figure:
    """Plot the photo plan on a 2D grid.

    Visualizes waypoints as markers connected by lines showing the flight path
    (lawn-mower pattern). Color indicates the row index for visual clarity.

    Args:
        photo_plans: List of waypoints for the photo plan.

    Returns:
        Plotly figure object.
    """
    x_coords = [wp.x for wp in photo_plans]
    y_coords = [wp.y for wp in photo_plans]

    fig = go.Figure()

    # Flight path as a line
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(color='royalblue', width=1, dash='dot'),
        name='Flight Path',
        showlegend=True,
    ))

    # Waypoints as markers
    speeds = [wp.speed for wp in photo_plans]
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            size=6,
            color=speeds,
            colorscale='Viridis',
            colorbar=dict(title='Speed (m/s)'),
        ),
        text=[f"({wp.x:.1f}, {wp.y:.1f})<br>z={wp.z:.1f}m<br>speed={wp.speed:.2f} m/s"
              for wp in photo_plans],
        hoverinfo='text',
        name='Waypoints',
        showlegend=True,
    ))

    # Mark start and end
    fig.add_trace(go.Scatter(
        x=[x_coords[0]],
        y=[y_coords[0]],
        mode='markers+text',
        marker=dict(size=12, color='green', symbol='star'),
        text=['Start'],
        textposition='top center',
        name='Start',
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[x_coords[-1]],
        y=[y_coords[-1]],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='x'),
        text=['End'],
        textposition='top center',
        name='End',
        showlegend=True,
    ))

    fig.update_layout(
        title='Drone Photo Plan – Lawn-Mower Grid',
        xaxis_title='X Position (m)',
        yaxis_title='Y Position (m)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        template='plotly_white',
        width=800,
        height=700,
    )

    return fig