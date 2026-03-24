"""Geometric quantities derived from a structured grid mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import build_axis_mappings
from .mapping import build_grid_point_coordinates
from .model import StructuredGridSpec


@dataclass(frozen=True)
class StructuredGridGeometry:
    """Structured grid coordinates and minimal geometric weights."""

    spec: StructuredGridSpec
    x_logical: np.ndarray
    y_logical: np.ndarray
    z_logical: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    z_coordinates: np.ndarray
    x_points: np.ndarray
    y_points: np.ndarray
    z_points: np.ndarray
    point_jacobian: np.ndarray
    cell_widths_x: np.ndarray
    cell_widths_y: np.ndarray
    cell_widths_z: np.ndarray
    cell_volumes: np.ndarray


def compute_cell_edges_1d(points: np.ndarray) -> np.ndarray:
    """Build point-centered cell edges from a monotone 1D grid."""

    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("At least two monotone 1D points are required.")
    if not np.all(np.diff(coordinates) > 0.0):
        raise ValueError("Cell edges require a strictly increasing 1D axis.")

    edges = np.empty(coordinates.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (coordinates[1:] + coordinates[:-1])
    edges[0] = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
    edges[-1] = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
    return edges


def compute_cell_widths_1d(points: np.ndarray) -> np.ndarray:
    """Compute positive point-centered 1D cell widths."""

    return np.diff(compute_cell_edges_1d(points))


def build_grid_geometry(spec: StructuredGridSpec) -> StructuredGridGeometry:
    """Build structured grid coordinates and minimal geometric weights."""

    x_mapping, y_mapping, z_mapping = build_axis_mappings(spec)
    x_points, y_points, z_points = build_grid_point_coordinates(
        x_mapping.physical_coordinates,
        y_mapping.physical_coordinates,
        z_mapping.physical_coordinates,
    )

    point_jacobian = (
        x_mapping.point_jacobian[:, None, None]
        * y_mapping.point_jacobian[None, :, None]
        * z_mapping.point_jacobian[None, None, :]
    )

    cell_widths_x = compute_cell_widths_1d(x_mapping.physical_coordinates)
    cell_widths_y = compute_cell_widths_1d(y_mapping.physical_coordinates)
    cell_widths_z = compute_cell_widths_1d(z_mapping.physical_coordinates)
    cell_volumes = (
        cell_widths_x[:, None, None]
        * cell_widths_y[None, :, None]
        * cell_widths_z[None, None, :]
    )

    return StructuredGridGeometry(
        spec=spec,
        x_logical=x_mapping.logical_coordinates,
        y_logical=y_mapping.logical_coordinates,
        z_logical=z_mapping.logical_coordinates,
        x_coordinates=x_mapping.physical_coordinates,
        y_coordinates=y_mapping.physical_coordinates,
        z_coordinates=z_mapping.physical_coordinates,
        x_points=x_points,
        y_points=y_points,
        z_points=z_points,
        point_jacobian=point_jacobian,
        cell_widths_x=cell_widths_x,
        cell_widths_y=cell_widths_y,
        cell_widths_z=cell_widths_z,
        cell_volumes=cell_volumes,
    )
