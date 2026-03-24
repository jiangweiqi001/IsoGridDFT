"""Structured logical-to-physical mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import AxisStretchSpec
from .model import StructuredGridSpec


@dataclass(frozen=True)
class AxisMapping:
    """One-dimensional logical and physical coordinates for one axis."""

    logical_coordinates: np.ndarray
    physical_coordinates: np.ndarray
    point_jacobian: np.ndarray


def logical_axis_coordinates(num_points: int) -> np.ndarray:
    """Create the canonical logical axis on [-1, 1]."""

    if num_points < 2:
        raise ValueError("At least two logical points are required per axis.")
    return np.linspace(-1.0, 1.0, num_points, dtype=np.float64)


def _stretched_fraction(fraction: np.ndarray, stretch: float) -> np.ndarray:
    if stretch == 0.0:
        return fraction
    return np.sinh(stretch * fraction) / np.sinh(stretch)


def _stretched_fraction_derivative(fraction: np.ndarray, stretch: float) -> np.ndarray:
    if stretch == 0.0:
        return np.ones_like(fraction)
    return stretch * np.cosh(stretch * fraction) / np.sinh(stretch)


def map_logical_to_physical_1d(
    logical_coordinates: np.ndarray,
    axis: AxisStretchSpec,
    center_coordinate: float,
) -> np.ndarray:
    """Map a logical 1D axis to physical coordinates.

    This first version uses a separable, piecewise-sinh stretch. It is a
    geometry-driven default, not a final adaptive strategy.
    """

    logical = np.asarray(logical_coordinates, dtype=np.float64)
    distance_from_center = np.abs(logical)
    stretched_fraction = _stretched_fraction(distance_from_center, axis.stretch)

    physical = np.empty_like(logical)
    negative_side = logical < 0.0
    positive_side = ~negative_side

    left_extent = -axis.lower_offset
    right_extent = axis.upper_offset

    physical[negative_side] = center_coordinate - left_extent * stretched_fraction[negative_side]
    physical[positive_side] = center_coordinate + right_extent * stretched_fraction[positive_side]
    return physical


def mapping_jacobian_1d(
    logical_coordinates: np.ndarray,
    axis: AxisStretchSpec,
) -> np.ndarray:
    """Return dx/du for one separable axis mapping."""

    logical = np.asarray(logical_coordinates, dtype=np.float64)
    distance_from_center = np.abs(logical)
    stretched_derivative = _stretched_fraction_derivative(distance_from_center, axis.stretch)

    jacobian = np.empty_like(logical)
    negative_side = logical < 0.0
    positive_side = ~negative_side

    left_extent = -axis.lower_offset
    right_extent = axis.upper_offset

    jacobian[negative_side] = left_extent * stretched_derivative[negative_side]
    jacobian[positive_side] = right_extent * stretched_derivative[positive_side]
    return jacobian


def build_axis_mapping(
    num_points: int,
    axis: AxisStretchSpec,
    center_coordinate: float,
) -> AxisMapping:
    """Build the logical and physical coordinates for one axis."""

    logical = logical_axis_coordinates(num_points)
    physical = map_logical_to_physical_1d(
        logical_coordinates=logical,
        axis=axis,
        center_coordinate=center_coordinate,
    )
    jacobian = mapping_jacobian_1d(logical_coordinates=logical, axis=axis)
    return AxisMapping(
        logical_coordinates=logical,
        physical_coordinates=physical,
        point_jacobian=jacobian,
    )


def build_axis_mappings(spec: StructuredGridSpec) -> tuple[AxisMapping, AxisMapping, AxisMapping]:
    """Build the three separable axis mappings for one grid specification."""

    center_x, center_y, center_z = spec.reference_center
    return (
        build_axis_mapping(spec.nx, spec.x_axis, center_x),
        build_axis_mapping(spec.ny, spec.y_axis, center_y),
        build_axis_mapping(spec.nz, spec.z_axis, center_z),
    )


def build_grid_point_coordinates(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine three 1D physical axes into a structured 3D grid."""

    return np.meshgrid(
        np.asarray(x_coordinates, dtype=np.float64),
        np.asarray(y_coordinates, dtype=np.float64),
        np.asarray(z_coordinates, dtype=np.float64),
        indexing="ij",
    )
