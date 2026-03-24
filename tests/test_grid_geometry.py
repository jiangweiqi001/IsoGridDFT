"""Sanity checks for the structured adaptive grid geometry layer."""

from __future__ import annotations

import numpy as np

from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_default_h2_grid_spec
from isogrid.grid import build_axis_mapping


def test_h2_default_grid_can_be_constructed() -> None:
    geometry = build_default_h2_grid_geometry()

    assert geometry.spec.name == "h2_r1p4_structured_grid"


def test_axis_mapping_is_monotone() -> None:
    spec = build_default_h2_grid_spec()
    x_mapping = build_axis_mapping(spec.nx, spec.x_axis, spec.reference_center[0])

    assert np.all(np.diff(x_mapping.logical_coordinates) > 0.0)
    assert np.all(np.diff(x_mapping.physical_coordinates) > 0.0)


def test_grid_coordinate_shapes_are_correct() -> None:
    geometry = build_default_h2_grid_geometry()
    expected_shape = geometry.spec.shape

    assert geometry.x_points.shape == expected_shape
    assert geometry.y_points.shape == expected_shape
    assert geometry.z_points.shape == expected_shape
    assert geometry.point_jacobian.shape == expected_shape
    assert geometry.cell_volumes.shape == expected_shape


def test_geometric_weights_are_positive() -> None:
    geometry = build_default_h2_grid_geometry()

    assert np.all(geometry.point_jacobian > 0.0)
    assert np.all(geometry.cell_widths_x > 0.0)
    assert np.all(geometry.cell_widths_y > 0.0)
    assert np.all(geometry.cell_widths_z > 0.0)
    assert np.all(geometry.cell_volumes > 0.0)


def test_default_h2_grid_preserves_basic_symmetry() -> None:
    geometry = build_default_h2_grid_geometry()
    center_x, center_y, center_z = geometry.spec.reference_center
    center_ix = geometry.spec.nx // 2
    center_iy = geometry.spec.ny // 2
    center_iz = geometry.spec.nz // 2

    assert np.isclose(geometry.x_coordinates[center_ix], center_x)
    assert np.isclose(geometry.y_coordinates[center_iy], center_y)
    assert np.isclose(geometry.z_coordinates[center_iz], center_z)

    assert np.allclose(geometry.x_coordinates - center_x, -(geometry.x_coordinates[::-1] - center_x))
    assert np.allclose(geometry.y_coordinates - center_y, -(geometry.y_coordinates[::-1] - center_y))
    assert np.allclose(geometry.z_coordinates - center_z, -(geometry.z_coordinates[::-1] - center_z))
