"""Focused tests for monitor-grid cell-local geometry helpers."""

from __future__ import annotations

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_monitor_cell_local_quadrature_recovers_box_volume_better_than_nodal_cell_volumes() -> None:
    from isogrid.grid.monitor_geometry import build_monitor_cell_local_quadrature

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(19, 19, 23),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    quadrature = build_monitor_cell_local_quadrature(grid_geometry)
    box_bounds = grid_geometry.spec.box_bounds
    physical_box_volume = (
        (box_bounds[0][1] - box_bounds[0][0])
        * (box_bounds[1][1] - box_bounds[1][0])
        * (box_bounds[2][1] - box_bounds[2][0])
    )
    nodal_error = abs(
        float(np.sum(grid_geometry.cell_volumes, dtype=np.float64)) - physical_box_volume
    )
    quadrature_error = abs(
        float(np.sum(quadrature.sample_weights, dtype=np.float64)) - physical_box_volume
    )

    assert quadrature_error < nodal_error
