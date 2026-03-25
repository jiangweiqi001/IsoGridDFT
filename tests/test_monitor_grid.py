"""Smoke tests for the new 3D monitor-driven grid core."""

import numpy as np

from isogrid.config import CO_AUDIT_CASE
from isogrid.config import H2O_AUDIT_CASE
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.config import N2_AUDIT_CASE
from isogrid.grid import build_monitor_grid_for_case


def _assert_valid_monitor_geometry(case) -> None:
    geometry = build_monitor_grid_for_case(case, shape=(21, 21, 21))

    assert geometry.x_points.shape == geometry.spec.shape
    assert geometry.monitor_field.values.shape == geometry.spec.shape
    assert np.all(np.isfinite(geometry.monitor_field.values))
    assert np.all(np.isfinite(geometry.jacobian))
    assert np.all(np.isfinite(geometry.cell_volumes))
    assert np.min(geometry.jacobian) > 0.0
    assert geometry.quality_report.has_nonpositive_jacobian is False
    assert geometry.quality_report.mean_near_atom_spacing < geometry.quality_report.mean_far_field_spacing
    assert geometry.metric_tensor.shape == geometry.spec.shape + (3, 3)
    assert geometry.inverse_metric_tensor.shape == geometry.spec.shape + (3, 3)
    assert len(geometry.patch_interfaces) == len(case.geometry.atoms)


def test_build_h2_monitor_geometry() -> None:
    _assert_valid_monitor_geometry(H2_BENCHMARK_CASE)


def test_build_n2_monitor_geometry() -> None:
    _assert_valid_monitor_geometry(N2_AUDIT_CASE)


def test_build_co_monitor_geometry() -> None:
    _assert_valid_monitor_geometry(CO_AUDIT_CASE)


def test_build_h2o_monitor_geometry() -> None:
    _assert_valid_monitor_geometry(H2O_AUDIT_CASE)
