"""Smoke tests for the lowest-2 local continuity/response audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_lowest2_local_response_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_lowest2_local_response_audit")

    assert hasattr(module, "run_h2_monitor_grid_lowest2_local_response_audit")
    assert hasattr(module, "print_h2_monitor_grid_lowest2_local_response_summary")


def test_h2_monitor_grid_lowest2_local_response_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_lowest2_local_response_audit import (
        run_h2_monitor_grid_lowest2_local_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_lowest2_local_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=6,
        probe_iteration=6,
        controller_name="generic_charge_spin_preconditioned",
    )

    for spin_audit in (result.singlet, result.triplet):
        for probe in (spin_audit.charge_probe, spin_audit.spin_probe):
            assert probe.raw_lowest_orbital_overlap_abs is not None
            assert isfinite(probe.raw_lowest_orbital_overlap_abs)
            assert probe.best_in_subspace_occupied_overlap_abs is not None
            assert isfinite(probe.best_in_subspace_occupied_overlap_abs)
            assert probe.lowest2_subspace_rotation_max_angle_deg is not None
            assert isfinite(probe.lowest2_subspace_rotation_max_angle_deg)
            assert probe.projector_drift_frobenius_norm is not None
            assert isfinite(probe.projector_drift_frobenius_norm)
            assert probe.internal_rotation_angle_deg is not None
            assert isfinite(probe.internal_rotation_angle_deg)


def test_h2_monitor_grid_lowest2_local_response_audit_distinguishes_internal_rotation_from_subspace_drift() -> None:
    from isogrid.audit.h2_monitor_grid_lowest2_local_response_audit import (
        run_h2_monitor_grid_lowest2_local_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(13, 13, 15),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_lowest2_local_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
    )

    singlet_charge = result.singlet.charge_probe
    singlet_spin = result.singlet.spin_probe
    triplet_charge = result.triplet.charge_probe

    assert singlet_charge.best_in_subspace_occupied_overlap_abs > singlet_charge.raw_lowest_orbital_overlap_abs
    assert singlet_spin.best_in_subspace_occupied_overlap_abs > singlet_spin.raw_lowest_orbital_overlap_abs
    assert singlet_charge.internal_rotation_angle_deg > 45.0
    assert singlet_spin.internal_rotation_angle_deg > 45.0
    assert singlet_charge.best_in_subspace_occupied_overlap_abs > 0.9
    assert singlet_spin.best_in_subspace_occupied_overlap_abs > 0.9
    assert triplet_charge.internal_rotation_angle_deg < 10.0
