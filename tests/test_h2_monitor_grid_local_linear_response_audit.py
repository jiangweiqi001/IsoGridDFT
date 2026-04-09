"""Smoke tests for the local linear-response H2 monitor-grid audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_local_linear_response_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_local_linear_response_audit")

    assert hasattr(module, "run_h2_monitor_grid_local_linear_response_audit")
    assert hasattr(module, "print_h2_monitor_grid_local_linear_response_summary")


def test_h2_monitor_grid_local_linear_response_audit_returns_finite_probe_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_local_linear_response_audit import (
        run_h2_monitor_grid_local_linear_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_local_linear_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=6,
        probe_iteration=6,
        controller_name="generic_charge_spin_preconditioned",
    )

    for spin_audit in (result.singlet, result.triplet):
        for probe in (spin_audit.charge_probe, spin_audit.spin_probe):
            assert probe.input_charge_perturbation_norm >= 0.0
            assert probe.input_spin_perturbation_norm >= 0.0
            assert probe.hartree_response_norm >= 0.0
            assert probe.output_charge_response_norm >= 0.0
            assert probe.output_spin_response_norm >= 0.0
            assert probe.hartree_gain is not None and isfinite(probe.hartree_gain)
            assert probe.charge_gain is not None and isfinite(probe.charge_gain)
            if probe.spin_gain is not None:
                assert isfinite(probe.spin_gain)
            assert probe.lowest2_subspace_rotation_max_angle_deg is not None


def test_h2_monitor_grid_local_linear_response_audit_distinguishes_charge_and_spin_channels() -> None:
    from isogrid.audit.h2_monitor_grid_local_linear_response_audit import (
        run_h2_monitor_grid_local_linear_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(13, 13, 15),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_local_linear_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.singlet.charge_probe.perturbation_channel == "charge"
    assert result.singlet.spin_probe.perturbation_channel == "spin"
    assert (
        result.singlet.charge_probe.input_charge_perturbation_norm
        > result.singlet.charge_probe.input_spin_perturbation_norm
    )
    assert (
        result.singlet.spin_probe.input_spin_perturbation_norm
        > result.singlet.spin_probe.input_charge_perturbation_norm
    )
    assert (
        result.singlet.charge_probe.hartree_gain
        > result.singlet.spin_probe.hartree_gain
    )
    assert result.triplet.charge_probe.perturbation_channel == "charge"
    assert result.triplet.spin_probe.perturbation_channel == "spin"
