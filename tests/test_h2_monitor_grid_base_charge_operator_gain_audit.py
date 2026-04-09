"""Smoke tests for the base charge-operator gain mismatch audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_base_charge_operator_gain_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_base_charge_operator_gain_audit")

    assert hasattr(module, "run_h2_monitor_grid_base_charge_operator_gain_audit")
    assert hasattr(module, "print_h2_monitor_grid_base_charge_operator_gain_summary")


def test_h2_monitor_grid_base_charge_operator_gain_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_base_charge_operator_gain_audit import (
        run_h2_monitor_grid_base_charge_operator_gain_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_base_charge_operator_gain_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.spin_state_label == "singlet"
    assert result.controller_name == "generic_charge_spin_preconditioned"
    assert 0.0 < result.principal_mode_explained_fraction <= 1.0
    assert isfinite(result.current_base_update_to_mode_ratio)
    assert isfinite(result.realized_residual_reduction_ratio)
    assert isfinite(result.base_gain_shortfall)
    assert len(result.samples) == 5


def test_h2_monitor_grid_base_charge_operator_gain_audit_identifies_base_gain_mismatch() -> None:
    from isogrid.audit.h2_monitor_grid_base_charge_operator_gain_audit import (
        run_h2_monitor_grid_base_charge_operator_gain_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_base_charge_operator_gain_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.principal_mode_explained_fraction > 0.5
    assert abs(result.base_gain_shortfall) > 0.0
