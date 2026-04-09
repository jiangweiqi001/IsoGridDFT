"""Smoke tests for the preconditioner-variable comparison audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_preconditioner_variable_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_preconditioner_variable_audit")

    assert hasattr(module, "run_h2_monitor_grid_preconditioner_variable_audit")
    assert hasattr(module, "print_h2_monitor_grid_preconditioner_variable_summary")


def test_h2_monitor_grid_preconditioner_variable_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_preconditioner_variable_audit import (
        run_h2_monitor_grid_preconditioner_variable_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_preconditioner_variable_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.spin_state_label == "singlet"
    assert result.controller_name == "generic_charge_spin_preconditioned"
    assert len(result.variables) >= 4
    for sample in result.variables:
        assert 0.0 < sample.principal_mode_explained_fraction <= 1.0
        assert isfinite(sample.current_mode_coefficient)
        assert isfinite(sample.current_update_projection)
        if sample.last_realized_mode_ratio is not None:
            assert isfinite(sample.last_realized_mode_ratio)


def test_h2_monitor_grid_preconditioner_variable_audit_identifies_best_alignment() -> None:
    from isogrid.audit.h2_monitor_grid_preconditioner_variable_audit import (
        run_h2_monitor_grid_preconditioner_variable_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_preconditioner_variable_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    names = {sample.variable_name for sample in result.variables}
    assert "charge_residual" in names
    assert "smoothed_charge_residual" in names
    assert "support_weighted_charge_residual" in names
    assert "realized_charge_update" in names
    assert result.best_alignment_variable_name in names
