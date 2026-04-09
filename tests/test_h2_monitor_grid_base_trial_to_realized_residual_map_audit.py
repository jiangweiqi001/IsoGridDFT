"""Smoke tests for the base-trial to realized-residual map audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_base_trial_to_realized_residual_map_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_base_trial_to_realized_residual_map_audit")

    assert hasattr(module, "run_h2_monitor_grid_base_trial_to_realized_residual_map_audit")
    assert hasattr(module, "print_h2_monitor_grid_base_trial_to_realized_residual_map_summary")


def test_h2_monitor_grid_base_trial_to_realized_residual_map_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_base_trial_to_realized_residual_map_audit import (
        run_h2_monitor_grid_base_trial_to_realized_residual_map_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_base_trial_to_realized_residual_map_audit(
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
    assert len(result.samples) == 5
    assert isfinite(max(abs(sample.mode_residual_coefficient) for sample in result.samples))


def test_h2_monitor_grid_base_trial_to_realized_residual_map_audit_identifies_a_dominant_loss_stage() -> None:
    from isogrid.audit.h2_monitor_grid_base_trial_to_realized_residual_map_audit import (
        run_h2_monitor_grid_base_trial_to_realized_residual_map_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_base_trial_to_realized_residual_map_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.dominant_loss_stage in {
        "trial_to_charge_next",
        "charge_next_to_postclip",
        "postclip_to_realized",
    }
