"""Smoke tests for the late-step charge-mode compression audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_charge_mode_compression_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_charge_mode_compression_audit")

    assert hasattr(module, "run_h2_monitor_grid_charge_mode_compression_audit")
    assert hasattr(module, "print_h2_monitor_grid_charge_mode_compression_summary")


def test_h2_monitor_grid_charge_mode_compression_audit_returns_finite_platform_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_charge_mode_compression_audit import (
        run_h2_monitor_grid_charge_mode_compression_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_charge_mode_compression_audit(
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
    assert isfinite(result.principal_mode_residual_norm)
    assert isfinite(result.current_update_mode_coefficient)
    assert isfinite(result.current_update_to_residual_ratio)
    assert len(result.samples) == 5
    assert result.samples[0].iteration == 8
    assert result.samples[-1].iteration == 12
    assert any(sample.realized_next_mode_ratio is not None for sample in result.samples[:-1])


def test_h2_monitor_grid_charge_mode_compression_audit_identifies_nontrivial_platform_mode() -> None:
    from isogrid.audit.h2_monitor_grid_charge_mode_compression_audit import (
        run_h2_monitor_grid_charge_mode_compression_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_charge_mode_compression_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.principal_mode_explained_fraction > 0.20
    assert max(abs(sample.mode_residual_coefficient) for sample in result.samples) > 0.0
    assert result.mode_energy_fraction_captured > 0.20
