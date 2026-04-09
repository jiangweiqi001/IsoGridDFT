"""Smoke tests for the charge postprocess / clipping audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_charge_postprocess_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_charge_postprocess_audit")

    assert hasattr(module, "run_h2_monitor_grid_charge_postprocess_audit")
    assert hasattr(module, "print_h2_monitor_grid_charge_postprocess_summary")


def test_h2_monitor_grid_charge_postprocess_audit_returns_finite_clipping_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_charge_postprocess_audit import (
        run_h2_monitor_grid_charge_postprocess_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_charge_postprocess_audit(
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
    assert result.samples[0].iteration == 8
    assert result.samples[-1].iteration == 12
    for sample in result.samples:
        assert isfinite(sample.rho_charge_trial_min_value)
        assert sample.clipped_negative_charge_integral >= 0.0
        assert isfinite(sample.preclip_mode_projection)
        assert isfinite(sample.postclip_mode_projection)
        assert isfinite(sample.mode_projection_delta)


def test_h2_monitor_grid_charge_postprocess_audit_detects_nontrivial_clipping_or_projection_distortion() -> None:
    from isogrid.audit.h2_monitor_grid_charge_postprocess_audit import (
        run_h2_monitor_grid_charge_postprocess_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_charge_postprocess_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert (
        any(sample.clipped_negative_charge_integral > 0.0 for sample in result.samples)
        or any(abs(sample.mode_projection_delta) > 0.0 for sample in result.samples)
    )


def test_h2_monitor_grid_charge_postprocess_audit_reports_raw_vs_limited_trial_distinction() -> None:
    from isogrid.audit.h2_monitor_grid_charge_postprocess_audit import (
        run_h2_monitor_grid_charge_postprocess_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_charge_postprocess_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert any(sample.rho_charge_unbounded_trial_min_value <= sample.rho_charge_trial_min_value for sample in result.samples)
