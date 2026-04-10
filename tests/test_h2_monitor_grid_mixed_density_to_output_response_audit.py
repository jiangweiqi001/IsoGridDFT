"""Smoke tests for the mixed-density to output-density response audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_mixed_density_to_output_response_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_mixed_density_to_output_response_audit")

    assert hasattr(module, "run_h2_monitor_grid_mixed_density_to_output_response_audit")
    assert hasattr(module, "print_h2_monitor_grid_mixed_density_to_output_response_summary")


def test_h2_monitor_grid_mixed_density_to_output_response_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_mixed_density_to_output_response_audit import (
        run_h2_monitor_grid_mixed_density_to_output_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_mixed_density_to_output_response_audit(
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
    assert len(result.samples) >= 2
    assert any(
        sample.next_output_response_to_mixed_ratio is not None for sample in result.samples[:-1]
    )
    assert isfinite(result.current_mixed_update_to_mode_ratio)


def test_h2_monitor_grid_mixed_density_to_output_response_audit_identifies_response_regime() -> None:
    from isogrid.audit.h2_monitor_grid_mixed_density_to_output_response_audit import (
        run_h2_monitor_grid_mixed_density_to_output_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_mixed_density_to_output_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.response_regime in {"follow", "neutral", "counteract"}


def test_h2_monitor_grid_mixed_density_to_output_response_audit_accepts_projector_route() -> None:
    from isogrid.audit.h2_monitor_grid_mixed_density_to_output_response_audit import (
        run_h2_monitor_grid_mixed_density_to_output_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_mixed_density_to_output_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
        singlet_experimental_route_name="projector_mixing",
    )

    assert result.spin_state_label == "singlet"
    assert result.response_regime in {"follow", "neutral", "counteract"}
