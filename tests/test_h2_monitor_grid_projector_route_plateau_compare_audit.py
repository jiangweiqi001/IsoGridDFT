"""Smoke tests for the projector-route plateau compare audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_projector_route_plateau_compare_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_projector_route_plateau_compare_audit")

    assert hasattr(module, "run_h2_monitor_grid_projector_route_plateau_compare_audit")
    assert hasattr(module, "print_h2_monitor_grid_projector_route_plateau_compare_summary")


def test_h2_monitor_grid_projector_route_plateau_compare_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_projector_route_plateau_compare_audit import (
        run_h2_monitor_grid_projector_route_plateau_compare_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_projector_route_plateau_compare_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.spin_state_label == "singlet"
    assert result.baseline_route_name == "none"
    assert result.projector_route_name == "projector_mixing"
    assert result.guarded_route_name == "guarded_projector_mixing"
    assert isfinite(result.baseline_occupied_density_signed_gain)
    assert isfinite(result.projector_occupied_density_signed_gain)
    assert isfinite(result.guarded_occupied_density_signed_gain)
    assert isfinite(result.baseline_current_mixed_update_to_mode_ratio)
    assert isfinite(result.projector_current_mixed_update_to_mode_ratio)
    assert isfinite(result.guarded_current_mixed_update_to_mode_ratio)


def test_h2_monitor_grid_projector_route_plateau_compare_audit_classifies_improvement_regime() -> None:
    from isogrid.audit.h2_monitor_grid_projector_route_plateau_compare_audit import (
        run_h2_monitor_grid_projector_route_plateau_compare_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(17, 17, 19),
        box_half_extents=(10.0, 10.0, 12.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_projector_route_plateau_compare_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.comparison_regime in {
        "response_amplified_only",
        "plateau_contraction_improved",
        "no_material_change",
    }
    assert result.guarded_next_output_response_regime in {
        "counteract",
        "neutral",
        "follow",
    }


def test_h2_monitor_grid_projector_route_plateau_compare_audit_exposes_guarded_route_delta() -> None:
    from isogrid.audit.h2_monitor_grid_projector_route_plateau_compare_audit import (
        run_h2_monitor_grid_projector_route_plateau_compare_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(17, 17, 19),
        box_half_extents=(10.0, 10.0, 12.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_projector_route_plateau_compare_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=16,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.guarded_route_name == "guarded_projector_mixing"
    assert result.guarded_next_output_response_regime in {"counteract", "neutral", "follow"}
    assert isfinite(result.guarded_occupied_density_signed_gain)
    assert isfinite(result.guarded_current_mixed_update_to_mode_ratio)
