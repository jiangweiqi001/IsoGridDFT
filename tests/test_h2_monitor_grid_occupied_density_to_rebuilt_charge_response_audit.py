"""Smoke tests for the occupied-density -> rebuilt-charge plateau-mode audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit_module_imports() -> None:
    module = import_module(
        "isogrid.audit.h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit"
    )

    assert hasattr(
        module,
        "run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit",
    )
    assert hasattr(
        module,
        "print_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_summary",
    )


def test_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit import (
        run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.spin_state_label == "singlet"
    assert result.controller_name == "generic_charge_spin_preconditioned"
    assert 0.0 < result.principal_mode_explained_fraction <= 1.0
    assert isfinite(result.occupied_density_signed_gain)
    assert isfinite(result.rebuilt_charge_density_signed_gain)
    assert isfinite(result.renormalized_residual_signed_gain)


def test_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit_identifies_sign_regime() -> None:
    from isogrid.audit.h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit import (
        run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.rebuilt_charge_sign_regime in {"positive", "negative", "near_zero"}


def test_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit_treats_tiny_stage_loss_as_inactive() -> None:
    from isogrid.audit.h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit import (
        run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert abs(result.occupied_to_rebuilt_stage_loss) < 1.0e-9
    assert "closely aligned" in result.verdict


def test_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit_accepts_projector_route() -> None:
    from isogrid.audit.h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit import (
        run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
        singlet_experimental_route_name="projector_mixing",
    )

    assert result.spin_state_label == "singlet"
    assert result.rebuilt_charge_sign_regime in {"positive", "negative", "near_zero"}
