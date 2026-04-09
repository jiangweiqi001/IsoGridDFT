"""Smoke tests for the plateau-mode lowest-state eigenvalue / occupation sensitivity audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit_module_imports() -> None:
    module = import_module(
        "isogrid.audit.h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit"
    )

    assert hasattr(
        module,
        "run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit",
    )
    assert hasattr(
        module,
        "print_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_summary",
    )


def test_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit import (
        run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit(
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
    assert isfinite(result.lowest_eigenvalue_signed_gain)
    assert isfinite(result.second_eigenvalue_signed_gain)
    assert isfinite(result.lowest_gap_signed_gain)
    assert isfinite(result.occupied_density_signed_gain)


def test_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit_classifies_spectral_regime() -> None:
    from isogrid.audit.h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit import (
        run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.spectral_sensitivity_regime in {"weak", "mixed", "strong"}


def test_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit_accepts_projector_route() -> None:
    from isogrid.audit.h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit import (
        run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        probe_iteration=12,
        controller_name="generic_charge_spin_preconditioned",
        singlet_experimental_route_name="projector_mixing",
    )

    assert result.spin_state_label == "singlet"
    assert result.spectral_sensitivity_regime in {"weak", "mixed", "strong"}
