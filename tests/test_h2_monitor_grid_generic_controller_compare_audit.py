"""Smoke tests for the generic-controller vs baseline H2 monitor-grid audit."""

from importlib import import_module

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_generic_controller_compare_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_generic_controller_compare_audit")

    assert hasattr(module, "run_h2_monitor_grid_generic_controller_compare_audit")
    assert hasattr(module, "print_h2_monitor_grid_generic_controller_compare_summary")


def test_h2_monitor_grid_generic_controller_compare_audit_improves_singlet_gap() -> None:
    from isogrid.audit.h2_monitor_grid_generic_controller_compare_audit import (
        run_h2_monitor_grid_generic_controller_compare_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_generic_controller_compare_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=6,
        short_run_iterations=6,
    )

    assert result.baseline.singlet_max_targeted_density_gap > 0.20
    assert result.generic_charge_spin.singlet_max_targeted_density_gap < 0.10
    assert (
        result.generic_charge_spin.singlet_max_targeted_density_gap
        < result.baseline.singlet_max_targeted_density_gap
    )
    assert (
        result.generic_charge_spin.triplet_final_density_residual
        <= result.baseline.triplet_final_density_residual + 0.05
    )


def test_h2_monitor_grid_generic_controller_compare_audit_avoids_large_case_late_bad_pair() -> None:
    from isogrid.audit.h2_monitor_grid_generic_controller_compare_audit import (
        run_h2_monitor_grid_generic_controller_compare_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(11, 11, 13),
        box_half_extents=(7.0, 7.0, 9.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_generic_controller_compare_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=10,
        short_run_iterations=10,
    )

    assert result.baseline.singlet_has_late_targeted_bad_pair
    assert result.generic_charge_spin.singlet_max_targeted_density_gap < 0.10
    assert not result.generic_charge_spin.singlet_has_late_targeted_bad_pair


def test_h2_monitor_grid_generic_controller_compare_audit_avoids_xlarge_early_blow_up() -> None:
    from isogrid.audit.h2_monitor_grid_generic_controller_compare_audit import (
        run_h2_monitor_grid_generic_controller_compare_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(13, 13, 15),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_generic_controller_compare_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=8,
        short_run_iterations=8,
    )

    assert result.baseline.singlet_max_targeted_density_gap > 0.20
    assert result.generic_charge_spin.singlet_max_targeted_density_gap < 0.10
