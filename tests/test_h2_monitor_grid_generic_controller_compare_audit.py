"""Smoke tests for the generic-controller vs baseline H2 monitor-grid audit."""

from importlib import import_module

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.audit.h2_monitor_grid_targeted_bad_pair_audit import (
    run_h2_monitor_grid_targeted_bad_pair_audit,
)
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.scf import run_h2_monitor_grid_scf_dry_run


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


def test_h2_monitor_grid_generic_controller_compare_audit_avoids_xlarge_late_bad_pair() -> None:
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
        source_iteration_count=12,
        short_run_iterations=12,
    )

    assert result.baseline.singlet_has_late_targeted_bad_pair
    assert result.generic_charge_spin.singlet_max_targeted_density_gap < 0.10
    assert not result.generic_charge_spin.singlet_has_late_targeted_bad_pair


def test_preconditioned_controller_reduces_xlarge_and_xxlarge_plateau() -> None:
    for shape, extents, expected_absolute_gain in (
        ((13, 13, 15), (8.0, 8.0, 10.0), 1.0e-2),
        ((15, 15, 17), (9.0, 9.0, 11.0), 0.0),
    ):
        grid_geometry = build_monitor_grid_for_case(
            H2_BENCHMARK_CASE,
            shape=shape,
            box_half_extents=extents,
            element_parameters=build_h2_local_patch_development_element_parameters(),
        )
        generic = run_h2_monitor_grid_scf_dry_run(
            "singlet",
            case=H2_BENCHMARK_CASE,
            grid_geometry=grid_geometry,
            max_iterations=12,
            mixing=0.2,
            density_tolerance=1.0e-2,
            energy_tolerance=1.0e-4,
            eigensolver_tolerance=1.0e-2,
            eigensolver_ncv=8,
            controller_name="generic_charge_spin",
        )
        preconditioned = run_h2_monitor_grid_scf_dry_run(
            "singlet",
            case=H2_BENCHMARK_CASE,
            grid_geometry=grid_geometry,
            max_iterations=12,
            mixing=0.2,
            density_tolerance=1.0e-2,
            energy_tolerance=1.0e-4,
            eigensolver_tolerance=1.0e-2,
            eigensolver_ncv=8,
            controller_name="generic_charge_spin_preconditioned",
        )
        targeted = run_h2_monitor_grid_targeted_bad_pair_audit(
            case=H2_BENCHMARK_CASE,
            grid_geometry=grid_geometry,
            source_iteration_count=12,
            controller_name="generic_charge_spin_preconditioned",
        )

        max_gap = max(
            (
                pair.baseline_minus_freeze_hartree_density_residual
                for pair in targeted.singlet.targeted_pairs
            ),
            default=0.0,
        )
        assert (
            preconditioned.density_residual_history[-1]
            <= generic.density_residual_history[-1] - expected_absolute_gain + 1.0e-9
        )
        assert max_gap < 0.10
