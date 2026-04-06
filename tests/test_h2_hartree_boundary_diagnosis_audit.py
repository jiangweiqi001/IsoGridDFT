"""Minimal smoke tests for the fixed-density Hartree boundary diagnosis audit."""

from importlib import import_module

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_monitor_grid


def test_hartree_boundary_diagnosis_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_hartree_boundary_diagnosis_audit")

    assert hasattr(module, "run_h2_hartree_boundary_diagnosis_audit")
    assert hasattr(module, "evaluate_fixed_density_hartree_route")


def test_h2_monitor_grid_baseline_shape_reflects_current_frozen_electrostatics_baseline() -> None:
    grid_geometry = build_h2_local_patch_development_monitor_grid()

    assert grid_geometry.spec.shape == (83, 83, 101)


def test_small_hartree_boundary_diagnosis_audit_is_finite() -> None:
    from isogrid.audit.h2_hartree_boundary_diagnosis_audit import (
        run_h2_hartree_boundary_diagnosis_audit,
    )

    result = run_h2_hartree_boundary_diagnosis_audit(
        case=H2_BENCHMARK_CASE,
        monitor_shape=(19, 19, 23),
        baseline_monitor_box_half_extents=(8.0, 8.0, 10.0),
        expanded_monitor_box_half_extents=(10.0, 10.0, 12.0),
        tolerance=1.0e-6,
        max_iterations=200,
    )

    assert result.primary_verdict in {
        "likely_electrostatics_or_boundary",
        "likely_scf_or_preconditioning",
        "mixed_or_inconclusive",
    }
    assert np.isfinite(result.gaussian_centered_monitor.total_charge)
    assert np.isfinite(result.gaussian_centered_monitor.dipole_norm)
    assert np.isfinite(result.gaussian_centered_monitor.quadrupole_norm)
    assert np.isfinite(result.monitor_volume_consistency.physical_box_volume)
    assert np.isfinite(result.monitor_volume_consistency.cell_volume_sum)
    assert np.isfinite(result.monitor_volume_consistency.trapezoidal_cell_volume_sum)
    assert np.isfinite(result.gaussian_representation_consistency.uniform_box_quadrupole_norm)
    assert np.isfinite(
        result.gaussian_representation_consistency.uniform_box_with_monitor_weights_quadrupole_norm
    )
    assert np.isfinite(
        result.gaussian_representation_consistency.mapped_monitor_with_uniform_weights_quadrupole_norm
    )
    assert np.isfinite(result.gaussian_representation_consistency.mapped_monitor_quadrupole_norm)
    assert np.isfinite(result.monitor_inversion_symmetry.coordinate_pairing_max_abs)
    assert np.isfinite(result.monitor_inversion_symmetry.cell_volume_pairing_max_abs)
    assert np.isfinite(result.monitor_inversion_symmetry.gaussian_density_pairing_rms)
    assert np.isfinite(result.monitor_inversion_symmetry.gaussian_dipole_integrand_pairing_rms)
    assert np.isfinite(result.gaussian_centered_difference.monitor_minus_legacy_hartree_energy_mha)
    assert np.isfinite(result.gaussian_shift_sensitivity.shifted_minus_centered_hartree_energy_mha)
    assert np.isfinite(result.h2_frozen_difference.monitor_minus_legacy_hartree_energy_mha)
    assert abs(result.monitor_volume_consistency.trapezoidal_relative_error) <= abs(
        result.monitor_volume_consistency.point_volume_relative_error
    )


def test_small_hartree_boundary_shape_sweep_is_finite_and_not_systematically_worse() -> None:
    from isogrid.audit.h2_hartree_boundary_diagnosis_audit import (
        run_h2_hartree_boundary_shape_sweep_audit,
    )

    result = run_h2_hartree_boundary_shape_sweep_audit(
        case=H2_BENCHMARK_CASE,
        shapes=((19, 19, 23), (23, 23, 27), (27, 27, 31)),
        baseline_monitor_box_half_extents=(8.0, 8.0, 10.0),
        expanded_monitor_box_half_extents=(10.0, 10.0, 12.0),
        tolerance=1.0e-6,
        max_iterations=200,
    )

    gaussian_quadrupoles = [point.gaussian_centered_monitor_quadrupole_norm for point in result.points]
    gaussian_gaps = [point.gaussian_monitor_minus_legacy_hartree_energy_mha for point in result.points]
    h2_gaps = [point.h2_frozen_monitor_minus_legacy_hartree_energy_mha for point in result.points]
    gaussian_box = [point.gaussian_box_expand_sensitivity_mha for point in result.points]
    h2_box = [point.h2_frozen_box_expand_sensitivity_mha for point in result.points]

    for sequence in (gaussian_quadrupoles, gaussian_gaps, h2_gaps, gaussian_box, h2_box):
        assert all(np.isfinite(sequence))

    assert result.trend_verdict in {
        "resolution_improving",
        "resolution_plateau",
        "resolution_mixed",
    }
    assert gaussian_quadrupoles[-1] <= gaussian_quadrupoles[0] + 1.0e-12
    assert gaussian_gaps[-1] <= gaussian_gaps[0] + 1.0e-12
    assert h2_gaps[-1] <= h2_gaps[0] + 1.0e-12
    assert gaussian_box[-1] <= gaussian_box[0] + 1.0e-12
    assert h2_box[-1] <= h2_box[0] + 1.0e-12


def test_small_hartree_measure_ledger_audit_is_finite() -> None:
    from isogrid.audit.h2_hartree_boundary_diagnosis_audit import (
        run_h2_hartree_measure_ledger_audit,
    )

    result = run_h2_hartree_measure_ledger_audit(
        case=H2_BENCHMARK_CASE,
        monitor_shape=(19, 19, 23),
        baseline_monitor_box_half_extents=(8.0, 8.0, 10.0),
    )

    assert len(result.path_summaries) >= 3
    assert result.path_summaries[0].measure_name == "cell_volumes"
    assert result.path_summaries[1].measure_name == "cell_volumes"
    assert result.path_summaries[2].measure_name == "identity_collocation"
    for path in result.path_summaries:
        for integral in path.integrals:
            assert np.isfinite(integral.value)
            assert np.isfinite(integral.reference_value)
            assert np.isfinite(integral.bias)


def test_small_hartree_geometry_representation_audit_is_finite() -> None:
    from isogrid.audit.h2_hartree_boundary_diagnosis_audit import (
        run_h2_hartree_geometry_representation_audit,
    )

    result = run_h2_hartree_geometry_representation_audit(
        case=H2_BENCHMARK_CASE,
        monitor_shape=(19, 19, 23),
        baseline_monitor_box_half_extents=(8.0, 8.0, 10.0),
    )

    assert np.isfinite(result.cell_volume_construction.logical_cell_volume)
    assert np.isfinite(result.cell_volume_construction.recomputed_cell_volume_sum)
    assert np.isfinite(result.cell_volume_construction.max_abs_cell_volume_difference)
    assert len(result.polynomial_exactness_rows) == 8
    for row in result.polynomial_exactness_rows:
        assert np.isfinite(row.reference_value)
        assert np.isfinite(row.uniform_weight_value)
        assert np.isfinite(row.current_cell_volume_value)
        assert np.isfinite(row.trapezoidal_adjusted_value)
        assert np.isfinite(row.uniform_coordinates_monitor_weights_value)
        assert np.isfinite(row.mapped_coordinates_uniform_weights_value)
    assert len(result.second_order_region_rows) == 4
    for row in result.second_order_region_rows:
        assert np.isfinite(row.boundary_mean_abs_mapping_distortion)
        assert np.isfinite(row.interior_mean_abs_mapping_distortion)
        assert np.isfinite(row.high_jacobian_mean_abs_mapping_distortion)
        assert np.isfinite(row.low_jacobian_mean_abs_mapping_distortion)
        assert np.isfinite(row.z_dominant_mean_abs_mapping_distortion)
        assert np.isfinite(row.xy_comparable_mean_abs_mapping_distortion)
    assert np.isfinite(result.error_region_summary.boundary_mean_abs_r2_mapping_distortion)
    assert np.isfinite(result.error_region_summary.high_jacobian_mean_abs_r2_mapping_distortion)
    assert np.isfinite(result.mapping_z_stretch_summary.uniform_physical_dz)
    assert np.isfinite(result.mapping_z_stretch_summary.high_jacobian_mean_abs_dz_dzeta)
    assert np.isfinite(result.mapping_z_stretch_summary.high_jacobian_mean_abs_second_z_variation)
