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
    assert np.isfinite(result.gaussian_centered_difference.monitor_minus_legacy_hartree_energy_mha)
    assert np.isfinite(result.gaussian_shift_sensitivity.shifted_minus_centered_hartree_energy_mha)
    assert np.isfinite(result.h2_frozen_difference.monitor_minus_legacy_hartree_energy_mha)
