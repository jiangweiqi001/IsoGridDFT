"""Smoke tests for the targeted bad-pair Hartree-gap audit."""

from importlib import import_module

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_targeted_bad_pair_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_targeted_bad_pair_audit")

    assert hasattr(module, "run_h2_monitor_grid_targeted_bad_pair_audit")
    assert hasattr(module, "print_h2_monitor_grid_targeted_bad_pair_summary")


def test_h2_monitor_grid_targeted_bad_pair_audit_reports_targeted_pairs() -> None:
    from isogrid.audit.h2_monitor_grid_targeted_bad_pair_audit import (
        run_h2_monitor_grid_targeted_bad_pair_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_targeted_bad_pair_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=4,
    )

    assert result.singlet.targeted_pairs
    assert result.triplet.targeted_pairs
    singlet_pair = result.singlet.targeted_pairs[0]
    assert singlet_pair.pair_iterations[0] < singlet_pair.pair_iterations[1]
    assert singlet_pair.baseline_minus_freeze_hartree_density_residual > 0.0
    assert singlet_pair.bad_pair_score > 0.0
    assert singlet_pair.baseline_hartree_share is None or np.isfinite(singlet_pair.baseline_hartree_share)
    assert isinstance(result.singlet.verdict, str)
    assert singlet_pair.baseline_minus_freeze_hartree_density_residual < 0.20
