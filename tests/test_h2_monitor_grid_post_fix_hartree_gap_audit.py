"""Smoke tests for the post-fix baseline vs freeze-Hartree gap audit."""

from importlib import import_module

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_post_fix_hartree_gap_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_post_fix_hartree_gap_audit")

    assert hasattr(module, "run_h2_monitor_grid_post_fix_hartree_gap_audit")
    assert hasattr(module, "print_h2_monitor_grid_post_fix_hartree_gap_summary")


def test_h2_monitor_grid_post_fix_hartree_gap_audit_reports_pair_gaps() -> None:
    from isogrid.audit.h2_monitor_grid_post_fix_hartree_gap_audit import (
        run_h2_monitor_grid_post_fix_hartree_gap_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_post_fix_hartree_gap_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=3,
    )

    assert result.singlet.pair_gaps
    assert result.triplet.pair_gaps
    singlet_pair = result.singlet.pair_gaps[0]
    assert singlet_pair.pair_iterations == (1, 2)
    assert np.isfinite(singlet_pair.baseline_minus_freeze_hartree_density_residual)
    assert np.isfinite(singlet_pair.baseline_minus_freeze_hartree_overlap_abs)
    assert singlet_pair.baseline_hartree_share is None or np.isfinite(singlet_pair.baseline_hartree_share)
    assert isinstance(result.singlet.verdict, str)
