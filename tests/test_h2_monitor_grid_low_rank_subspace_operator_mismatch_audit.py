"""Smoke tests for the low-rank subspace operator mismatch audit."""

from importlib import import_module
from math import isfinite

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_low_rank_subspace_operator_mismatch_audit")

    assert hasattr(module, "run_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit")
    assert hasattr(module, "print_h2_monitor_grid_low_rank_subspace_operator_mismatch_summary")


def test_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit_returns_finite_metrics() -> None:
    from isogrid.audit.h2_monitor_grid_low_rank_subspace_operator_mismatch_audit import (
        run_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    assert result.spin_state_label == "singlet"
    assert result.controller_name == "generic_charge_spin_preconditioned"
    assert result.subspace_rank >= 1
    assert len(result.mode_explained_fractions) == result.subspace_rank
    assert isfinite(result.operator_mismatch_frobenius_norm)
    assert isfinite(result.diagonal_gain_shortfall_norm)
    assert isfinite(result.off_diagonal_coupling_norm)
    assert len(result.samples) >= 2


def test_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit_exposes_square_operators() -> None:
    from isogrid.audit.h2_monitor_grid_low_rank_subspace_operator_mismatch_audit import (
        run_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=12,
        late_window_size=5,
        controller_name="generic_charge_spin_preconditioned",
    )

    rank = result.subspace_rank
    assert len(result.projected_update_operator) == rank
    assert len(result.realized_next_residual_operator) == rank
    assert len(result.residual_reduction_operator) == rank
    assert all(len(row) == rank for row in result.projected_update_operator)
    assert all(len(row) == rank for row in result.realized_next_residual_operator)
    assert all(len(row) == rank for row in result.residual_reduction_operator)
