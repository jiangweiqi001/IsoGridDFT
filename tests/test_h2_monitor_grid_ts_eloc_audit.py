"""Minimal smoke tests for the H2 monitor-grid `T_s + E_loc,ion` audit."""

import numpy as np

from isogrid.audit.h2_monitor_grid_ts_eloc_audit import H2GridEnergyGeometrySummary
from isogrid.audit.h2_monitor_grid_ts_eloc_audit import H2TsElocGridResult
from isogrid.audit.h2_monitor_grid_ts_eloc_audit import evaluate_h2_singlet_ts_eloc_on_monitor_grid
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops import apply_monitor_grid_kinetic_operator
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential


def test_monitor_grid_kinetic_and_local_paths_run() -> None:
    grid_geometry = build_monitor_grid_for_case(H2_BENCHMARK_CASE, shape=(21, 21, 21))
    orbital = np.exp(
        -0.7
        * (
            (grid_geometry.x_points) ** 2
            + (grid_geometry.y_points) ** 2
            + (grid_geometry.z_points) ** 2
        )
    )
    kinetic_action = apply_monitor_grid_kinetic_operator(orbital, grid_geometry=grid_geometry)
    local_eval = evaluate_monitor_grid_local_ionic_potential(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )

    assert kinetic_action.shape == grid_geometry.spec.shape
    assert local_eval.total_local_potential.shape == grid_geometry.spec.shape
    assert np.all(np.isfinite(kinetic_action))
    assert np.all(np.isfinite(local_eval.total_local_potential))


def test_h2_monitor_grid_ts_eloc_values_are_finite() -> None:
    grid_geometry = build_monitor_grid_for_case(H2_BENCHMARK_CASE, shape=(21, 21, 21))
    kinetic, local_ionic, total, summary = evaluate_h2_singlet_ts_eloc_on_monitor_grid(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )

    assert np.isfinite(kinetic)
    assert np.isfinite(local_ionic)
    assert np.isfinite(total)
    assert summary.grid_type == "monitor_a_grid"
    assert summary.near_atom_spacing_bohr < summary.far_field_spacing_bohr


def test_construct_ts_eloc_result_object() -> None:
    summary = H2GridEnergyGeometrySummary(
        grid_type="monitor_a_grid",
        grid_shape=(29, 29, 29),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        min_spacing_estimate_bohr=0.12,
        near_core_min_spacing_bohr=0.11,
        near_atom_spacing_bohr=0.28,
        far_field_spacing_bohr=0.44,
        min_jacobian=1.0,
        max_jacobian=2.0,
        center_line_local_potential_center=-1.0,
        center_line_local_potential_near_atom=-2.0,
    )
    result = H2TsElocGridResult(
        grid_type="monitor_a_grid",
        geometry_summary=summary,
        kinetic_energy=0.8,
        local_ionic_energy=-2.0,
        ts_plus_eloc_energy=-1.2,
        reference_pyscf_total_energy=-1.13,
        reference_offset_ha=-0.07,
        reference_offset_mha=-70.0,
        improvement_vs_legacy_ha=0.01,
        improvement_vs_legacy_mha=10.0,
    )

    assert result.grid_type == "monitor_a_grid"
    assert result.ts_plus_eloc_energy == -1.2
    assert result.improvement_vs_legacy_mha == 10.0
