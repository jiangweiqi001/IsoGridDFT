"""Smoke tests for the early-step SCF amplification/channel-ablation audit."""

from importlib import import_module

import numpy as np

from isogrid.audit.h2_monitor_grid_scf_amplification_ablation_audit import (
    H2MonitorGridScfAmplificationAblationAuditResult,
)
from isogrid.audit.h2_monitor_grid_scf_amplification_ablation_audit import (
    H2MonitorGridScfAmplificationPairAudit,
)
from isogrid.audit.h2_monitor_grid_scf_amplification_ablation_audit import (
    H2MonitorGridScfAmplificationReplayRoute,
)
from isogrid.audit.h2_monitor_grid_scf_amplification_ablation_audit import (
    H2MonitorGridScfAmplificationSpinAudit,
)
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_scf_amplification_ablation_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_scf_amplification_ablation_audit")

    assert hasattr(module, "run_h2_monitor_grid_scf_amplification_ablation_audit")
    assert hasattr(module, "print_h2_monitor_grid_scf_amplification_ablation_summary")


def test_construct_h2_monitor_grid_scf_amplification_ablation_result() -> None:
    route = H2MonitorGridScfAmplificationReplayRoute(
        route_label="baseline",
        pair_iterations=(1, 2),
        density_residual=0.3,
        density_residual_ratio=0.9,
        delta_hartree_potential_rms=0.2,
        delta_xc_potential_rms=0.1,
        delta_local_potential_rms=0.05,
        hartree_share=0.57,
        xc_share=0.29,
        local_share=0.14,
        occupied_orbital_overlap_abs=0.98,
        lowest2_subspace_overlap_min_singular_value=0.96,
        lowest2_subspace_rotation_max_angle_deg=8.0,
        lowest_gap_ha=0.05,
        lowest_gap_delta_ha=-0.01,
    )
    pair = H2MonitorGridScfAmplificationPairAudit(
        pair_iterations=(1, 2),
        baseline=route,
        freeze_hartree=route,
        freeze_xc=route,
    )
    spin = H2MonitorGridScfAmplificationSpinAudit(
        spin_state_label="singlet",
        source_iteration_count=2,
        density_residual_history=(0.5, 0.3),
        pair_audits=(pair,),
    )
    result = H2MonitorGridScfAmplificationAblationAuditResult(
        case_name=H2_BENCHMARK_CASE.name,
        grid_parameter_summary="shape=(9, 9, 11), box=(6, 6, 8)",
        singlet=spin,
        triplet=spin,
        note="smoke",
    )

    assert result.singlet.pair_audits[0].baseline.hartree_share == 0.57
    assert result.triplet.pair_audits[0].freeze_xc.lowest_gap_ha == 0.05


def test_h2_monitor_grid_scf_amplification_ablation_audit_reports_three_replay_routes() -> None:
    from isogrid.audit.h2_monitor_grid_scf_amplification_ablation_audit import (
        run_h2_monitor_grid_scf_amplification_ablation_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_scf_amplification_ablation_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=2,
        track_lowest_two_states=True,
    )

    singlet_pair = result.singlet.pair_audits[0]
    assert singlet_pair.baseline.route_label == "baseline"
    assert singlet_pair.freeze_hartree.route_label == "freeze_hartree"
    assert singlet_pair.freeze_xc.route_label == "freeze_xc"
    assert singlet_pair.freeze_hartree.delta_hartree_potential_rms <= 1.0e-10
    assert singlet_pair.freeze_xc.delta_xc_potential_rms <= 1.0e-10
    assert singlet_pair.baseline.occupied_orbital_overlap_abs is not None
    assert np.isfinite(singlet_pair.baseline.lowest2_subspace_rotation_max_angle_deg)
    assert np.isfinite(result.triplet.pair_audits[0].baseline.lowest_gap_ha)


def test_h2_monitor_grid_singlet_baseline_keeps_occupied_orbital_continuity() -> None:
    from isogrid.audit.h2_monitor_grid_scf_amplification_ablation_audit import (
        run_h2_monitor_grid_scf_amplification_ablation_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_scf_amplification_ablation_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=2,
        track_lowest_two_states=True,
    )

    singlet_pair = result.singlet.pair_audits[0]
    assert singlet_pair.baseline.occupied_orbital_overlap_abs is not None
    assert singlet_pair.baseline.occupied_orbital_overlap_abs >= 0.40
