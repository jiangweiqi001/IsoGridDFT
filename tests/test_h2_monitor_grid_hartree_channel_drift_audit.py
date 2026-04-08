"""Lightweight smoke tests for the H2 Hartree-channel drift audit."""

from importlib import import_module

import numpy as np

from isogrid.audit.h2_monitor_grid_hartree_channel_drift_audit import (
    H2MonitorGridHartreeChannelDriftAuditResult,
)
from isogrid.audit.h2_monitor_grid_hartree_channel_drift_audit import (
    H2MonitorGridHartreeChannelDriftComparison,
)
from isogrid.audit.h2_monitor_grid_hartree_channel_drift_audit import (
    H2MonitorGridHartreeChannelDriftRoute,
)
from isogrid.audit.h2_monitor_grid_hartree_channel_drift_audit import (
    H2MonitorGridHartreeChannelDriftSecantPair,
)
from isogrid.audit.h2_monitor_grid_hartree_channel_drift_audit import (
    H2MonitorGridHartreeChannelDriftStep,
)
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_hartree_channel_drift_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_hartree_channel_drift_audit")

    assert hasattr(module, "run_h2_monitor_grid_hartree_channel_drift_audit")
    assert hasattr(module, "print_h2_monitor_grid_hartree_channel_drift_summary")


def test_construct_h2_monitor_grid_hartree_channel_drift_result() -> None:
    route = H2MonitorGridHartreeChannelDriftRoute(
        spin_state_label="singlet",
        boundary_construction_mode="corrected_moments",
        iteration_limit=3,
        steps=(
            H2MonitorGridHartreeChannelDriftStep(
                iteration=1,
                density_residual=0.2,
                boundary_value_correction_rms=1.0e-3,
                boundary_value_correction_max_abs=2.0e-3,
                hartree_tail_far_field_mean_abs=0.04,
                hartree_tail_far_field_signed_mean=0.03,
                effective_hartree_potential_rms=0.7,
                effective_total_potential_rms=1.6,
                effective_hartree_component_share=0.44,
            ),
        ),
        secant_pairs=(
            H2MonitorGridHartreeChannelDriftSecantPair(
                pair_iterations=(1, 2),
                density_secant_norm=0.1,
                density_residual_ratio=0.95,
                hartree_contribution_share=0.65,
                xc_contribution_share=0.15,
                local_orbital_contribution_share=0.20,
            ),
        ),
    )
    comparison = H2MonitorGridHartreeChannelDriftComparison(
        spin_state_label="singlet",
        legacy_split=route,
        corrected_moments=route,
        boundary_value_correction_rms_reduction=0.0,
        final_density_residual_change=0.0,
        last_secant_hartree_share_change=0.0,
        verdict="neutral",
    )
    result = H2MonitorGridHartreeChannelDriftAuditResult(
        case_name=H2_BENCHMARK_CASE.name,
        grid_parameter_summary="shape=(9, 9, 11), box=(6, 6, 8)",
        singlet=comparison,
        triplet=comparison,
        note="smoke",
    )

    assert result.singlet.corrected_moments.steps[0].effective_hartree_component_share == 0.44
    assert result.triplet.legacy_split.secant_pairs[0].pair_iterations == (1, 2)


def test_h2_monitor_grid_hartree_channel_drift_audit_reports_both_boundary_modes() -> None:
    from isogrid.audit.h2_monitor_grid_hartree_channel_drift_audit import (
        run_h2_monitor_grid_hartree_channel_drift_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_hartree_channel_drift_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        iteration_limit=2,
    )

    assert result.singlet.legacy_split.boundary_construction_mode == "legacy_split"
    assert result.singlet.corrected_moments.boundary_construction_mode == "corrected_moments"
    assert len(result.singlet.corrected_moments.steps) == 2
    assert len(result.triplet.legacy_split.steps) == 2
    assert result.singlet.corrected_moments.steps[-1].boundary_value_correction_rms is not None
    assert result.singlet.legacy_split.steps[-1].boundary_value_correction_rms is not None
    assert (
        result.singlet.corrected_moments.steps[-1].boundary_value_correction_rms
        >= 0.0
    )
    assert np.isfinite(result.triplet.corrected_moments.steps[-1].density_residual)
