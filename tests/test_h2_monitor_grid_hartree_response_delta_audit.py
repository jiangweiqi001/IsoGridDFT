"""Smoke tests for the fixed-density Hartree response delta audit."""

from importlib import import_module

import numpy as np

from isogrid.audit.h2_monitor_grid_hartree_response_delta_audit import (
    H2MonitorGridHartreeResponseDeltaAuditResult,
)
from isogrid.audit.h2_monitor_grid_hartree_response_delta_audit import (
    H2MonitorGridHartreeResponseDeltaComparison,
)
from isogrid.audit.h2_monitor_grid_hartree_response_delta_audit import (
    H2MonitorGridHartreeResponseDeltaRoute,
)
from isogrid.audit.h2_monitor_grid_hartree_response_delta_audit import (
    H2MonitorGridHartreeResponseDeltaSnapshot,
)
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_h2_monitor_grid_hartree_response_delta_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_hartree_response_delta_audit")

    assert hasattr(module, "run_h2_monitor_grid_hartree_response_delta_audit")
    assert hasattr(module, "print_h2_monitor_grid_hartree_response_delta_summary")


def test_construct_h2_monitor_grid_hartree_response_delta_result() -> None:
    route = H2MonitorGridHartreeResponseDeltaRoute(
        spin_state_label="singlet",
        boundary_construction_mode="corrected_moments",
        source_snapshot_count=2,
        snapshots=(
            H2MonitorGridHartreeResponseDeltaSnapshot(
                source_snapshot_label="iter1_input",
                density_electrons=2.0,
                boundary_value_correction_rms=1.0e-3,
                corrected_moment_boundary_rms_mismatch=0.0,
                boundary_value_rms=0.2,
                boundary_source_laplacian_rms=0.3,
                rhs_l2_norm=0.4,
                interior_potential_rms=0.5,
                interior_poisson_residual_l2_norm=1.0e-8,
                interior_poisson_residual_max_abs=2.0e-8,
                hartree_tail_far_field_mean_abs=0.1,
                hartree_tail_far_field_signed_mean=0.1,
                hartree_potential_rms=0.8,
                effective_hartree_component_share=0.6,
            ),
        ),
    )
    comparison = H2MonitorGridHartreeResponseDeltaComparison(
        spin_state_label="singlet",
        source_snapshot_label="iter1_input",
        legacy_split=route.snapshots[0],
        corrected_moments=route.snapshots[0],
        hartree_potential_delta_rms=0.0,
        hartree_potential_relative_delta=0.0,
        effective_potential_delta_rms=0.0,
        boundary_mismatch_delta=-1.0e-3,
        boundary_source_laplacian_rms_delta=0.0,
        interior_potential_rms_delta=0.0,
        hartree_tail_mean_abs_delta=0.0,
        verdict="smoke",
    )
    result = H2MonitorGridHartreeResponseDeltaAuditResult(
        case_name=H2_BENCHMARK_CASE.name,
        grid_parameter_summary="shape=(9, 9, 11), box=(6, 6, 8)",
        singlet_routes=(route, route),
        triplet_routes=(route, route),
        singlet_comparisons=(comparison,),
        triplet_comparisons=(comparison,),
        note="smoke",
    )

    assert result.singlet_routes[0].snapshots[0].effective_hartree_component_share == 0.6
    assert result.triplet_comparisons[0].boundary_mismatch_delta == -1.0e-3


def test_h2_monitor_grid_hartree_response_delta_audit_reports_fixed_density_deltas() -> None:
    from isogrid.audit.h2_monitor_grid_hartree_response_delta_audit import (
        run_h2_monitor_grid_hartree_response_delta_audit,
    )

    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    result = run_h2_monitor_grid_hartree_response_delta_audit(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        source_iteration_count=2,
    )

    assert result.singlet_routes[0].boundary_construction_mode == "legacy_split"
    assert result.singlet_routes[1].boundary_construction_mode == "corrected_moments"
    assert len(result.singlet_comparisons) == 2
    assert len(result.triplet_comparisons) == 2
    assert result.singlet_comparisons[0].legacy_split.corrected_moment_boundary_rms_mismatch > 0.0
    assert result.singlet_comparisons[0].corrected_moments.corrected_moment_boundary_rms_mismatch < 1.0e-10
    assert result.singlet_comparisons[0].hartree_potential_delta_rms >= 0.0
    assert np.isfinite(result.triplet_comparisons[-1].effective_potential_delta_rms)
