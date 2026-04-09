"""Lowest-2 local continuity/response audit on the H2 monitor grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ks import weighted_overlap_matrix
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf.driver import _is_h2_closed_shell_singlet
from isogrid.scf.driver import _select_closed_shell_singlet_tracked_occupied_orbital
from isogrid.scf.driver import resolve_h2_spin_occupations

from .h2_monitor_grid_local_linear_response_audit import _build_context
from .h2_monitor_grid_local_linear_response_audit import _density_from_tracked_solve
from .h2_monitor_grid_local_linear_response_audit import _localized_bump
from .h2_monitor_grid_local_linear_response_audit import _perturbed_input_density
from .h2_monitor_grid_scf_amplification_ablation_audit import _baseline_track_solves as _reuse_track_solves
from .h2_monitor_grid_scf_amplification_ablation_audit import _k_track
from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result
from .h2_monitor_grid_scf_amplification_ablation_audit import _solve_track_block

_DEFAULT_SHAPE = (9, 9, 11)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (6.0, 6.0, 8.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 6
_DEFAULT_PROBE_ITERATION = 6
_DEFAULT_PERTURBATION_AMPLITUDE = 0.03
_DEFAULT_PERTURBATION_WIDTH_BOHR = 0.9


@dataclass(frozen=True)
class H2MonitorGridLowest2LocalResponseProbe:
    """One local perturbation probe decomposed into subspace drift vs in-subspace mixing."""

    spin_state_label: str
    perturbation_channel: str
    controller_name: str
    probe_iteration: int
    source_density_residual: float
    source_hartree_share: float | None
    raw_lowest_orbital_overlap_abs: float | None
    best_in_subspace_occupied_overlap_abs: float | None
    internal_rotation_angle_deg: float | None
    projector_drift_frobenius_norm: float | None
    lowest2_overlap_matrix: tuple[tuple[float, float], tuple[float, float]]
    lowest2_subspace_overlap_singular_values: tuple[float, float]
    lowest2_subspace_overlap_min_singular_value: float | None
    lowest2_subspace_rotation_max_angle_deg: float | None
    lowest_gap_ha: float | None
    lowest_gap_delta_ha: float | None
    verdict: str


@dataclass(frozen=True)
class H2MonitorGridLowest2LocalResponseSpinAudit:
    """Paired charge/spin lowest-2 probes for one spin state."""

    spin_state_label: str
    charge_probe: H2MonitorGridLowest2LocalResponseProbe
    spin_probe: H2MonitorGridLowest2LocalResponseProbe


@dataclass(frozen=True)
class H2MonitorGridLowest2LocalResponseAuditResult:
    """Top-level lowest-2 local continuity/response audit result."""

    case_name: str
    grid_parameter_summary: str
    perturbation_amplitude: float
    perturbation_width_bohr: float
    singlet: H2MonitorGridLowest2LocalResponseSpinAudit
    triplet: H2MonitorGridLowest2LocalResponseSpinAudit
    note: str


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1])),
        (float(matrix[1, 0]), float(matrix[1, 1])),
    )


def _overlap_decomposition(
    *,
    previous_block: np.ndarray,
    current_block: np.ndarray,
    previous_occupied: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[float, float],
    float,
    float,
    float,
]:
    overlap = np.asarray(
        weighted_overlap_matrix(
            current_block,
            grid_geometry=grid_geometry,
            other=previous_block,
        ),
        dtype=np.float64,
    )
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(np.asarray(singular_values, dtype=np.float64), 0.0, 1.0)
    min_singular = float(np.min(singular_values))
    max_angle_deg = float(np.degrees(np.arccos(min_singular)))
    projector_drift = float(
        np.sqrt(max(0.0, 2.0 * float(np.sum(1.0 - singular_values * singular_values, dtype=np.float64))))
    )

    occupied_coefficients = np.asarray(
        weighted_overlap_matrix(
            current_block,
            grid_geometry=grid_geometry,
            other=previous_occupied[:1],
        )[:, 0],
        dtype=np.complex128,
    )
    coefficient_abs = np.abs(occupied_coefficients).astype(np.float64)
    raw_lowest_overlap_abs = float(coefficient_abs[0])
    best_in_subspace_overlap_abs = float(min(1.0, np.linalg.norm(coefficient_abs)))
    if best_in_subspace_overlap_abs <= 1.0e-16:
        internal_rotation_angle_deg = 0.0
    else:
        internal_rotation_angle_deg = float(
            np.degrees(
                np.arctan2(
                    coefficient_abs[1],
                    max(coefficient_abs[0], 1.0e-16),
                )
            )
        )
    return (
        _matrix_tuple(overlap),
        (float(singular_values[0]), float(singular_values[1])),
        raw_lowest_overlap_abs,
        best_in_subspace_overlap_abs,
        internal_rotation_angle_deg,
        projector_drift,
    )


def _probe_spin_response(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_iteration_count: int,
    probe_iteration: int,
    perturbation_amplitude: float,
    perturbation_width_bohr: float,
    perturbation_channel: str,
    controller_name: str,
) -> H2MonitorGridLowest2LocalResponseProbe:
    occupations = resolve_h2_spin_occupations(spin_label=spin_label, case=case)
    source_result: H2StaticLocalScfDryRunResult = _shared_source_result(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        controller_name=controller_name,
    )
    probe_index = max(0, min(int(probe_iteration) - 1, len(source_result.history) - 1))
    probe_record = source_result.history[probe_index]

    baseline_context = _build_context(
        case=case,
        grid_geometry=grid_geometry,
        rho_up=probe_record.input_rho_up,
        rho_down=probe_record.input_rho_down,
    )
    track_count = _k_track(occupations, track_lowest_two_states=True)
    baseline_solves, baseline_tracked_occupied = _reuse_track_solves(
        contexts=(baseline_context,),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    baseline_solve = baseline_solves[0]
    baseline_tracked = baseline_tracked_occupied[0]
    perturb_center = case.geometry.atoms[0].position
    bump = _localized_bump(
        grid_geometry=grid_geometry,
        center=perturb_center,
        width_bohr=perturbation_width_bohr,
    )
    rho_up_perturbed, rho_down_perturbed = _perturbed_input_density(
        rho_up=probe_record.input_rho_up,
        rho_down=probe_record.input_rho_down,
        occupations=occupations,
        grid_geometry=grid_geometry,
        local_weight=bump,
        amplitude=perturbation_amplitude,
        perturbation_channel=perturbation_channel,
    )
    perturbed_context = _build_context(
        case=case,
        grid_geometry=grid_geometry,
        rho_up=rho_up_perturbed,
        rho_down=rho_down_perturbed,
    )
    perturbed_solve = _solve_track_block(
        operator_context=perturbed_context,
        k=track_count,
        case=case,
        initial_guess_orbitals=np.asarray(baseline_solve.orbitals[:track_count], dtype=np.float64),
    )
    if _is_h2_closed_shell_singlet(occupations):
        perturbed_tracked = _select_closed_shell_singlet_tracked_occupied_orbital(
            perturbed_solve.orbitals,
            reference_orbital=baseline_tracked,
            grid_geometry=grid_geometry,
        )
    else:
        perturbed_tracked = np.asarray(
            perturbed_solve.orbitals[: occupations.n_alpha],
            dtype=np.float64,
        )
    _density_from_tracked_solve(
        solve_up=perturbed_solve,
        tracked_occupied_orbitals=perturbed_tracked,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )

    (
        overlap_matrix,
        singular_values,
        raw_lowest_overlap_abs,
        best_in_subspace_overlap_abs,
        internal_rotation_angle_deg,
        projector_drift,
    ) = _overlap_decomposition(
        previous_block=np.asarray(baseline_solve.orbitals[:2], dtype=np.float64),
        current_block=np.asarray(perturbed_solve.orbitals[:2], dtype=np.float64),
        previous_occupied=np.asarray(baseline_tracked, dtype=np.float64),
        grid_geometry=grid_geometry,
    )
    current_gap = float(perturbed_solve.eigenvalues[1] - perturbed_solve.eigenvalues[0])
    previous_gap = float(baseline_solve.eigenvalues[1] - baseline_solve.eigenvalues[0])
    source_signal = (
        source_result.controller_signals_history[probe_index]
        if probe_index < len(source_result.controller_signals_history)
        else None
    )
    if (
        best_in_subspace_overlap_abs >= 0.95
        and raw_lowest_overlap_abs <= 0.2
        and internal_rotation_angle_deg >= 45.0
    ):
        verdict = (
            "The lowest-2 subspace remains locally available, but the occupied direction rotates "
            "strongly inside that subspace under this perturbation."
        )
    elif singular_values[0] <= 0.8:
        verdict = "The perturbation moves the whole lowest-2 subspace, not just the occupied direction."
    else:
        verdict = "The perturbation produces only moderate lowest-2 subspace drift and internal mixing."

    return H2MonitorGridLowest2LocalResponseProbe(
        spin_state_label=spin_label,
        perturbation_channel=perturbation_channel,
        controller_name=controller_name,
        probe_iteration=int(probe_record.iteration),
        source_density_residual=float(probe_record.density_residual),
        source_hartree_share=(None if source_signal is None else source_signal.hartree_share),
        raw_lowest_orbital_overlap_abs=raw_lowest_overlap_abs,
        best_in_subspace_occupied_overlap_abs=best_in_subspace_overlap_abs,
        internal_rotation_angle_deg=internal_rotation_angle_deg,
        projector_drift_frobenius_norm=projector_drift,
        lowest2_overlap_matrix=overlap_matrix,
        lowest2_subspace_overlap_singular_values=singular_values,
        lowest2_subspace_overlap_min_singular_value=float(min(singular_values)),
        lowest2_subspace_rotation_max_angle_deg=float(np.degrees(np.arccos(min(singular_values)))),
        lowest_gap_ha=current_gap,
        lowest_gap_delta_ha=float(current_gap - previous_gap),
        verdict=verdict,
    )


def _probe_spin_audit(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_iteration_count: int,
    probe_iteration: int,
    perturbation_amplitude: float,
    perturbation_width_bohr: float,
    controller_name: str,
) -> H2MonitorGridLowest2LocalResponseSpinAudit:
    return H2MonitorGridLowest2LocalResponseSpinAudit(
        spin_state_label=spin_label,
        charge_probe=_probe_spin_response(
            spin_label=spin_label,
            case=case,
            grid_geometry=grid_geometry,
            source_iteration_count=source_iteration_count,
            probe_iteration=probe_iteration,
            perturbation_amplitude=perturbation_amplitude,
            perturbation_width_bohr=perturbation_width_bohr,
            perturbation_channel="charge",
            controller_name=controller_name,
        ),
        spin_probe=_probe_spin_response(
            spin_label=spin_label,
            case=case,
            grid_geometry=grid_geometry,
            source_iteration_count=source_iteration_count,
            probe_iteration=probe_iteration,
            perturbation_amplitude=perturbation_amplitude,
            perturbation_width_bohr=perturbation_width_bohr,
            perturbation_channel="spin",
            controller_name=controller_name,
        ),
    )


def run_h2_monitor_grid_lowest2_local_response_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    probe_iteration: int = _DEFAULT_PROBE_ITERATION,
    perturbation_amplitude: float = _DEFAULT_PERTURBATION_AMPLITUDE,
    perturbation_width_bohr: float = _DEFAULT_PERTURBATION_WIDTH_BOHR,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridLowest2LocalResponseAuditResult:
    """Measure whether local perturbations move the whole lowest-2 subspace or only its occupied direction."""

    if grid_geometry is None:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=_DEFAULT_SHAPE,
            box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
            element_parameters=build_h2_local_patch_development_element_parameters(),
        )
    bounds = grid_geometry.spec.box_bounds
    box_half_extents_bohr = (
        0.5 * float(bounds[0][1] - bounds[0][0]),
        0.5 * float(bounds[1][1] - bounds[1][0]),
        0.5 * float(bounds[2][1] - bounds[2][0]),
    )
    singlet = _probe_spin_audit(
        spin_label="singlet",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        probe_iteration=probe_iteration,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_width_bohr=perturbation_width_bohr,
        controller_name=controller_name,
    )
    triplet = _probe_spin_audit(
        spin_label="triplet",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        probe_iteration=probe_iteration,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_width_bohr=perturbation_width_bohr,
        controller_name=controller_name,
    )
    return H2MonitorGridLowest2LocalResponseAuditResult(
        case_name=case.name,
        grid_parameter_summary=(
            f"shape={grid_geometry.spec.shape}, "
            f"box_half_extents_bohr={box_half_extents_bohr}"
        ),
        perturbation_amplitude=float(perturbation_amplitude),
        perturbation_width_bohr=float(perturbation_width_bohr),
        singlet=singlet,
        triplet=triplet,
        note=(
            "This local-only audit freezes one SCF input-density snapshot, applies paired local "
            "charge/spin perturbations, and decomposes the resulting lowest-2 response into whole-"
            "subspace drift versus in-subspace occupied-direction rotation. "
            f"controller={controller_name}"
        ),
    )


def print_h2_monitor_grid_lowest2_local_response_summary(
    result: H2MonitorGridLowest2LocalResponseAuditResult,
) -> None:
    print("IsoGridDFT H2 monitor-grid lowest-2 local continuity/response audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(
        "perturbation: "
        f"amplitude={result.perturbation_amplitude}, "
        f"width_bohr={result.perturbation_width_bohr}"
    )
    print(f"note: {result.note}")
    for spin_audit in (result.singlet, result.triplet):
        print()
        print(f"spin: {spin_audit.spin_state_label}")
        for probe in (spin_audit.charge_probe, spin_audit.spin_probe):
            print(f"  channel: {probe.perturbation_channel}")
            print(f"    probe_iteration: {probe.probe_iteration}")
            print(f"    source_density_residual: {probe.source_density_residual}")
            print(f"    source_hartree_share: {probe.source_hartree_share}")
            print(f"    raw_lowest_overlap_abs: {probe.raw_lowest_orbital_overlap_abs}")
            print(f"    best_in_subspace_overlap_abs: {probe.best_in_subspace_occupied_overlap_abs}")
            print(f"    internal_rotation_angle_deg: {probe.internal_rotation_angle_deg}")
            print(f"    projector_drift_fro_norm: {probe.projector_drift_frobenius_norm}")
            print(f"    lowest2_overlap_matrix: {probe.lowest2_overlap_matrix}")
            print(f"    lowest2_singular_values: {probe.lowest2_subspace_overlap_singular_values}")
            print(f"    subspace_rotation_deg: {probe.lowest2_subspace_rotation_max_angle_deg}")
            print(f"    gap_ha: {probe.lowest_gap_ha}")
            print(f"    gap_delta_ha: {probe.lowest_gap_delta_ha}")
            print(f"    verdict: {probe.verdict}")


def main() -> int:
    result = run_h2_monitor_grid_lowest2_local_response_audit()
    print_h2_monitor_grid_lowest2_local_response_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
