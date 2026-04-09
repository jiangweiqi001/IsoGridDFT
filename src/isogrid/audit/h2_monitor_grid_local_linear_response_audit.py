"""Local linear-response / perturbation-gain audit on the H2 monitor grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ks import FixedPotentialStaticLocalOperatorContext
from isogrid.ks import prepare_fixed_potential_static_local_operator
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import run_h2_monitor_grid_scf_dry_run
from isogrid.scf.driver import _build_density_from_occupied_orbitals
from isogrid.scf.driver import _closed_shell_singlet_tracking_solve_count
from isogrid.scf.driver import _default_monitor_patch_parameters
from isogrid.scf.driver import _is_h2_closed_shell_singlet
from isogrid.scf.driver import _renormalize_density
from isogrid.scf.driver import _select_closed_shell_singlet_tracked_occupied_orbital
from isogrid.scf.driver import resolve_h2_spin_occupations

from .h2_monitor_grid_scf_amplification_ablation_audit import _baseline_track_solves as _reuse_track_solves
from .h2_monitor_grid_scf_amplification_ablation_audit import _k_track
from .h2_monitor_grid_scf_amplification_ablation_audit import _overlap_tracking
from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result
from .h2_monitor_grid_scf_amplification_ablation_audit import _solve_track_block
from .h2_monitor_grid_scf_amplification_ablation_audit import _weighted_field_norm

_DEFAULT_SHAPE = (9, 9, 11)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (6.0, 6.0, 8.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 6
_DEFAULT_PROBE_ITERATION = 6
_DEFAULT_PERTURBATION_AMPLITUDE = 0.03
_DEFAULT_PERTURBATION_WIDTH_BOHR = 0.9


@dataclass(frozen=True)
class H2MonitorGridLocalLinearResponseProbe:
    """One spin-resolved local perturbation / response measurement."""

    spin_state_label: str
    perturbation_channel: str
    controller_name: str
    probe_iteration: int
    source_density_residual: float
    source_hartree_share: float | None
    input_charge_perturbation_norm: float
    input_spin_perturbation_norm: float
    input_charge_perturbation_peak_abs: float
    hartree_response_norm: float
    output_charge_response_norm: float
    output_spin_response_norm: float
    hartree_gain: float | None
    charge_gain: float | None
    spin_gain: float | None
    occupied_orbital_overlap_abs: float | None
    lowest2_subspace_overlap_min_singular_value: float | None
    lowest2_subspace_rotation_max_angle_deg: float | None
    lowest_gap_ha: float | None
    lowest_gap_delta_ha: float | None
    verdict: str


@dataclass(frozen=True)
class H2MonitorGridLocalLinearResponseSpinAudit:
    """Paired charge/spin perturbation probes for one spin state."""

    spin_state_label: str
    charge_probe: H2MonitorGridLocalLinearResponseProbe
    spin_probe: H2MonitorGridLocalLinearResponseProbe


@dataclass(frozen=True)
class H2MonitorGridLocalLinearResponseAuditResult:
    """Top-level paired local linear-response audit result."""

    case_name: str
    grid_parameter_summary: str
    perturbation_amplitude: float
    perturbation_width_bohr: float
    singlet: H2MonitorGridLocalLinearResponseProbe
    triplet: H2MonitorGridLocalLinearResponseProbe
    note: str


def _build_context(
    *,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
) -> FixedPotentialStaticLocalOperatorContext:
    return prepare_fixed_potential_static_local_operator(
        grid_geometry=grid_geometry,
        rho_up=np.asarray(rho_up, dtype=np.float64),
        rho_down=np.asarray(rho_down, dtype=np.float64),
        spin_channel="up",
        case=case,
        use_monitor_patch=True,
        patch_parameters=_default_monitor_patch_parameters(),
        kinetic_version="trial_fix",
        hartree_backend="python",
        monitor_boundary_construction_mode="corrected_moments",
    )


def _density_from_tracked_solve(
    *,
    solve_up,
    tracked_occupied_orbitals: np.ndarray,
    occupations,
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    if _is_h2_closed_shell_singlet(occupations):
        rho_up = _build_density_from_occupied_orbitals(
            tracked_occupied_orbitals,
            occupations.occupations_up,
            grid_geometry=grid_geometry,
        )
        rho_down = _build_density_from_occupied_orbitals(
            tracked_occupied_orbitals,
            occupations.occupations_down,
            grid_geometry=grid_geometry,
        )
        return (
            _renormalize_density(rho_up, occupations.n_alpha, grid_geometry=grid_geometry),
            _renormalize_density(rho_down, occupations.n_beta, grid_geometry=grid_geometry),
        )

    rho_up = _build_density_from_occupied_orbitals(
        solve_up.orbitals[: occupations.n_alpha],
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    )
    rho_down = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    return (
        _renormalize_density(rho_up, occupations.n_alpha, grid_geometry=grid_geometry),
        _renormalize_density(rho_down, occupations.n_beta, grid_geometry=grid_geometry),
    )


def _localized_bump(
    *,
    grid_geometry: MonitorGridGeometry,
    center: tuple[float, float, float],
    width_bohr: float,
) -> np.ndarray:
    dx = np.asarray(grid_geometry.x_points, dtype=np.float64) - float(center[0])
    dy = np.asarray(grid_geometry.y_points, dtype=np.float64) - float(center[1])
    dz = np.asarray(grid_geometry.z_points, dtype=np.float64) - float(center[2])
    radius_sq = dx * dx + dy * dy + dz * dz
    return np.asarray(
        np.exp(-0.5 * radius_sq / (float(width_bohr) ** 2)),
        dtype=np.float64,
    )


def _perturbed_input_density(
    *,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    occupations,
    grid_geometry: MonitorGridGeometry,
    local_weight: np.ndarray,
    amplitude: float,
    perturbation_channel: str,
) -> tuple[np.ndarray, np.ndarray]:
    if perturbation_channel == "charge":
        factor_up = np.asarray(1.0 + float(amplitude) * local_weight, dtype=np.float64)
        factor_down = np.asarray(1.0 + float(amplitude) * local_weight, dtype=np.float64)
    elif perturbation_channel == "spin":
        factor_up = np.asarray(1.0 + float(amplitude) * local_weight, dtype=np.float64)
        factor_down = np.asarray(1.0 - float(amplitude) * local_weight, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported perturbation_channel `{perturbation_channel}`.")
    rho_up_perturbed = _renormalize_density(
        np.asarray(rho_up, dtype=np.float64) * factor_up,
        occupations.n_alpha,
        grid_geometry=grid_geometry,
    )
    rho_down_perturbed = _renormalize_density(
        np.asarray(rho_down, dtype=np.float64) * factor_down,
        occupations.n_beta,
        grid_geometry=grid_geometry,
    )
    return rho_up_perturbed, rho_down_perturbed


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
) -> H2MonitorGridLocalLinearResponseProbe:
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
    baseline_rho_up_out, baseline_rho_down_out = _density_from_tracked_solve(
        solve_up=baseline_solve,
        tracked_occupied_orbitals=baseline_tracked,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )

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
    perturbed_rho_up_out, perturbed_rho_down_out = _density_from_tracked_solve(
        solve_up=perturbed_solve,
        tracked_occupied_orbitals=perturbed_tracked,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )

    baseline_charge_in = np.asarray(
        probe_record.input_rho_up + probe_record.input_rho_down,
        dtype=np.float64,
    )
    baseline_spin_in = np.asarray(
        probe_record.input_rho_up - probe_record.input_rho_down,
        dtype=np.float64,
    )
    perturbed_charge_in = np.asarray(rho_up_perturbed + rho_down_perturbed, dtype=np.float64)
    perturbed_spin_in = np.asarray(rho_up_perturbed - rho_down_perturbed, dtype=np.float64)
    input_charge_perturbation = np.asarray(
        perturbed_charge_in - baseline_charge_in,
        dtype=np.float64,
    )
    input_spin_perturbation = np.asarray(
        perturbed_spin_in - baseline_spin_in,
        dtype=np.float64,
    )
    baseline_charge_out = np.asarray(baseline_rho_up_out + baseline_rho_down_out, dtype=np.float64)
    baseline_spin_out = np.asarray(baseline_rho_up_out - baseline_rho_down_out, dtype=np.float64)
    perturbed_charge_out = np.asarray(perturbed_rho_up_out + perturbed_rho_down_out, dtype=np.float64)
    perturbed_spin_out = np.asarray(perturbed_rho_up_out - perturbed_rho_down_out, dtype=np.float64)
    output_charge_response = np.asarray(
        perturbed_charge_out - baseline_charge_out,
        dtype=np.float64,
    )
    output_spin_response = np.asarray(
        perturbed_spin_out - baseline_spin_out,
        dtype=np.float64,
    )
    hartree_response = np.asarray(
        perturbed_context.hartree_potential - baseline_context.hartree_potential,
        dtype=np.float64,
    )

    input_charge_norm = _weighted_field_norm(input_charge_perturbation, grid_geometry=grid_geometry)
    input_spin_norm = _weighted_field_norm(input_spin_perturbation, grid_geometry=grid_geometry)
    output_charge_norm = _weighted_field_norm(output_charge_response, grid_geometry=grid_geometry)
    output_spin_norm = _weighted_field_norm(output_spin_response, grid_geometry=grid_geometry)
    hartree_norm = _weighted_field_norm(hartree_response, grid_geometry=grid_geometry)
    (
        occupied_overlap_abs,
        min_singular,
        max_angle_deg,
        current_gap,
        gap_delta,
    ) = _overlap_tracking(
        previous_solve=baseline_solve,
        current_solve=perturbed_solve,
        previous_occupied_orbitals=baseline_tracked,
        current_occupied_orbitals=perturbed_tracked,
        grid_geometry=grid_geometry,
        track_lowest_two_states=True,
    )

    dominant_input_norm = max(input_charge_norm, input_spin_norm)
    charge_gain = None if input_charge_norm <= 1.0e-16 else float(output_charge_norm / input_charge_norm)
    spin_gain = None if input_spin_norm <= 1.0e-16 else float(output_spin_norm / input_spin_norm)
    hartree_gain = None if dominant_input_norm <= 1.0e-16 else float(hartree_norm / dominant_input_norm)
    if dominant_input_norm <= 1.0e-16:
        verdict = "Input perturbation norm was too small to form a stable local response gain."
    elif perturbation_channel == "charge" and charge_gain is not None and charge_gain > 1.0:
        verdict = "The local charge perturbation is amplified by the one-step map at this probe."
    elif perturbation_channel == "spin" and spin_gain is not None and spin_gain > 1.0:
        verdict = "The local spin perturbation is amplified by the one-step map at this probe."
    else:
        verdict = (
            f"The local {perturbation_channel} perturbation is not amplified by the one-step "
            "map at this probe."
        )

    source_signal = (
        source_result.controller_signals_history[probe_index]
        if probe_index < len(source_result.controller_signals_history)
        else None
    )
    return H2MonitorGridLocalLinearResponseProbe(
        spin_state_label=spin_label,
        perturbation_channel=perturbation_channel,
        controller_name=controller_name,
        probe_iteration=int(probe_record.iteration),
        source_density_residual=float(probe_record.density_residual),
        source_hartree_share=(None if source_signal is None else source_signal.hartree_share),
        input_charge_perturbation_norm=float(input_charge_norm),
        input_spin_perturbation_norm=float(input_spin_norm),
        input_charge_perturbation_peak_abs=float(np.max(np.abs(input_charge_perturbation))),
        hartree_response_norm=float(hartree_norm),
        output_charge_response_norm=float(output_charge_norm),
        output_spin_response_norm=float(output_spin_norm),
        hartree_gain=hartree_gain,
        charge_gain=charge_gain,
        spin_gain=spin_gain,
        occupied_orbital_overlap_abs=occupied_overlap_abs,
        lowest2_subspace_overlap_min_singular_value=min_singular,
        lowest2_subspace_rotation_max_angle_deg=max_angle_deg,
        lowest_gap_ha=current_gap,
        lowest_gap_delta_ha=gap_delta,
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
) -> H2MonitorGridLocalLinearResponseSpinAudit:
    return H2MonitorGridLocalLinearResponseSpinAudit(
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


def run_h2_monitor_grid_local_linear_response_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    probe_iteration: int = _DEFAULT_PROBE_ITERATION,
    perturbation_amplitude: float = _DEFAULT_PERTURBATION_AMPLITUDE,
    perturbation_width_bohr: float = _DEFAULT_PERTURBATION_WIDTH_BOHR,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridLocalLinearResponseAuditResult:
    """Measure one-step local charge/Hartree/subspace gains around one SCF probe state."""

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
    return H2MonitorGridLocalLinearResponseAuditResult(
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
            "This local-only audit freezes one SCF input-density snapshot, applies a small "
            "localized closed-shell charge perturbation near one H atom, then compares the "
            "one-step Hartree / output-density / lowest-two-state response between the "
            "unperturbed and perturbed maps. "
            f"controller={controller_name}"
        ),
    )


def print_h2_monitor_grid_local_linear_response_summary(
    result: H2MonitorGridLocalLinearResponseAuditResult,
) -> None:
    print("IsoGridDFT H2 monitor-grid local linear-response audit")
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
            print(f"    input_charge_perturbation_norm: {probe.input_charge_perturbation_norm}")
            print(f"    input_spin_perturbation_norm: {probe.input_spin_perturbation_norm}")
            print(f"    input_charge_perturbation_peak_abs: {probe.input_charge_perturbation_peak_abs}")
            print(f"    hartree_response_norm: {probe.hartree_response_norm}")
            print(f"    output_charge_response_norm: {probe.output_charge_response_norm}")
            print(f"    output_spin_response_norm: {probe.output_spin_response_norm}")
            print(f"    hartree_gain: {probe.hartree_gain}")
            print(f"    charge_gain: {probe.charge_gain}")
            print(f"    spin_gain: {probe.spin_gain}")
            print(f"    overlap_abs: {probe.occupied_orbital_overlap_abs}")
            print(f"    subspace_rotation_deg: {probe.lowest2_subspace_rotation_max_angle_deg}")
            print(f"    gap_ha: {probe.lowest_gap_ha}")
            print(f"    gap_delta_ha: {probe.lowest_gap_delta_ha}")
            print(f"    verdict: {probe.verdict}")


def main() -> int:
    result = run_h2_monitor_grid_local_linear_response_audit()
    print_h2_monitor_grid_local_linear_response_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
