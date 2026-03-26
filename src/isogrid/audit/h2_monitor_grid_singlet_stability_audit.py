"""Very small singlet-only SCF stability audit on the repaired A-grid path.

This audit intentionally stays narrow:

- H2 singlet only
- current A-grid+patch+kinetic-trial-fix baseline only
- no nonlocal migration
- only one minimal DIIS prototype beyond linear mixing

The only comparison here is whether a more conservative linear mixing value,
plus one explicit minimal DIIS branch, changes the previously observed weak
singlet two-cycle behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.audit.baselines import H2_SCF_DRY_RUN_BASELINE
from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.ops import integrate_field
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_SINGLET_STABILITY_MAX_ITERATIONS = 10
_SINGLET_STABILITY_DENSITY_TOLERANCE = 5.0e-3
_SINGLET_STABILITY_ENERGY_TOLERANCE = 5.0e-5
_SINGLET_STABILITY_EIGENSOLVER_TOLERANCE = 1.0e-3
_SINGLET_STABILITY_EIGENSOLVER_NCV = 20
_SINGLET_STABILITY_DIIS_WARMUP_ITERATIONS = 3
_SINGLET_STABILITY_DIIS_HISTORY_LENGTH = 4


@dataclass(frozen=True)
class H2SingletStabilityParameterSummary:
    """Fixed parameter summary for one singlet stability scheme."""

    grid_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    weight_scale: float
    radius_scale: float
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    mixing: float
    max_iterations: int
    density_tolerance: float
    energy_tolerance: float
    eigensolver_tolerance: float
    eigensolver_ncv: int
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str


@dataclass(frozen=True)
class H2TwoCycleDiagnostics:
    """Lightweight tail diagnostics for the singlet oscillation pattern."""

    detected_two_cycle: bool
    tail_length: int
    even_energy_mean_ha: float | None
    odd_energy_mean_ha: float | None
    even_odd_energy_gap_ha: float | None
    even_energy_std_ha: float | None
    odd_energy_std_ha: float | None
    even_residual_mean: float | None
    odd_residual_mean: float | None
    even_odd_residual_gap: float | None
    even_residual_std: float | None
    odd_residual_std: float | None
    verdict: str


@dataclass(frozen=True)
class H2SingletStabilityRouteResult:
    """Compact result for one singlet stability scheme."""

    scheme_label: str
    path_type: str
    spin_state_label: str
    kinetic_version: str
    includes_nonlocal: bool
    parameter_summary: H2SingletStabilityParameterSummary
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str
    diis_used_iterations: tuple[int, ...]
    diis_history_sizes: tuple[int, ...]
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    energy_history_ha: tuple[float, ...]
    density_residual_history: tuple[float, ...]
    energy_change_history_ha: tuple[float | None, ...]
    final_density_residual: float | None
    final_energy_change_ha: float | None
    final_rho_up_electrons: float
    final_rho_down_electrons: float
    final_energy_components: SinglePointEnergyComponents
    two_cycle: H2TwoCycleDiagnostics


@dataclass(frozen=True)
class H2SingletStabilityAuditResult:
    """Top-level H2 singlet stability audit result."""

    baseline_route: H2SingletStabilityRouteResult
    smaller_mixing_route: H2SingletStabilityRouteResult
    diis_prototype_route: H2SingletStabilityRouteResult
    note: str


def _parameter_summary(
    result: H2StaticLocalScfDryRunResult,
    *,
    mixing: float,
) -> H2SingletStabilityParameterSummary:
    parameters = result.parameter_summary
    return H2SingletStabilityParameterSummary(
        grid_shape=parameters.grid_shape,
        box_half_extents_bohr=parameters.box_half_extents_bohr,
        weight_scale=parameters.weight_scale,
        radius_scale=parameters.radius_scale,
        patch_radius_scale=parameters.patch_radius_scale,
        patch_grid_shape=parameters.patch_grid_shape,
        correction_strength=parameters.correction_strength,
        interpolation_neighbors=parameters.interpolation_neighbors,
        kinetic_version=parameters.kinetic_version,
        mixing=float(mixing),
        max_iterations=_SINGLET_STABILITY_MAX_ITERATIONS,
        density_tolerance=_SINGLET_STABILITY_DENSITY_TOLERANCE,
        energy_tolerance=_SINGLET_STABILITY_ENERGY_TOLERANCE,
        eigensolver_tolerance=_SINGLET_STABILITY_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_SINGLET_STABILITY_EIGENSOLVER_NCV,
        diis_enabled=parameters.diis_enabled,
        diis_warmup_iterations=parameters.diis_warmup_iterations,
        diis_history_length=parameters.diis_history_length,
        diis_residual_definition=parameters.diis_residual_definition,
    )


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.std(values))


def _build_two_cycle_diagnostics(
    energy_history_ha: tuple[float, ...],
    density_residual_history: tuple[float, ...],
    *,
    converged: bool,
) -> H2TwoCycleDiagnostics:
    if converged or len(energy_history_ha) < 6 or len(density_residual_history) < 6:
        verdict = "converged" if converged else "insufficient_tail"
        return H2TwoCycleDiagnostics(
            detected_two_cycle=False,
            tail_length=0,
            even_energy_mean_ha=None,
            odd_energy_mean_ha=None,
            even_odd_energy_gap_ha=None,
            even_energy_std_ha=None,
            odd_energy_std_ha=None,
            even_residual_mean=None,
            odd_residual_mean=None,
            even_odd_residual_gap=None,
            even_residual_std=None,
            odd_residual_std=None,
            verdict=verdict,
        )

    tail_start = len(energy_history_ha) // 2
    tail_energies = np.asarray(energy_history_ha[tail_start:], dtype=np.float64)
    tail_residuals = np.asarray(density_residual_history[tail_start:], dtype=np.float64)
    if tail_energies.size % 2 != 0:
        tail_energies = tail_energies[1:]
        tail_residuals = tail_residuals[1:]

    even_energies = tail_energies[::2]
    odd_energies = tail_energies[1::2]
    even_residuals = tail_residuals[::2]
    odd_residuals = tail_residuals[1::2]

    even_energy_mean = _safe_mean(even_energies)
    odd_energy_mean = _safe_mean(odd_energies)
    even_residual_mean = _safe_mean(even_residuals)
    odd_residual_mean = _safe_mean(odd_residuals)
    even_energy_std = _safe_std(even_energies)
    odd_energy_std = _safe_std(odd_energies)
    even_residual_std = _safe_std(even_residuals)
    odd_residual_std = _safe_std(odd_residuals)

    energy_gap = (
        None
        if even_energy_mean is None or odd_energy_mean is None
        else float(abs(even_energy_mean - odd_energy_mean))
    )
    residual_gap = (
        None
        if even_residual_mean is None or odd_residual_mean is None
        else float(abs(even_residual_mean - odd_residual_mean))
    )

    energy_noise_floor = max(even_energy_std or 0.0, odd_energy_std or 0.0, 1.0e-16)
    residual_noise_floor = max(even_residual_std or 0.0, odd_residual_std or 0.0, 1.0e-16)
    detected_two_cycle = bool(
        energy_gap is not None
        and residual_gap is not None
        and energy_gap > 5.0 * energy_noise_floor
        and residual_gap > 5.0 * residual_noise_floor
    )
    if detected_two_cycle:
        verdict = "weak_two_cycle"
    elif abs(tail_residuals[-1] - tail_residuals[0]) < 5.0e-3:
        verdict = "stable_not_converged"
    elif tail_residuals[-1] > tail_residuals[0]:
        verdict = "diverging"
    else:
        verdict = "slow_monotone_or_damped"

    return H2TwoCycleDiagnostics(
        detected_two_cycle=detected_two_cycle,
        tail_length=int(tail_energies.size),
        even_energy_mean_ha=even_energy_mean,
        odd_energy_mean_ha=odd_energy_mean,
        even_odd_energy_gap_ha=energy_gap,
        even_energy_std_ha=even_energy_std,
        odd_energy_std_ha=odd_energy_std,
        even_residual_mean=even_residual_mean,
        odd_residual_mean=odd_residual_mean,
        even_odd_residual_gap=residual_gap,
        even_residual_std=even_residual_std,
        odd_residual_std=odd_residual_std,
        verdict=verdict,
    )


def _build_route_result(
    scheme_label: str,
    *,
    mixing: float,
    result: H2StaticLocalScfDryRunResult,
) -> H2SingletStabilityRouteResult:
    grid_geometry = (
        result.solve_up.operator_context.grid_geometry
        if result.solve_up is not None
        else result.solve_down.operator_context.grid_geometry
    )
    return H2SingletStabilityRouteResult(
        scheme_label=scheme_label,
        path_type=result.path_type,
        spin_state_label=result.spin_state_label,
        kinetic_version=result.kinetic_version,
        includes_nonlocal=result.includes_nonlocal,
        parameter_summary=_parameter_summary(result, mixing=mixing),
        diis_enabled=result.diis_enabled,
        diis_warmup_iterations=result.diis_warmup_iterations,
        diis_history_length=result.diis_history_length,
        diis_residual_definition=result.diis_residual_definition,
        diis_used_iterations=result.diis_used_iterations,
        diis_history_sizes=result.diis_history_sizes,
        converged=result.converged,
        iteration_count=result.iteration_count,
        final_total_energy_ha=float(result.energy.total),
        final_lowest_eigenvalue_ha=result.lowest_eigenvalue,
        energy_history_ha=tuple(float(value) for value in result.energy_history),
        density_residual_history=tuple(float(value) for value in result.density_residual_history),
        energy_change_history_ha=tuple(
            None if record.energy_change is None else float(record.energy_change)
            for record in result.history
        ),
        final_density_residual=(
            None if not result.history else float(result.history[-1].density_residual)
        ),
        final_energy_change_ha=(
            None
            if not result.history or result.history[-1].energy_change is None
            else float(result.history[-1].energy_change)
        ),
        final_rho_up_electrons=float(integrate_field(result.rho_up, grid_geometry=grid_geometry)),
        final_rho_down_electrons=float(integrate_field(result.rho_down, grid_geometry=grid_geometry)),
        final_energy_components=result.energy,
        two_cycle=_build_two_cycle_diagnostics(
            energy_history_ha=tuple(float(value) for value in result.energy_history),
            density_residual_history=tuple(float(value) for value in result.density_residual_history),
            converged=bool(result.converged),
        ),
    )


def _run_monitor_singlet_scheme(
    *,
    scheme_label: str,
    mixing: float,
    enable_diis: bool,
    case: BenchmarkCase,
) -> H2SingletStabilityRouteResult:
    result = run_h2_monitor_grid_scf_dry_run(
        "singlet",
        case=case,
        mixing=mixing,
        max_iterations=_SINGLET_STABILITY_MAX_ITERATIONS,
        density_tolerance=_SINGLET_STABILITY_DENSITY_TOLERANCE,
        energy_tolerance=_SINGLET_STABILITY_ENERGY_TOLERANCE,
        eigensolver_tolerance=_SINGLET_STABILITY_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_SINGLET_STABILITY_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        enable_diis=enable_diis,
        diis_warmup_iterations=_SINGLET_STABILITY_DIIS_WARMUP_ITERATIONS,
        diis_history_length=_SINGLET_STABILITY_DIIS_HISTORY_LENGTH,
    )
    return _build_route_result(scheme_label, mixing=mixing, result=result)


def run_h2_monitor_grid_singlet_stability_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2SingletStabilityAuditResult:
    """Run a very small singlet-only stability audit on the repaired A-grid path."""

    baseline_route = _run_monitor_singlet_scheme(
        scheme_label="baseline",
        mixing=0.20,
        enable_diis=False,
        case=case,
    )
    smaller_mixing_route = _run_monitor_singlet_scheme(
        scheme_label="smaller-mixing",
        mixing=0.10,
        enable_diis=False,
        case=case,
    )
    diis_prototype_route = _run_monitor_singlet_scheme(
        scheme_label="diis-prototype",
        mixing=0.10,
        enable_diis=True,
        case=case,
    )
    return H2SingletStabilityAuditResult(
        baseline_route=baseline_route,
        smaller_mixing_route=smaller_mixing_route,
        diis_prototype_route=diis_prototype_route,
        note=(
            "This is a very small singlet-only stability audit on the current "
            "A-grid+patch+kinetic-trial-fix dry-run path. Nonlocal remains absent. "
            "The only added stabilization prototype beyond linear mixing is a tiny "
            "density-DIIS branch on the A-grid singlet path: after a short warmup, "
            "it stores the last few mixed densities together with the fixed-point "
            "density residual fields r = rho_out - rho_in and solves a small Pulay "
            "system for the next mixed density. The audit keeps the iteration budget "
            "to 10 steps on purpose: the goal is to classify the singlet stability "
            "mode, not to start a wider SCF tuning exercise."
        ),
    )


def print_h2_monitor_grid_singlet_stability_summary(
    result: H2SingletStabilityAuditResult,
) -> None:
    """Print the compact singlet stability audit summary."""

    print("IsoGridDFT H2 singlet stability audit")
    print(f"note: {result.note}")
    print(
        "baseline reference: current frozen regression says monitor singlet dry-run "
        f"ended unconverged after {H2_SCF_DRY_RUN_BASELINE.monitor_singlet_route.iteration_count} iterations"
    )
    for route in (result.baseline_route, result.smaller_mixing_route, result.diis_prototype_route):
        print()
        print(f"scheme: {route.scheme_label}")
        print(f"  converged: {route.converged}")
        print(f"  iterations: {route.iteration_count}")
        print(
            "  diis: "
            f"enabled={route.diis_enabled}, "
            f"warmup={route.diis_warmup_iterations}, "
            f"history_length={route.diis_history_length}, "
            f"used_iterations={route.diis_used_iterations}, "
            f"history_sizes={route.diis_history_sizes}"
        )
        print(f"  final total energy [Ha]: {route.final_total_energy_ha:.12f}")
        print(
            "  final lowest eigenvalue [Ha]: "
            f"{route.final_lowest_eigenvalue_ha:.12f}" if route.final_lowest_eigenvalue_ha is not None else "  final lowest eigenvalue [Ha]: n/a"
        )
        print(
            "  final electrons: "
            f"rho_up={route.final_rho_up_electrons:.12f}, "
            f"rho_down={route.final_rho_down_electrons:.12f}"
        )
        print(
            "  final components [Ha]: "
            f"T={route.final_energy_components.kinetic:.12f}, "
            f"V_loc={route.final_energy_components.local_ionic:.12f}, "
            f"E_H={route.final_energy_components.hartree:.12f}, "
            f"E_xc={route.final_energy_components.xc:.12f}, "
            f"E_II={route.final_energy_components.ion_ion_repulsion:.12f}"
        )
        print(
            "  two-cycle diagnostics: "
            f"detected={route.two_cycle.detected_two_cycle}, "
            f"verdict={route.two_cycle.verdict}, "
            f"energy_gap={route.two_cycle.even_odd_energy_gap_ha}, "
            f"residual_gap={route.two_cycle.even_odd_residual_gap}"
        )
        print("  history:")
        for index, energy in enumerate(route.energy_history_ha):
            residual = route.density_residual_history[index]
            energy_change = route.energy_change_history_ha[index]
            if energy_change is None:
                print(
                    f"    iter={index + 1:02d}, E={energy:+.12f} Ha, "
                    f"density_residual={residual:.6e}, energy_change=n/a"
                )
            else:
                print(
                    f"    iter={index + 1:02d}, E={energy:+.12f} Ha, "
                    f"density_residual={residual:.6e}, energy_change={energy_change:+.6e} Ha"
                )


def main() -> int:
    result = run_h2_monitor_grid_singlet_stability_audit()
    print_h2_monitor_grid_singlet_stability_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
