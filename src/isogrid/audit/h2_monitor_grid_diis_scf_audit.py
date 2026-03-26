"""Small H2 A-grid SCF audit for linear mixing versus a minimal DIIS prototype.

This audit intentionally stays inside the current repaired A-grid dry-run path:

- H2 only
- A-grid + patch + kinetic trial-fix only
- local static chain only: T + V_loc,ion + V_H + V_xc
- no nonlocal migration
- no legacy changes

The goal is narrow and explicit: check whether a small Pulay/DIIS prototype is
enough to move the current singlet path from "more stable but not converged"
into actual convergence, while confirming that triplet remains healthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.ops import integrate_field
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_A_GRID_DIIS_SCF_MAX_ITERATIONS = 20
_A_GRID_DIIS_SCF_DENSITY_TOLERANCE = 5.0e-3
_A_GRID_DIIS_SCF_ENERGY_TOLERANCE = 5.0e-5
_A_GRID_DIIS_SCF_EIGENSOLVER_TOLERANCE = 1.0e-3
_A_GRID_DIIS_SCF_EIGENSOLVER_NCV = 20
_A_GRID_DIIS_SCF_DIIS_WARMUP_ITERATIONS = 3
_A_GRID_DIIS_SCF_DIIS_HISTORY_LENGTH = 4


@dataclass(frozen=True)
class H2DiisScfParameterSummary:
    """Fixed parameter summary for one A-grid DIIS audit scheme."""

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
class H2DiisScfRouteResult:
    """Compact result for one spin state and one mixer scheme."""

    spin_state_label: str
    scheme_label: str
    path_type: str
    kinetic_version: str
    includes_nonlocal: bool
    parameter_summary: H2DiisScfParameterSummary
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
    trajectory_verdict: str
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str
    diis_used_iterations: tuple[int, ...]
    diis_history_sizes: tuple[int, ...]
    diis_fallback_iterations: tuple[int, ...]


@dataclass(frozen=True)
class H2DiisScfSpinAuditResult:
    """Three-scheme DIIS audit summary for one spin state."""

    spin_state_label: str
    baseline_route: H2DiisScfRouteResult
    smaller_mixing_route: H2DiisScfRouteResult
    diis_prototype_route: H2DiisScfRouteResult


@dataclass(frozen=True)
class H2MonitorGridDiisScfAuditResult:
    """Top-level H2 A-grid DIIS SCF audit result."""

    singlet: H2DiisScfSpinAuditResult
    triplet: H2DiisScfSpinAuditResult
    note: str


def _parameter_summary(
    result: H2StaticLocalScfDryRunResult,
    *,
    mixing: float,
) -> H2DiisScfParameterSummary:
    parameters = result.parameter_summary
    return H2DiisScfParameterSummary(
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
        max_iterations=_A_GRID_DIIS_SCF_MAX_ITERATIONS,
        density_tolerance=_A_GRID_DIIS_SCF_DENSITY_TOLERANCE,
        energy_tolerance=_A_GRID_DIIS_SCF_ENERGY_TOLERANCE,
        eigensolver_tolerance=_A_GRID_DIIS_SCF_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_A_GRID_DIIS_SCF_EIGENSOLVER_NCV,
        diis_enabled=parameters.diis_enabled,
        diis_warmup_iterations=parameters.diis_warmup_iterations,
        diis_history_length=parameters.diis_history_length,
        diis_residual_definition=parameters.diis_residual_definition,
    )


def _classify_trajectory(
    energy_history_ha: tuple[float, ...],
    density_residual_history: tuple[float, ...],
    *,
    converged: bool,
) -> str:
    if converged:
        return "converged"
    if len(energy_history_ha) < 6 or len(density_residual_history) < 6:
        return "stable_not_converged"

    tail_start = len(energy_history_ha) // 2
    tail_energies = np.asarray(energy_history_ha[tail_start:], dtype=np.float64)
    tail_residuals = np.asarray(density_residual_history[tail_start:], dtype=np.float64)
    if tail_energies.size % 2 != 0:
        tail_energies = tail_energies[1:]
        tail_residuals = tail_residuals[1:]
    if tail_energies.size < 4:
        return "stable_not_converged"

    even_energies = tail_energies[::2]
    odd_energies = tail_energies[1::2]
    even_residuals = tail_residuals[::2]
    odd_residuals = tail_residuals[1::2]
    energy_gap = abs(float(np.mean(even_energies) - np.mean(odd_energies)))
    residual_gap = abs(float(np.mean(even_residuals) - np.mean(odd_residuals)))
    energy_noise = max(float(np.std(even_energies)), float(np.std(odd_energies)), 1.0e-16)
    residual_noise = max(float(np.std(even_residuals)), float(np.std(odd_residuals)), 1.0e-16)

    if energy_gap > 5.0 * energy_noise and residual_gap > 5.0 * residual_noise:
        return "weak_two_cycle"
    if tail_residuals[-1] > 1.05 * tail_residuals[0]:
        return "diverging"
    if tail_residuals[-1] < 0.98 * tail_residuals[0]:
        return "slow_monotone_or_damped"
    return "stable_not_converged"


def _build_route_result(
    scheme_label: str,
    *,
    mixing: float,
    result: H2StaticLocalScfDryRunResult,
) -> H2DiisScfRouteResult:
    grid_geometry = (
        result.solve_up.operator_context.grid_geometry
        if result.solve_up is not None
        else result.solve_down.operator_context.grid_geometry
    )
    energy_history = tuple(float(value) for value in result.energy_history)
    density_residual_history = tuple(float(value) for value in result.density_residual_history)
    return H2DiisScfRouteResult(
        spin_state_label=result.spin_state_label,
        scheme_label=scheme_label,
        path_type=result.path_type,
        kinetic_version=result.kinetic_version,
        includes_nonlocal=bool(result.includes_nonlocal),
        parameter_summary=_parameter_summary(result, mixing=mixing),
        converged=bool(result.converged),
        iteration_count=int(result.iteration_count),
        final_total_energy_ha=float(result.energy.total),
        final_lowest_eigenvalue_ha=result.lowest_eigenvalue,
        energy_history_ha=energy_history,
        density_residual_history=density_residual_history,
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
        trajectory_verdict=_classify_trajectory(
            energy_history,
            density_residual_history,
            converged=bool(result.converged),
        ),
        diis_enabled=bool(result.diis_enabled),
        diis_warmup_iterations=int(result.diis_warmup_iterations),
        diis_history_length=int(result.diis_history_length),
        diis_residual_definition=result.diis_residual_definition,
        diis_used_iterations=result.diis_used_iterations,
        diis_history_sizes=result.diis_history_sizes,
        diis_fallback_iterations=result.diis_fallback_iterations,
    )


def _run_monitor_spin_scheme(
    spin_label: str,
    *,
    scheme_label: str,
    mixing: float,
    enable_diis: bool,
    case: BenchmarkCase,
) -> H2DiisScfRouteResult:
    result = run_h2_monitor_grid_scf_dry_run(
        spin_label,
        case=case,
        max_iterations=_A_GRID_DIIS_SCF_MAX_ITERATIONS,
        mixing=mixing,
        density_tolerance=_A_GRID_DIIS_SCF_DENSITY_TOLERANCE,
        energy_tolerance=_A_GRID_DIIS_SCF_ENERGY_TOLERANCE,
        eigensolver_tolerance=_A_GRID_DIIS_SCF_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_A_GRID_DIIS_SCF_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        enable_diis=enable_diis,
        diis_warmup_iterations=_A_GRID_DIIS_SCF_DIIS_WARMUP_ITERATIONS,
        diis_history_length=_A_GRID_DIIS_SCF_DIIS_HISTORY_LENGTH,
    )
    return _build_route_result(
        scheme_label,
        mixing=mixing,
        result=result,
    )


def _run_spin_audit(
    spin_label: str,
    *,
    case: BenchmarkCase,
) -> H2DiisScfSpinAuditResult:
    baseline_route = _run_monitor_spin_scheme(
        spin_label,
        scheme_label="baseline",
        mixing=0.20,
        enable_diis=False,
        case=case,
    )
    smaller_mixing_route = _run_monitor_spin_scheme(
        spin_label,
        scheme_label="smaller-mixing",
        mixing=0.10,
        enable_diis=False,
        case=case,
    )
    diis_prototype_route = _run_monitor_spin_scheme(
        spin_label,
        scheme_label="diis-prototype",
        mixing=0.10,
        enable_diis=True,
        case=case,
    )
    return H2DiisScfSpinAuditResult(
        spin_state_label=spin_label,
        baseline_route=baseline_route,
        smaller_mixing_route=smaller_mixing_route,
        diis_prototype_route=diis_prototype_route,
    )


def run_h2_monitor_grid_diis_scf_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridDiisScfAuditResult:
    """Run the small A-grid DIIS SCF audit on singlet and triplet."""

    return H2MonitorGridDiisScfAuditResult(
        singlet=_run_spin_audit("singlet", case=case),
        triplet=_run_spin_audit("triplet", case=case),
        note=(
            "This is an A-grid-only dry-run audit on the current repaired "
            "A-grid+patch+kinetic-trial-fix path. The Hamiltonian still contains "
            "only T + V_loc,ion + V_H + V_xc. The comparison is limited to three "
            "small mixer schemes: baseline linear mixing, smaller linear mixing, "
            "and a tiny warmup+DIIS prototype with fallback to linear mixing."
        ),
    )


def _print_route(route: H2DiisScfRouteResult) -> None:
    print(f"  scheme: {route.scheme_label}")
    print(f"    converged: {route.converged}")
    print(f"    iterations: {route.iteration_count}")
    print(
        "    final total energy [Ha]: "
        f"{route.final_total_energy_ha:.12f}"
    )
    print(
        "    final lowest eigenvalue [Ha]: "
        f"{route.final_lowest_eigenvalue_ha:.12f}"
        if route.final_lowest_eigenvalue_ha is not None
        else "    final lowest eigenvalue [Ha]: n/a"
    )
    print(
        "    final electrons: "
        f"rho_up={route.final_rho_up_electrons:.12f}, "
        f"rho_down={route.final_rho_down_electrons:.12f}"
    )
    print(
        "    DIIS: "
        f"enabled={route.diis_enabled}, "
        f"warmup={route.diis_warmup_iterations}, "
        f"history={route.diis_history_length}, "
        f"used={route.diis_used_iterations}, "
        f"fallback={route.diis_fallback_iterations}"
    )
    print(f"    verdict: {route.trajectory_verdict}")
    print("    history:")
    for index, energy in enumerate(route.energy_history_ha):
        residual = route.density_residual_history[index]
        energy_change = route.energy_change_history_ha[index]
        if energy_change is None:
            print(
                f"      iter={index + 1:02d}, E={energy:+.12f} Ha, "
                f"density_residual={residual:.6e}, energy_change=n/a"
            )
        else:
            print(
                f"      iter={index + 1:02d}, E={energy:+.12f} Ha, "
                f"density_residual={residual:.6e}, energy_change={energy_change:+.6e} Ha"
            )


def print_h2_monitor_grid_diis_scf_summary(
    result: H2MonitorGridDiisScfAuditResult,
) -> None:
    """Print the small H2 A-grid DIIS SCF audit summary."""

    print("IsoGridDFT H2 A-grid DIIS SCF audit")
    print(f"note: {result.note}")
    for spin_result in (result.singlet, result.triplet):
        print()
        print(f"spin: {spin_result.spin_state_label}")
        _print_route(spin_result.baseline_route)
        _print_route(spin_result.smaller_mixing_route)
        _print_route(spin_result.diis_prototype_route)


def main() -> int:
    result = run_h2_monitor_grid_diis_scf_audit()
    print_h2_monitor_grid_diis_scf_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
