"""H2 minimal SCF dry-run audit on the repaired A-grid static-local path.

This audit intentionally keeps the A-grid Hamiltonian limited to

    T + V_loc,ion + V_H + V_xc

with the current patch-assisted local ionic path, repaired monitor-grid
Hartree/Poisson, and the kinetic trial-fix branch. Nonlocal ionic action is
still excluded on the A-grid side, so this is a dry-run of the new main-grid
line rather than a final production SCF benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.ops import integrate_field
from isogrid.scf import H2ScfResult
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_minimal_scf
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_A_GRID_DRY_RUN_MAX_ITERATIONS = 20
_A_GRID_DRY_RUN_MIXING = 0.20
_A_GRID_DRY_RUN_DENSITY_TOLERANCE = 5.0e-3
_A_GRID_DRY_RUN_ENERGY_TOLERANCE = 5.0e-5
_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE = 1.0e-3
_A_GRID_DRY_RUN_EIGENSOLVER_NCV = 20


@dataclass(frozen=True)
class H2ScfDryRunRouteResult:
    """Compact SCF dry-run summary for one route and one spin state."""

    path_type: str
    spin_state_label: str
    kinetic_version: str
    includes_nonlocal: bool
    parameter_summary: str
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    lowest_eigenvalue_ha: float | None
    energy_history_ha: tuple[float, ...]
    density_residual_history: tuple[float, ...]
    energy_change_history_ha: tuple[float | None, ...]
    final_density_residual: float | None
    final_energy_change_ha: float | None
    final_rho_up_electrons: float
    final_rho_down_electrons: float
    final_energy_components: SinglePointEnergyComponents


@dataclass(frozen=True)
class H2MonitorGridScfDryRunAuditResult:
    """Top-level H2 SCF dry-run audit result for legacy and A-grid routes."""

    legacy_singlet: H2ScfDryRunRouteResult
    monitor_singlet: H2ScfDryRunRouteResult
    legacy_triplet: H2ScfDryRunRouteResult | None
    monitor_triplet: H2ScfDryRunRouteResult | None
    note: str


def _legacy_parameter_summary() -> str:
    return (
        "legacy minimal SCF baseline "
        "(includes nonlocal ionic action; mixing=0.60, max_iterations=8, "
        "density_tol=2.5e-3, energy_tol=1.0e-5, eig_tol=5.0e-3, ncv=20)"
    )


def _monitor_parameter_summary(result: H2StaticLocalScfDryRunResult) -> str:
    parameters = result.parameter_summary
    return (
        "A-grid+patch+trial-fix static-local dry-run: "
        f"shape={parameters.grid_shape}, "
        f"box={parameters.box_half_extents_bohr}, "
        f"weight_scale={parameters.weight_scale:.2f}, "
        f"radius_scale={parameters.radius_scale:.2f}, "
        f"patch_radius_scale={parameters.patch_radius_scale:.2f}, "
        f"patch_grid_shape={parameters.patch_grid_shape}, "
        f"strength={parameters.correction_strength:.2f}, "
        f"neighbors={parameters.interpolation_neighbors}, "
        f"mixing={_A_GRID_DRY_RUN_MIXING:.2f}, "
        f"max_iterations={_A_GRID_DRY_RUN_MAX_ITERATIONS}, "
        f"density_tol={_A_GRID_DRY_RUN_DENSITY_TOLERANCE:.1e}, "
        f"energy_tol={_A_GRID_DRY_RUN_ENERGY_TOLERANCE:.1e}, "
        f"eig_tol={_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE:.1e}, "
        f"ncv={_A_GRID_DRY_RUN_EIGENSOLVER_NCV}"
    )


def _lowest_eigenvalue(up: np.ndarray, down: np.ndarray) -> float | None:
    candidates: list[float] = []
    if up.size:
        candidates.append(float(up[0]))
    if down.size:
        candidates.append(float(down[0]))
    return min(candidates) if candidates else None


def _build_route_result_from_legacy(result: H2ScfResult) -> H2ScfDryRunRouteResult:
    grid_geometry = (
        result.solve_up.operator_context.grid_geometry
        if result.solve_up is not None
        else result.solve_down.operator_context.grid_geometry
    )
    return H2ScfDryRunRouteResult(
        path_type="legacy",
        spin_state_label=result.spin_state_label,
        kinetic_version="production",
        includes_nonlocal=True,
        parameter_summary=_legacy_parameter_summary(),
        converged=bool(result.converged),
        iteration_count=int(result.iteration_count),
        final_total_energy_ha=float(result.energy.total),
        lowest_eigenvalue_ha=_lowest_eigenvalue(result.eigenvalues_up, result.eigenvalues_down),
        energy_history_ha=tuple(float(record.energy.total) for record in result.history),
        density_residual_history=tuple(float(record.density_residual) for record in result.history),
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
    )


def _build_route_result_from_monitor(
    result: H2StaticLocalScfDryRunResult,
) -> H2ScfDryRunRouteResult:
    grid_geometry = (
        result.solve_up.operator_context.grid_geometry
        if result.solve_up is not None
        else result.solve_down.operator_context.grid_geometry
    )
    return H2ScfDryRunRouteResult(
        path_type=result.path_type,
        spin_state_label=result.spin_state_label,
        kinetic_version=result.kinetic_version,
        includes_nonlocal=bool(result.includes_nonlocal),
        parameter_summary=_monitor_parameter_summary(result),
        converged=bool(result.converged),
        iteration_count=int(result.iteration_count),
        final_total_energy_ha=float(result.energy.total),
        lowest_eigenvalue_ha=result.lowest_eigenvalue,
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
    )


def run_h2_monitor_grid_scf_dry_run_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    *,
    attempt_triplet: bool = True,
) -> H2MonitorGridScfDryRunAuditResult:
    """Run the H2 minimal SCF dry-run audit on legacy and A-grid routes."""

    legacy_singlet = _build_route_result_from_legacy(run_h2_minimal_scf("singlet", case=case))
    monitor_singlet = _build_route_result_from_monitor(
        run_h2_monitor_grid_scf_dry_run(
            "singlet",
            case=case,
            max_iterations=_A_GRID_DRY_RUN_MAX_ITERATIONS,
            mixing=_A_GRID_DRY_RUN_MIXING,
            density_tolerance=_A_GRID_DRY_RUN_DENSITY_TOLERANCE,
            energy_tolerance=_A_GRID_DRY_RUN_ENERGY_TOLERANCE,
            eigensolver_tolerance=_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE,
            eigensolver_ncv=_A_GRID_DRY_RUN_EIGENSOLVER_NCV,
        )
    )

    legacy_triplet = None
    monitor_triplet = None
    if attempt_triplet:
        legacy_triplet = _build_route_result_from_legacy(run_h2_minimal_scf("triplet", case=case))
        monitor_triplet = _build_route_result_from_monitor(
            run_h2_monitor_grid_scf_dry_run(
                "triplet",
                case=case,
                max_iterations=_A_GRID_DRY_RUN_MAX_ITERATIONS,
                mixing=_A_GRID_DRY_RUN_MIXING,
                density_tolerance=_A_GRID_DRY_RUN_DENSITY_TOLERANCE,
                energy_tolerance=_A_GRID_DRY_RUN_ENERGY_TOLERANCE,
                eigensolver_tolerance=_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE,
                eigensolver_ncv=_A_GRID_DRY_RUN_EIGENSOLVER_NCV,
            )
        )

    return H2MonitorGridScfDryRunAuditResult(
        legacy_singlet=legacy_singlet,
        monitor_singlet=monitor_singlet,
        legacy_triplet=legacy_triplet,
        monitor_triplet=monitor_triplet,
        note=(
            "This is an H2 SCF dry-run only. The A-grid route uses the repaired "
            "monitor Poisson path, patch-assisted local ionic energy, and the "
            "kinetic trial-fix branch, but still excludes nonlocal ionic action. "
            "The legacy route remains the current minimal SCF baseline with "
            "nonlocal included."
        ),
    )


def _print_route(route: H2ScfDryRunRouteResult) -> None:
    print(f"path: {route.path_type}")
    print(f"  spin: {route.spin_state_label}")
    print(f"  kinetic version: {route.kinetic_version}")
    print(f"  includes nonlocal: {route.includes_nonlocal}")
    print(f"  parameter summary: {route.parameter_summary}")
    print(f"  converged: {route.converged}")
    print(f"  iterations: {route.iteration_count}")
    print(f"  final total energy [Ha]: {route.final_total_energy_ha:.12f}")
    print(
        "  lowest eigenvalue [Ha]: "
        f"{route.lowest_eigenvalue_ha:.12f}" if route.lowest_eigenvalue_ha is not None else "  lowest eigenvalue [Ha]: n/a"
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
        f"V_nl={route.final_energy_components.nonlocal_ionic:.12f}, "
        f"E_H={route.final_energy_components.hartree:.12f}, "
        f"E_xc={route.final_energy_components.xc:.12f}, "
        f"E_II={route.final_energy_components.ion_ion_repulsion:.12f}"
    )
    print("  history:")
    for index, energy in enumerate(route.energy_history_ha):
        residual = route.density_residual_history[index]
        energy_change = route.energy_change_history_ha[index]
        print(
            "    "
            f"iter={index + 1:02d}, "
            f"E={energy:+.12f} Ha, "
            f"density_residual={residual:.6e}, "
            f"energy_change={energy_change:+.6e} Ha" if energy_change is not None else
            "    "
            f"iter={index + 1:02d}, "
            f"E={energy:+.12f} Ha, "
            f"density_residual={residual:.6e}, "
            "energy_change=n/a"
        )


def print_h2_monitor_grid_scf_dry_run_summary(
    result: H2MonitorGridScfDryRunAuditResult,
) -> None:
    """Print the compact H2 SCF dry-run summary."""

    print("IsoGridDFT H2 SCF dry-run audit")
    print(f"note: {result.note}")
    print(
        "important: the A-grid route currently contains only "
        "T + V_loc,ion + V_H + V_xc; nonlocal remains on the legacy path"
    )
    print()
    _print_route(result.legacy_singlet)
    print()
    _print_route(result.monitor_singlet)
    if result.legacy_triplet is not None and result.monitor_triplet is not None:
        print()
        _print_route(result.legacy_triplet)
        print()
        _print_route(result.monitor_triplet)
    print()
    print("dry-run verdict:")
    print(
        "  singlet total-energy delta A-grid minus legacy [Ha]: "
        f"{result.monitor_singlet.final_total_energy_ha - result.legacy_singlet.final_total_energy_ha:+.12f}"
    )
    print(
        "  singlet lowest-eigenvalue delta A-grid minus legacy [Ha]: "
        f"{(result.monitor_singlet.lowest_eigenvalue_ha or 0.0) - (result.legacy_singlet.lowest_eigenvalue_ha or 0.0):+.12f}"
    )
    print(
        "  singlet SCF ready for next-stage A-grid work: "
        f"{result.monitor_singlet.converged}"
    )


def main() -> int:
    result = run_h2_monitor_grid_scf_dry_run_audit()
    print_h2_monitor_grid_scf_dry_run_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
