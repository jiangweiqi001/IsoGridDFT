"""H2 singlet static local-chain audit for legacy, A-grid, and A-grid+patch.

This audit keeps the same frozen H2 singlet trial orbital and density on all
three routes and compares only the static local chain

    T_s + E_loc,ion + E_H + E_xc

The monitor-grid patch still corrects only the near-core local-GTH energy. It
does not alter the trial density, Hartree solve, or LSDA evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.poisson import solve_hartree_potential
from isogrid.pseudo import LocalPotentialPatchParameters
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential_with_patch
from isogrid.xc import evaluate_lsda_energy

from .h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary
from .h2_monitor_grid_patch_local_audit import _patch_summary
from .h2_monitor_grid_ts_eloc_audit import H2GridEnergyGeometrySummary
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital
from .h2_monitor_grid_ts_eloc_audit import evaluate_h2_singlet_ts_eloc_on_legacy_grid
from .h2_monitor_grid_ts_eloc_audit import evaluate_h2_singlet_ts_eloc_on_monitor_grid
from .pyscf_h2_reference import run_reference_spin_state


@dataclass(frozen=True)
class H2StaticLocalRouteResult:
    """Resolved H2 singlet static-local audit result for one route."""

    path_type: str
    grid_parameter_summary: str
    geometry_summary: H2GridEnergyGeometrySummary
    patch_parameter_summary: H2MonitorPatchParameterSummary | None
    kinetic_energy: float
    local_ionic_energy: float
    hartree_energy: float
    xc_energy: float
    static_local_energy: float
    reference_pyscf_total_energy: float
    reference_offset_ha: float
    reference_offset_mha: float
    delta_vs_legacy_ha: float
    delta_vs_legacy_mha: float
    delta_vs_unpatched_monitor_ha: float
    delta_vs_unpatched_monitor_mha: float
    improvement_vs_unpatched_monitor_ha: float
    improvement_vs_unpatched_monitor_mha: float
    patch_correction_ha: float
    patch_correction_mha: float
    hartree_solver_method: str
    hartree_solver_iterations: int
    hartree_residual_max: float


@dataclass(frozen=True)
class H2MonitorPatchHartreeXCAuditResult:
    """Top-level H2 singlet static local-chain audit result."""

    legacy_result: H2StaticLocalRouteResult
    monitor_unpatched_result: H2StaticLocalRouteResult
    patch_recheck_results: tuple[H2StaticLocalRouteResult, ...]
    best_patch_result: H2StaticLocalRouteResult
    note: str


def _find_h2_singlet(case: BenchmarkCase):
    for spin_state in case.spin_states:
        if spin_state.label.lower() == "singlet":
            return spin_state
    raise ValueError(f"`{case.name}` does not define a singlet state.")


def _grid_parameter_summary() -> str:
    return (
        "A-grid baseline: "
        f"shape={H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE}, "
        f"box={H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR}, "
        f"weight_scale={H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE:.2f}, "
        f"radius_scale={H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE:.2f}"
    )


def _default_patch_recheck() -> tuple[LocalPotentialPatchParameters, ...]:
    return (
        LocalPotentialPatchParameters(
            patch_radius_scale=0.75,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.20,
            interpolation_neighbors=8,
        ),
        LocalPotentialPatchParameters(
            patch_radius_scale=0.75,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.30,
            interpolation_neighbors=8,
        ),
    )


def _build_singlet_spin_densities(orbital: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orbital_density = np.abs(orbital) ** 2
    rho_up = np.asarray(orbital_density, dtype=np.float64)
    rho_down = np.asarray(orbital_density, dtype=np.float64)
    return rho_up, rho_down, rho_up + rho_down


def _resolve_hartree_and_xc(
    *,
    grid_geometry,
    orbital: np.ndarray,
) -> tuple[float, float, object]:
    rho_up, rho_down, rho_total = _build_singlet_spin_densities(orbital)
    hartree_result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho_total,
    )
    hartree_energy = 0.5 * float(
        np.sum(
            rho_total * hartree_result.potential * grid_geometry.cell_volumes,
            dtype=np.float64,
        )
    )
    xc_energy = evaluate_lsda_energy(
        rho_up=rho_up,
        rho_down=rho_down,
        grid_geometry=grid_geometry,
    )
    return hartree_energy, xc_energy, hartree_result


def _build_route_result(
    *,
    path_type: str,
    grid_parameter_summary: str,
    geometry_summary: H2GridEnergyGeometrySummary,
    kinetic_energy: float,
    local_ionic_energy: float,
    hartree_energy: float,
    xc_energy: float,
    reference_pyscf_total_energy: float,
    legacy_static_local_energy: float,
    unpatched_monitor_static_local_energy: float,
    hartree_result,
    patch_parameter_summary: H2MonitorPatchParameterSummary | None = None,
    patch_correction_ha: float = 0.0,
) -> H2StaticLocalRouteResult:
    static_local_energy = float(kinetic_energy + local_ionic_energy + hartree_energy + xc_energy)
    reference_offset_ha = static_local_energy - reference_pyscf_total_energy
    improvement_vs_unpatched = (
        abs(unpatched_monitor_static_local_energy - reference_pyscf_total_energy)
        - abs(reference_offset_ha)
    )
    return H2StaticLocalRouteResult(
        path_type=path_type,
        grid_parameter_summary=grid_parameter_summary,
        geometry_summary=geometry_summary,
        patch_parameter_summary=patch_parameter_summary,
        kinetic_energy=float(kinetic_energy),
        local_ionic_energy=float(local_ionic_energy),
        hartree_energy=float(hartree_energy),
        xc_energy=float(xc_energy),
        static_local_energy=static_local_energy,
        reference_pyscf_total_energy=float(reference_pyscf_total_energy),
        reference_offset_ha=float(reference_offset_ha),
        reference_offset_mha=float(reference_offset_ha * 1000.0),
        delta_vs_legacy_ha=float(static_local_energy - legacy_static_local_energy),
        delta_vs_legacy_mha=float((static_local_energy - legacy_static_local_energy) * 1000.0),
        delta_vs_unpatched_monitor_ha=float(
            static_local_energy - unpatched_monitor_static_local_energy
        ),
        delta_vs_unpatched_monitor_mha=float(
            (static_local_energy - unpatched_monitor_static_local_energy) * 1000.0
        ),
        improvement_vs_unpatched_monitor_ha=float(improvement_vs_unpatched),
        improvement_vs_unpatched_monitor_mha=float(improvement_vs_unpatched * 1000.0),
        patch_correction_ha=float(patch_correction_ha),
        patch_correction_mha=float(patch_correction_ha * 1000.0),
        hartree_solver_method=hartree_result.solver_method,
        hartree_solver_iterations=int(hartree_result.solver_iterations),
        hartree_residual_max=float(hartree_result.residual_max),
    )


def _pick_best_patch_result(
    results: tuple[H2StaticLocalRouteResult, ...],
) -> H2StaticLocalRouteResult:
    return max(
        results,
        key=lambda result: (
            result.improvement_vs_unpatched_monitor_ha,
            result.improvement_vs_unpatched_monitor_mha,
        ),
    )


def run_h2_monitor_grid_patch_hartree_xc_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    patch_recheck: tuple[LocalPotentialPatchParameters, ...] | None = None,
) -> H2MonitorPatchHartreeXCAuditResult:
    """Audit `T_s + E_loc,ion + E_H + E_xc` for legacy, A-grid, and A-grid+patch."""

    singlet = _find_h2_singlet(case)
    reference_result = run_reference_spin_state(case=case, spin_state=singlet)

    legacy_t, legacy_eloc, _, legacy_summary = evaluate_h2_singlet_ts_eloc_on_legacy_grid(case=case)
    legacy_geometry = build_default_h2_grid_geometry(case=case)
    legacy_orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=legacy_geometry)
    legacy_hartree_energy, legacy_xc_energy, legacy_hartree_result = _resolve_hartree_and_xc(
        grid_geometry=legacy_geometry,
        orbital=legacy_orbital,
    )
    legacy_static_local_energy = legacy_t + legacy_eloc + legacy_hartree_energy + legacy_xc_energy
    legacy_result = _build_route_result(
        path_type="legacy",
        grid_parameter_summary="legacy structured sinh baseline",
        geometry_summary=legacy_summary,
        kinetic_energy=legacy_t,
        local_ionic_energy=legacy_eloc,
        hartree_energy=legacy_hartree_energy,
        xc_energy=legacy_xc_energy,
        reference_pyscf_total_energy=reference_result.total_energy,
        legacy_static_local_energy=legacy_static_local_energy,
        unpatched_monitor_static_local_energy=legacy_static_local_energy,
        hartree_result=legacy_hartree_result,
    )

    monitor_grid = build_h2_local_patch_development_monitor_grid()
    monitor_t, monitor_eloc, _, monitor_summary = evaluate_h2_singlet_ts_eloc_on_monitor_grid(
        case=case,
        grid_geometry=monitor_grid,
    )
    monitor_orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=monitor_grid)
    monitor_hartree_energy, monitor_xc_energy, monitor_hartree_result = _resolve_hartree_and_xc(
        grid_geometry=monitor_grid,
        orbital=monitor_orbital,
    )
    monitor_static_local_energy = monitor_t + monitor_eloc + monitor_hartree_energy + monitor_xc_energy
    monitor_unpatched_result = _build_route_result(
        path_type="monitor_a_grid",
        grid_parameter_summary=_grid_parameter_summary(),
        geometry_summary=monitor_summary,
        kinetic_energy=monitor_t,
        local_ionic_energy=monitor_eloc,
        hartree_energy=monitor_hartree_energy,
        xc_energy=monitor_xc_energy,
        reference_pyscf_total_energy=reference_result.total_energy,
        legacy_static_local_energy=legacy_static_local_energy,
        unpatched_monitor_static_local_energy=monitor_static_local_energy,
        hartree_result=monitor_hartree_result,
    )

    rho_total = _build_singlet_spin_densities(monitor_orbital)[2]
    base_local_evaluation = evaluate_monitor_grid_local_ionic_potential(
        case=case,
        grid_geometry=monitor_grid,
    )
    if patch_recheck is None:
        patch_recheck = _default_patch_recheck()

    patch_recheck_results = tuple(
        _build_route_result(
            path_type="monitor_a_grid_plus_patch",
            grid_parameter_summary=_grid_parameter_summary(),
            geometry_summary=monitor_summary,
            kinetic_energy=monitor_t,
            local_ionic_energy=patch_evaluation.corrected_local_energy,
            hartree_energy=monitor_hartree_energy,
            xc_energy=monitor_xc_energy,
            reference_pyscf_total_energy=reference_result.total_energy,
            legacy_static_local_energy=legacy_static_local_energy,
            unpatched_monitor_static_local_energy=monitor_static_local_energy,
            hartree_result=monitor_hartree_result,
            patch_parameter_summary=_patch_summary(patch_evaluation.patch_parameters),
            patch_correction_ha=patch_evaluation.total_patch_correction,
        )
        for patch_evaluation in (
            evaluate_monitor_grid_local_ionic_potential_with_patch(
                case=case,
                grid_geometry=monitor_grid,
                density_field=rho_total,
                patch_parameters=patch_parameters,
                base_evaluation=base_local_evaluation,
            )
            for patch_parameters in patch_recheck
        )
    )
    return H2MonitorPatchHartreeXCAuditResult(
        legacy_result=legacy_result,
        monitor_unpatched_result=monitor_unpatched_result,
        patch_recheck_results=patch_recheck_results,
        best_patch_result=_pick_best_patch_result(patch_recheck_results),
        note=(
            "This audit compares only the static local chain "
            "T_s + E_loc,ion + E_H + E_xc on H2 singlet. "
            "The monitor-grid Hartree term now uses the repaired monitor Poisson split. "
            "The patch still corrects only the local-GTH near-core energy; "
            "nonlocal, eigensolver, and SCF remain on their current paths."
        ),
    )


def _print_route_result(result: H2StaticLocalRouteResult) -> None:
    print(f"path: {result.path_type}")
    print(f"  grid summary: {result.grid_parameter_summary}")
    print(f"  T_s [Ha]: {result.kinetic_energy:.12f}")
    print(f"  E_loc,ion [Ha]: {result.local_ionic_energy:.12f}")
    print(f"  E_H [Ha]: {result.hartree_energy:.12f}")
    print(f"  E_xc [Ha]: {result.xc_energy:.12f}")
    print(f"  static local sum [Ha]: {result.static_local_energy:.12f}")
    print(f"  offset vs PySCF singlet total [mHa]: {result.reference_offset_mha:+.3f}")
    print(f"  delta vs legacy [mHa]: {result.delta_vs_legacy_mha:+.3f}")
    print(f"  delta vs unpatched A-grid [mHa]: {result.delta_vs_unpatched_monitor_mha:+.3f}")
    print(
        "  improvement vs unpatched A-grid [mHa]: "
        f"{result.improvement_vs_unpatched_monitor_mha:+.3f}"
    )
    print(
        "  Hartree solver: "
        f"{result.hartree_solver_method}, "
        f"iterations={result.hartree_solver_iterations}, "
        f"residual={result.hartree_residual_max:.3e}"
    )
    if result.patch_parameter_summary is not None:
        patch = result.patch_parameter_summary
        print(
            "  patch params: "
            f"radius_scale={patch.patch_radius_scale:.2f}, "
            f"grid_shape={patch.patch_grid_shape}, "
            f"strength={patch.correction_strength:.2f}, "
            f"neighbors={patch.interpolation_neighbors}"
        )
        print(f"  patch correction [mHa]: {result.patch_correction_mha:+.3f}")


def print_h2_monitor_grid_patch_hartree_xc_summary(
    result: H2MonitorPatchHartreeXCAuditResult,
) -> None:
    """Print the compact H2 singlet static local-chain audit summary."""

    print("IsoGridDFT H2 singlet A-grid patch Hartree/XC audit")
    print(f"note: {result.note}")
    print(
        "important: patch changes only E_loc,ion; E_H and E_xc are evaluated from the "
        "same frozen density on the same grid."
    )
    print()
    _print_route_result(result.legacy_result)
    print()
    _print_route_result(result.monitor_unpatched_result)
    print()
    print("patch recheck points:")
    for patch_result in result.patch_recheck_results:
        _print_route_result(patch_result)
        print()
    best_patch = result.best_patch_result.patch_parameter_summary
    print("best patch point:")
    print(
        "  params: "
        f"radius_scale={best_patch.patch_radius_scale:.2f}, "
        f"grid_shape={best_patch.patch_grid_shape}, "
        f"strength={best_patch.correction_strength:.2f}, "
        f"neighbors={best_patch.interpolation_neighbors}"
    )
    print(f"  status vs legacy [mHa]: {result.best_patch_result.delta_vs_legacy_mha:+.3f}")
    print(
        "  improvement vs unpatched A-grid [mHa]: "
        f"{result.best_patch_result.improvement_vs_unpatched_monitor_mha:+.3f}"
    )
    print("formal frozen baseline:")
    print("  route: monitor_a_grid_plus_patch @ radius_scale=0.75, grid_shape=(25, 25, 25), strength=1.30, neighbors=8")
    print(
        "  remaining component deltas vs legacy [mHa]: "
        f"T_s={(result.best_patch_result.kinetic_energy - result.legacy_result.kinetic_energy) * 1000.0:+.3f}, "
        f"E_loc,ion={(result.best_patch_result.local_ionic_energy - result.legacy_result.local_ionic_energy) * 1000.0:+.3f}, "
        f"E_H={(result.best_patch_result.hartree_energy - result.legacy_result.hartree_energy) * 1000.0:+.3f}, "
        f"E_xc={(result.best_patch_result.xc_energy - result.legacy_result.xc_energy) * 1000.0:+.3f}"
    )
    print(
        "  net status vs legacy [mHa]: "
        f"{result.best_patch_result.delta_vs_legacy_mha:+.3f}"
    )


def main() -> int:
    result = run_h2_monitor_grid_patch_hartree_xc_audit()
    print_h2_monitor_grid_patch_hartree_xc_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
