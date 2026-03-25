"""H2 singlet audit for patch-assisted local-GTH correction on the A-grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.pseudo import LocalIonicPotentialPatchEvaluation
from isogrid.pseudo import LocalPotentialPatchParameters
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential_with_patch

from .h2_monitor_grid_ts_eloc_audit import H2GridEnergyGeometrySummary
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital
from .h2_monitor_grid_ts_eloc_audit import evaluate_h2_singlet_ts_eloc_on_legacy_grid
from .h2_monitor_grid_ts_eloc_audit import evaluate_h2_singlet_ts_eloc_on_monitor_grid
from .pyscf_h2_reference import run_reference_spin_state

_SINGLET_OCCUPATION = 2.0


@dataclass(frozen=True)
class H2MonitorPatchParameterSummary:
    """Patch-parameter summary for one local-GTH patch scan point."""

    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int


@dataclass(frozen=True)
class H2TsElocPatchRouteResult:
    """Resolved H2 singlet `T_s + E_loc,ion` result for one route."""

    path_type: str
    grid_parameter_summary: str
    geometry_summary: H2GridEnergyGeometrySummary
    patch_parameter_summary: H2MonitorPatchParameterSummary | None
    kinetic_energy: float
    local_ionic_energy: float
    ts_plus_eloc_energy: float
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


@dataclass(frozen=True)
class H2MonitorPatchLocalAuditResult:
    """Top-level H2 singlet patch-local audit result."""

    legacy_result: H2TsElocPatchRouteResult
    monitor_unpatched_result: H2TsElocPatchRouteResult
    patch_scan_results: tuple[H2TsElocPatchRouteResult, ...]
    best_patch_result: H2TsElocPatchRouteResult
    note: str


def _find_h2_singlet(case: BenchmarkCase):
    for spin_state in case.spin_states:
        if spin_state.label.lower() == "singlet":
            return spin_state
    raise ValueError(f"`{case.name}` does not define a singlet state.")


def _default_patch_scan() -> tuple[LocalPotentialPatchParameters, ...]:
    return (
        LocalPotentialPatchParameters(
            patch_radius_scale=0.60,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.00,
            interpolation_neighbors=8,
        ),
        LocalPotentialPatchParameters(
            patch_radius_scale=0.75,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.00,
            interpolation_neighbors=8,
        ),
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


def _grid_parameter_summary() -> str:
    return (
        "A-grid baseline: "
        f"shape={H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE}, "
        f"box={H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR}, "
        f"weight_scale={H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE:.2f}, "
        f"radius_scale={H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE:.2f}"
    )


def _patch_summary(
    patch_parameters: LocalPotentialPatchParameters,
) -> H2MonitorPatchParameterSummary:
    return H2MonitorPatchParameterSummary(
        patch_radius_scale=patch_parameters.patch_radius_scale,
        patch_grid_shape=patch_parameters.patch_grid_shape,
        correction_strength=patch_parameters.correction_strength,
        interpolation_neighbors=patch_parameters.interpolation_neighbors,
    )


def _build_route_result(
    *,
    path_type: str,
    grid_parameter_summary: str,
    geometry_summary: H2GridEnergyGeometrySummary,
    kinetic_energy: float,
    local_ionic_energy: float,
    reference_pyscf_total_energy: float,
    legacy_ts_plus_eloc: float,
    legacy_reference_offset_ha: float,
    unpatched_monitor_ts_plus_eloc: float,
    unpatched_monitor_reference_offset_ha: float,
    patch_parameter_summary: H2MonitorPatchParameterSummary | None = None,
    patch_correction_ha: float = 0.0,
) -> H2TsElocPatchRouteResult:
    ts_plus_eloc_energy = float(kinetic_energy + local_ionic_energy)
    reference_offset_ha = ts_plus_eloc_energy - reference_pyscf_total_energy
    improvement_vs_unpatched = (
        abs(unpatched_monitor_reference_offset_ha) - abs(reference_offset_ha)
    )
    return H2TsElocPatchRouteResult(
        path_type=path_type,
        grid_parameter_summary=grid_parameter_summary,
        geometry_summary=geometry_summary,
        patch_parameter_summary=patch_parameter_summary,
        kinetic_energy=float(kinetic_energy),
        local_ionic_energy=float(local_ionic_energy),
        ts_plus_eloc_energy=ts_plus_eloc_energy,
        reference_pyscf_total_energy=float(reference_pyscf_total_energy),
        reference_offset_ha=float(reference_offset_ha),
        reference_offset_mha=float(reference_offset_ha * 1000.0),
        delta_vs_legacy_ha=float(ts_plus_eloc_energy - legacy_ts_plus_eloc),
        delta_vs_legacy_mha=float((ts_plus_eloc_energy - legacy_ts_plus_eloc) * 1000.0),
        delta_vs_unpatched_monitor_ha=float(ts_plus_eloc_energy - unpatched_monitor_ts_plus_eloc),
        delta_vs_unpatched_monitor_mha=float(
            (ts_plus_eloc_energy - unpatched_monitor_ts_plus_eloc) * 1000.0
        ),
        improvement_vs_unpatched_monitor_ha=float(improvement_vs_unpatched),
        improvement_vs_unpatched_monitor_mha=float(improvement_vs_unpatched * 1000.0),
        patch_correction_ha=float(patch_correction_ha),
        patch_correction_mha=float(patch_correction_ha * 1000.0),
    )


def _build_patch_route_result(
    *,
    patch_evaluation: LocalIonicPotentialPatchEvaluation,
    geometry_summary: H2GridEnergyGeometrySummary,
    kinetic_energy: float,
    reference_pyscf_total_energy: float,
    legacy_result: H2TsElocPatchRouteResult,
    unpatched_result: H2TsElocPatchRouteResult,
) -> H2TsElocPatchRouteResult:
    return _build_route_result(
        path_type="monitor_a_grid_plus_patch",
        grid_parameter_summary=unpatched_result.grid_parameter_summary,
        geometry_summary=geometry_summary,
        kinetic_energy=kinetic_energy,
        local_ionic_energy=patch_evaluation.corrected_local_energy,
        reference_pyscf_total_energy=reference_pyscf_total_energy,
        legacy_ts_plus_eloc=legacy_result.ts_plus_eloc_energy,
        legacy_reference_offset_ha=legacy_result.reference_offset_ha,
        unpatched_monitor_ts_plus_eloc=unpatched_result.ts_plus_eloc_energy,
        unpatched_monitor_reference_offset_ha=unpatched_result.reference_offset_ha,
        patch_parameter_summary=_patch_summary(patch_evaluation.patch_parameters),
        patch_correction_ha=patch_evaluation.total_patch_correction,
    )


def _pick_best_patch_result(
    results: tuple[H2TsElocPatchRouteResult, ...],
) -> H2TsElocPatchRouteResult:
    return max(
        results,
        key=lambda result: (
            result.improvement_vs_unpatched_monitor_ha,
            result.improvement_vs_unpatched_monitor_mha,
        ),
    )


def run_h2_monitor_grid_patch_local_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    patch_scan: tuple[LocalPotentialPatchParameters, ...] | None = None,
) -> H2MonitorPatchLocalAuditResult:
    """Audit the patch-assisted local-GTH correction on the H2 singlet A-grid."""

    singlet = _find_h2_singlet(case)
    reference_result = run_reference_spin_state(case=case, spin_state=singlet)

    legacy_t, legacy_eloc, _, legacy_summary = evaluate_h2_singlet_ts_eloc_on_legacy_grid(case=case)
    legacy_result = _build_route_result(
        path_type="legacy",
        grid_parameter_summary="legacy structured sinh baseline",
        geometry_summary=legacy_summary,
        kinetic_energy=legacy_t,
        local_ionic_energy=legacy_eloc,
        reference_pyscf_total_energy=reference_result.total_energy,
        legacy_ts_plus_eloc=legacy_t + legacy_eloc,
        legacy_reference_offset_ha=(legacy_t + legacy_eloc) - reference_result.total_energy,
        unpatched_monitor_ts_plus_eloc=legacy_t + legacy_eloc,
        unpatched_monitor_reference_offset_ha=(legacy_t + legacy_eloc) - reference_result.total_energy,
    )

    monitor_grid = build_h2_local_patch_development_monitor_grid()
    monitor_t, monitor_eloc, _, monitor_summary = evaluate_h2_singlet_ts_eloc_on_monitor_grid(
        case=case,
        grid_geometry=monitor_grid,
    )
    unpatched_result = _build_route_result(
        path_type="monitor_a_grid",
        grid_parameter_summary=_grid_parameter_summary(),
        geometry_summary=monitor_summary,
        kinetic_energy=monitor_t,
        local_ionic_energy=monitor_eloc,
        reference_pyscf_total_energy=reference_result.total_energy,
        legacy_ts_plus_eloc=legacy_result.ts_plus_eloc_energy,
        legacy_reference_offset_ha=legacy_result.reference_offset_ha,
        unpatched_monitor_ts_plus_eloc=monitor_t + monitor_eloc,
        unpatched_monitor_reference_offset_ha=(monitor_t + monitor_eloc) - reference_result.total_energy,
    )

    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=monitor_grid)
    rho_total = _SINGLET_OCCUPATION * np.abs(orbital) ** 2
    base_local_evaluation = evaluate_monitor_grid_local_ionic_potential(
        case=case,
        grid_geometry=monitor_grid,
    )
    if patch_scan is None:
        patch_scan = _default_patch_scan()

    patch_scan_results = tuple(
        _build_patch_route_result(
            patch_evaluation=evaluate_monitor_grid_local_ionic_potential_with_patch(
                case=case,
                grid_geometry=monitor_grid,
                density_field=rho_total,
                patch_parameters=patch_parameters,
                base_evaluation=base_local_evaluation,
            ),
            geometry_summary=monitor_summary,
            kinetic_energy=monitor_t,
            reference_pyscf_total_energy=reference_result.total_energy,
            legacy_result=legacy_result,
            unpatched_result=unpatched_result,
        )
        for patch_parameters in patch_scan
    )
    return H2MonitorPatchLocalAuditResult(
        legacy_result=legacy_result,
        monitor_unpatched_result=unpatched_result,
        patch_scan_results=patch_scan_results,
        best_patch_result=_pick_best_patch_result(patch_scan_results),
        note=(
            "This audit changes only the A-grid local-GTH near-core contribution. "
            "Nonlocal, Hartree, XC, eigensolver, and SCF remain on their current paths."
        ),
    )


def _print_route_result(result: H2TsElocPatchRouteResult) -> None:
    print(f"path: {result.path_type}")
    print(f"  grid summary: {result.grid_parameter_summary}")
    print(f"  T_s [Ha]: {result.kinetic_energy:.12f}")
    print(f"  E_loc,ion [Ha]: {result.local_ionic_energy:.12f}")
    print(f"  T_s + E_loc,ion [Ha]: {result.ts_plus_eloc_energy:.12f}")
    print(f"  offset vs PySCF singlet total [mHa]: {result.reference_offset_mha:+.3f}")
    print(f"  delta vs legacy [mHa]: {result.delta_vs_legacy_mha:+.3f}")
    print(f"  delta vs unpatched A-grid [mHa]: {result.delta_vs_unpatched_monitor_mha:+.3f}")
    print(
        f"  improvement vs unpatched A-grid [mHa]: "
        f"{result.improvement_vs_unpatched_monitor_mha:+.3f}"
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


def print_h2_monitor_grid_patch_local_summary(
    result: H2MonitorPatchLocalAuditResult,
) -> None:
    """Print the compact H2 singlet A-grid local-patch audit summary."""

    print("IsoGridDFT H2 singlet A-grid local-GTH patch audit")
    print(f"note: {result.note}")
    print()
    _print_route_result(result.legacy_result)
    print()
    _print_route_result(result.monitor_unpatched_result)
    print()
    print("patch scan points:")
    for scan_result in result.patch_scan_results:
        _print_route_result(scan_result)
        print()
    patch = result.best_patch_result.patch_parameter_summary
    print("best patch point:")
    print(
        "  params: "
        f"radius_scale={patch.patch_radius_scale:.2f}, "
        f"grid_shape={patch.patch_grid_shape}, "
        f"strength={patch.correction_strength:.2f}, "
        f"neighbors={patch.interpolation_neighbors}"
    )
    print(
        "  status vs legacy [mHa]: "
        f"{result.best_patch_result.delta_vs_legacy_mha:+.3f}"
    )
    print(
        "  improvement vs unpatched A-grid [mHa]: "
        f"{result.best_patch_result.improvement_vs_unpatched_monitor_mha:+.3f}"
    )


def main() -> int:
    result = run_h2_monitor_grid_patch_local_audit()
    print_h2_monitor_grid_patch_local_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
