"""Very small H2 k=2 subspace audit on legacy and A-grid+patch+trial-fix routes.

This audit does not change the eigensolver, SCF, or any operator kernel. It
only asks whether the current near-degenerate A-grid k=2 subspace can be
reorganized into a more interpretable basis by a tiny in-subspace analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem

from .h2_monitor_grid_operator_audit import _weighted_inner_product
from .h2_monitor_grid_orbital_shape_audit import H2OrbitalShapeResult
from .h2_monitor_grid_orbital_shape_audit import H2MonitorPatchParameterSummary
from .h2_monitor_grid_orbital_shape_audit import _build_frozen_spin_densities
from .h2_monitor_grid_orbital_shape_audit import _default_patch_parameters
from .h2_monitor_grid_orbital_shape_audit import _evaluate_route
from .h2_monitor_grid_orbital_shape_audit import _grid_parameter_summary
from .h2_monitor_grid_orbital_shape_audit import _orbital_result
from .h2_monitor_grid_orbital_shape_audit import _patch_summary
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital


@dataclass(frozen=True)
class H2K2SubspaceMatrixSummary:
    """Small 2x2 subspace matrix summary."""

    label: str
    matrix: tuple[tuple[float, float], tuple[float, float]]
    eigenvalues: tuple[float, float]


@dataclass(frozen=True)
class H2K2SubspaceRotationSummary:
    """Chosen very small in-subspace rotation summary."""

    rotation_label: str
    rotation_matrix: tuple[tuple[float, float], tuple[float, float]]
    raw_bonding_overlaps: tuple[float, float]
    rotated_bonding_overlaps: tuple[float, float]
    rotated_orbitals: tuple[H2OrbitalShapeResult, H2OrbitalShapeResult]
    note: str


@dataclass(frozen=True)
class H2K2SubspaceRouteResult:
    """k=2 subspace audit result for one route."""

    path_type: str
    kinetic_version: str
    grid_parameter_summary: str
    patch_parameter_summary: H2MonitorPatchParameterSummary | None
    raw_k2_eigenvalues_ha: tuple[float, float]
    raw_k2_gap_ha: float
    raw_k2_orbitals: tuple[H2OrbitalShapeResult, H2OrbitalShapeResult]
    inversion_matrix: H2K2SubspaceMatrixSummary
    z_mirror_matrix: H2K2SubspaceMatrixSummary
    bonding_rotation: H2K2SubspaceRotationSummary | None


@dataclass(frozen=True)
class H2MonitorGridK2SubspaceAuditResult:
    """Top-level H2 k=2 subspace audit."""

    legacy_route: H2K2SubspaceRouteResult
    monitor_patch_trial_fix_route: H2K2SubspaceRouteResult
    diagnosis: str
    note: str


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1])),
        (float(matrix[1, 0]), float(matrix[1, 1])),
    )


def _subspace_matrix(
    orbitals: np.ndarray,
    transformed_orbitals: np.ndarray,
    grid_geometry,
    label: str,
) -> H2K2SubspaceMatrixSummary:
    matrix = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            matrix[i, j] = float(np.real(_weighted_inner_product(orbitals[i], transformed_orbitals[j], grid_geometry)))
    eigenvalues = np.linalg.eigvalsh(matrix)
    return H2K2SubspaceMatrixSummary(
        label=label,
        matrix=_matrix_tuple(matrix),
        eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
    )


def _bonding_rotation_summary(
    *,
    orbitals: np.ndarray,
    raw_orbitals: tuple[H2OrbitalShapeResult, H2OrbitalShapeResult],
    grid_geometry,
    case: BenchmarkCase,
) -> H2K2SubspaceRotationSummary:
    bonding_template, _, _ = _build_frozen_spin_densities(case=case, grid_geometry=grid_geometry)
    raw_bonding_overlaps = np.asarray(
        [float(np.real(_weighted_inner_product(orbitals[i], bonding_template, grid_geometry))) for i in range(2)],
        dtype=np.float64,
    )
    overlap_norm = float(np.linalg.norm(raw_bonding_overlaps))
    if overlap_norm <= 1.0e-14:
        rotation_matrix = np.eye(2, dtype=np.float64)
        note = "Bonding-template overlap in the k=2 subspace is numerically zero; no rotation was applied."
    else:
        vector = raw_bonding_overlaps / overlap_norm
        rotation_matrix = np.array(
            [
                [vector[0], -vector[1]],
                [vector[1], vector[0]],
            ],
            dtype=np.float64,
        )
        note = (
            "This very small rotation chooses the first orbital as the maximum-overlap bonding-like "
            "direction inside the raw k=2 subspace, and the second as its orthogonal complement."
        )

    rotated_orbitals = np.einsum("ip,ixyz->pxyz", rotation_matrix, orbitals)
    rotated_bonding_overlaps = tuple(
        float(np.real(_weighted_inner_product(rotated_orbitals[i], bonding_template, grid_geometry)))
        for i in range(2)
    )
    rotated_orbital_results = (
        _orbital_result(
            solve_label="rotated_k2",
            orbital_index=0,
            orbital=rotated_orbitals[0],
            eigenvalue=float(raw_orbitals[0].eigenvalue_ha),
            residual_norm=float(raw_orbitals[0].residual_norm),
            grid_geometry=grid_geometry,
            case=case,
        ),
        _orbital_result(
            solve_label="rotated_k2",
            orbital_index=1,
            orbital=rotated_orbitals[1],
            eigenvalue=float(raw_orbitals[1].eigenvalue_ha),
            residual_norm=float(raw_orbitals[1].residual_norm),
            grid_geometry=grid_geometry,
            case=case,
        ),
    )
    return H2K2SubspaceRotationSummary(
        rotation_label="bonding_overlap_rotation",
        rotation_matrix=_matrix_tuple(rotation_matrix),
        raw_bonding_overlaps=(float(raw_bonding_overlaps[0]), float(raw_bonding_overlaps[1])),
        rotated_bonding_overlaps=rotated_bonding_overlaps,
        rotated_orbitals=rotated_orbital_results,
        note=note,
    )


def _route_result(
    *,
    case: BenchmarkCase,
    path_type: str,
    kinetic_version: str,
) -> H2K2SubspaceRouteResult:
    route = _evaluate_route(case=case, path_type=path_type, kinetic_version=kinetic_version)
    if path_type == "legacy":
        grid_geometry = build_default_h2_grid_geometry(case=case)
        patch_summary = None
    elif path_type == "monitor_a_grid_plus_patch":
        grid_geometry = build_h2_local_patch_development_monitor_grid()
        patch_summary = _patch_summary(_default_patch_parameters())
    else:
        raise ValueError(f"Unsupported path_type `{path_type}`.")

    _, rho_up, rho_down = _build_frozen_spin_densities(case=case, grid_geometry=grid_geometry)

    k2_result = solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=2,
        case=case,
        tolerance=1.0e-3,
        ncv=20,
        use_monitor_patch=(path_type == "monitor_a_grid_plus_patch"),
        patch_parameters=_default_patch_parameters() if path_type == "monitor_a_grid_plus_patch" else None,
        kinetic_version=kinetic_version,
    )
    orbitals = np.asarray(k2_result.orbitals, dtype=np.float64)
    inversion = np.asarray([orbitals[0][::-1, ::-1, ::-1], orbitals[1][::-1, ::-1, ::-1]], dtype=np.float64)
    z_mirror = np.asarray([orbitals[0][:, :, ::-1], orbitals[1][:, :, ::-1]], dtype=np.float64)

    bonding_rotation = None
    if path_type == "monitor_a_grid_plus_patch":
        bonding_rotation = _bonding_rotation_summary(
            orbitals=orbitals,
            raw_orbitals=route.k2_orbitals,
            grid_geometry=grid_geometry,
            case=case,
        )

    return H2K2SubspaceRouteResult(
        path_type=path_type,
        kinetic_version=kinetic_version,
        grid_parameter_summary=_grid_parameter_summary(path_type),
        patch_parameter_summary=patch_summary,
        raw_k2_eigenvalues_ha=(float(k2_result.eigenvalues[0]), float(k2_result.eigenvalues[1])),
        raw_k2_gap_ha=float(k2_result.eigenvalues[1] - k2_result.eigenvalues[0]),
        raw_k2_orbitals=route.k2_orbitals,
        inversion_matrix=_subspace_matrix(orbitals, inversion, grid_geometry, "inversion"),
        z_mirror_matrix=_subspace_matrix(orbitals, z_mirror, grid_geometry, "z_mirror"),
        bonding_rotation=bonding_rotation,
    )


def _diagnosis(
    legacy_route: H2K2SubspaceRouteResult,
    monitor_route: H2K2SubspaceRouteResult,
) -> str:
    rotated = monitor_route.bonding_rotation
    if rotated is None:
        return "No monitor-route bonding rotation was available."

    raw_gap = monitor_route.raw_k2_gap_ha
    raw0, raw1 = monitor_route.raw_k2_orbitals
    rot0, rot1 = rotated.rotated_orbitals
    if (
        raw_gap < 1.0e-3
        and raw0.node_summary.centerline_sign_changes == 0
        and raw1.node_summary.centerline_sign_changes == 0
        and rot1.node_summary.centerline_sign_changes > 10
        and rot1.boundary_summary.boundary_layer_norm_fraction < 0.01
    ):
        return (
            "The A-grid k=2 pair behaves like a near-degenerate mixed subspace, but the simplest "
            "bonding-overlap rotation does not recover a clean legacy-like antibonding orbital. It "
            "produces one bonding-like state and one strongly oscillatory orthogonal complement, so "
            "the subspace is still not physically organized enough for a confident A-grid SCF dry-run."
        )
    return (
        "The k=2 subspace audit did not recover a clearly interpretable bonding/antibonding pair "
        "from the current A-grid near-degenerate subspace."
    )


def run_h2_monitor_grid_k2_subspace_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridK2SubspaceAuditResult:
    """Run the very small H2 k=2 subspace audit."""

    legacy_route = _route_result(case=case, path_type="legacy", kinetic_version="production")
    monitor_route = _route_result(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
    )
    return H2MonitorGridK2SubspaceAuditResult(
        legacy_route=legacy_route,
        monitor_patch_trial_fix_route=monitor_route,
        diagnosis=_diagnosis(legacy_route, monitor_route),
        note=(
            "This is a very small k=2 subspace audit only. It keeps the current fixed-potential "
            "legacy and A-grid+patch+trial-fix routes unchanged, and asks only whether the raw "
            "A-grid near-degenerate pair can be reorganized into a more interpretable basis."
        ),
    )


def _print_orbital(label: str, result: H2OrbitalShapeResult) -> None:
    print(
        f"  {label}: eig={result.eigenvalue_ha:+.12f} Ha, res={result.residual_norm:.6e}, "
        f"z-mirror mismatch={result.symmetry_summary.z_mirror_best_mismatch:.6e}, "
        f"nodes={result.node_summary.centerline_sign_changes}, "
        f"boundary={result.boundary_summary.boundary_layer_norm_fraction:.6f}"
    )
    for sample in result.centerline_samples:
        print(
            "    "
            f"z[{sample.sample_index:02d}]={sample.z_coordinate_bohr:+.6f} -> {sample.orbital_value:+.12f}"
        )


def _print_route(route: H2K2SubspaceRouteResult) -> None:
    print(f"path: {route.path_type}")
    print(f"  kinetic version: {route.kinetic_version}")
    print(f"  raw k2 eigenvalues [Ha]: {list(route.raw_k2_eigenvalues_ha)}")
    print(f"  raw k2 gap [Ha]: {route.raw_k2_gap_ha:+.12f}")
    print(f"  inversion matrix: {route.inversion_matrix.matrix}")
    print(f"  z-mirror matrix: {route.z_mirror_matrix.matrix}")
    _print_orbital("raw orbital[0]", route.raw_k2_orbitals[0])
    _print_orbital("raw orbital[1]", route.raw_k2_orbitals[1])
    if route.bonding_rotation is not None:
        print(f"  bonding rotation matrix: {route.bonding_rotation.rotation_matrix}")
        print(f"  raw bonding overlaps: {route.bonding_rotation.raw_bonding_overlaps}")
        print(f"  rotated bonding overlaps: {route.bonding_rotation.rotated_bonding_overlaps}")
        _print_orbital("rotated orbital[0]", route.bonding_rotation.rotated_orbitals[0])
        _print_orbital("rotated orbital[1]", route.bonding_rotation.rotated_orbitals[1])
        print(f"  rotation note: {route.bonding_rotation.note}")


def print_h2_monitor_grid_k2_subspace_audit_summary(
    result: H2MonitorGridK2SubspaceAuditResult,
) -> None:
    print("IsoGridDFT H2 k=2 subspace audit")
    print(f"note: {result.note}")
    print()
    _print_route(result.legacy_route)
    print()
    _print_route(result.monitor_patch_trial_fix_route)
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_k2_subspace_audit()
    print_h2_monitor_grid_k2_subspace_audit_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
