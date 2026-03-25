"""Very small H2 fixed-potential orbital-shape audit on legacy and A-grid paths.

This audit does not modify the eigensolver or SCF. It inspects the spatial
shape of the already-converged fixed-potential orbitals on

    T + V_loc,ion + V_H + V_xc

for the legacy grid and the current A-grid+patch+kinetic-trial-fix route.
The goal is to decide whether the fixed-potential A-grid orbitals are already
physically healthy enough to justify an eventual A-grid SCF dry-run.
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
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field
from isogrid.pseudo import LocalPotentialPatchParameters

from .baselines import H2_FIXED_POTENTIAL_EIGENSOLVER_TRIAL_FIX_BASELINE
from .h2_monitor_grid_operator_audit import _compute_region_masks
from .h2_monitor_grid_operator_audit import _weighted_inner_product
from .h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary
from .h2_monitor_grid_patch_local_audit import _patch_summary
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class OrbitalCenterlineSample:
    """One center-line sample for orbital-shape diagnostics."""

    sample_index: int
    z_coordinate_bohr: float
    orbital_value: float


@dataclass(frozen=True)
class OrbitalSymmetrySummary:
    """Parity and inversion diagnostics for one orbital."""

    inversion_overlap: float
    inversion_best_parity: str
    inversion_best_mismatch: float
    z_mirror_overlap: float
    z_mirror_best_parity: str
    z_mirror_best_mismatch: float
    z_center_of_mass_bohr: float


@dataclass(frozen=True)
class OrbitalNodeSummary:
    """Center-line node summary for one orbital."""

    centerline_sign_changes: int
    node_positions_bohr: tuple[float, ...]
    far_field_sign_changes: int
    left_endpoint_value: float
    center_value: float
    right_endpoint_value: float


@dataclass(frozen=True)
class OrbitalBoundarySummary:
    """Regionwise norm fractions for one orbital."""

    near_core_norm_fraction: float
    center_norm_fraction: float
    far_field_norm_fraction: float
    boundary_layer_norm_fraction: float
    far_field_max_abs_value: float
    boundary_layer_max_abs_value: float


@dataclass(frozen=True)
class H2OrbitalShapeResult:
    """Shape audit result for one orbital from one solve."""

    solve_label: str
    orbital_index: int
    eigenvalue_ha: float
    weighted_norm: float
    residual_norm: float
    symmetry_summary: OrbitalSymmetrySummary
    node_summary: OrbitalNodeSummary
    boundary_summary: OrbitalBoundarySummary
    centerline_samples: tuple[OrbitalCenterlineSample, ...]


@dataclass(frozen=True)
class H2OrbitalShapeRouteResult:
    """Fixed-potential orbital-shape audit result for one route."""

    path_type: str
    kinetic_version: str
    grid_parameter_summary: str
    patch_parameter_summary: H2MonitorPatchParameterSummary | None
    frozen_density_integral: float
    converged_k1: bool
    converged_k2: bool
    k1_orbital: H2OrbitalShapeResult
    k2_orbitals: tuple[H2OrbitalShapeResult, H2OrbitalShapeResult]
    k2_gap_ha: float


@dataclass(frozen=True)
class H2MonitorGridOrbitalShapeAuditResult:
    """Top-level H2 orbital-shape audit on legacy and A-grid+patch+trial-fix."""

    legacy_route: H2OrbitalShapeRouteResult
    monitor_patch_trial_fix_route: H2OrbitalShapeRouteResult
    diagnosis: str
    note: str


def _default_patch_parameters() -> LocalPotentialPatchParameters:
    return LocalPotentialPatchParameters(
        patch_radius_scale=0.75,
        patch_grid_shape=(25, 25, 25),
        correction_strength=1.30,
        interpolation_neighbors=8,
    )


def _grid_parameter_summary(path_type: str) -> str:
    if path_type == "legacy":
        return "legacy structured sinh baseline"
    return (
        "A-grid baseline: "
        f"shape={H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE}, "
        f"box={H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR}, "
        f"weight_scale={H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE:.2f}, "
        f"radius_scale={H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE:.2f}"
    )


def _build_frozen_spin_densities(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    orbital_density = np.asarray(np.abs(orbital) ** 2, dtype=np.float64)
    rho_up = orbital_density.copy()
    rho_down = orbital_density.copy()
    return orbital, rho_up, rho_down


def _weighted_norm(field: np.ndarray, grid_geometry: GridGeometryLike) -> float:
    norm2 = float(np.real_if_close(_weighted_inner_product(field, field, grid_geometry)))
    return float(np.sqrt(max(norm2, 0.0)))


def _normalize_orbital(field: np.ndarray, grid_geometry: GridGeometryLike) -> np.ndarray:
    orbital = validate_orbital_field(field, grid_geometry=grid_geometry, name="orbital")
    return np.asarray(orbital / _weighted_norm(orbital, grid_geometry), dtype=np.float64)


def _centerline_values(field: np.ndarray, grid_geometry: GridGeometryLike) -> tuple[np.ndarray, np.ndarray]:
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    return (
        np.asarray(grid_geometry.z_points[center_ix, center_iy, :], dtype=np.float64),
        np.asarray(field[center_ix, center_iy, :], dtype=np.float64),
    )


def _centerline_samples(
    field: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> tuple[OrbitalCenterlineSample, ...]:
    z_coordinates, values = _centerline_values(field, grid_geometry)
    sample_indices = (
        0,
        len(z_coordinates) // 4,
        len(z_coordinates) // 2,
        3 * len(z_coordinates) // 4,
        len(z_coordinates) - 1,
    )
    return tuple(
        OrbitalCenterlineSample(
            sample_index=index,
            z_coordinate_bohr=float(z_coordinates[index]),
            orbital_value=float(values[index]),
        )
        for index in sample_indices
    )


def _weighted_mismatch_ratio(
    left: np.ndarray,
    right: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> float:
    diff = np.asarray(left - right, dtype=np.float64)
    numerator = float(np.real_if_close(_weighted_inner_product(diff, diff, grid_geometry)))
    denominator = float(np.real_if_close(_weighted_inner_product(left, left, grid_geometry)))
    return float(np.sqrt(max(numerator, 0.0) / max(denominator, 1.0e-30)))


def _symmetry_summary(
    field: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> OrbitalSymmetrySummary:
    orbital = _normalize_orbital(field, grid_geometry)
    inversion_field = orbital[::-1, ::-1, ::-1]
    z_mirror_field = orbital[:, :, ::-1]
    denominator = float(np.real_if_close(_weighted_inner_product(orbital, orbital, grid_geometry)))
    inversion_overlap = float(
        np.real_if_close(_weighted_inner_product(orbital, inversion_field, grid_geometry) / denominator)
    )
    z_overlap = float(
        np.real_if_close(_weighted_inner_product(orbital, z_mirror_field, grid_geometry) / denominator)
    )

    inversion_even_mismatch = _weighted_mismatch_ratio(orbital, inversion_field, grid_geometry)
    inversion_odd_mismatch = _weighted_mismatch_ratio(orbital, -inversion_field, grid_geometry)
    z_even_mismatch = _weighted_mismatch_ratio(orbital, z_mirror_field, grid_geometry)
    z_odd_mismatch = _weighted_mismatch_ratio(orbital, -z_mirror_field, grid_geometry)

    density = np.abs(orbital) ** 2
    density_norm = integrate_field(density, grid_geometry=grid_geometry)
    z_center_of_mass = float(
        integrate_field(density * grid_geometry.z_points, grid_geometry=grid_geometry) / density_norm
    )
    return OrbitalSymmetrySummary(
        inversion_overlap=inversion_overlap,
        inversion_best_parity="even" if inversion_even_mismatch <= inversion_odd_mismatch else "odd",
        inversion_best_mismatch=min(inversion_even_mismatch, inversion_odd_mismatch),
        z_mirror_overlap=z_overlap,
        z_mirror_best_parity="even" if z_even_mismatch <= z_odd_mismatch else "odd",
        z_mirror_best_mismatch=min(z_even_mismatch, z_odd_mismatch),
        z_center_of_mass_bohr=z_center_of_mass,
    )


def _node_summary(
    field: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> OrbitalNodeSummary:
    z_coordinates, values = _centerline_values(field, grid_geometry)
    threshold = max(1.0e-8, 1.0e-3 * float(np.max(np.abs(values))))
    sign_changes = 0
    far_field_sign_changes = 0
    node_positions: list[float] = []
    for index in range(len(values) - 1):
        left = values[index]
        right = values[index + 1]
        if abs(left) < threshold or abs(right) < threshold:
            continue
        if left * right < 0.0:
            sign_changes += 1
            node_positions.append(float(0.5 * (z_coordinates[index] + z_coordinates[index + 1])))
            if max(abs(z_coordinates[index]), abs(z_coordinates[index + 1])) > 0.7 * float(np.max(np.abs(z_coordinates))):
                far_field_sign_changes += 1

    midpoint = len(values) // 2
    return OrbitalNodeSummary(
        centerline_sign_changes=sign_changes,
        node_positions_bohr=tuple(node_positions),
        far_field_sign_changes=far_field_sign_changes,
        left_endpoint_value=float(values[0]),
        center_value=float(values[midpoint]),
        right_endpoint_value=float(values[-1]),
    )


def _boundary_summary(
    field: np.ndarray,
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase,
) -> OrbitalBoundarySummary:
    orbital = _normalize_orbital(field, grid_geometry)
    density = np.abs(orbital) ** 2
    total_norm = integrate_field(density, grid_geometry=grid_geometry)
    masks = dict(_compute_region_masks(case=case, grid_geometry=grid_geometry))

    bounds = (
        (float(np.min(grid_geometry.x_points)), float(np.max(grid_geometry.x_points))),
        (float(np.min(grid_geometry.y_points)), float(np.max(grid_geometry.y_points))),
        (float(np.min(grid_geometry.z_points)), float(np.max(grid_geometry.z_points))),
    )
    distance_to_box = np.minimum.reduce(
        (
            grid_geometry.x_points - bounds[0][0],
            bounds[0][1] - grid_geometry.x_points,
            grid_geometry.y_points - bounds[1][0],
            bounds[1][1] - grid_geometry.y_points,
            grid_geometry.z_points - bounds[2][0],
            bounds[2][1] - grid_geometry.z_points,
        )
    )
    boundary_layer_mask = distance_to_box < 1.0

    def _fraction(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(integrate_field(density * mask.astype(np.float64), grid_geometry=grid_geometry) / total_norm)

    far_field_mask = masks["far_field"]
    return OrbitalBoundarySummary(
        near_core_norm_fraction=_fraction(masks["near_core"]),
        center_norm_fraction=_fraction(masks["center"]),
        far_field_norm_fraction=_fraction(far_field_mask),
        boundary_layer_norm_fraction=_fraction(boundary_layer_mask),
        far_field_max_abs_value=float(np.max(np.abs(orbital[far_field_mask]))) if np.any(far_field_mask) else 0.0,
        boundary_layer_max_abs_value=float(np.max(np.abs(orbital[boundary_layer_mask]))) if np.any(boundary_layer_mask) else 0.0,
    )


def _orbital_result(
    *,
    solve_label: str,
    orbital_index: int,
    orbital: np.ndarray,
    eigenvalue: float,
    residual_norm: float,
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase,
) -> H2OrbitalShapeResult:
    normalized = _normalize_orbital(orbital, grid_geometry)
    return H2OrbitalShapeResult(
        solve_label=solve_label,
        orbital_index=orbital_index,
        eigenvalue_ha=float(eigenvalue),
        weighted_norm=_weighted_norm(normalized, grid_geometry),
        residual_norm=float(residual_norm),
        symmetry_summary=_symmetry_summary(normalized, grid_geometry),
        node_summary=_node_summary(normalized, grid_geometry),
        boundary_summary=_boundary_summary(normalized, grid_geometry, case),
        centerline_samples=_centerline_samples(normalized, grid_geometry),
    )


def _evaluate_route(
    *,
    case: BenchmarkCase,
    path_type: str,
    kinetic_version: str,
) -> H2OrbitalShapeRouteResult:
    if path_type == "legacy":
        grid_geometry = build_default_h2_grid_geometry(case=case)
        patch_parameters = None
        use_monitor_patch = False
        patch_summary = None
    elif path_type == "monitor_a_grid_plus_patch":
        grid_geometry = build_h2_local_patch_development_monitor_grid()
        patch_parameters = _default_patch_parameters()
        use_monitor_patch = True
        patch_summary = _patch_summary(patch_parameters)
    else:
        raise ValueError(f"Unsupported path_type `{path_type}`.")

    _, rho_up, rho_down = _build_frozen_spin_densities(case=case, grid_geometry=grid_geometry)
    rho_total = rho_up + rho_down

    k1_result = solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=1,
        case=case,
        tolerance=1.0e-3,
        ncv=20,
        use_monitor_patch=use_monitor_patch,
        patch_parameters=patch_parameters,
        kinetic_version=kinetic_version,
    )
    k2_result = solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=2,
        case=case,
        tolerance=1.0e-3,
        ncv=20,
        use_monitor_patch=use_monitor_patch,
        patch_parameters=patch_parameters,
        kinetic_version=kinetic_version,
    )

    return H2OrbitalShapeRouteResult(
        path_type=path_type,
        kinetic_version=kinetic_version,
        grid_parameter_summary=_grid_parameter_summary(path_type),
        patch_parameter_summary=patch_summary,
        frozen_density_integral=float(integrate_field(rho_total, grid_geometry=grid_geometry)),
        converged_k1=bool(k1_result.converged),
        converged_k2=bool(k2_result.converged),
        k1_orbital=_orbital_result(
            solve_label="k1",
            orbital_index=0,
            orbital=k1_result.orbitals[0],
            eigenvalue=float(k1_result.eigenvalues[0]),
            residual_norm=float(k1_result.residual_norms[0]),
            grid_geometry=grid_geometry,
            case=case,
        ),
        k2_orbitals=(
            _orbital_result(
                solve_label="k2",
                orbital_index=0,
                orbital=k2_result.orbitals[0],
                eigenvalue=float(k2_result.eigenvalues[0]),
                residual_norm=float(k2_result.residual_norms[0]),
                grid_geometry=grid_geometry,
                case=case,
            ),
            _orbital_result(
                solve_label="k2",
                orbital_index=1,
                orbital=k2_result.orbitals[1],
                eigenvalue=float(k2_result.eigenvalues[1]),
                residual_norm=float(k2_result.residual_norms[1]),
                grid_geometry=grid_geometry,
                case=case,
            ),
        ),
        k2_gap_ha=float(k2_result.eigenvalues[1] - k2_result.eigenvalues[0]),
    )


def _diagnosis(
    legacy_route: H2OrbitalShapeRouteResult,
    monitor_route: H2OrbitalShapeRouteResult,
) -> str:
    monitor_k1 = monitor_route.k1_orbital
    legacy_k2_second = legacy_route.k2_orbitals[1]
    monitor_k2_first, monitor_k2_second = monitor_route.k2_orbitals
    if (
        monitor_k1.symmetry_summary.z_mirror_best_parity == "even"
        and monitor_k1.boundary_summary.boundary_layer_norm_fraction < 0.05
        and monitor_k2_first.boundary_summary.boundary_layer_norm_fraction < 0.05
        and monitor_k2_second.boundary_summary.boundary_layer_norm_fraction < 0.05
        and monitor_route.k2_gap_ha < 1.0e-3
    ):
        return (
            "The A-grid+patch+trial-fix k=1 orbital looks physically healthy enough for a bonding-like "
            "ground state, but the k=2 solve returns a near-degenerate pair rather than a clear legacy-like "
            "bonding/antibonding split. The dominant remaining risk is shape mixing inside that nearly "
            "degenerate subspace, not obvious boundary pollution."
        )
    return (
        "The orbital-shape audit still sees unresolved structure in the A-grid k=2 pair relative to "
        "the legacy fixed-potential reference."
    )


def run_h2_monitor_grid_orbital_shape_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridOrbitalShapeAuditResult:
    """Run the very small H2 orbital-shape audit on legacy and A-grid+patch+trial-fix."""

    legacy_route = _evaluate_route(case=case, path_type="legacy", kinetic_version="production")
    monitor_route = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
    )
    return H2MonitorGridOrbitalShapeAuditResult(
        legacy_route=legacy_route,
        monitor_patch_trial_fix_route=monitor_route,
        diagnosis=_diagnosis(legacy_route, monitor_route),
        note=(
            "This is a very small fixed-potential orbital-shape audit only. It compares legacy and "
            "A-grid+patch+kinetic-trial-fix routes on the same H2 singlet frozen density, without "
            "nonlocal or SCF."
        ),
    )


def _print_orbital_result(result: H2OrbitalShapeResult) -> None:
    symmetry = result.symmetry_summary
    node = result.node_summary
    boundary = result.boundary_summary
    print(
        f"  {result.solve_label} orbital[{result.orbital_index}]: "
        f"eig={result.eigenvalue_ha:+.12f} Ha, "
        f"res={result.residual_norm:.6e}, "
        f"norm={result.weighted_norm:.12f}"
    )
    print(
        "    symmetry: "
        f"inversion={symmetry.inversion_best_parity} mismatch={symmetry.inversion_best_mismatch:.6e}, "
        f"z-mirror={symmetry.z_mirror_best_parity} mismatch={symmetry.z_mirror_best_mismatch:.6e}, "
        f"z-COM={symmetry.z_center_of_mass_bohr:+.6e} Bohr"
    )
    print(
        "    nodes: "
        f"count={node.centerline_sign_changes}, "
        f"far-field-count={node.far_field_sign_changes}, "
        f"positions={list(node.node_positions_bohr)}"
    )
    print(
        "    boundary fractions: "
        f"near-core={boundary.near_core_norm_fraction:.6f}, "
        f"center={boundary.center_norm_fraction:.6f}, "
        f"far-field={boundary.far_field_norm_fraction:.6f}, "
        f"boundary-layer={boundary.boundary_layer_norm_fraction:.6f}"
    )
    print("    center-line samples:")
    for sample in result.centerline_samples:
        print(
            "      "
            f"z[{sample.sample_index:02d}]={sample.z_coordinate_bohr:+.6f} Bohr -> "
            f"{sample.orbital_value:+.12f}"
        )


def _print_route(route: H2OrbitalShapeRouteResult) -> None:
    print(f"path: {route.path_type}")
    print(f"  kinetic version: {route.kinetic_version}")
    print(f"  grid summary: {route.grid_parameter_summary}")
    print(f"  frozen density integral: {route.frozen_density_integral:.12f}")
    print(f"  converged(k1/k2): {route.converged_k1} / {route.converged_k2}")
    print(f"  k2 gap [Ha]: {route.k2_gap_ha:+.12f}")
    if route.patch_parameter_summary is not None:
        patch = route.patch_parameter_summary
        print(
            "  patch params: "
            f"radius_scale={patch.patch_radius_scale:.2f}, "
            f"grid_shape={patch.patch_grid_shape}, "
            f"strength={patch.correction_strength:.2f}, "
            f"neighbors={patch.interpolation_neighbors}"
        )
    _print_orbital_result(route.k1_orbital)
    _print_orbital_result(route.k2_orbitals[0])
    _print_orbital_result(route.k2_orbitals[1])


def print_h2_monitor_grid_orbital_shape_audit_summary(
    result: H2MonitorGridOrbitalShapeAuditResult,
) -> None:
    """Print the compact H2 orbital-shape audit summary."""

    print("IsoGridDFT H2 fixed-potential orbital-shape audit")
    print(f"note: {result.note}")
    print(
        "trial-fix eigensolver baseline: "
        f"k1={H2_FIXED_POTENTIAL_EIGENSOLVER_TRIAL_FIX_BASELINE.monitor_patch_trial_fix_k1_route.eigenvalues_ha[0]:+.12f} Ha, "
        f"k2={list(H2_FIXED_POTENTIAL_EIGENSOLVER_TRIAL_FIX_BASELINE.monitor_patch_trial_fix_k2_route.eigenvalues_ha)}"
    )
    print()
    _print_route(result.legacy_route)
    print()
    _print_route(result.monitor_patch_trial_fix_route)
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_orbital_shape_audit()
    print_h2_monitor_grid_orbital_shape_audit_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
