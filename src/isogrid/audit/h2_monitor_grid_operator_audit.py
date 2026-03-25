"""Operator-level audit for the H2 frozen-density static-local Hamiltonian.

This audit does not modify the eigensolver or SCF. It isolates the current
fixed-potential static-local operator

    H_local = T + V_loc,ion + V_H + V_xc

on the legacy grid, the raw A-grid, and the A-grid plus frozen patch embedding.
The goal is to diagnose why the A-grid path can look favorable at the static
local-chain energy level but still fail badly once the fixed-potential
eigensolver acts on it.
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
from isogrid.ks import FixedPotentialStaticLocalOperatorContext
from isogrid.ks import apply_fixed_potential_static_local_operator
from isogrid.ks import prepare_fixed_potential_static_local_operator
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ops import apply_kinetic_operator
from isogrid.ops import validate_orbital_field
from isogrid.ops import weighted_l2_norm
from isogrid.ops.kinetic import apply_monitor_grid_kinetic_operator_trial_boundary_fix
from isogrid.pseudo import LocalPotentialPatchParameters

from .baselines import H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE
from .h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary
from .h2_monitor_grid_patch_local_audit import _patch_summary
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class ScalarFieldSummary:
    """Compact scalar-field summary for one 3D field."""

    minimum: float
    maximum: float
    mean: float
    rms: float


@dataclass(frozen=True)
class WeightedExpectationSummary:
    """Weighted Rayleigh quotient and component expectations for one orbital."""

    weighted_norm: float
    denominator: float
    rayleigh_quotient: float
    kinetic_expectation: float
    local_ionic_expectation: float
    hartree_expectation: float
    xc_expectation: float


@dataclass(frozen=True)
class ResidualRegionSummary:
    """Regionwise residual diagnostics for one operator residual field."""

    region_name: str
    point_fraction: float
    residual_summary: ScalarFieldSummary
    weighted_rms: float
    mean_signed_residual: float


@dataclass(frozen=True)
class SelfAdjointnessProbe:
    """Weighted self-adjointness probe for one operator or sub-operator."""

    absolute_difference: float
    relative_difference: float
    left_inner_product_real: float
    right_inner_product_real: float


@dataclass(frozen=True)
class OperatorCenterlineSample:
    """One center-line sample for orbital or residual diagnostics."""

    sample_index: int
    z_coordinate_bohr: float
    orbital_value: float
    residual_value: float


@dataclass(frozen=True)
class H2StaticLocalOperatorRouteResult:
    """Operator-level audit result for one grid/path type."""

    path_type: str
    kinetic_version: str
    grid_parameter_summary: str
    patch_parameter_summary: H2MonitorPatchParameterSummary | None
    frozen_density_integral: float
    trial_expectation: WeightedExpectationSummary
    eigen_expectation: WeightedExpectationSummary
    eigenvalue: float
    weighted_residual_norm: float
    residual_summary: ScalarFieldSummary
    residual_centerline_samples: tuple[OperatorCenterlineSample, ...]
    residual_regions: tuple[ResidualRegionSummary, ...]
    self_adjoint_probe_total: SelfAdjointnessProbe
    self_adjoint_probe_kinetic: SelfAdjointnessProbe
    self_adjoint_probe_local_potential: SelfAdjointnessProbe
    converged: bool
    patch_embedding_energy_mismatch: float | None
    patch_embedded_correction_mha: float | None


@dataclass(frozen=True)
class H2MonitorGridOperatorAuditResult:
    """Top-level operator-level audit for legacy, A-grid, and A-grid+patch."""

    legacy_result: H2StaticLocalOperatorRouteResult
    monitor_patch_production_result: H2StaticLocalOperatorRouteResult
    monitor_patch_trial_fix_result: H2StaticLocalOperatorRouteResult
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


def _build_frozen_density(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    orbital_density = np.asarray(np.abs(orbital) ** 2, dtype=np.float64)
    rho_up = orbital_density.copy()
    rho_down = orbital_density.copy()
    return orbital, rho_up, rho_down, rho_up + rho_down


def _weighted_inner_product(
    left: np.ndarray,
    right: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> complex:
    left_field = validate_orbital_field(left, grid_geometry=grid_geometry, name="left")
    right_field = validate_orbital_field(right, grid_geometry=grid_geometry, name="right")
    return complex(
        np.sum(
            np.conjugate(left_field) * right_field * grid_geometry.cell_volumes,
            dtype=np.complex128,
        )
    )


def _field_summary(field: np.ndarray, mask: np.ndarray | None = None) -> ScalarFieldSummary:
    values = np.asarray(field, dtype=np.float64)
    if mask is not None:
        values = values[mask]
    return ScalarFieldSummary(
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        rms=float(np.sqrt(np.mean(values * values))),
    )


def _compute_region_masks(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> tuple[tuple[str, np.ndarray], ...]:
    center = tuple(
        sum(atom.position[axis] for atom in case.geometry.atoms) / float(len(case.geometry.atoms))
        for axis in range(3)
    )
    nearest_atom_distance = np.full(grid_geometry.spec.shape, np.inf, dtype=np.float64)
    for atom in case.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        radius = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float64)
        nearest_atom_distance = np.minimum(nearest_atom_distance, radius)

    center_radius = np.sqrt(
        (grid_geometry.x_points - center[0]) ** 2
        + (grid_geometry.y_points - center[1]) ** 2
        + (grid_geometry.z_points - center[2]) ** 2,
        dtype=np.float64,
    )

    if isinstance(grid_geometry, MonitorGridGeometry):
        bounds = grid_geometry.spec.box_bounds
    else:
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

    near_core_mask = nearest_atom_distance < 0.9
    center_mask = (center_radius < 0.8) & (~near_core_mask)
    far_field_mask = (distance_to_box < 1.5) & (~near_core_mask)
    if not np.any(center_mask):
        center_mask = center_radius < 1.2
    if not np.any(far_field_mask):
        far_field_mask = center_radius > np.quantile(center_radius, 0.85)
    return (
        ("near_core", near_core_mask),
        ("center", center_mask),
        ("far_field", far_field_mask),
    )


def _residual_region_summaries(
    residual_field: np.ndarray,
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase,
) -> tuple[ResidualRegionSummary, ...]:
    summaries = []
    total_points = float(np.prod(grid_geometry.spec.shape))
    for name, mask in _compute_region_masks(case=case, grid_geometry=grid_geometry):
        if not np.any(mask):
            continue
        residual_summary = _field_summary(residual_field, mask=mask)
        weights = grid_geometry.cell_volumes[mask]
        values = np.asarray(residual_field, dtype=np.float64)[mask]
        weighted_rms = float(np.sqrt(np.sum(values * values * weights) / np.sum(weights)))
        mean_signed = float(np.sum(values * weights) / np.sum(weights))
        summaries.append(
            ResidualRegionSummary(
                region_name=name,
                point_fraction=float(np.sum(mask) / total_points),
                residual_summary=residual_summary,
                weighted_rms=weighted_rms,
                mean_signed_residual=mean_signed,
            )
        )
    return tuple(summaries)


def _centerline_samples(
    orbital: np.ndarray,
    residual_field: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> tuple[OperatorCenterlineSample, ...]:
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    z_coordinates = grid_geometry.z_points[center_ix, center_iy, :]
    sample_indices = (
        0,
        len(z_coordinates) // 4,
        len(z_coordinates) // 2,
        3 * len(z_coordinates) // 4,
        len(z_coordinates) - 1,
    )
    return tuple(
        OperatorCenterlineSample(
            sample_index=index,
            z_coordinate_bohr=float(z_coordinates[index]),
            orbital_value=float(orbital[center_ix, center_iy, index]),
            residual_value=float(residual_field[center_ix, center_iy, index]),
        )
        for index in sample_indices
    )


def _component_actions(
    psi: np.ndarray,
    operator_context: FixedPotentialStaticLocalOperatorContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    field = validate_orbital_field(psi, grid_geometry=operator_context.grid_geometry, name="psi")
    if (
        operator_context.kinetic_version == "trial_fix"
        and isinstance(operator_context.grid_geometry, MonitorGridGeometry)
    ):
        kinetic_action = apply_monitor_grid_kinetic_operator_trial_boundary_fix(
            field,
            grid_geometry=operator_context.grid_geometry,
        )
    else:
        kinetic_action = apply_kinetic_operator(field, grid_geometry=operator_context.grid_geometry)
    local_action = operator_context.local_ionic_potential * field
    hartree_action = operator_context.hartree_potential * field
    xc_action = operator_context.xc_potential * field
    total_action = kinetic_action + local_action + hartree_action + xc_action
    return kinetic_action, local_action, hartree_action, xc_action, total_action


def _expectation_summary(
    psi: np.ndarray,
    operator_context: FixedPotentialStaticLocalOperatorContext,
) -> WeightedExpectationSummary:
    field = validate_orbital_field(psi, grid_geometry=operator_context.grid_geometry, name="psi")
    kinetic_action, local_action, hartree_action, xc_action, total_action = _component_actions(
        field,
        operator_context=operator_context,
    )
    denominator = _weighted_inner_product(field, field, operator_context.grid_geometry)
    denominator_real = float(np.real_if_close(denominator))
    return WeightedExpectationSummary(
        weighted_norm=float(np.sqrt(max(denominator_real, 0.0))),
        denominator=denominator_real,
        rayleigh_quotient=float(np.real_if_close(_weighted_inner_product(field, total_action, operator_context.grid_geometry) / denominator)),
        kinetic_expectation=float(np.real_if_close(_weighted_inner_product(field, kinetic_action, operator_context.grid_geometry) / denominator)),
        local_ionic_expectation=float(np.real_if_close(_weighted_inner_product(field, local_action, operator_context.grid_geometry) / denominator)),
        hartree_expectation=float(np.real_if_close(_weighted_inner_product(field, hartree_action, operator_context.grid_geometry) / denominator)),
        xc_expectation=float(np.real_if_close(_weighted_inner_product(field, xc_action, operator_context.grid_geometry) / denominator)),
    )


def _build_probe_field(
    kind: str,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    if isinstance(grid_geometry, MonitorGridGeometry):
        center = (
            0.5 * (grid_geometry.spec.box_bounds[0][0] + grid_geometry.spec.box_bounds[0][1]),
            0.5 * (grid_geometry.spec.box_bounds[1][0] + grid_geometry.spec.box_bounds[1][1]),
            0.5 * (grid_geometry.spec.box_bounds[2][0] + grid_geometry.spec.box_bounds[2][1]),
        )
    else:
        center = grid_geometry.spec.reference_center

    x_shift = grid_geometry.x_points - center[0]
    y_shift = grid_geometry.y_points - center[1]
    z_shift = grid_geometry.z_points - center[2]
    radius_squared = x_shift * x_shift + y_shift * y_shift + z_shift * z_shift
    if kind == "u":
        field = np.exp(-0.55 * radius_squared)
    elif kind == "w":
        field = (1.0 + 0.35 * z_shift) * np.exp(-0.45 * radius_squared)
    else:
        raise ValueError(f"Unsupported probe field `{kind}`.")

    norm = weighted_l2_norm(field, grid_geometry=grid_geometry)
    return np.asarray(field / norm, dtype=np.float64)


def _self_adjoint_probe(
    operator_context: FixedPotentialStaticLocalOperatorContext,
    *,
    apply_operator,
) -> SelfAdjointnessProbe:
    u = _build_probe_field("u", operator_context.grid_geometry)
    w = _build_probe_field("w", operator_context.grid_geometry)
    left = _weighted_inner_product(u, apply_operator(w), operator_context.grid_geometry)
    right = _weighted_inner_product(apply_operator(u), w, operator_context.grid_geometry)
    absolute_difference = abs(left - right)
    scale = max(abs(left), abs(right), 1.0e-30)
    return SelfAdjointnessProbe(
        absolute_difference=float(absolute_difference),
        relative_difference=float(absolute_difference / scale),
        left_inner_product_real=float(np.real_if_close(left)),
        right_inner_product_real=float(np.real_if_close(right)),
    )


def _prepare_operator_context(
    *,
    case: BenchmarkCase,
    path_type: str,
    kinetic_version: str = "production",
) -> tuple[GridGeometryLike, np.ndarray, np.ndarray, np.ndarray, FixedPotentialStaticLocalOperatorContext]:
    if path_type == "legacy":
        grid_geometry = build_default_h2_grid_geometry(case=case)
        use_monitor_patch = False
        patch_parameters = None
    elif path_type == "monitor_a_grid_plus_patch":
        grid_geometry = build_h2_local_patch_development_monitor_grid()
        use_monitor_patch = True
        patch_parameters = _default_patch_parameters()
    else:
        raise ValueError(f"Unsupported path_type `{path_type}`.")

    trial_orbital, rho_up, rho_down, rho_total = _build_frozen_density(case=case, grid_geometry=grid_geometry)
    operator_context = prepare_fixed_potential_static_local_operator(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        case=case,
        use_monitor_patch=use_monitor_patch,
        patch_parameters=patch_parameters,
        kinetic_version=kinetic_version,
    )
    return grid_geometry, trial_orbital, rho_up, rho_down, operator_context


def _evaluate_route(
    *,
    case: BenchmarkCase,
    path_type: str,
    kinetic_version: str = "production",
) -> H2StaticLocalOperatorRouteResult:
    grid_geometry, trial_orbital, rho_up, rho_down, operator_context = _prepare_operator_context(
        case=case,
        path_type=path_type,
        kinetic_version=kinetic_version,
    )
    eigensolver_result = solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=1,
        case=case,
        tolerance=1.0e-3,
        ncv=20,
        use_monitor_patch=(path_type == "monitor_a_grid_plus_patch"),
        patch_parameters=_default_patch_parameters() if path_type == "monitor_a_grid_plus_patch" else None,
        kinetic_version=kinetic_version,
    )
    eigen_orbital = eigensolver_result.orbitals[0]
    eigenvalue = float(eigensolver_result.eigenvalues[0])
    residual_field = apply_fixed_potential_static_local_operator(
        eigen_orbital,
        operator_context=operator_context,
    ) - eigenvalue * eigen_orbital

    def _apply_total(field: np.ndarray) -> np.ndarray:
        return apply_fixed_potential_static_local_operator(field, operator_context=operator_context)

    def _apply_kinetic(field: np.ndarray) -> np.ndarray:
        if kinetic_version == "trial_fix" and isinstance(grid_geometry, MonitorGridGeometry):
            return apply_monitor_grid_kinetic_operator_trial_boundary_fix(
                field,
                grid_geometry=grid_geometry,
            )
        return apply_kinetic_operator(field, grid_geometry=grid_geometry)

    def _apply_local(field: np.ndarray) -> np.ndarray:
        return operator_context.effective_local_potential * validate_orbital_field(
            field,
            grid_geometry=grid_geometry,
            name="field",
        )

    patch_summary = None
    patch_embedding_energy_mismatch = None
    patch_embedded_correction_mha = None
    if path_type == "monitor_a_grid_plus_patch":
        patch_summary = _patch_summary(_default_patch_parameters())
        embedding = operator_context.frozen_patch_local_embedding
        patch_embedding_energy_mismatch = float(embedding.embedding_energy_mismatch)
        patch_embedded_correction_mha = float(embedding.embedded_patch_correction_energy * 1000.0)

    return H2StaticLocalOperatorRouteResult(
        path_type=path_type,
        kinetic_version=kinetic_version,
        grid_parameter_summary=_grid_parameter_summary(path_type),
        patch_parameter_summary=patch_summary,
        frozen_density_integral=float(_weighted_inner_product(rho_up + rho_down, np.ones_like(rho_up), grid_geometry).real),
        trial_expectation=_expectation_summary(trial_orbital, operator_context),
        eigen_expectation=_expectation_summary(eigen_orbital, operator_context),
        eigenvalue=eigenvalue,
        weighted_residual_norm=float(weighted_l2_norm(residual_field, grid_geometry=grid_geometry)),
        residual_summary=_field_summary(residual_field),
        residual_centerline_samples=_centerline_samples(eigen_orbital, residual_field, grid_geometry),
        residual_regions=_residual_region_summaries(residual_field, grid_geometry, case),
        self_adjoint_probe_total=_self_adjoint_probe(operator_context, apply_operator=_apply_total),
        self_adjoint_probe_kinetic=_self_adjoint_probe(operator_context, apply_operator=_apply_kinetic),
        self_adjoint_probe_local_potential=_self_adjoint_probe(operator_context, apply_operator=_apply_local),
        converged=bool(eigensolver_result.converged),
        patch_embedding_energy_mismatch=patch_embedding_energy_mismatch,
        patch_embedded_correction_mha=patch_embedded_correction_mha,
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


def _diagnosis(
    legacy_result: H2StaticLocalOperatorRouteResult,
    monitor_patch_production_result: H2StaticLocalOperatorRouteResult,
    monitor_patch_trial_fix_result: H2StaticLocalOperatorRouteResult,
) -> str:
    residual_drop = (
        monitor_patch_production_result.weighted_residual_norm
        - monitor_patch_trial_fix_result.weighted_residual_norm
    )
    eigenvalue_lift = (
        monitor_patch_trial_fix_result.eigenvalue
        - monitor_patch_production_result.eigenvalue
    )
    if (
        abs(monitor_patch_trial_fix_result.self_adjoint_probe_total.relative_difference) < 1.0e-10
        and abs(monitor_patch_trial_fix_result.self_adjoint_probe_kinetic.relative_difference) < 1.0e-10
        and residual_drop > 0.1
        and eigenvalue_lift > 0.1
    ):
        return (
            "The kinetic trial-fix materially improves the A-grid static-local operator: "
            "the bad k=1 eigenvalue moves upward and the weighted residual drops. The strongest "
            "change remains in the kinetic expectation on the eigensolver-selected orbital, "
            "which supports the earlier boundary/ghost-closure diagnosis."
        )
    if (
        abs(monitor_patch_trial_fix_result.self_adjoint_probe_total.relative_difference) < 1.0e-10
        and abs(monitor_patch_trial_fix_result.self_adjoint_probe_kinetic.relative_difference) < 1.0e-10
    ):
        return (
            "The trial-fix branch preserves weighted self-adjointness but has not yet delivered "
            "a decisive operator-level recovery. The A-grid static-local failure still appears "
            "upstream of SCF and centered on the kinetic path."
        )
    return (
        "The operator-level audit remains inconclusive at the diagnosis level, but the current "
        "A-grid failure is clearly upstream of SCF and persists even before nonlocal terms enter."
    )


def run_h2_monitor_grid_operator_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridOperatorAuditResult:
    """Run the H2 static-local operator-level audit on legacy and A-grid routes."""

    legacy_result = _evaluate_route(case=case, path_type="legacy")
    monitor_patch_production_result = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="production",
    )
    monitor_patch_trial_fix_result = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
    )
    return H2MonitorGridOperatorAuditResult(
        legacy_result=legacy_result,
        monitor_patch_production_result=monitor_patch_production_result,
        monitor_patch_trial_fix_result=monitor_patch_trial_fix_result,
        diagnosis=_diagnosis(
            legacy_result,
            monitor_patch_production_result,
            monitor_patch_trial_fix_result,
        ),
        note=(
            "This is an operator-level audit only. It keeps the H2 singlet frozen density fixed "
            "and inspects the static-local operator T + V_loc,ion + V_H + V_xc on legacy and "
            "A-grid+patch routes with production versus kinetic-trial-fix branches. Nonlocal "
            "and SCF are still absent."
        ),
    )


def _print_expectation(label: str, summary: WeightedExpectationSummary) -> None:
    print(f"  {label}:")
    print(f"    weighted norm: {summary.weighted_norm:.12f}")
    print(f"    Rayleigh [Ha]: {summary.rayleigh_quotient:.12f}")
    print(f"    <T> [Ha]: {summary.kinetic_expectation:.12f}")
    print(f"    <V_loc> [Ha]: {summary.local_ionic_expectation:.12f}")
    print(f"    <V_H> [Ha]: {summary.hartree_expectation:.12f}")
    print(f"    <V_xc> [Ha]: {summary.xc_expectation:.12f}")


def _print_route(result: H2StaticLocalOperatorRouteResult) -> None:
    print(f"path: {result.path_type}")
    print(f"  kinetic version: {result.kinetic_version}")
    print(f"  grid summary: {result.grid_parameter_summary}")
    print(f"  frozen density integral: {result.frozen_density_integral:.12f}")
    print(f"  converged: {result.converged}")
    print(f"  eigenvalue [Ha]: {result.eigenvalue:.12f}")
    print(f"  weighted residual norm: {result.weighted_residual_norm:.12e}")
    print(
        "  self-adjoint probe total: "
        f"abs={result.self_adjoint_probe_total.absolute_difference:.3e}, "
        f"rel={result.self_adjoint_probe_total.relative_difference:.3e}"
    )
    print(
        "  self-adjoint probe kinetic: "
        f"abs={result.self_adjoint_probe_kinetic.absolute_difference:.3e}, "
        f"rel={result.self_adjoint_probe_kinetic.relative_difference:.3e}"
    )
    print(
        "  self-adjoint probe local potential: "
        f"abs={result.self_adjoint_probe_local_potential.absolute_difference:.3e}, "
        f"rel={result.self_adjoint_probe_local_potential.relative_difference:.3e}"
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
        print(
            "  frozen patch embedding: "
            f"embedded correction={result.patch_embedded_correction_mha:+.3f} mHa, "
            f"mismatch={result.patch_embedding_energy_mismatch:+.3e} Ha"
        )
    _print_expectation("trial orbital", result.trial_expectation)
    _print_expectation("eigensolver orbital", result.eigen_expectation)
    print("  residual regions:")
    for region in result.residual_regions:
        print(
            "    "
            f"{region.region_name}: frac={region.point_fraction:.4f}, "
            f"rms={region.residual_summary.rms:.6e}, "
            f"weighted_rms={region.weighted_rms:.6e}, "
            f"mean={region.mean_signed_residual:+.6e}"
        )
    print("  residual center-line samples:")
    for sample in result.residual_centerline_samples:
        print(
            "    "
            f"z[{sample.sample_index:02d}]={sample.z_coordinate_bohr:+.6f} Bohr -> "
            f"psi={sample.orbital_value:+.12f}, "
            f"res={sample.residual_value:+.12e}"
        )


def print_h2_monitor_grid_operator_audit_summary(
    result: H2MonitorGridOperatorAuditResult,
) -> None:
    """Print the compact H2 static-local operator audit summary."""

    print("IsoGridDFT H2 static-local operator audit")
    print(f"note: {result.note}")
    print(
        "reference failure baseline: "
        f"legacy k=1={H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE.legacy_route.eigenvalue_ha:+.12f} Ha, "
        f"A-grid+patch production k=1={H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE.monitor_patch_route.eigenvalue_ha:+.12f} Ha"
    )
    print()
    _print_route(result.legacy_result)
    print()
    _print_route(result.monitor_patch_production_result)
    print()
    _print_route(result.monitor_patch_trial_fix_result)
    print()
    print("trial-fix delta vs production:")
    print(
        "  eigenvalue [Ha]: "
        f"{result.monitor_patch_trial_fix_result.eigenvalue - result.monitor_patch_production_result.eigenvalue:+.12f}"
    )
    print(
        "  weighted residual norm: "
        f"{result.monitor_patch_trial_fix_result.weighted_residual_norm - result.monitor_patch_production_result.weighted_residual_norm:+.12e}"
    )
    print(
        "  eigen <T> delta [Ha]: "
        f"{result.monitor_patch_trial_fix_result.eigen_expectation.kinetic_expectation - result.monitor_patch_production_result.eigen_expectation.kinetic_expectation:+.12f}"
    )
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_operator_audit()
    print_h2_monitor_grid_operator_audit_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
