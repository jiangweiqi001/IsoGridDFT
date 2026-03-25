r"""Discrete Green-identity audit for the H2 monitor-grid kinetic path.

This audit checks a trial-fix branch for the monitor-grid kinetic path. The
production path is left intact; the audit only swaps the A-grid boundary/ghost
handling to a centered zero-ghost closure and then rechecks the logical-cube
Green identity

    T psi = -1/2 * (1/J) d_a [ F^a ]
    F^a = J g^{ab} d_b psi

For a real field psi, the continuous identity is

    K_op   = <psi, T psi>
    K_grad = 1/2 \int (d_a psi) F^a d\xi
    K_bdry = -1/2 \oint psi F^a n_a dS_\xi
    K_op = K_grad + K_bdry

The audit computes the operator energy, a gradient-form reference energy, and
an explicit boundary-flux proxy on the same A-grid geometry in order to check
whether the current bad eigensolver orbital violates that identity mainly
through a boundary-term mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.grid import MonitorGridGeometry
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ops import validate_orbital_field
from isogrid.ops.kinetic import apply_monitor_grid_kinetic_operator_trial_boundary_fix
from isogrid.ops.kinetic import compute_monitor_grid_contravariant_flux_components
from isogrid.pseudo import LocalPotentialPatchParameters

from .baselines import H2_GEOMETRY_CONSISTENCY_AUDIT_BASELINE
from .baselines import H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE
from .h2_monitor_grid_kinetic_operator_audit import _MONITOR_FINER_SHAPE
from .h2_monitor_grid_kinetic_operator_audit import _build_monitor_grid
from .h2_monitor_grid_kinetic_operator_audit import _build_smooth_field
from .h2_monitor_grid_operator_audit import ScalarFieldSummary
from .h2_monitor_grid_operator_audit import _build_frozen_density
from .h2_monitor_grid_operator_audit import _compute_region_masks
from .h2_monitor_grid_operator_audit import _field_summary
from .h2_monitor_grid_operator_audit import _weighted_inner_product


@dataclass(frozen=True)
class BoundaryFaceContribution:
    """One face contribution to the Green-identity boundary term."""

    face_label: str
    contribution_ha: float


@dataclass(frozen=True)
class GreenIdentityCenterlineSample:
    """Center-line sample for operator/gradient/boundary identity diagnostics."""

    sample_index: int
    z_coordinate_bohr: float
    orbital_value: float
    operator_indicator: float
    gradient_indicator: float
    boundary_indicator: float
    closure_indicator: float


@dataclass(frozen=True)
class GreenIdentityRegionSummary:
    """Regionwise Green-identity diagnostics for one field."""

    region_name: str
    point_fraction: float
    operator_indicator_summary: ScalarFieldSummary
    gradient_indicator_summary: ScalarFieldSummary
    boundary_indicator_summary: ScalarFieldSummary
    closure_indicator_summary: ScalarFieldSummary
    operator_weighted_contribution_ha: float
    gradient_weighted_contribution_ha: float
    boundary_weighted_contribution_ha: float
    delta_weighted_contribution_mha: float
    closure_weighted_contribution_mha: float


@dataclass(frozen=True)
class GreenIdentityFieldResult:
    """Discrete Green-identity diagnostics for one field."""

    shape_label: str
    field_label: str
    weighted_norm: float
    operator_kinetic_ha: float
    gradient_kinetic_ha: float
    delta_kinetic_mha: float
    boundary_term_ha: float
    closure_mismatch_mha: float
    operator_indicator_summary: ScalarFieldSummary
    gradient_indicator_summary: ScalarFieldSummary
    boundary_indicator_summary: ScalarFieldSummary
    closure_indicator_summary: ScalarFieldSummary
    face_contributions: tuple[BoundaryFaceContribution, ...]
    region_summaries: tuple[GreenIdentityRegionSummary, ...]
    centerline_samples: tuple[GreenIdentityCenterlineSample, ...]
    source_eigenvalue_ha: float | None
    source_residual_norm: float | None
    source_converged: bool | None


@dataclass(frozen=True)
class H2MonitorGridKineticGreenIdentityAuditResult:
    """Top-level H2 discrete Green-identity audit result."""

    frozen_trial_baseline: GreenIdentityFieldResult
    bad_eigen_baseline: GreenIdentityFieldResult
    bad_eigen_finer_shape: GreenIdentityFieldResult
    smooth_field_results: tuple[GreenIdentityFieldResult, ...]
    diagnosis: str
    note: str


def _default_patch_parameters() -> LocalPotentialPatchParameters:
    return LocalPotentialPatchParameters(
        patch_radius_scale=0.75,
        patch_grid_shape=(25, 25, 25),
        correction_strength=1.30,
        interpolation_neighbors=8,
    )


def _logical_spacing(axis: np.ndarray) -> float:
    coordinates = np.asarray(axis, dtype=np.float64)
    spacing = np.diff(coordinates)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("Green-identity audit expects uniform logical coordinates.")
    return float(spacing[0])


def _flux_components(
    orbital: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name="orbital")
    flux_x, flux_y, flux_z = compute_monitor_grid_contravariant_flux_components(
        field,
        grid_geometry,
        use_trial_boundary_fix=True,
    )
    return flux_x, flux_y, flux_z


def _gradient_reference_density(
    orbital: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> np.ndarray:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name="orbital")
    gradients = np.gradient(
        field,
        grid_geometry.logical_x,
        grid_geometry.logical_y,
        grid_geometry.logical_z,
        edge_order=2,
    )
    inverse_metric = np.asarray(grid_geometry.inverse_metric_tensor, dtype=np.float64)
    density = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    for axis_a in range(3):
        for axis_b in range(3):
            density += np.real(
                np.conjugate(gradients[axis_a])
                * inverse_metric[..., axis_a, axis_b]
                * gradients[axis_b]
            )
    return 0.5 * density


def _boundary_term_proxy(
    orbital: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, tuple[BoundaryFaceContribution, ...], float]:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name="orbital")
    flux_x, flux_y, flux_z = _flux_components(field, grid_geometry)
    dxi = _logical_spacing(grid_geometry.logical_x)
    deta = _logical_spacing(grid_geometry.logical_y)
    dzeta = _logical_spacing(grid_geometry.logical_z)
    area_yz = deta * dzeta
    area_xz = dxi * dzeta
    area_xy = dxi * deta
    boundary_density = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    face_contributions: list[BoundaryFaceContribution] = []

    face_specs = (
        ("x_min", (0, slice(1, -1), slice(1, -1)), flux_x, +0.5, area_yz),
        ("x_max", (-1, slice(1, -1), slice(1, -1)), flux_x, -0.5, area_yz),
        ("y_min", (slice(1, -1), 0, slice(1, -1)), flux_y, +0.5, area_xz),
        ("y_max", (slice(1, -1), -1, slice(1, -1)), flux_y, -0.5, area_xz),
        ("z_min", (slice(1, -1), slice(1, -1), 0), flux_z, +0.5, area_xy),
        ("z_max", (slice(1, -1), slice(1, -1), -1), flux_z, -0.5, area_xy),
    )

    for face_label, face_slice, flux_component, sign, area_element in face_specs:
        contribution_density = sign * np.real(
            np.conjugate(field[face_slice]) * flux_component[face_slice]
        )
        contribution_ha = float(np.sum(contribution_density) * area_element)
        cell_volume = np.asarray(grid_geometry.cell_volumes[face_slice], dtype=np.float64)
        boundary_density[face_slice] += contribution_density * area_element / cell_volume
        face_contributions.append(
            BoundaryFaceContribution(face_label=face_label, contribution_ha=contribution_ha)
        )

    total_boundary_term = float(
        np.sum(boundary_density * grid_geometry.cell_volumes, dtype=np.float64)
    )
    return boundary_density, tuple(face_contributions), total_boundary_term


def _centerline_samples(
    orbital: np.ndarray,
    operator_indicator: np.ndarray,
    gradient_indicator: np.ndarray,
    boundary_indicator: np.ndarray,
    closure_indicator: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[GreenIdentityCenterlineSample, ...]:
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
        GreenIdentityCenterlineSample(
            sample_index=index,
            z_coordinate_bohr=float(z_coordinates[index]),
            orbital_value=float(orbital[center_ix, center_iy, index]),
            operator_indicator=float(operator_indicator[center_ix, center_iy, index]),
            gradient_indicator=float(gradient_indicator[center_ix, center_iy, index]),
            boundary_indicator=float(boundary_indicator[center_ix, center_iy, index]),
            closure_indicator=float(closure_indicator[center_ix, center_iy, index]),
        )
        for index in sample_indices
    )


def _region_summaries(
    *,
    operator_indicator: np.ndarray,
    gradient_indicator: np.ndarray,
    boundary_indicator: np.ndarray,
    closure_indicator: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    case: BenchmarkCase,
    denominator: float,
) -> tuple[GreenIdentityRegionSummary, ...]:
    summaries: list[GreenIdentityRegionSummary] = []
    total_points = float(np.prod(grid_geometry.spec.shape))
    for region_name, mask in _compute_region_masks(case=case, grid_geometry=grid_geometry):
        if not np.any(mask):
            continue
        weights = grid_geometry.cell_volumes[mask]
        op_values = operator_indicator[mask]
        grad_values = gradient_indicator[mask]
        bdry_values = boundary_indicator[mask]
        closure_values = closure_indicator[mask]
        summaries.append(
            GreenIdentityRegionSummary(
                region_name=region_name,
                point_fraction=float(np.sum(mask) / total_points),
                operator_indicator_summary=_field_summary(op_values),
                gradient_indicator_summary=_field_summary(grad_values),
                boundary_indicator_summary=_field_summary(bdry_values),
                closure_indicator_summary=_field_summary(closure_values),
                operator_weighted_contribution_ha=float(np.sum(op_values * weights) / denominator),
                gradient_weighted_contribution_ha=float(np.sum(grad_values * weights) / denominator),
                boundary_weighted_contribution_ha=float(np.sum(bdry_values * weights) / denominator),
                delta_weighted_contribution_mha=float(
                    1000.0 * np.sum((op_values - grad_values) * weights) / denominator
                ),
                closure_weighted_contribution_mha=float(
                    1000.0 * np.sum(closure_values * weights) / denominator
                ),
            )
        )
    return tuple(summaries)


def _evaluate_field(
    *,
    orbital: np.ndarray,
    field_label: str,
    shape_label: str,
    grid_geometry: MonitorGridGeometry,
    case: BenchmarkCase,
    source_eigenvalue_ha: float | None,
    source_residual_norm: float | None,
    source_converged: bool | None,
) -> GreenIdentityFieldResult:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name=field_label)
    kinetic_action = apply_monitor_grid_kinetic_operator_trial_boundary_fix(
        field,
        grid_geometry=grid_geometry,
    )
    operator_indicator = np.real(np.conjugate(field) * kinetic_action)
    gradient_indicator = _gradient_reference_density(field, grid_geometry)
    boundary_indicator, face_contributions, boundary_term = _boundary_term_proxy(field, grid_geometry)
    closure_indicator = operator_indicator - gradient_indicator - boundary_indicator
    denominator = float(np.real_if_close(_weighted_inner_product(field, field, grid_geometry)))
    operator_kinetic = float(
        np.real_if_close(_weighted_inner_product(field, kinetic_action, grid_geometry) / denominator)
    )
    gradient_kinetic = float(np.sum(gradient_indicator * grid_geometry.cell_volumes) / denominator)
    return GreenIdentityFieldResult(
        shape_label=shape_label,
        field_label=field_label,
        weighted_norm=float(np.sqrt(max(denominator, 0.0))),
        operator_kinetic_ha=operator_kinetic,
        gradient_kinetic_ha=gradient_kinetic,
        delta_kinetic_mha=float((operator_kinetic - gradient_kinetic) * 1000.0),
        boundary_term_ha=boundary_term / denominator,
        closure_mismatch_mha=float(
            1000.0 * np.sum(closure_indicator * grid_geometry.cell_volumes) / denominator
        ),
        operator_indicator_summary=_field_summary(operator_indicator),
        gradient_indicator_summary=_field_summary(gradient_indicator),
        boundary_indicator_summary=_field_summary(boundary_indicator),
        closure_indicator_summary=_field_summary(closure_indicator),
        face_contributions=face_contributions,
        region_summaries=_region_summaries(
            operator_indicator=operator_indicator,
            gradient_indicator=gradient_indicator,
            boundary_indicator=boundary_indicator,
            closure_indicator=closure_indicator,
            grid_geometry=grid_geometry,
            case=case,
            denominator=denominator,
        ),
        centerline_samples=_centerline_samples(
            field,
            operator_indicator,
            gradient_indicator,
            boundary_indicator,
            closure_indicator,
            grid_geometry,
        ),
        source_eigenvalue_ha=source_eigenvalue_ha,
        source_residual_norm=source_residual_norm,
        source_converged=source_converged,
    )


def _evaluate_bad_eigen_orbital(
    *,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, float, float, bool]:
    _, rho_up, rho_down, _ = _build_frozen_density(case=case, grid_geometry=grid_geometry)
    result = solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=1,
        case=case,
        tolerance=1.0e-3,
        ncv=20,
        use_monitor_patch=True,
        patch_parameters=_default_patch_parameters(),
    )
    return (
        result.orbitals[0],
        float(result.eigenvalues[0]),
        float(result.residual_norms[0]),
        bool(result.converged),
    )


def _diagnosis(result: H2MonitorGridKineticGreenIdentityAuditResult) -> str:
    frozen = result.frozen_trial_baseline
    bad = result.bad_eigen_baseline
    finer = result.bad_eigen_finer_shape
    smooth_small = all(abs(item.closure_mismatch_mha) < 10.0 for item in result.smooth_field_results)
    bad_far = next(region for region in bad.region_summaries if region.region_name == "far_field")
    if abs(frozen.closure_mismatch_mha) < 1.0e-6 and smooth_small:
        if (
            abs(bad_far.delta_weighted_contribution_mha) > 1_000.0
            and abs(bad_far.boundary_weighted_contribution_ha) > 0.5
            and abs(bad_far.closure_weighted_contribution_mha) > 1_000.0
        ):
            return (
                "The trial boundary-fix branch materially reduces the bad-orbital Green-identity "
                "failure while leaving the frozen and smooth fields well behaved. The remaining "
                "mismatch is still far-field dominated, so the boundary/ghost diagnosis survives, "
                "but the prototype closure change is hitting the right symptom."
            )
    return (
        "The trial boundary-fix branch changes the Green-identity balance, but the bad-orbital "
        "mismatch is not yet collapsed enough to call the issue solved."
    )


def run_h2_monitor_grid_kinetic_green_identity_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridKineticGreenIdentityAuditResult:
    """Run the H2 monitor-grid discrete Green-identity audit."""

    baseline_grid = build_h2_local_patch_development_monitor_grid()
    finer_grid = _build_monitor_grid(_MONITOR_FINER_SHAPE)

    frozen_trial_orbital, _, _, _ = _build_frozen_density(case=case, grid_geometry=baseline_grid)
    bad_orbital_baseline, eigenvalue_baseline, residual_baseline, converged_baseline = (
        _evaluate_bad_eigen_orbital(case=case, grid_geometry=baseline_grid)
    )
    bad_orbital_finer, eigenvalue_finer, residual_finer, converged_finer = (
        _evaluate_bad_eigen_orbital(case=case, grid_geometry=finer_grid)
    )

    frozen_trial_baseline = _evaluate_field(
        orbital=frozen_trial_orbital,
        field_label="frozen_trial_orbital",
        shape_label="baseline",
        grid_geometry=baseline_grid,
        case=case,
        source_eigenvalue_ha=None,
        source_residual_norm=None,
        source_converged=None,
    )
    bad_eigen_baseline = _evaluate_field(
        orbital=bad_orbital_baseline,
        field_label="bad_eigensolver_orbital_k1",
        shape_label="baseline",
        grid_geometry=baseline_grid,
        case=case,
        source_eigenvalue_ha=eigenvalue_baseline,
        source_residual_norm=residual_baseline,
        source_converged=converged_baseline,
    )
    bad_eigen_finer_shape = _evaluate_field(
        orbital=bad_orbital_finer,
        field_label="bad_eigensolver_orbital_k1",
        shape_label="finer-shape",
        grid_geometry=finer_grid,
        case=case,
        source_eigenvalue_ha=eigenvalue_finer,
        source_residual_norm=residual_finer,
        source_converged=converged_finer,
    )
    smooth_field_results = (
        _evaluate_field(
            orbital=_build_smooth_field("gaussian", baseline_grid),
            field_label="smooth_gaussian",
            shape_label="baseline",
            grid_geometry=baseline_grid,
            case=case,
            source_eigenvalue_ha=None,
            source_residual_norm=None,
            source_converged=None,
        ),
        _evaluate_field(
            orbital=_build_smooth_field("cosine", baseline_grid),
            field_label="smooth_cosine",
            shape_label="baseline",
            grid_geometry=baseline_grid,
            case=case,
            source_eigenvalue_ha=None,
            source_residual_norm=None,
            source_converged=None,
        ),
    )
    result = H2MonitorGridKineticGreenIdentityAuditResult(
        frozen_trial_baseline=frozen_trial_baseline,
        bad_eigen_baseline=bad_eigen_baseline,
        bad_eigen_finer_shape=bad_eigen_finer_shape,
        smooth_field_results=smooth_field_results,
        diagnosis="",
        note=(
            "This audit keeps the H2 singlet frozen density fixed and checks the discrete "
            "Green identity K_op = K_grad + K_bdry on the monitor-grid kinetic trial-fix path. "
            "The production kinetic implementation remains available; this audit only exercises "
            "the centered zero-ghost boundary-closure prototype."
        ),
    )
    return H2MonitorGridKineticGreenIdentityAuditResult(
        frozen_trial_baseline=result.frozen_trial_baseline,
        bad_eigen_baseline=result.bad_eigen_baseline,
        bad_eigen_finer_shape=result.bad_eigen_finer_shape,
        smooth_field_results=result.smooth_field_results,
        diagnosis=_diagnosis(result),
        note=result.note,
    )


def _print_field_result(result: GreenIdentityFieldResult) -> None:
    print(f"field: {result.field_label} ({result.shape_label})")
    print(f"  weighted norm: {result.weighted_norm:.12f}")
    print(
        f"  K_op [Ha]: {result.operator_kinetic_ha:+.12f} | "
        f"K_grad [Ha]: {result.gradient_kinetic_ha:+.12f} | "
        f"Delta K [mHa]: {result.delta_kinetic_mha:+.3f}"
    )
    print(
        f"  K_bdry [Ha]: {result.boundary_term_ha:+.12f} | "
        f"closure mismatch [mHa]: {result.closure_mismatch_mha:+.3f}"
    )
    if result.source_eigenvalue_ha is not None:
        print(
            f"  eig info: eps={result.source_eigenvalue_ha:+.12f} Ha, "
            f"res={result.source_residual_norm:.6e}, converged={result.source_converged}"
        )
    print("  face contributions [Ha]:")
    for face in result.face_contributions:
        print(f"    {face.face_label}: {face.contribution_ha:+.12f}")
    print("  region diagnostics:")
    for region in result.region_summaries:
        print(
            "    "
            f"{region.region_name}: Delta K={region.delta_weighted_contribution_mha:+.3f} mHa, "
            f"K_bdry={region.boundary_weighted_contribution_ha:+.12f} Ha, "
            f"closure={region.closure_weighted_contribution_mha:+.3f} mHa"
        )
    print("  center-line samples:")
    for sample in result.centerline_samples:
        print(
            "    "
            f"z[{sample.sample_index:02d}]={sample.z_coordinate_bohr:+.6f} -> "
            f"psi={sample.orbital_value:+.6e}, "
            f"op={sample.operator_indicator:+.6e}, "
            f"grad={sample.gradient_indicator:+.6e}, "
            f"bdry={sample.boundary_indicator:+.6e}, "
            f"closure={sample.closure_indicator:+.6e}"
        )


def print_h2_monitor_grid_kinetic_green_identity_audit_summary(
    result: H2MonitorGridKineticGreenIdentityAuditResult,
) -> None:
    """Print the compact H2 monitor-grid Green-identity audit summary."""

    print("IsoGridDFT H2 kinetic Green-identity audit")
    print(f"note: {result.note}")
    print(f"geometry-consistency baseline diagnosis: {H2_GEOMETRY_CONSISTENCY_AUDIT_BASELINE.diagnosis}")
    print(
        "pre-fix bad-eigen baseline: "
        f"Delta K={H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE.bad_eigen_baseline.delta_kinetic_mha:+.3f} mHa, "
        f"K_bdry={H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE.bad_eigen_baseline.boundary_term_ha:+.12f} Ha, "
        f"closure={H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE.bad_eigen_baseline.closure_mismatch_mha:+.3f} mHa"
    )
    print()
    _print_field_result(result.frozen_trial_baseline)
    print()
    _print_field_result(result.bad_eigen_baseline)
    print()
    _print_field_result(result.bad_eigen_finer_shape)
    print()
    print("smooth probes:")
    for field in result.smooth_field_results:
        print(
            "  "
            f"{field.field_label}: Delta K={field.delta_kinetic_mha:+.6f} mHa, "
            f"K_bdry={field.boundary_term_ha:+.12f} Ha, "
            f"closure={field.closure_mismatch_mha:+.6f} mHa"
        )
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_kinetic_green_identity_audit()
    print_h2_monitor_grid_kinetic_green_identity_audit_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
