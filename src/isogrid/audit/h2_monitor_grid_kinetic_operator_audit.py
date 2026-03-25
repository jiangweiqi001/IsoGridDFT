"""Kinetic-operator audit for the H2 frozen-density A-grid failure mode.

This audit isolates only the kinetic sub-operator

    T = -1/2 nabla^2

on the legacy grid, the raw A-grid, and the A-grid+patch fixed-potential path.
Patch does not directly modify kinetic; it only changes which orbital the
fixed-potential eigensolver selects on the frozen static-local operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ops import apply_kinetic_operator
from isogrid.ops import validate_orbital_field
from isogrid.ops import weighted_l2_norm
from isogrid.pseudo import LocalPotentialPatchParameters

from .h2_monitor_grid_operator_audit import ScalarFieldSummary
from .h2_monitor_grid_operator_audit import SelfAdjointnessProbe
from .h2_monitor_grid_operator_audit import _build_frozen_density
from .h2_monitor_grid_operator_audit import _build_probe_field
from .h2_monitor_grid_operator_audit import _compute_region_masks
from .h2_monitor_grid_operator_audit import _default_patch_parameters
from .h2_monitor_grid_operator_audit import _field_summary
from .h2_monitor_grid_operator_audit import _grid_parameter_summary
from .h2_monitor_grid_operator_audit import _weighted_inner_product
from .h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary
from .h2_monitor_grid_patch_local_audit import _patch_summary

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry
_MONITOR_FINER_SHAPE = (75, 75, 91)


@dataclass(frozen=True)
class KineticCenterlineSample:
    """One center-line kinetic sample for one orbital."""

    sample_index: int
    z_coordinate_bohr: float
    orbital_value: float
    kinetic_action_value: float
    local_kinetic_indicator: float


@dataclass(frozen=True)
class KineticRegionSummary:
    """Regionwise kinetic diagnostics for one orbital."""

    region_name: str
    point_fraction: float
    psi_abs_mean: float
    kinetic_action_summary: ScalarFieldSummary
    local_indicator_mean: float
    weighted_indicator_contribution_ha: float
    negative_indicator_fraction: float


@dataclass(frozen=True)
class KineticOrbitalSummary:
    """Kinetic diagnostics for one orbital on one route."""

    orbital_label: str
    weighted_norm: float
    kinetic_rayleigh_quotient: float
    kinetic_action_summary: ScalarFieldSummary
    local_indicator_summary: ScalarFieldSummary
    negative_indicator_fraction: float
    centerline_samples: tuple[KineticCenterlineSample, ...]
    region_summaries: tuple[KineticRegionSummary, ...]


@dataclass(frozen=True)
class KineticRouteAuditResult:
    """Kinetic audit result for one route and one monitor shape label."""

    path_type: str
    shape_label: str
    grid_parameter_summary: str
    patch_parameter_summary: H2MonitorPatchParameterSummary | None
    frozen_density_integral: float
    frozen_orbital_summary: KineticOrbitalSummary
    eigen_orbital_summary: KineticOrbitalSummary
    eigenvalue_ha: float
    eigensolver_converged: bool
    eigensolver_weighted_residual_norm: float
    self_adjoint_probe: SelfAdjointnessProbe


@dataclass(frozen=True)
class SmoothFieldKineticResult:
    """Kinetic audit result for one simple smooth test field."""

    path_type: str
    field_label: str
    kinetic_rayleigh_quotient: float
    weighted_norm: float
    kinetic_action_summary: ScalarFieldSummary
    negative_indicator_fraction: float


@dataclass(frozen=True)
class H2MonitorGridKineticOperatorAuditResult:
    """Top-level H2 kinetic-operator audit result."""

    legacy_result: KineticRouteAuditResult
    monitor_unpatched_baseline_result: KineticRouteAuditResult
    monitor_patch_baseline_result: KineticRouteAuditResult
    monitor_patch_finer_shape_result: KineticRouteAuditResult
    smooth_field_results: tuple[SmoothFieldKineticResult, ...]
    diagnosis: str
    note: str


def _build_monitor_grid(shape: tuple[int, int, int]) -> MonitorGridGeometry:
    return build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=shape,
        box_half_extents=H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def _centerline_samples(
    orbital: np.ndarray,
    kinetic_action: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> tuple[KineticCenterlineSample, ...]:
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    z_coordinates = grid_geometry.z_points[center_ix, center_iy, :]
    local_indicator = np.real(np.conjugate(orbital) * kinetic_action)
    sample_indices = (
        0,
        len(z_coordinates) // 4,
        len(z_coordinates) // 2,
        3 * len(z_coordinates) // 4,
        len(z_coordinates) - 1,
    )
    return tuple(
        KineticCenterlineSample(
            sample_index=index,
            z_coordinate_bohr=float(z_coordinates[index]),
            orbital_value=float(orbital[center_ix, center_iy, index]),
            kinetic_action_value=float(kinetic_action[center_ix, center_iy, index]),
            local_kinetic_indicator=float(local_indicator[center_ix, center_iy, index]),
        )
        for index in sample_indices
    )


def _kinetic_region_summaries(
    orbital: np.ndarray,
    kinetic_action: np.ndarray,
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase,
) -> tuple[KineticRegionSummary, ...]:
    denominator = float(
        np.real_if_close(_weighted_inner_product(orbital, orbital, grid_geometry))
    )
    local_indicator = np.real(np.conjugate(orbital) * kinetic_action)
    summaries: list[KineticRegionSummary] = []
    total_points = float(np.prod(grid_geometry.spec.shape))
    for name, mask in _compute_region_masks(case=case, grid_geometry=grid_geometry):
        if not np.any(mask):
            continue
        weights = grid_geometry.cell_volumes[mask]
        psi_values = np.asarray(orbital, dtype=np.float64)[mask]
        indicator_values = local_indicator[mask]
        kinetic_values = np.asarray(kinetic_action, dtype=np.float64)[mask]
        summaries.append(
            KineticRegionSummary(
                region_name=name,
                point_fraction=float(np.sum(mask) / total_points),
                psi_abs_mean=float(np.mean(np.abs(psi_values))),
                kinetic_action_summary=_field_summary(kinetic_values),
                local_indicator_mean=float(np.mean(indicator_values)),
                weighted_indicator_contribution_ha=float(
                    np.sum(indicator_values * weights) / denominator
                ),
                negative_indicator_fraction=float(np.mean(indicator_values < 0.0)),
            )
        )
    return tuple(summaries)


def _summarize_kinetic_orbital(
    orbital: np.ndarray,
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase,
    orbital_label: str,
) -> KineticOrbitalSummary:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name=orbital_label)
    kinetic_action = apply_kinetic_operator(field, grid_geometry=grid_geometry)
    denominator = _weighted_inner_product(field, field, grid_geometry)
    denominator_real = float(np.real_if_close(denominator))
    local_indicator = np.real(np.conjugate(field) * kinetic_action)
    return KineticOrbitalSummary(
        orbital_label=orbital_label,
        weighted_norm=float(np.sqrt(max(denominator_real, 0.0))),
        kinetic_rayleigh_quotient=float(
            np.real_if_close(
                _weighted_inner_product(field, kinetic_action, grid_geometry) / denominator
            )
        ),
        kinetic_action_summary=_field_summary(kinetic_action),
        local_indicator_summary=_field_summary(local_indicator),
        negative_indicator_fraction=float(np.mean(local_indicator < 0.0)),
        centerline_samples=_centerline_samples(field, kinetic_action, grid_geometry),
        region_summaries=_kinetic_region_summaries(field, kinetic_action, grid_geometry, case),
    )


def _self_adjoint_kinetic_probe(grid_geometry: GridGeometryLike) -> SelfAdjointnessProbe:
    u = _build_probe_field("u", grid_geometry)
    w = _build_probe_field("w", grid_geometry)
    kinetic_u = apply_kinetic_operator(u, grid_geometry=grid_geometry)
    kinetic_w = apply_kinetic_operator(w, grid_geometry=grid_geometry)
    left = _weighted_inner_product(u, kinetic_w, grid_geometry)
    right = _weighted_inner_product(kinetic_u, w, grid_geometry)
    absolute_difference = abs(left - right)
    scale = max(abs(left), abs(right), 1.0e-30)
    return SelfAdjointnessProbe(
        absolute_difference=float(absolute_difference),
        relative_difference=float(absolute_difference / scale),
        left_inner_product_real=float(np.real_if_close(left)),
        right_inner_product_real=float(np.real_if_close(right)),
    )


def _build_smooth_field(
    field_label: str,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    if isinstance(grid_geometry, MonitorGridGeometry):
        center = (
            0.5 * (grid_geometry.spec.box_bounds[0][0] + grid_geometry.spec.box_bounds[0][1]),
            0.5 * (grid_geometry.spec.box_bounds[1][0] + grid_geometry.spec.box_bounds[1][1]),
            0.5 * (grid_geometry.spec.box_bounds[2][0] + grid_geometry.spec.box_bounds[2][1]),
        )
        half_extents = (
            0.5 * (grid_geometry.spec.box_bounds[0][1] - grid_geometry.spec.box_bounds[0][0]),
            0.5 * (grid_geometry.spec.box_bounds[1][1] - grid_geometry.spec.box_bounds[1][0]),
            0.5 * (grid_geometry.spec.box_bounds[2][1] - grid_geometry.spec.box_bounds[2][0]),
        )
    else:
        center = grid_geometry.spec.reference_center
        half_extents = (
            0.5 * (float(np.max(grid_geometry.x_points)) - float(np.min(grid_geometry.x_points))),
            0.5 * (float(np.max(grid_geometry.y_points)) - float(np.min(grid_geometry.y_points))),
            0.5 * (float(np.max(grid_geometry.z_points)) - float(np.min(grid_geometry.z_points))),
        )

    x_shift = grid_geometry.x_points - center[0]
    y_shift = grid_geometry.y_points - center[1]
    z_shift = grid_geometry.z_points - center[2]
    radius_squared = x_shift * x_shift + y_shift * y_shift + z_shift * z_shift
    if field_label == "gaussian":
        field = np.exp(-0.45 * radius_squared)
    elif field_label == "cosine":
        field = (
            np.cos(0.5 * np.pi * x_shift / half_extents[0])
            * np.cos(0.5 * np.pi * y_shift / half_extents[1])
            * np.cos(0.5 * np.pi * z_shift / half_extents[2])
        )
    else:
        raise ValueError(f"Unsupported smooth field `{field_label}`.")

    norm = weighted_l2_norm(field, grid_geometry=grid_geometry)
    return np.asarray(field / norm, dtype=np.float64)


def _evaluate_smooth_field(
    *,
    field_label: str,
    path_type: str,
    grid_geometry: GridGeometryLike,
) -> SmoothFieldKineticResult:
    field = _build_smooth_field(field_label, grid_geometry)
    kinetic_action = apply_kinetic_operator(field, grid_geometry=grid_geometry)
    denominator = _weighted_inner_product(field, field, grid_geometry)
    denominator_real = float(np.real_if_close(denominator))
    local_indicator = np.real(np.conjugate(field) * kinetic_action)
    return SmoothFieldKineticResult(
        path_type=path_type,
        field_label=field_label,
        weighted_norm=float(np.sqrt(max(denominator_real, 0.0))),
        kinetic_rayleigh_quotient=float(
            np.real_if_close(
                _weighted_inner_product(field, kinetic_action, grid_geometry) / denominator
            )
        ),
        kinetic_action_summary=_field_summary(kinetic_action),
        negative_indicator_fraction=float(np.mean(local_indicator < 0.0)),
    )


def _build_route_geometry(
    *,
    path_type: str,
    shape_label: str,
) -> tuple[GridGeometryLike, H2MonitorPatchParameterSummary | None, bool, LocalPotentialPatchParameters | None]:
    if path_type == "legacy":
        return build_default_h2_grid_geometry(case=H2_BENCHMARK_CASE), None, False, None
    if path_type == "monitor_a_grid":
        return build_h2_local_patch_development_monitor_grid(), None, False, None
    if path_type == "monitor_a_grid_plus_patch":
        patch_parameters = _default_patch_parameters()
        patch_summary = _patch_summary(patch_parameters)
        if shape_label == "baseline":
            geometry = build_h2_local_patch_development_monitor_grid()
        elif shape_label == "finer-shape":
            geometry = _build_monitor_grid(_MONITOR_FINER_SHAPE)
        else:
            raise ValueError(f"Unsupported monitor shape label `{shape_label}`.")
        return geometry, patch_summary, True, patch_parameters
    raise ValueError(f"Unsupported path_type `{path_type}`.")


def _evaluate_route(
    *,
    case: BenchmarkCase,
    path_type: str,
    shape_label: str,
) -> KineticRouteAuditResult:
    grid_geometry, patch_summary, use_monitor_patch, patch_parameters = _build_route_geometry(
        path_type=path_type,
        shape_label=shape_label,
    )
    trial_orbital, rho_up, rho_down, rho_total = _build_frozen_density(case=case, grid_geometry=grid_geometry)
    eigensolver_result = solve_fixed_potential_static_local_eigenproblem(
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
    )
    eigen_orbital = eigensolver_result.orbitals[0]
    return KineticRouteAuditResult(
        path_type=path_type,
        shape_label=shape_label,
        grid_parameter_summary=(
            _grid_parameter_summary(path_type)
            if shape_label == "baseline"
            else (
                _grid_parameter_summary(path_type)
                + f", shape={_MONITOR_FINER_SHAPE}"
            )
        ),
        patch_parameter_summary=patch_summary,
        frozen_density_integral=float(
            np.real_if_close(_weighted_inner_product(rho_total, np.ones_like(rho_total), grid_geometry))
        ),
        frozen_orbital_summary=_summarize_kinetic_orbital(
            trial_orbital,
            grid_geometry,
            case,
            orbital_label="frozen_trial_orbital",
        ),
        eigen_orbital_summary=_summarize_kinetic_orbital(
            eigen_orbital,
            grid_geometry,
            case,
            orbital_label="eigensolver_orbital_k1",
        ),
        eigenvalue_ha=float(eigensolver_result.eigenvalues[0]),
        eigensolver_converged=bool(eigensolver_result.converged),
        eigensolver_weighted_residual_norm=float(eigensolver_result.residual_norms[0]),
        self_adjoint_probe=_self_adjoint_kinetic_probe(grid_geometry),
    )


def _diagnosis(result: H2MonitorGridKineticOperatorAuditResult) -> str:
    baseline = result.monitor_patch_baseline_result
    finer = result.monitor_patch_finer_shape_result
    smooth_monitor = [
        item for item in result.smooth_field_results if item.path_type == "monitor_a_grid"
    ]
    smooth_positive = all(item.kinetic_rayleigh_quotient > 0.0 for item in smooth_monitor)
    finer_improvement = (
        finer.eigen_orbital_summary.kinetic_rayleigh_quotient
        - baseline.eigen_orbital_summary.kinetic_rayleigh_quotient
    )
    if (
        baseline.self_adjoint_probe.relative_difference < 1.0e-10
        and result.monitor_unpatched_baseline_result.self_adjoint_probe.relative_difference < 1.0e-10
        and smooth_positive
    ):
        if baseline.frozen_orbital_summary.kinetic_rayleigh_quotient > 0.0:
            if finer_improvement >= 0.0:
                return (
                    "The kinetic failure does not look like a weighted self-adjointness bug. "
                    "Smooth test fields and the frozen trial orbital still give positive kinetic "
                    "quotients, while the A-grid eigensolver orbitals drive <psi|T|psi> strongly "
                    "negative. The current symptom looks more like the A-grid kinetic "
                    "discretization admitting pathological modes for the fixed-potential-selected "
                    "orbital, with only weak evidence for a simple resolution-tail explanation."
                )
            return (
                "The kinetic failure does not look like a weighted self-adjointness bug. Smooth "
                "test fields and the frozen trial orbital still give positive kinetic quotients, "
                "but the A-grid eigensolver orbitals drive <psi|T|psi> strongly negative. The "
                "very small finer-shape recheck makes the bad kinetic mode even deeper rather than "
                "improving it, so the dominant problem now looks more like an operator-form or "
                "geometry/kinetic consistency defect than a simple resolution tail."
            )
    return (
        "The kinetic audit confirms a severe A-grid eigensolver-orbital anomaly, but the exact "
        "mechanism remains unresolved."
    )


def run_h2_monitor_grid_kinetic_operator_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridKineticOperatorAuditResult:
    """Run the H2 kinetic-operator audit on legacy and A-grid routes."""

    legacy_result = _evaluate_route(case=case, path_type="legacy", shape_label="baseline")
    monitor_unpatched_baseline_result = _evaluate_route(
        case=case,
        path_type="monitor_a_grid",
        shape_label="baseline",
    )
    monitor_patch_baseline_result = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        shape_label="baseline",
    )
    monitor_patch_finer_shape_result = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        shape_label="finer-shape",
    )
    smooth_field_results = (
        _evaluate_smooth_field(
            field_label="gaussian",
            path_type="legacy",
            grid_geometry=build_default_h2_grid_geometry(case=case),
        ),
        _evaluate_smooth_field(
            field_label="gaussian",
            path_type="monitor_a_grid",
            grid_geometry=build_h2_local_patch_development_monitor_grid(),
        ),
        _evaluate_smooth_field(
            field_label="cosine",
            path_type="legacy",
            grid_geometry=build_default_h2_grid_geometry(case=case),
        ),
        _evaluate_smooth_field(
            field_label="cosine",
            path_type="monitor_a_grid",
            grid_geometry=build_h2_local_patch_development_monitor_grid(),
        ),
    )
    result = H2MonitorGridKineticOperatorAuditResult(
        legacy_result=legacy_result,
        monitor_unpatched_baseline_result=monitor_unpatched_baseline_result,
        monitor_patch_baseline_result=monitor_patch_baseline_result,
        monitor_patch_finer_shape_result=monitor_patch_finer_shape_result,
        smooth_field_results=smooth_field_results,
        diagnosis="",
        note=(
            "This is a kinetic-operator audit only. It keeps the H2 singlet frozen density "
            "fixed and compares legacy, A-grid, and A-grid+patch fixed-potential orbitals, "
            "but patch does not directly modify T. Nonlocal and SCF are still absent."
        ),
    )
    return H2MonitorGridKineticOperatorAuditResult(
        legacy_result=result.legacy_result,
        monitor_unpatched_baseline_result=result.monitor_unpatched_baseline_result,
        monitor_patch_baseline_result=result.monitor_patch_baseline_result,
        monitor_patch_finer_shape_result=result.monitor_patch_finer_shape_result,
        smooth_field_results=result.smooth_field_results,
        diagnosis=_diagnosis(result),
        note=result.note,
    )


def _print_orbital_summary(summary: KineticOrbitalSummary) -> None:
    print(f"    orbital: {summary.orbital_label}")
    print(f"      weighted norm: {summary.weighted_norm:.12f}")
    print(f"      <T> [Ha]: {summary.kinetic_rayleigh_quotient:+.12f}")
    print(
        "      T psi summary: "
        f"min={summary.kinetic_action_summary.minimum:+.6e}, "
        f"max={summary.kinetic_action_summary.maximum:+.6e}, "
        f"rms={summary.kinetic_action_summary.rms:.6e}"
    )
    print(
        "      local indicator summary: "
        f"min={summary.local_indicator_summary.minimum:+.6e}, "
        f"max={summary.local_indicator_summary.maximum:+.6e}, "
        f"rms={summary.local_indicator_summary.rms:.6e}, "
        f"negative_frac={summary.negative_indicator_fraction:.6f}"
    )
    print("      region diagnostics:")
    for region in summary.region_summaries:
        print(
            "        "
            f"{region.region_name}: frac={region.point_fraction:.4f}, "
            f"|psi|mean={region.psi_abs_mean:.6e}, "
            f"Tpsi_rms={region.kinetic_action_summary.rms:.6e}, "
            f"weighted_contribution={region.weighted_indicator_contribution_ha:+.12f} Ha, "
            f"neg_frac={region.negative_indicator_fraction:.6f}"
        )
    print("      center-line samples:")
    for sample in summary.centerline_samples:
        print(
            "        "
            f"z[{sample.sample_index:02d}]={sample.z_coordinate_bohr:+.6f} Bohr -> "
            f"psi={sample.orbital_value:+.12f}, "
            f"Tpsi={sample.kinetic_action_value:+.12e}, "
            f"psi*Tpsi={sample.local_kinetic_indicator:+.12e}"
        )


def _print_route_result(result: KineticRouteAuditResult) -> None:
    print(f"path: {result.path_type} ({result.shape_label})")
    print(f"  grid summary: {result.grid_parameter_summary}")
    print(f"  frozen density integral: {result.frozen_density_integral:.12f}")
    print(f"  converged: {result.eigensolver_converged}")
    print(f"  eigenvalue [Ha]: {result.eigenvalue_ha:+.12f}")
    print(f"  eig residual norm: {result.eigensolver_weighted_residual_norm:.12e}")
    print(
        "  self-adjoint kinetic probe: "
        f"abs={result.self_adjoint_probe.absolute_difference:.3e}, "
        f"rel={result.self_adjoint_probe.relative_difference:.3e}"
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
    _print_orbital_summary(result.frozen_orbital_summary)
    _print_orbital_summary(result.eigen_orbital_summary)


def print_h2_monitor_grid_kinetic_operator_audit_summary(
    result: H2MonitorGridKineticOperatorAuditResult,
) -> None:
    """Print the compact H2 kinetic-operator audit summary."""

    print("IsoGridDFT H2 kinetic operator audit")
    print(f"note: {result.note}")
    print()
    _print_route_result(result.legacy_result)
    print()
    _print_route_result(result.monitor_unpatched_baseline_result)
    print()
    _print_route_result(result.monitor_patch_baseline_result)
    print()
    _print_route_result(result.monitor_patch_finer_shape_result)
    print()
    print("smooth-field probes:")
    for item in result.smooth_field_results:
        print(
            "  "
            f"{item.path_type} / {item.field_label}: "
            f"<T>={item.kinetic_rayleigh_quotient:+.12f} Ha, "
            f"Tpsi_rms={item.kinetic_action_summary.rms:.6e}, "
            f"negative_frac={item.negative_indicator_fraction:.6f}"
        )
    print()
    baseline = result.monitor_patch_baseline_result
    finer = result.monitor_patch_finer_shape_result
    print("very small shape recheck:")
    print(
        "  A-grid+patch k=1 <T> delta [Ha]: "
        f"{finer.eigen_orbital_summary.kinetic_rayleigh_quotient - baseline.eigen_orbital_summary.kinetic_rayleigh_quotient:+.12f}"
    )
    print(
        "  A-grid+patch k=1 residual delta: "
        f"{finer.eigensolver_weighted_residual_norm - baseline.eigensolver_weighted_residual_norm:+.12e}"
    )
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_kinetic_operator_audit()
    print_h2_monitor_grid_kinetic_operator_audit_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
