"""Production-vs-reference audit for the A-grid kinetic operator form.

This audit does not modify the production kinetic implementation. It compares
the current monitor-grid kinetic operator against a more direct reference
discretization on the same monitor-grid geometry and the same H2 singlet frozen
density / fixed-potential orbitals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ops import apply_monitor_grid_kinetic_operator
from isogrid.ops import validate_orbital_field

from .h2_monitor_grid_kinetic_operator_audit import _MONITOR_FINER_SHAPE
from .h2_monitor_grid_kinetic_operator_audit import _build_monitor_grid
from .h2_monitor_grid_kinetic_operator_audit import _build_smooth_field
from .h2_monitor_grid_operator_audit import ScalarFieldSummary
from .h2_monitor_grid_operator_audit import SelfAdjointnessProbe
from .h2_monitor_grid_operator_audit import _build_frozen_density
from .h2_monitor_grid_operator_audit import _build_probe_field
from .h2_monitor_grid_operator_audit import _compute_region_masks
from .h2_monitor_grid_operator_audit import _default_patch_parameters
from .h2_monitor_grid_operator_audit import _field_summary
from .h2_monitor_grid_operator_audit import _weighted_inner_product


@dataclass(frozen=True)
class KineticFormCenterlineSample:
    """Center-line sample for production/reference kinetic comparison."""

    sample_index: int
    z_coordinate_bohr: float
    orbital_value: float
    production_tpsi: float
    reference_tpsi: float
    delta_tpsi: float


@dataclass(frozen=True)
class KineticFormRegionSummary:
    """Regionwise summary for one kinetic field or difference field."""

    region_name: str
    point_fraction: float
    field_summary: ScalarFieldSummary
    weighted_mean: float
    weighted_rms: float
    negative_fraction: float


@dataclass(frozen=True)
class KineticFormComparisonResult:
    """Production-vs-reference kinetic comparison for one field/orbital."""

    shape_label: str
    orbital_label: str
    production_label: str
    reference_label: str
    weighted_norm: float
    production_kinetic_quotient: float
    reference_kinetic_quotient: float
    delta_kinetic_quotient_mha: float
    production_tpsi_summary: ScalarFieldSummary
    reference_tpsi_summary: ScalarFieldSummary
    delta_tpsi_summary: ScalarFieldSummary
    production_region_summaries: tuple[KineticFormRegionSummary, ...]
    reference_region_summaries: tuple[KineticFormRegionSummary, ...]
    delta_region_summaries: tuple[KineticFormRegionSummary, ...]
    centerline_samples: tuple[KineticFormCenterlineSample, ...]
    eigensolver_eigenvalue_ha: float | None
    eigensolver_residual_norm: float | None
    eigensolver_converged: bool | None


@dataclass(frozen=True)
class KineticFormSelfAdjointnessComparison:
    """Self-adjointness comparison for production and reference kinetic paths."""

    shape_label: str
    production_probe: SelfAdjointnessProbe
    reference_probe: SelfAdjointnessProbe


@dataclass(frozen=True)
class H2MonitorGridKineticFormAuditResult:
    """Top-level A-grid kinetic production-vs-reference audit result."""

    frozen_trial_baseline: KineticFormComparisonResult
    bad_eigen_baseline: KineticFormComparisonResult
    bad_eigen_finer_shape: KineticFormComparisonResult
    smooth_field_results: tuple[KineticFormComparisonResult, ...]
    self_adjointness_baseline: KineticFormSelfAdjointnessComparison
    self_adjointness_finer_shape: KineticFormSelfAdjointnessComparison
    diagnosis: str
    note: str


def apply_monitor_grid_reference_kinetic_operator(
    psi: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> np.ndarray:
    """Apply a more direct reference kinetic discretization on the A-grid.

    Production path:

        T_prod psi = -1/2 * (1/J) * d_a [ J g^{ab} d_b psi ]

    Reference path used here:

        T_ref psi = -1/2 * [ g^{ab} d_ab psi + b^b d_b psi ]
        b^b = (1/J) d_a (J g^{ab})

    The reference form expands the divergence explicitly, uses symmetrized mixed
    second derivatives, and therefore avoids the production flux-divergence
    assembly as a direct audit-side comparison.
    """

    field = validate_orbital_field(psi, grid_geometry=grid_geometry, name="psi")
    logical_axes = (
        np.asarray(grid_geometry.logical_x, dtype=np.float64),
        np.asarray(grid_geometry.logical_y, dtype=np.float64),
        np.asarray(grid_geometry.logical_z, dtype=np.float64),
    )
    jacobian = np.asarray(grid_geometry.jacobian, dtype=np.float64)
    inverse_metric = np.asarray(grid_geometry.inverse_metric_tensor, dtype=np.float64)
    coefficient_tensor = jacobian[..., None, None] * inverse_metric

    first_derivatives = np.gradient(field, *logical_axes, edge_order=2)
    second_derivatives = [[None] * 3 for _ in range(3)]
    for axis_a in range(3):
        for axis_b in range(3):
            second_derivatives[axis_a][axis_b] = np.gradient(
                first_derivatives[axis_b],
                logical_axes[axis_a],
                axis=axis_a,
                edge_order=2,
            )

    laplacian_second_order = np.zeros_like(field, dtype=np.float64)
    for axis_a in range(3):
        for axis_b in range(3):
            if axis_a == axis_b:
                second_order = second_derivatives[axis_a][axis_b]
            else:
                second_order = 0.5 * (
                    second_derivatives[axis_a][axis_b]
                    + second_derivatives[axis_b][axis_a]
                )
            laplacian_second_order += inverse_metric[..., axis_a, axis_b] * second_order

    drift_term = np.zeros_like(field, dtype=np.float64)
    for axis_b in range(3):
        drift_component = np.zeros_like(field, dtype=np.float64)
        for axis_a in range(3):
            drift_component += np.gradient(
                coefficient_tensor[..., axis_a, axis_b],
                logical_axes[axis_a],
                axis=axis_a,
                edge_order=2,
            )
        drift_term += (drift_component / jacobian) * first_derivatives[axis_b]

    return -0.5 * (laplacian_second_order + drift_term)


def _region_summaries(
    field: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    case: BenchmarkCase,
) -> tuple[KineticFormRegionSummary, ...]:
    summaries: list[KineticFormRegionSummary] = []
    total_points = float(np.prod(grid_geometry.spec.shape))
    values = np.asarray(field, dtype=np.float64)
    for name, mask in _compute_region_masks(case=case, grid_geometry=grid_geometry):
        if not np.any(mask):
            continue
        masked_values = values[mask]
        weights = grid_geometry.cell_volumes[mask]
        summaries.append(
            KineticFormRegionSummary(
                region_name=name,
                point_fraction=float(np.sum(mask) / total_points),
                field_summary=_field_summary(masked_values),
                weighted_mean=float(np.sum(masked_values * weights) / np.sum(weights)),
                weighted_rms=float(np.sqrt(np.sum(masked_values * masked_values * weights) / np.sum(weights))),
                negative_fraction=float(np.mean(masked_values < 0.0)),
            )
        )
    return tuple(summaries)


def _centerline_samples(
    orbital: np.ndarray,
    production_tpsi: np.ndarray,
    reference_tpsi: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[KineticFormCenterlineSample, ...]:
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
        KineticFormCenterlineSample(
            sample_index=index,
            z_coordinate_bohr=float(z_coordinates[index]),
            orbital_value=float(orbital[center_ix, center_iy, index]),
            production_tpsi=float(production_tpsi[center_ix, center_iy, index]),
            reference_tpsi=float(reference_tpsi[center_ix, center_iy, index]),
            delta_tpsi=float(
                production_tpsi[center_ix, center_iy, index]
                - reference_tpsi[center_ix, center_iy, index]
            ),
        )
        for index in sample_indices
    )


def _kinetic_quotient(
    orbital: np.ndarray,
    kinetic_action: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[float, float]:
    denominator = _weighted_inner_product(orbital, orbital, grid_geometry)
    denominator_real = float(np.real_if_close(denominator))
    quotient = float(
        np.real_if_close(_weighted_inner_product(orbital, kinetic_action, grid_geometry) / denominator)
    )
    return float(np.sqrt(max(denominator_real, 0.0))), quotient


def _build_comparison(
    *,
    orbital: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    case: BenchmarkCase,
    orbital_label: str,
    shape_label: str,
    eigenvalue_ha: float | None,
    residual_norm: float | None,
    converged: bool | None,
) -> KineticFormComparisonResult:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name=orbital_label)
    production_tpsi = apply_monitor_grid_kinetic_operator(field, grid_geometry=grid_geometry)
    reference_tpsi = apply_monitor_grid_reference_kinetic_operator(field, grid_geometry=grid_geometry)
    delta_tpsi = production_tpsi - reference_tpsi
    weighted_norm, production_q = _kinetic_quotient(field, production_tpsi, grid_geometry)
    _, reference_q = _kinetic_quotient(field, reference_tpsi, grid_geometry)
    return KineticFormComparisonResult(
        shape_label=shape_label,
        orbital_label=orbital_label,
        production_label="monitor_production",
        reference_label="monitor_reference",
        weighted_norm=weighted_norm,
        production_kinetic_quotient=production_q,
        reference_kinetic_quotient=reference_q,
        delta_kinetic_quotient_mha=float((production_q - reference_q) * 1000.0),
        production_tpsi_summary=_field_summary(production_tpsi),
        reference_tpsi_summary=_field_summary(reference_tpsi),
        delta_tpsi_summary=_field_summary(delta_tpsi),
        production_region_summaries=_region_summaries(production_tpsi, grid_geometry, case),
        reference_region_summaries=_region_summaries(reference_tpsi, grid_geometry, case),
        delta_region_summaries=_region_summaries(delta_tpsi, grid_geometry, case),
        centerline_samples=_centerline_samples(field, production_tpsi, reference_tpsi, grid_geometry),
        eigensolver_eigenvalue_ha=eigenvalue_ha,
        eigensolver_residual_norm=residual_norm,
        eigensolver_converged=converged,
    )


def _self_adjoint_probe(
    grid_geometry: MonitorGridGeometry,
    *,
    apply_operator,
) -> SelfAdjointnessProbe:
    u = _build_probe_field("u", grid_geometry)
    w = _build_probe_field("w", grid_geometry)
    left = _weighted_inner_product(u, apply_operator(w), grid_geometry)
    right = _weighted_inner_product(apply_operator(u), w, grid_geometry)
    absolute_difference = abs(left - right)
    scale = max(abs(left), abs(right), 1.0e-30)
    return SelfAdjointnessProbe(
        absolute_difference=float(absolute_difference),
        relative_difference=float(absolute_difference / scale),
        left_inner_product_real=float(np.real_if_close(left)),
        right_inner_product_real=float(np.real_if_close(right)),
    )


def _evaluate_bad_eigen_orbital(
    *,
    grid_geometry: MonitorGridGeometry,
    case: BenchmarkCase,
) -> tuple[np.ndarray, float, float, bool]:
    _, rho_up, rho_down, _ = _build_frozen_density(case=case, grid_geometry=grid_geometry)
    eigensolver_result = solve_fixed_potential_static_local_eigenproblem(
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
        eigensolver_result.orbitals[0],
        float(eigensolver_result.eigenvalues[0]),
        float(eigensolver_result.residual_norms[0]),
        bool(eigensolver_result.converged),
    )


def _diagnosis(result: H2MonitorGridKineticFormAuditResult) -> str:
    frozen = result.frozen_trial_baseline
    bad = result.bad_eigen_baseline
    finer = result.bad_eigen_finer_shape
    smooth_ok = all(
        item.production_kinetic_quotient > 0.0 and item.reference_kinetic_quotient > 0.0
        for item in result.smooth_field_results
    )
    bad_split_small = abs(bad.delta_kinetic_quotient_mha) < 0.1
    finer_split_small = abs(finer.delta_kinetic_quotient_mha) < 0.01
    frozen_split_small = abs(frozen.delta_kinetic_quotient_mha) < 10.0
    if smooth_ok and frozen.production_kinetic_quotient > 0.0 and frozen.reference_kinetic_quotient > 0.0:
        if (
            bad.production_kinetic_quotient < 0.0
            and bad.reference_kinetic_quotient < 0.0
            and bad_split_small
            and finer.production_kinetic_quotient < 0.0
            and finer.reference_kinetic_quotient < 0.0
            and finer_split_small
            and frozen_split_small
        ):
            return (
                "Production and reference kinetic forms stay close on the same A-grid geometry: the "
                "frozen trial orbital and smooth fields remain positive, while the bad eigensolver "
                "orbital is strongly negative in both discretizations with only sub-mHa production/"
                "reference differences. The finer-shape recheck makes that shared negative kinetic "
                "mode deeper. This points away from a production-only flux/divergence bug and more "
                "toward a geometry/metric consistency defect that both discretizations inherit on the "
                "same monitor-grid geometry."
            )
    return (
        "The kinetic form audit shows a meaningful production/reference split, but the exact root "
        "cause still needs a narrower operator inspection."
    )


def run_h2_monitor_grid_kinetic_form_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridKineticFormAuditResult:
    """Run the H2 A-grid production-vs-reference kinetic form audit."""

    baseline_grid = _build_monitor_grid((67, 67, 81))
    finer_grid = _build_monitor_grid(_MONITOR_FINER_SHAPE)

    frozen_trial_orbital, _, _, _ = _build_frozen_density(case=case, grid_geometry=baseline_grid)
    bad_orbital_baseline, eigenvalue_baseline, residual_baseline, converged_baseline = (
        _evaluate_bad_eigen_orbital(grid_geometry=baseline_grid, case=case)
    )
    bad_orbital_finer, eigenvalue_finer, residual_finer, converged_finer = (
        _evaluate_bad_eigen_orbital(grid_geometry=finer_grid, case=case)
    )

    frozen_trial_baseline = _build_comparison(
        orbital=frozen_trial_orbital,
        grid_geometry=baseline_grid,
        case=case,
        orbital_label="frozen_trial_orbital",
        shape_label="baseline",
        eigenvalue_ha=None,
        residual_norm=None,
        converged=None,
    )
    bad_eigen_baseline = _build_comparison(
        orbital=bad_orbital_baseline,
        grid_geometry=baseline_grid,
        case=case,
        orbital_label="bad_eigensolver_orbital_k1",
        shape_label="baseline",
        eigenvalue_ha=eigenvalue_baseline,
        residual_norm=residual_baseline,
        converged=converged_baseline,
    )
    bad_eigen_finer_shape = _build_comparison(
        orbital=bad_orbital_finer,
        grid_geometry=finer_grid,
        case=case,
        orbital_label="bad_eigensolver_orbital_k1",
        shape_label="finer-shape",
        eigenvalue_ha=eigenvalue_finer,
        residual_norm=residual_finer,
        converged=converged_finer,
    )

    smooth_field_results = (
        _build_comparison(
            orbital=_build_smooth_field("gaussian", baseline_grid),
            grid_geometry=baseline_grid,
            case=case,
            orbital_label="smooth_gaussian",
            shape_label="baseline",
            eigenvalue_ha=None,
            residual_norm=None,
            converged=None,
        ),
        _build_comparison(
            orbital=_build_smooth_field("cosine", baseline_grid),
            grid_geometry=baseline_grid,
            case=case,
            orbital_label="smooth_cosine",
            shape_label="baseline",
            eigenvalue_ha=None,
            residual_norm=None,
            converged=None,
        ),
    )

    result = H2MonitorGridKineticFormAuditResult(
        frozen_trial_baseline=frozen_trial_baseline,
        bad_eigen_baseline=bad_eigen_baseline,
        bad_eigen_finer_shape=bad_eigen_finer_shape,
        smooth_field_results=smooth_field_results,
        self_adjointness_baseline=KineticFormSelfAdjointnessComparison(
            shape_label="baseline",
            production_probe=_self_adjoint_probe(
                baseline_grid,
                apply_operator=lambda field: apply_monitor_grid_kinetic_operator(field, baseline_grid),
            ),
            reference_probe=_self_adjoint_probe(
                baseline_grid,
                apply_operator=lambda field: apply_monitor_grid_reference_kinetic_operator(field, baseline_grid),
            ),
        ),
        self_adjointness_finer_shape=KineticFormSelfAdjointnessComparison(
            shape_label="finer-shape",
            production_probe=_self_adjoint_probe(
                finer_grid,
                apply_operator=lambda field: apply_monitor_grid_kinetic_operator(field, finer_grid),
            ),
            reference_probe=_self_adjoint_probe(
                finer_grid,
                apply_operator=lambda field: apply_monitor_grid_reference_kinetic_operator(field, finer_grid),
            ),
        ),
        diagnosis="",
        note=(
            "This is a production-vs-reference kinetic-form audit on the A-grid only. It keeps "
            "the H2 singlet frozen density fixed and compares the current monitor-grid kinetic "
            "operator against a more direct expanded-form reference discretization on the same "
            "geometry. Patch does not directly modify T."
        ),
    )
    return H2MonitorGridKineticFormAuditResult(
        frozen_trial_baseline=result.frozen_trial_baseline,
        bad_eigen_baseline=result.bad_eigen_baseline,
        bad_eigen_finer_shape=result.bad_eigen_finer_shape,
        smooth_field_results=result.smooth_field_results,
        self_adjointness_baseline=result.self_adjointness_baseline,
        self_adjointness_finer_shape=result.self_adjointness_finer_shape,
        diagnosis=_diagnosis(result),
        note=result.note,
    )


def _print_comparison(result: KineticFormComparisonResult) -> None:
    print(f"field: {result.orbital_label} ({result.shape_label})")
    print(f"  weighted norm: {result.weighted_norm:.12f}")
    print(
        f"  prod <T> [Ha]: {result.production_kinetic_quotient:+.12f} | "
        f"ref <T> [Ha]: {result.reference_kinetic_quotient:+.12f} | "
        f"delta [mHa]: {result.delta_kinetic_quotient_mha:+.3f}"
    )
    if result.eigensolver_eigenvalue_ha is not None:
        print(
            f"  eig info: eps={result.eigensolver_eigenvalue_ha:+.12f} Ha, "
            f"res={result.eigensolver_residual_norm:.6e}, converged={result.eigensolver_converged}"
        )
    print(
        "  Tpsi rms: "
        f"prod={result.production_tpsi_summary.rms:.6e}, "
        f"ref={result.reference_tpsi_summary.rms:.6e}, "
        f"delta={result.delta_tpsi_summary.rms:.6e}"
    )
    print("  delta region diagnostics:")
    for region in result.delta_region_summaries:
        print(
            "    "
            f"{region.region_name}: rms={region.weighted_rms:.6e}, "
            f"mean={region.weighted_mean:+.6e}, "
            f"neg_frac={region.negative_fraction:.6f}"
        )
    print("  center-line samples:")
    for sample in result.centerline_samples:
        print(
            "    "
            f"z[{sample.sample_index:02d}]={sample.z_coordinate_bohr:+.6f} -> "
            f"prod={sample.production_tpsi:+.6e}, "
            f"ref={sample.reference_tpsi:+.6e}, "
            f"delta={sample.delta_tpsi:+.6e}"
        )


def print_h2_monitor_grid_kinetic_form_audit_summary(
    result: H2MonitorGridKineticFormAuditResult,
) -> None:
    """Print the compact H2 kinetic-form audit summary."""

    print("IsoGridDFT H2 kinetic form audit")
    print(f"note: {result.note}")
    print()
    print("self-adjointness probes:")
    for probe in (result.self_adjointness_baseline, result.self_adjointness_finer_shape):
        print(
            "  "
            f"{probe.shape_label}: prod_rel={probe.production_probe.relative_difference:.3e}, "
            f"ref_rel={probe.reference_probe.relative_difference:.3e}"
        )
    print()
    _print_comparison(result.frozen_trial_baseline)
    print()
    _print_comparison(result.bad_eigen_baseline)
    print()
    _print_comparison(result.bad_eigen_finer_shape)
    print()
    print("smooth-field comparisons:")
    for item in result.smooth_field_results:
        print(
            "  "
            f"{item.orbital_label}: prod={item.production_kinetic_quotient:+.12f} Ha, "
            f"ref={item.reference_kinetic_quotient:+.12f} Ha, "
            f"delta={item.delta_kinetic_quotient_mha:+.3f} mHa"
        )
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_kinetic_form_audit()
    print_h2_monitor_grid_kinetic_form_audit_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
