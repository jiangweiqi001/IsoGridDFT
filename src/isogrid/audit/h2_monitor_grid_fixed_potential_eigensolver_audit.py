"""H2 fixed-potential eigensolver audit on legacy and A-grid+patch routes.

This audit migrates only the static local chain

    T + V_loc,ion + V_H + V_xc

to the current A-grid+patch development baseline. Nonlocal ionic action and
SCF updates are intentionally left on their current paths.
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
from isogrid.ks import FixedPotentialStaticLocalOperatorContext
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ks import weighted_orbital_norms
from isogrid.ops import integrate_field
from isogrid.pseudo import LocalPotentialPatchParameters

from .baselines import H2_HARTREE_TAIL_RECHECK_BASELINE
from .baselines import H2_FIXED_POTENTIAL_EIGENSOLVER_BASELINE
from .baselines import H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE
from .h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary
from .h2_monitor_grid_patch_local_audit import _patch_summary
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital


@dataclass(frozen=True)
class H2FixedPotentialCenterlineSample:
    """One center-line orbital sample for the fixed-potential audit."""

    orbital_index: int
    sample_index: int
    z_coordinate_bohr: float
    orbital_value: float


@dataclass(frozen=True)
class H2FixedPotentialRouteResult:
    """Resolved fixed-potential eigensolver audit result for one route."""

    path_type: str
    kinetic_version: str
    grid_parameter_summary: str
    patch_parameter_summary: H2MonitorPatchParameterSummary | None
    target_orbitals: int
    solver_backend: str
    use_scipy_fallback: bool
    iteration_count: int
    eigenvalues: np.ndarray
    orbital_weighted_norms: np.ndarray
    max_orthogonality_error: float
    residual_norms: np.ndarray
    converged: bool
    solver_method: str
    solver_note: str
    frozen_density_integral: float
    rho_up_integral: float
    rho_down_integral: float
    patch_embedding_energy_mismatch: float | None
    patch_embedded_correction_mha: float | None
    centerline_samples: tuple[H2FixedPotentialCenterlineSample, ...]
    use_jax_block_kernels: bool = False
    use_jax_cached_kernels: bool = False
    wall_time_seconds: float | None = None


@dataclass(frozen=True)
class H2MonitorGridFixedPotentialEigensolverAuditResult:
    """Top-level H2 fixed-potential audit on legacy and A-grid+patch routes."""

    legacy_k1_result: H2FixedPotentialRouteResult
    monitor_patch_trial_fix_old_hotpath_k1_result: H2FixedPotentialRouteResult
    monitor_patch_trial_fix_jax_hotpath_k1_result: H2FixedPotentialRouteResult
    legacy_k2_result: H2FixedPotentialRouteResult | None
    monitor_patch_trial_fix_old_hotpath_k2_result: H2FixedPotentialRouteResult | None
    monitor_patch_trial_fix_jax_hotpath_k2_result: H2FixedPotentialRouteResult | None
    note: str


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


def _default_patch_parameters() -> LocalPotentialPatchParameters:
    return LocalPotentialPatchParameters(
        patch_radius_scale=0.75,
        patch_grid_shape=(25, 25, 25),
        correction_strength=1.30,
        interpolation_neighbors=8,
    )


def _build_frozen_spin_densities(
    case: BenchmarkCase,
    grid_geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    orbital_density = np.asarray(np.abs(orbital) ** 2, dtype=np.float64)
    rho_up = orbital_density.copy()
    rho_down = orbital_density.copy()
    return orbital, rho_up, rho_down


def _centerline_samples(orbitals: np.ndarray, grid_geometry) -> tuple[H2FixedPotentialCenterlineSample, ...]:
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
    samples: list[H2FixedPotentialCenterlineSample] = []
    for orbital_index, orbital in enumerate(orbitals):
        for sample_index in sample_indices:
            samples.append(
                H2FixedPotentialCenterlineSample(
                    orbital_index=orbital_index,
                    sample_index=sample_index,
                    z_coordinate_bohr=float(z_coordinates[sample_index]),
                    orbital_value=float(orbital[center_ix, center_iy, sample_index]),
                )
            )
    return tuple(samples)


def _evaluate_route(
    *,
    case: BenchmarkCase,
    path_type: str,
    k: int,
    kinetic_version: str = "production",
    use_jax_block_kernels: bool = False,
) -> H2FixedPotentialRouteResult:
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
    result = solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=k,
        case=case,
        tolerance=1.0e-3,
        ncv=20,
        use_monitor_patch=use_monitor_patch,
        patch_parameters=patch_parameters,
        kinetic_version=kinetic_version,
        use_jax_block_kernels=use_jax_block_kernels,
    )
    operator_context = result.operator_context
    if not isinstance(operator_context, FixedPotentialStaticLocalOperatorContext):
        raise TypeError("Expected a static-local operator context for the fixed-potential audit.")

    patch_embedding_energy_mismatch = None
    patch_embedded_correction_mha = None
    if operator_context.frozen_patch_local_embedding is not None:
        embedding = operator_context.frozen_patch_local_embedding
        patch_embedding_energy_mismatch = float(embedding.embedding_energy_mismatch)
        patch_embedded_correction_mha = float(embedding.embedded_patch_correction_energy * 1000.0)

    return H2FixedPotentialRouteResult(
        path_type=path_type,
        kinetic_version=kinetic_version,
        grid_parameter_summary=_grid_parameter_summary(path_type),
        patch_parameter_summary=patch_summary,
        target_orbitals=k,
        solver_backend=result.solver_backend,
        use_scipy_fallback=bool(result.use_scipy_fallback),
        iteration_count=int(result.iteration_count),
        eigenvalues=np.asarray(result.eigenvalues, dtype=np.float64),
        orbital_weighted_norms=weighted_orbital_norms(
            result.orbitals,
            grid_geometry=grid_geometry,
            use_jax_block_kernels=use_jax_block_kernels,
        ),
        max_orthogonality_error=float(result.max_orthogonality_error),
        residual_norms=np.asarray(result.residual_norms, dtype=np.float64),
        converged=bool(result.converged),
        solver_method=result.solver_method,
        solver_note=result.solver_note,
        use_jax_block_kernels=bool(result.use_jax_block_kernels),
        use_jax_cached_kernels=bool(result.use_jax_cached_kernels),
        wall_time_seconds=result.wall_time_seconds,
        frozen_density_integral=float(integrate_field(rho_total, grid_geometry=grid_geometry)),
        rho_up_integral=float(integrate_field(rho_up, grid_geometry=grid_geometry)),
        rho_down_integral=float(integrate_field(rho_down, grid_geometry=grid_geometry)),
        patch_embedding_energy_mismatch=patch_embedding_energy_mismatch,
        patch_embedded_correction_mha=patch_embedded_correction_mha,
        centerline_samples=_centerline_samples(result.orbitals, grid_geometry=grid_geometry),
    )


def run_h2_monitor_grid_fixed_potential_eigensolver_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridFixedPotentialEigensolverAuditResult:
    """Run the H2 fixed-potential eigensolver audit on legacy and A-grid+patch."""

    legacy_k1 = _evaluate_route(case=case, path_type="legacy", k=1)
    monitor_patch_trial_fix_old_hotpath_k1 = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        k=1,
        kinetic_version="trial_fix",
        use_jax_block_kernels=False,
    )
    monitor_patch_trial_fix_jax_hotpath_k1 = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        k=1,
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
    )
    legacy_k2 = _evaluate_route(case=case, path_type="legacy", k=2)
    monitor_patch_trial_fix_old_hotpath_k2 = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        k=2,
        kinetic_version="trial_fix",
        use_jax_block_kernels=False,
    )
    monitor_patch_trial_fix_jax_hotpath_k2 = _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        k=2,
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
    )
    return H2MonitorGridFixedPotentialEigensolverAuditResult(
        legacy_k1_result=legacy_k1,
        monitor_patch_trial_fix_old_hotpath_k1_result=monitor_patch_trial_fix_old_hotpath_k1,
        monitor_patch_trial_fix_jax_hotpath_k1_result=monitor_patch_trial_fix_jax_hotpath_k1,
        legacy_k2_result=legacy_k2,
        monitor_patch_trial_fix_old_hotpath_k2_result=monitor_patch_trial_fix_old_hotpath_k2,
        monitor_patch_trial_fix_jax_hotpath_k2_result=monitor_patch_trial_fix_jax_hotpath_k2,
        note=(
            "This audit migrates only the static local chain "
            "T + V_loc,ion + V_H + V_xc to the A-grid+patch fixed-potential eigensolver, "
            "and now compares the explicit SciPy fallback route against the JAX-native "
            "fixed-potential solver on top of the repaired kinetic-trial-fix branch. "
            "Nonlocal ionic action and SCF are still not on the A-grid path. "
            "The current static-local regression baseline and Hartree tail baseline remain "
            f"{H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE.monitor_patch_vs_legacy_delta_mha:+.3f} mHa "
            f"and {H2_HARTREE_TAIL_RECHECK_BASELINE.baseline_point.hartree_delta_vs_legacy_mha:+.3f} mHa "
            "relative to legacy. The formal fixed-potential regression baseline is now "
            f"{H2_FIXED_POTENTIAL_EIGENSOLVER_BASELINE.monitor_patch_k1_route.max_residual_norm:.3f} "
            "for the A-grid+patch k=1 residual norm."
        ),
    )


def _print_route_result(result: H2FixedPotentialRouteResult) -> None:
    print(f"path: {result.path_type}")
    print(f"  kinetic version: {result.kinetic_version}")
    print(f"  jax block hot path: {result.use_jax_block_kernels}")
    print(f"  jax cache/reuse: {result.use_jax_cached_kernels}")
    print(f"  grid summary: {result.grid_parameter_summary}")
    print(f"  target orbitals: {result.target_orbitals}")
    print(f"  solver backend: {result.solver_backend}")
    print(f"  uses SciPy fallback: {result.use_scipy_fallback}")
    print(f"  solver: {result.solver_method}")
    print(f"  converged: {result.converged}")
    print(f"  iteration count: {result.iteration_count}")
    print(f"  solver note: {result.solver_note}")
    if result.wall_time_seconds is not None:
        print(f"  wall time [s]: {result.wall_time_seconds:.6f}")
    print(f"  eigenvalues [Ha]: {result.eigenvalues.tolist()}")
    print(f"  weighted norms: {result.orbital_weighted_norms.tolist()}")
    print(f"  max orthogonality error: {result.max_orthogonality_error:.6e}")
    print(f"  residual norms: {result.residual_norms.tolist()}")
    print(
        "  frozen density integrals: "
        f"rho_up={result.rho_up_integral:.12f}, "
        f"rho_down={result.rho_down_integral:.12f}, "
        f"rho_total={result.frozen_density_integral:.12f}"
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
            f"energy mismatch={result.patch_embedding_energy_mismatch:+.3e} Ha"
        )
    print("  center-line orbital samples:")
    for sample in result.centerline_samples:
        print(
            "    "
            f"orbital[{sample.orbital_index}] z[{sample.sample_index:02d}] "
            f"= {sample.z_coordinate_bohr:+.6f} Bohr -> "
            f"{sample.orbital_value:+.12f}"
        )


def print_h2_monitor_grid_fixed_potential_eigensolver_summary(
    result: H2MonitorGridFixedPotentialEigensolverAuditResult,
) -> None:
    """Print the compact H2 fixed-potential eigensolver audit summary."""

    print("IsoGridDFT H2 fixed-potential eigensolver audit")
    print(f"note: {result.note}")
    print(
        "important: this route contains only T + V_loc,ion + V_H + V_xc; "
        "nonlocal and SCF are still absent from the A-grid path"
    )
    print()
    _print_route_result(result.legacy_k1_result)
    print()
    _print_route_result(result.monitor_patch_trial_fix_old_hotpath_k1_result)
    print()
    _print_route_result(result.monitor_patch_trial_fix_jax_hotpath_k1_result)
    print()
    print("very small recheck (k=2):")
    _print_route_result(result.legacy_k2_result)
    print()
    _print_route_result(result.monitor_patch_trial_fix_old_hotpath_k2_result)
    print()
    _print_route_result(result.monitor_patch_trial_fix_jax_hotpath_k2_result)
    print()
    print("verdict:")
    print(
        "  old hot path k=1 eigenvalue delta vs legacy [Ha]: "
        f"{result.monitor_patch_trial_fix_old_hotpath_k1_result.eigenvalues[0] - result.legacy_k1_result.eigenvalues[0]:+.12f}"
    )
    print(
        "  jax-native k=1 eigenvalue delta vs scipy fallback [Ha]: "
        f"{result.monitor_patch_trial_fix_jax_hotpath_k1_result.eigenvalues[0] - result.monitor_patch_trial_fix_old_hotpath_k1_result.eigenvalues[0]:+.12f}"
    )
    print(
        "  jax-native k=1 residual ratio vs scipy fallback: "
        f"{result.monitor_patch_trial_fix_jax_hotpath_k1_result.residual_norms[0] / result.monitor_patch_trial_fix_old_hotpath_k1_result.residual_norms[0]:.3e}"
    )
    print(
        "  current A-grid+patch fixed-potential path ready for A-grid SCF: "
        f"{result.monitor_patch_trial_fix_jax_hotpath_k1_result.converged and result.monitor_patch_trial_fix_jax_hotpath_k2_result.converged}"
    )


def main() -> int:
    result = run_h2_monitor_grid_fixed_potential_eigensolver_audit()
    print_h2_monitor_grid_fixed_potential_eigensolver_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
