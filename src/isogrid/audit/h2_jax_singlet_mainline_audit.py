"""Formal H2 singlet audit on the current frozen A-grid local-only mainline.

This audit is intentionally narrow:

- H2 singlet only
- current frozen A-grid mainline only
- local-only Hamiltonian
- no new mixer or preconditioner experiments

The goal is to answer whether the current triplet-proven local-only A-grid line
is now good enough for the H2 singlet to be considered mainline-ready, or
whether the remaining obstacle is still SCF fixed-point behavior rather than a
hot-path backend defect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.audit.baselines import H2_SCF_DRY_RUN_BASELINE
from isogrid.audit.baselines import H2_SINGLET_STABILITY_BASELINE
from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_SINGLET_MAINLINE_MAX_ITERATIONS = 20
_SINGLET_MAINLINE_MIXING = 0.20
_SINGLET_MAINLINE_DENSITY_TOLERANCE = 5.0e-3
_SINGLET_MAINLINE_ENERGY_TOLERANCE = 5.0e-5
_SINGLET_MAINLINE_EIGENSOLVER_TOLERANCE = 1.0e-3
_SINGLET_MAINLINE_EIGENSOLVER_NCV = 20


@dataclass(frozen=True)
class H2JaxSingletMainlineTimingBreakdown:
    """Very rough timing buckets for the frozen singlet mainline route."""

    eigensolver_wall_time_seconds: float
    static_local_prepare_wall_time_seconds: float
    hartree_solve_wall_time_seconds: float
    energy_evaluation_wall_time_seconds: float
    density_update_wall_time_seconds: float
    bookkeeping_wall_time_seconds: float


@dataclass(frozen=True)
class H2JaxSingletMainlineBehavior:
    """Very small convergence-behavior summary for the singlet route."""

    detected_two_cycle: bool
    tail_length: int
    even_odd_energy_gap_ha: float | None
    even_odd_residual_gap: float | None
    verdict: str


@dataclass(frozen=True)
class H2JaxSingletMainlineParameterSummary:
    """Frozen parameter summary for the singlet mainline audit."""

    grid_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    weight_scale: float
    radius_scale: float
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    mixing: float
    max_iterations: int
    density_tolerance: float
    energy_tolerance: float
    eigensolver_tolerance: float
    eigensolver_ncv: int
    hartree_backend: str
    use_jax_hartree_cached_operator: bool
    jax_hartree_cg_impl: str
    jax_hartree_cg_preconditioner: str
    jax_hartree_line_preconditioner_impl: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool


@dataclass(frozen=True)
class H2JaxSingletMainlineAuditResult:
    """Compact audit result for the frozen H2 singlet mainline route."""

    path_label: str
    spin_state_label: str
    path_type: str
    kinetic_version: str
    includes_nonlocal: bool
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    final_energy_change_ha: float | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    parameter_summary: H2JaxSingletMainlineParameterSummary
    timing_breakdown: H2JaxSingletMainlineTimingBreakdown
    behavior: H2JaxSingletMainlineBehavior
    final_energy_components: SinglePointEnergyComponents
    note: str


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.std(values))


def _build_behavior(
    energy_history_ha: tuple[float, ...],
    density_residual_history: tuple[float, ...],
    *,
    converged: bool,
) -> H2JaxSingletMainlineBehavior:
    if converged or len(energy_history_ha) < 6 or len(density_residual_history) < 6:
        return H2JaxSingletMainlineBehavior(
            detected_two_cycle=False,
            tail_length=0,
            even_odd_energy_gap_ha=None,
            even_odd_residual_gap=None,
            verdict="converged" if converged else "insufficient_tail",
        )

    tail_start = len(energy_history_ha) // 2
    tail_energies = np.asarray(energy_history_ha[tail_start:], dtype=np.float64)
    tail_residuals = np.asarray(density_residual_history[tail_start:], dtype=np.float64)
    if tail_energies.size % 2 != 0:
        tail_energies = tail_energies[1:]
        tail_residuals = tail_residuals[1:]

    even_energies = tail_energies[::2]
    odd_energies = tail_energies[1::2]
    even_residuals = tail_residuals[::2]
    odd_residuals = tail_residuals[1::2]

    even_energy_mean = _safe_mean(even_energies)
    odd_energy_mean = _safe_mean(odd_energies)
    even_residual_mean = _safe_mean(even_residuals)
    odd_residual_mean = _safe_mean(odd_residuals)
    energy_gap = (
        None
        if even_energy_mean is None or odd_energy_mean is None
        else float(abs(even_energy_mean - odd_energy_mean))
    )
    residual_gap = (
        None
        if even_residual_mean is None or odd_residual_mean is None
        else float(abs(even_residual_mean - odd_residual_mean))
    )
    energy_noise_floor = max(_safe_std(even_energies) or 0.0, _safe_std(odd_energies) or 0.0, 1.0e-16)
    residual_noise_floor = max(
        _safe_std(even_residuals) or 0.0,
        _safe_std(odd_residuals) or 0.0,
        1.0e-16,
    )
    detected_two_cycle = bool(
        energy_gap is not None
        and residual_gap is not None
        and energy_gap > 5.0 * energy_noise_floor
        and residual_gap > 5.0 * residual_noise_floor
    )
    if detected_two_cycle:
        verdict = "weak_two_cycle"
    elif abs(tail_residuals[-1] - tail_residuals[0]) < 5.0e-3:
        verdict = "stable_not_converged"
    elif tail_residuals[-1] > tail_residuals[0]:
        verdict = "diverging"
    else:
        verdict = "slow_monotone_or_damped"
    return H2JaxSingletMainlineBehavior(
        detected_two_cycle=detected_two_cycle,
        tail_length=int(tail_energies.size),
        even_odd_energy_gap_ha=energy_gap,
        even_odd_residual_gap=residual_gap,
        verdict=verdict,
    )


def _build_parameter_summary(
    result: H2StaticLocalScfDryRunResult,
) -> H2JaxSingletMainlineParameterSummary:
    parameters = result.parameter_summary
    return H2JaxSingletMainlineParameterSummary(
        grid_shape=parameters.grid_shape,
        box_half_extents_bohr=parameters.box_half_extents_bohr,
        weight_scale=parameters.weight_scale,
        radius_scale=parameters.radius_scale,
        patch_radius_scale=parameters.patch_radius_scale,
        patch_grid_shape=parameters.patch_grid_shape,
        correction_strength=parameters.correction_strength,
        interpolation_neighbors=parameters.interpolation_neighbors,
        kinetic_version=parameters.kinetic_version,
        mixing=_SINGLET_MAINLINE_MIXING,
        max_iterations=_SINGLET_MAINLINE_MAX_ITERATIONS,
        density_tolerance=_SINGLET_MAINLINE_DENSITY_TOLERANCE,
        energy_tolerance=_SINGLET_MAINLINE_ENERGY_TOLERANCE,
        eigensolver_tolerance=_SINGLET_MAINLINE_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_SINGLET_MAINLINE_EIGENSOLVER_NCV,
        hartree_backend=parameters.hartree_backend,
        use_jax_hartree_cached_operator=parameters.use_jax_hartree_cached_operator,
        jax_hartree_cg_impl=parameters.jax_hartree_cg_impl,
        jax_hartree_cg_preconditioner=parameters.jax_hartree_cg_preconditioner,
        jax_hartree_line_preconditioner_impl=parameters.jax_hartree_line_preconditioner_impl,
        use_jax_block_kernels=parameters.use_jax_block_kernels,
        use_step_local_static_local_reuse=parameters.use_step_local_static_local_reuse,
    )


def _build_result(result: H2StaticLocalScfDryRunResult) -> H2JaxSingletMainlineAuditResult:
    return H2JaxSingletMainlineAuditResult(
        path_label="jax-singlet-mainline",
        spin_state_label=result.spin_state_label,
        path_type=result.path_type,
        kinetic_version=result.kinetic_version,
        includes_nonlocal=result.includes_nonlocal,
        converged=result.converged,
        iteration_count=result.iteration_count,
        final_total_energy_ha=float(result.energy.total),
        final_lowest_eigenvalue_ha=result.lowest_eigenvalue,
        final_density_residual=(
            None if not result.history else float(result.history[-1].density_residual)
        ),
        final_energy_change_ha=(
            None
            if not result.history or result.history[-1].energy_change is None
            else float(result.history[-1].energy_change)
        ),
        total_wall_time_seconds=float(result.total_wall_time_seconds),
        average_iteration_wall_time_seconds=result.average_iteration_wall_time_seconds,
        parameter_summary=_build_parameter_summary(result),
        timing_breakdown=H2JaxSingletMainlineTimingBreakdown(
            eigensolver_wall_time_seconds=float(result.eigensolver_wall_time_seconds),
            static_local_prepare_wall_time_seconds=float(result.static_local_prepare_wall_time_seconds),
            hartree_solve_wall_time_seconds=float(result.hartree_solve_wall_time_seconds),
            energy_evaluation_wall_time_seconds=float(result.energy_evaluation_wall_time_seconds),
            density_update_wall_time_seconds=float(result.density_update_wall_time_seconds),
            bookkeeping_wall_time_seconds=float(result.bookkeeping_wall_time_seconds),
        ),
        behavior=_build_behavior(
            tuple(float(value) for value in result.energy_history),
            tuple(float(value) for value in result.density_residual_history),
            converged=bool(result.converged),
        ),
        final_energy_components=result.energy,
        note=(
            "Formal H2 singlet audit on the current frozen A-grid local-only mainline: "
            "use_jax_block_kernels=True, use_step_local_static_local_reuse=True, "
            "hartree_backend='jax', use_jax_hartree_cached_operator=True, "
            "cg_impl='jax_loop', cg_preconditioner='none'. Nonlocal remains absent."
        ),
    )


def run_h2_jax_singlet_mainline_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2JaxSingletMainlineAuditResult:
    """Run the frozen H2 singlet audit on the current A-grid local-only mainline."""

    result = run_h2_monitor_grid_scf_dry_run(
        "singlet",
        case=case,
        max_iterations=_SINGLET_MAINLINE_MAX_ITERATIONS,
        mixing=_SINGLET_MAINLINE_MIXING,
        density_tolerance=_SINGLET_MAINLINE_DENSITY_TOLERANCE,
        energy_tolerance=_SINGLET_MAINLINE_ENERGY_TOLERANCE,
        eigensolver_tolerance=_SINGLET_MAINLINE_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_SINGLET_MAINLINE_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        hartree_backend="jax",
        use_jax_hartree_cached_operator=True,
        jax_hartree_cg_impl="jax_loop",
        jax_hartree_cg_preconditioner="none",
        jax_hartree_line_preconditioner_impl="baseline",
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
    )
    return _build_result(result)


def print_h2_jax_singlet_mainline_summary(result: H2JaxSingletMainlineAuditResult) -> None:
    """Print the compact H2 singlet mainline audit summary."""

    print("IsoGridDFT H2 singlet mainline audit")
    print(f"note: {result.note}")
    print(
        "previous references: "
        f"dry-run baseline converged={H2_SCF_DRY_RUN_BASELINE.monitor_singlet_route.converged}, "
        f"iters={H2_SCF_DRY_RUN_BASELINE.monitor_singlet_route.iteration_count}; "
        f"stability smaller-mixing verdict={H2_SINGLET_STABILITY_BASELINE.smaller_mixing_route.two_cycle_verdict}"
    )
    print(f"path: {result.path_label}")
    print(f"converged: {result.converged}")
    print(f"iterations: {result.iteration_count}")
    print(f"final total energy [Ha]: {result.final_total_energy_ha:.12f}")
    if result.final_lowest_eigenvalue_ha is None:
        print("final lowest eigenvalue [Ha]: n/a")
    else:
        print(f"final lowest eigenvalue [Ha]: {result.final_lowest_eigenvalue_ha:.12f}")
    print(f"final density residual: {result.final_density_residual}")
    print(f"final energy change [Ha]: {result.final_energy_change_ha}")
    print(
        "timing [s]: "
        f"total={result.total_wall_time_seconds:.6f}, "
        f"avg/iter={0.0 if result.average_iteration_wall_time_seconds is None else result.average_iteration_wall_time_seconds:.6f}, "
        f"eigensolver={result.timing_breakdown.eigensolver_wall_time_seconds:.6f}, "
        f"prepare={result.timing_breakdown.static_local_prepare_wall_time_seconds:.6f}, "
        f"hartree={result.timing_breakdown.hartree_solve_wall_time_seconds:.6f}, "
        f"energy_eval={result.timing_breakdown.energy_evaluation_wall_time_seconds:.6f}"
    )
    print(
        "behavior: "
        f"verdict={result.behavior.verdict}, "
        f"detected_two_cycle={result.behavior.detected_two_cycle}, "
        f"energy_gap={result.behavior.even_odd_energy_gap_ha}, "
        f"residual_gap={result.behavior.even_odd_residual_gap}"
    )


def main() -> int:
    result = run_h2_jax_singlet_mainline_audit()
    print_h2_jax_singlet_mainline_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
