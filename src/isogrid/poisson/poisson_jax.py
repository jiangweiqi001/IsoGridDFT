"""JAX hot kernels for the monitor-grid open-boundary Poisson path.

This module intentionally migrates only the numerically hot pieces of the
current monitor-grid Poisson route:

- monitor-grid Poisson operator apply
- one small JAX conjugate-gradient solve for the interior unknown

The physical boundary model is unchanged. Boundary values still come from the
existing open-boundary multipole approximation, which remains in the Python
audit/fallback layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from isogrid.config.runtime_jax import get_configured_jax
from isogrid.grid import MonitorGridGeometry
from isogrid.ops.kinetic_jax import build_monitor_grid_laplacian_operator_jax

from .hartree import validate_density_field
from .open_boundary import _FOUR_PI
from .open_boundary import _boundary_mask
from .open_boundary import _compute_multipole_boundary_condition
from .open_boundary import OpenBoundaryPoissonResult


@dataclass(frozen=True)
class MonitorPoissonJaxSolveDiagnostics:
    """Compact diagnostics for the JAX monitor-grid Poisson solve."""

    solver_method: str
    iteration_count: int
    residual_max: float
    converged: bool
    cg_impl: str = "baseline"
    total_wall_time_seconds: float = 0.0
    boundary_condition_wall_time_seconds: float = 0.0
    build_wall_time_seconds: float = 0.0
    rhs_assembly_wall_time_seconds: float = 0.0
    cg_wall_time_seconds: float = 0.0
    cg_other_overhead_wall_time_seconds: float = 0.0
    matvec_call_count: int = 0
    matvec_wall_time_seconds: float = 0.0
    matvec_timing_is_estimated: bool = False
    used_cached_operator: bool = False
    first_solve_for_cached_operator: bool = False


@dataclass(frozen=True)
class MonitorPoissonJaxCachedSolveKernels:
    """Cached monitor-grid JAX Poisson kernels for one geometry context."""

    interior_mask: np.ndarray
    interior_size: int
    scatter: object
    gather: object
    boundary_laplacian: object
    matvec: object
    matvec_probe: object


_MONITOR_POISSON_JAX_KERNEL_CACHE: dict[
    tuple[int, tuple[int, int, int]],
    MonitorPoissonJaxCachedSolveKernels,
] = {}
_MONITOR_POISSON_JAX_CG_LOOP_SOLVER_CACHE: dict[
    tuple[int, tuple[int, int, int], int, int],
    object,
] = {}
_LAST_MONITOR_POISSON_JAX_DIAGNOSTICS: MonitorPoissonJaxSolveDiagnostics | None = None
_VALID_MONITOR_POISSON_JAX_CG_IMPLS = {"baseline", "jax_loop"}


def apply_monitor_open_boundary_poisson_operator_jax(
    potential,
    *,
    grid_geometry: MonitorGridGeometry,
):
    """Apply the monitor-grid Poisson operator `-L(v)` with JAX."""

    jax = get_configured_jax()
    jnp = jax.numpy
    values = jnp.asarray(potential, dtype=jnp.float64)
    return -apply_monitor_grid_laplacian_operator_jax(values, grid_geometry=grid_geometry)


def _build_monitor_interior_scatter(grid_geometry: MonitorGridGeometry):
    jax = get_configured_jax()
    jnp = jax.numpy
    interior_mask = ~_boundary_mask(grid_geometry.spec.shape)
    interior_indices = np.flatnonzero(interior_mask.reshape(-1))
    shape = grid_geometry.spec.shape
    size = int(interior_indices.size)
    interior_indices_jax = jnp.asarray(interior_indices, dtype=jnp.int32)

    def _scatter(interior_values):
        full = jnp.zeros(shape, dtype=jnp.float64).reshape(-1)
        scattered = full.at[interior_indices_jax].set(interior_values)
        return scattered.reshape(shape)

    def _gather(full_field):
        flattened = jnp.asarray(full_field, dtype=jnp.float64).reshape(-1)
        return flattened[interior_indices_jax]

    return interior_mask, size, _scatter, _gather


def build_monitor_open_boundary_poisson_matvec_jax(
    *,
    grid_geometry: MonitorGridGeometry,
):
    """Return the interior Poisson matvec `x -> -L(x)` for zero-boundary unknowns."""

    jax = get_configured_jax()
    _, _, scatter, gather = _build_monitor_interior_scatter(grid_geometry)
    laplacian = build_monitor_grid_laplacian_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=False,
    )

    @jax.jit
    def _matvec(interior_values):
        full_field = scatter(interior_values)
        action = -laplacian(full_field)
        return gather(action)

    @jax.jit
    def _matvec_probe(interior_values):
        return _matvec(interior_values)

    return laplacian, _matvec, _matvec_probe


def clear_monitor_poisson_jax_kernel_cache() -> None:
    """Clear the thin monitor-grid JAX Poisson kernel cache."""

    global _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS
    _MONITOR_POISSON_JAX_KERNEL_CACHE.clear()
    _MONITOR_POISSON_JAX_CG_LOOP_SOLVER_CACHE.clear()
    _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS = None


def get_last_monitor_poisson_jax_solve_diagnostics() -> MonitorPoissonJaxSolveDiagnostics | None:
    """Return the most recent monitor-grid JAX Poisson solve diagnostics."""

    return _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS


def _normalize_monitor_poisson_jax_cg_impl(cg_impl: str) -> str:
    normalized = cg_impl.strip().lower()
    if normalized not in _VALID_MONITOR_POISSON_JAX_CG_IMPLS:
        raise ValueError(
            "cg_impl must be `baseline` or `jax_loop`; "
            f"received `{cg_impl}`."
        )
    return normalized


def _build_monitor_poisson_jax_cache_key(
    grid_geometry: MonitorGridGeometry,
) -> tuple[int, tuple[int, int, int]]:
    return (id(grid_geometry), tuple(grid_geometry.spec.shape))


def _get_monitor_open_boundary_poisson_kernels_jax(
    *,
    grid_geometry: MonitorGridGeometry,
    use_cached_operator: bool,
) -> tuple[MonitorPoissonJaxCachedSolveKernels, bool]:
    if not use_cached_operator:
        interior_mask, interior_size, scatter, gather = _build_monitor_interior_scatter(grid_geometry)
        boundary_laplacian, matvec, matvec_probe = build_monitor_open_boundary_poisson_matvec_jax(
            grid_geometry=grid_geometry,
        )
        return (
            MonitorPoissonJaxCachedSolveKernels(
                interior_mask=interior_mask,
                interior_size=interior_size,
                scatter=scatter,
                gather=gather,
                boundary_laplacian=boundary_laplacian,
                matvec=matvec,
                matvec_probe=matvec_probe,
            ),
            False,
        )

    cache_key = _build_monitor_poisson_jax_cache_key(grid_geometry)
    cached = _MONITOR_POISSON_JAX_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached, False

    interior_mask, interior_size, scatter, gather = _build_monitor_interior_scatter(grid_geometry)
    boundary_laplacian, matvec, matvec_probe = build_monitor_open_boundary_poisson_matvec_jax(
        grid_geometry=grid_geometry,
    )
    cached = MonitorPoissonJaxCachedSolveKernels(
        interior_mask=interior_mask,
        interior_size=interior_size,
        scatter=scatter,
        gather=gather,
        boundary_laplacian=boundary_laplacian,
        matvec=matvec,
        matvec_probe=matvec_probe,
    )
    _MONITOR_POISSON_JAX_KERNEL_CACHE[cache_key] = cached
    return cached, True


def _run_jax_cg(
    *,
    matvec,
    rhs,
    tolerance: float,
    max_iterations: int,
):
    jax = get_configured_jax()
    jnp = jax.numpy
    matvec_call_count = 0
    matvec_wall_time_seconds = 0.0

    def _timed_matvec(values):
        nonlocal matvec_call_count, matvec_wall_time_seconds
        start = perf_counter()
        result = matvec(values)
        matvec_wall_time_seconds += perf_counter() - start
        matvec_call_count += 1
        return result

    rhs_values = jnp.asarray(rhs, dtype=jnp.float64)
    x = jnp.zeros_like(rhs_values)
    r = rhs_values - _timed_matvec(x)
    p = r
    rr = jnp.vdot(r, r)

    converged = False
    iteration_count = 0
    residual_max = float(jnp.max(jnp.abs(r)))
    for iteration in range(1, max_iterations + 1):
        ap = _timed_matvec(p)
        denominator = jnp.vdot(p, ap)
        if float(jnp.abs(denominator)) <= 1.0e-20:
            break
        alpha = rr / denominator
        x = x + alpha * p
        r = r - alpha * ap
        residual_max = float(jnp.max(jnp.abs(r)))
        iteration_count = iteration
        if residual_max < tolerance:
            converged = True
            break
        rr_new = jnp.vdot(r, r)
        if float(jnp.abs(rr)) <= 1.0e-30:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return np.asarray(x, dtype=np.float64), MonitorPoissonJaxSolveDiagnostics(
        solver_method="jax_cg_monitor",
        iteration_count=iteration_count,
        residual_max=residual_max,
        converged=converged,
        cg_impl="baseline",
        cg_other_overhead_wall_time_seconds=0.0,
        matvec_call_count=int(matvec_call_count),
        matvec_wall_time_seconds=float(matvec_wall_time_seconds),
        matvec_timing_is_estimated=False,
    )


def _build_jax_loop_cg_solver(
    *,
    matvec,
    tolerance: float,
    max_iterations: int,
):
    jax = get_configured_jax()
    jnp = jax.numpy
    lax = jax.lax
    tolerance_value = float(tolerance)
    max_iterations_value = int(max_iterations)

    @jax.jit
    def _solve(rhs_values):
        rhs_values = jnp.asarray(rhs_values, dtype=jnp.float64)
        x0 = jnp.zeros_like(rhs_values)
        r0 = rhs_values - matvec(x0)
        p0 = r0
        rr0 = jnp.vdot(r0, r0)
        residual0 = jnp.max(jnp.abs(r0))

        def _body(_, state):
            active, iteration_count, x, r, p, rr, residual_max = state

            def _inactive(_state):
                return _state

            def _active(_state):
                _, iteration_count, x, r, p, rr, residual_max = _state
                ap = matvec(p)
                denominator = jnp.vdot(p, ap)
                denominator_ok = jnp.abs(denominator) > 1.0e-20

                def _denominator_fail(__):
                    return (
                        jnp.asarray(False),
                        iteration_count,
                        x,
                        r,
                        p,
                        rr,
                        residual_max,
                    )

                def _denominator_ok(__):
                    alpha = rr / denominator
                    x_next = x + alpha * p
                    r_next = r - alpha * ap
                    residual_next = jnp.max(jnp.abs(r_next))
                    iteration_next = iteration_count + 1
                    converged_next = residual_next < tolerance_value
                    rr_next = jnp.vdot(r_next, r_next)
                    rr_ok = jnp.abs(rr) > 1.0e-30

                    def _converged(___):
                        return (
                            jnp.asarray(False),
                            iteration_next,
                            x_next,
                            r_next,
                            p,
                            rr_next,
                            residual_next,
                        )

                    def _not_converged(___):
                        def _rr_fail(____):
                            return (
                                jnp.asarray(False),
                                iteration_next,
                                x_next,
                                r_next,
                                p,
                                rr_next,
                                residual_next,
                            )

                        def _rr_ok(____):
                            beta = rr_next / rr
                            p_next = r_next + beta * p
                            still_active = iteration_next < max_iterations_value
                            return (
                                still_active,
                                iteration_next,
                                x_next,
                                r_next,
                                p_next,
                                rr_next,
                                residual_next,
                            )

                        return lax.cond(rr_ok, _rr_ok, _rr_fail, operand=None)

                    return lax.cond(converged_next, _converged, _not_converged, operand=None)

                return lax.cond(denominator_ok, _denominator_ok, _denominator_fail, operand=None)

            return lax.cond(active, _active, _inactive, state)

        final_state = lax.fori_loop(
            0,
            max_iterations_value,
            _body,
            (
                jnp.asarray(residual0 >= tolerance_value),
                jnp.asarray(0, dtype=jnp.int32),
                x0,
                r0,
                p0,
                rr0,
                residual0,
            ),
        )
        active, iteration_count, x, r, _, _, residual_max = final_state
        converged = jnp.logical_not(active)
        return x, iteration_count, residual_max, converged

    return _solve


def _build_monitor_poisson_cg_loop_cache_key(
    grid_geometry: MonitorGridGeometry,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[int, tuple[int, int, int], int, int]:
    tolerance_key = int(round(float(tolerance) * 1.0e16))
    return (
        id(grid_geometry),
        tuple(grid_geometry.spec.shape),
        int(max_iterations),
        tolerance_key,
    )


def _get_monitor_poisson_cg_loop_solver(
    *,
    grid_geometry: MonitorGridGeometry,
    matvec,
    tolerance: float,
    max_iterations: int,
    use_cached_operator: bool,
):
    if not use_cached_operator:
        return _build_jax_loop_cg_solver(
            matvec=matvec,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    cache_key = _build_monitor_poisson_cg_loop_cache_key(
        grid_geometry,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    cached_solver = _MONITOR_POISSON_JAX_CG_LOOP_SOLVER_CACHE.get(cache_key)
    if cached_solver is not None:
        return cached_solver

    cached_solver = _build_jax_loop_cg_solver(
        matvec=matvec,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _MONITOR_POISSON_JAX_CG_LOOP_SOLVER_CACHE[cache_key] = cached_solver
    return cached_solver


def _estimate_matvec_wall_time_seconds(
    *,
    matvec_probe,
    interior_values: np.ndarray,
    call_count: int,
) -> float:
    if call_count <= 0:
        return 0.0
    jax = get_configured_jax()
    jnp = jax.numpy
    values = jnp.asarray(interior_values, dtype=jnp.float64)
    warmup = matvec_probe(values)
    if hasattr(warmup, "block_until_ready"):
        warmup.block_until_ready()
    else:
        np.asarray(warmup, dtype=np.float64)
    start = perf_counter()
    probe = matvec_probe(values)
    if hasattr(probe, "block_until_ready"):
        probe.block_until_ready()
    else:
        np.asarray(probe, dtype=np.float64)
    single_call_elapsed = perf_counter() - start
    return float(single_call_elapsed * call_count)


def _run_jax_cg_loop(
    *,
    grid_geometry: MonitorGridGeometry,
    matvec,
    matvec_probe,
    rhs,
    tolerance: float,
    max_iterations: int,
    use_cached_operator: bool,
):
    solver = _get_monitor_poisson_cg_loop_solver(
        grid_geometry=grid_geometry,
        matvec=matvec,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_cached_operator=use_cached_operator,
    )
    jax = get_configured_jax()
    jnp = jax.numpy
    rhs_values = jnp.asarray(rhs, dtype=jnp.float64)
    solve_start = perf_counter()
    solution, iteration_count, residual_max, converged = solver(rhs_values)
    if hasattr(solution, "block_until_ready"):
        solution.block_until_ready()
    solve_elapsed = perf_counter() - solve_start
    iteration_count_value = int(iteration_count)
    matvec_call_count = int(iteration_count_value + 1)
    estimated_matvec_wall_time_seconds = _estimate_matvec_wall_time_seconds(
        matvec_probe=matvec_probe,
        interior_values=np.asarray(rhs_values, dtype=np.float64),
        call_count=matvec_call_count,
    )
    return np.asarray(solution, dtype=np.float64), MonitorPoissonJaxSolveDiagnostics(
        solver_method="jax_cg_monitor",
        iteration_count=iteration_count_value,
        residual_max=float(residual_max),
        converged=bool(converged),
        cg_impl="jax_loop",
        cg_wall_time_seconds=float(solve_elapsed),
        cg_other_overhead_wall_time_seconds=max(
            0.0,
            float(solve_elapsed) - float(estimated_matvec_wall_time_seconds),
        ),
        matvec_call_count=matvec_call_count,
        matvec_wall_time_seconds=float(estimated_matvec_wall_time_seconds),
        matvec_timing_is_estimated=True,
    )


def solve_open_boundary_poisson_monitor_jax(
    *,
    grid_geometry: MonitorGridGeometry,
    rho: np.ndarray,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
    use_cached_operator: bool = False,
    cg_impl: str = "baseline",
) -> tuple[OpenBoundaryPoissonResult, MonitorPoissonJaxSolveDiagnostics]:
    """Solve the monitor-grid open-boundary Poisson problem with JAX CG."""

    global _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS

    solve_start = perf_counter()
    density = validate_density_field(rho, grid_geometry=grid_geometry)
    normalized_cg_impl = _normalize_monitor_poisson_jax_cg_impl(cg_impl)
    boundary_start = perf_counter()
    boundary_condition = _compute_multipole_boundary_condition(
        grid_geometry=grid_geometry,
        rho=density,
        multipole_order=multipole_order,
    )
    boundary_elapsed = perf_counter() - boundary_start
    build_start = perf_counter()
    kernels, first_cached_solve = _get_monitor_open_boundary_poisson_kernels_jax(
        grid_geometry=grid_geometry,
        use_cached_operator=use_cached_operator,
    )
    build_elapsed = perf_counter() - build_start
    interior_mask = kernels.interior_mask
    scatter = kernels.scatter
    gather = kernels.gather
    boundary_laplacian = kernels.boundary_laplacian
    matvec = kernels.matvec
    matvec_probe = kernels.matvec_probe
    jax = get_configured_jax()
    jnp = jax.numpy

    rhs_start = perf_counter()
    boundary_field = np.array(boundary_condition.boundary_values, copy=True)
    boundary_field[interior_mask] = 0.0
    rhs_full = _FOUR_PI * density + np.asarray(
        boundary_laplacian(
            jnp.asarray(boundary_field, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    rhs_interior = gather(rhs_full)
    rhs_elapsed = perf_counter() - rhs_start
    if normalized_cg_impl == "jax_loop":
        interior_solution, diagnostics = _run_jax_cg_loop(
            grid_geometry=grid_geometry,
            matvec=matvec,
            matvec_probe=matvec_probe,
            rhs=rhs_interior,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_cached_operator=use_cached_operator,
        )
        cg_elapsed = diagnostics.cg_wall_time_seconds
    else:
        cg_start = perf_counter()
        interior_solution, diagnostics = _run_jax_cg(
            matvec=matvec,
            rhs=rhs_interior,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        cg_elapsed = perf_counter() - cg_start
    full_interior = np.asarray(scatter(jnp.asarray(interior_solution, dtype=jnp.float64)), dtype=np.float64)
    potential = np.array(boundary_condition.boundary_values, copy=True)
    potential[interior_mask] = full_interior[interior_mask]
    total_elapsed = perf_counter() - solve_start
    diagnostics = MonitorPoissonJaxSolveDiagnostics(
        solver_method=diagnostics.solver_method,
        iteration_count=diagnostics.iteration_count,
        residual_max=diagnostics.residual_max,
        converged=diagnostics.converged,
        cg_impl=normalized_cg_impl,
        total_wall_time_seconds=float(total_elapsed),
        boundary_condition_wall_time_seconds=float(boundary_elapsed),
        build_wall_time_seconds=float(build_elapsed),
        rhs_assembly_wall_time_seconds=float(rhs_elapsed),
        cg_wall_time_seconds=float(cg_elapsed),
        cg_other_overhead_wall_time_seconds=float(diagnostics.cg_other_overhead_wall_time_seconds)
        if normalized_cg_impl == "jax_loop"
        else max(
            0.0,
            float(cg_elapsed) - float(diagnostics.matvec_wall_time_seconds),
        ),
        matvec_call_count=int(diagnostics.matvec_call_count),
        matvec_wall_time_seconds=float(diagnostics.matvec_wall_time_seconds),
        matvec_timing_is_estimated=bool(diagnostics.matvec_timing_is_estimated),
        used_cached_operator=bool(use_cached_operator),
        first_solve_for_cached_operator=bool(use_cached_operator and first_cached_solve),
    )
    _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS = diagnostics
    result = OpenBoundaryPoissonResult(
        rho=density,
        potential=potential,
        boundary_condition=boundary_condition,
        solver_method=diagnostics.solver_method,
        solver_iterations=diagnostics.iteration_count,
        residual_max=diagnostics.residual_max,
        description=(
            "Finite-domain Poisson solve on the monitor-driven A-grid with the same "
            "free-space multipole boundary model as the NumPy/SciPy route, but using "
            "a JAX CG interior solve."
        ),
    )
    return result, diagnostics


__all__ = [
    "MonitorPoissonJaxSolveDiagnostics",
    "clear_monitor_poisson_jax_kernel_cache",
    "get_last_monitor_poisson_jax_solve_diagnostics",
    "apply_monitor_open_boundary_poisson_operator_jax",
    "build_monitor_open_boundary_poisson_matvec_jax",
    "solve_open_boundary_poisson_monitor_jax",
]
