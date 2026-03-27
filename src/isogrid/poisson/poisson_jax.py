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
    total_wall_time_seconds: float = 0.0
    boundary_condition_wall_time_seconds: float = 0.0
    build_wall_time_seconds: float = 0.0
    rhs_assembly_wall_time_seconds: float = 0.0
    cg_wall_time_seconds: float = 0.0
    matvec_call_count: int = 0
    matvec_wall_time_seconds: float = 0.0
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


_MONITOR_POISSON_JAX_KERNEL_CACHE: dict[
    tuple[int, tuple[int, int, int]],
    MonitorPoissonJaxCachedSolveKernels,
] = {}
_LAST_MONITOR_POISSON_JAX_DIAGNOSTICS: MonitorPoissonJaxSolveDiagnostics | None = None


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

    return laplacian, _matvec


def clear_monitor_poisson_jax_kernel_cache() -> None:
    """Clear the thin monitor-grid JAX Poisson kernel cache."""

    global _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS
    _MONITOR_POISSON_JAX_KERNEL_CACHE.clear()
    _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS = None


def get_last_monitor_poisson_jax_solve_diagnostics() -> MonitorPoissonJaxSolveDiagnostics | None:
    """Return the most recent monitor-grid JAX Poisson solve diagnostics."""

    return _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS


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
        boundary_laplacian, matvec = build_monitor_open_boundary_poisson_matvec_jax(
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
            ),
            False,
        )

    cache_key = _build_monitor_poisson_jax_cache_key(grid_geometry)
    cached = _MONITOR_POISSON_JAX_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached, False

    interior_mask, interior_size, scatter, gather = _build_monitor_interior_scatter(grid_geometry)
    boundary_laplacian, matvec = build_monitor_open_boundary_poisson_matvec_jax(
        grid_geometry=grid_geometry,
    )
    cached = MonitorPoissonJaxCachedSolveKernels(
        interior_mask=interior_mask,
        interior_size=interior_size,
        scatter=scatter,
        gather=gather,
        boundary_laplacian=boundary_laplacian,
        matvec=matvec,
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
        matvec_call_count=int(matvec_call_count),
        matvec_wall_time_seconds=float(matvec_wall_time_seconds),
    )


def solve_open_boundary_poisson_monitor_jax(
    *,
    grid_geometry: MonitorGridGeometry,
    rho: np.ndarray,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
    use_cached_operator: bool = False,
) -> tuple[OpenBoundaryPoissonResult, MonitorPoissonJaxSolveDiagnostics]:
    """Solve the monitor-grid open-boundary Poisson problem with JAX CG."""

    global _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS

    solve_start = perf_counter()
    density = validate_density_field(rho, grid_geometry=grid_geometry)
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
        total_wall_time_seconds=float(total_elapsed),
        boundary_condition_wall_time_seconds=float(boundary_elapsed),
        build_wall_time_seconds=float(build_elapsed),
        rhs_assembly_wall_time_seconds=float(rhs_elapsed),
        cg_wall_time_seconds=float(cg_elapsed),
        matvec_call_count=int(diagnostics.matvec_call_count),
        matvec_wall_time_seconds=float(diagnostics.matvec_wall_time_seconds),
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
