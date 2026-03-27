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
    cg_preconditioner: str = "none"
    total_wall_time_seconds: float = 0.0
    boundary_condition_wall_time_seconds: float = 0.0
    build_wall_time_seconds: float = 0.0
    rhs_assembly_wall_time_seconds: float = 0.0
    cg_wall_time_seconds: float = 0.0
    cg_other_overhead_wall_time_seconds: float = 0.0
    matvec_call_count: int = 0
    matvec_wall_time_seconds: float = 0.0
    matvec_timing_is_estimated: bool = False
    preconditioner_apply_count: int = 0
    preconditioner_setup_wall_time_seconds: float = 0.0
    preconditioner_apply_wall_time_seconds: float = 0.0
    preconditioner_axis_reorder_wall_time_seconds: float = 0.0
    preconditioner_tridiagonal_solve_wall_time_seconds: float = 0.0
    preconditioner_other_overhead_wall_time_seconds: float = 0.0
    preconditioner_timing_is_estimated: bool = False
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
    inverse_preconditioner_diagonal: object


@dataclass(frozen=True)
class MonitorPoissonJaxSeparablePreconditionerContext:
    """Cached separable preconditioner data for one monitor-grid geometry."""

    interior_shape: tuple[int, int, int]
    coefficient_x: float
    coefficient_y: float
    coefficient_z: float
    apply: object


@dataclass(frozen=True)
class MonitorPoissonJaxLinePreconditionerContext:
    """Cached metric-aware line preconditioner for one monitor-grid geometry."""

    axis: int
    axis_label: str
    line_length: int
    average_line_diagonal_shift: float
    setup_wall_time_seconds: float
    apply: object
    apply_probe: object
    reorder_to_lines_probe: object
    axis_reorder_probe: object
    tridiagonal_solve_probe: object


_MONITOR_POISSON_JAX_KERNEL_CACHE: dict[
    tuple[int, tuple[int, int, int]],
    MonitorPoissonJaxCachedSolveKernels,
] = {}
_MONITOR_POISSON_JAX_SEPARABLE_PRECONDITIONER_CACHE: dict[
    tuple[int, tuple[int, int, int]],
    MonitorPoissonJaxSeparablePreconditionerContext,
] = {}
_MONITOR_POISSON_JAX_LINE_PRECONDITIONER_CACHE: dict[
    tuple[int, tuple[int, int, int]],
    MonitorPoissonJaxLinePreconditionerContext,
] = {}
_MONITOR_POISSON_JAX_CG_LOOP_SOLVER_CACHE: dict[
    tuple[int, tuple[int, int, int], int, int],
    object,
] = {}
_LAST_MONITOR_POISSON_JAX_DIAGNOSTICS: MonitorPoissonJaxSolveDiagnostics | None = None
_VALID_MONITOR_POISSON_JAX_CG_IMPLS = {"baseline", "jax_loop"}
_VALID_MONITOR_POISSON_JAX_CG_PRECONDITIONERS = {"none", "diag", "jacobi", "separable", "line"}


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
    _MONITOR_POISSON_JAX_SEPARABLE_PRECONDITIONER_CACHE.clear()
    _MONITOR_POISSON_JAX_LINE_PRECONDITIONER_CACHE.clear()
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


def _normalize_monitor_poisson_jax_cg_preconditioner(cg_preconditioner: str) -> str:
    normalized = cg_preconditioner.strip().lower()
    if normalized not in _VALID_MONITOR_POISSON_JAX_CG_PRECONDITIONERS:
        raise ValueError(
            "cg_preconditioner must be `none`, `diag`, `jacobi`, `separable`, or `line`; "
            f"received `{cg_preconditioner}`."
        )
    if normalized == "jacobi":
        return "diag"
    return normalized


def _logical_spacing(logical_coordinates: np.ndarray) -> float:
    coordinates = np.asarray(logical_coordinates, dtype=np.float64)
    spacings = np.diff(coordinates)
    if not np.allclose(spacings, spacings[0]):
        raise ValueError(
            "The current monitor-grid JAX Poisson kernels expect uniform logical axes."
        )
    return float(spacings[0])


def _build_monitor_open_boundary_inverse_preconditioner_diagonal_jax(
    *,
    grid_geometry: MonitorGridGeometry,
    gather,
):
    jax = get_configured_jax()
    jnp = jax.numpy
    inverse_metric = np.asarray(grid_geometry.inverse_metric_tensor, dtype=np.float64)
    dx = _logical_spacing(grid_geometry.logical_x)
    dy = _logical_spacing(grid_geometry.logical_y)
    dz = _logical_spacing(grid_geometry.logical_z)
    diagonal_surrogate = 2.0 * (
        inverse_metric[..., 0, 0] / (dx * dx)
        + inverse_metric[..., 1, 1] / (dy * dy)
        + inverse_metric[..., 2, 2] / (dz * dz)
    )
    positive_values = diagonal_surrogate[np.isfinite(diagonal_surrogate) & (diagonal_surrogate > 0.0)]
    if positive_values.size == 0:
        raise ValueError(
            "The monitor-grid Poisson diagonal surrogate must stay positive to build the "
            "very small Jacobi preconditioner."
        )
    floor = max(1.0e-12, 1.0e-10 * float(np.median(positive_values)))
    safe_diagonal = np.maximum(diagonal_surrogate, floor)
    gathered = gather(safe_diagonal)
    return jnp.asarray(1.0 / gathered, dtype=jnp.float64)


def _axis_stiffness_from_monitor_operator_diagonal(
    *,
    jacobian: np.ndarray,
    diagonal_flux_weight: np.ndarray,
    spacing: float,
    axis: int,
) -> float:
    center_slice = [slice(1, -1), slice(1, -1), slice(1, -1)]
    lower_slice = [slice(1, -1), slice(1, -1), slice(1, -1)]
    upper_slice = [slice(1, -1), slice(1, -1), slice(1, -1)]
    center_slice[axis] = slice(1, -1)
    lower_slice[axis] = slice(0, -2)
    upper_slice[axis] = slice(2, None)
    center = diagonal_flux_weight[tuple(center_slice)]
    lower = diagonal_flux_weight[tuple(lower_slice)]
    upper = diagonal_flux_weight[tuple(upper_slice)]
    jacobian_center = jacobian[1:-1, 1:-1, 1:-1]
    diagonal_contribution = (
        0.5 * (center + lower) + 0.5 * (center + upper)
    ) / (jacobian_center * spacing * spacing)
    coefficient = 0.5 * float(np.mean(diagonal_contribution)) * (spacing * spacing)
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError(
            "The separable monitor-grid preconditioner requires positive axis stiffness "
            "coefficients."
        )
    return coefficient


def _build_dirichlet_sine_basis_matrix(size: int) -> np.ndarray:
    indices = np.arange(1, size + 1, dtype=np.float64)
    basis = np.sin(np.pi * np.outer(indices, indices) / (size + 1))
    basis *= np.sqrt(2.0 / (size + 1))
    return np.asarray(basis, dtype=np.float64)


def _build_monitor_open_boundary_separable_preconditioner_context_jax(
    *,
    grid_geometry: MonitorGridGeometry,
) -> MonitorPoissonJaxSeparablePreconditionerContext:
    jax = get_configured_jax()
    jnp = jax.numpy
    jacobian = np.asarray(grid_geometry.jacobian, dtype=np.float64)
    inverse_metric = np.asarray(grid_geometry.inverse_metric_tensor, dtype=np.float64)
    dx = _logical_spacing(grid_geometry.logical_x)
    dy = _logical_spacing(grid_geometry.logical_y)
    dz = _logical_spacing(grid_geometry.logical_z)
    nx, ny, nz = grid_geometry.spec.shape
    interior_shape = (nx - 2, ny - 2, nz - 2)
    if min(interior_shape) <= 0:
        raise ValueError(
            "The separable monitor-grid preconditioner requires at least one interior point per axis."
        )

    jacobian_weighted_metric_x = jacobian * inverse_metric[..., 0, 0]
    jacobian_weighted_metric_y = jacobian * inverse_metric[..., 1, 1]
    jacobian_weighted_metric_z = jacobian * inverse_metric[..., 2, 2]
    coefficient_x = _axis_stiffness_from_monitor_operator_diagonal(
        jacobian=jacobian,
        diagonal_flux_weight=jacobian_weighted_metric_x,
        spacing=dx,
        axis=0,
    )
    coefficient_y = _axis_stiffness_from_monitor_operator_diagonal(
        jacobian=jacobian,
        diagonal_flux_weight=jacobian_weighted_metric_y,
        spacing=dy,
        axis=1,
    )
    coefficient_z = _axis_stiffness_from_monitor_operator_diagonal(
        jacobian=jacobian,
        diagonal_flux_weight=jacobian_weighted_metric_z,
        spacing=dz,
        axis=2,
    )

    sine_x = jnp.asarray(_build_dirichlet_sine_basis_matrix(interior_shape[0]), dtype=jnp.float64)
    sine_y = jnp.asarray(_build_dirichlet_sine_basis_matrix(interior_shape[1]), dtype=jnp.float64)
    sine_z = jnp.asarray(_build_dirichlet_sine_basis_matrix(interior_shape[2]), dtype=jnp.float64)
    modes_x = np.arange(1, interior_shape[0] + 1, dtype=np.float64)
    modes_y = np.arange(1, interior_shape[1] + 1, dtype=np.float64)
    modes_z = np.arange(1, interior_shape[2] + 1, dtype=np.float64)
    eigenvalues_x = 4.0 * np.sin(np.pi * modes_x / (2.0 * (interior_shape[0] + 1))) ** 2 / (dx * dx)
    eigenvalues_y = 4.0 * np.sin(np.pi * modes_y / (2.0 * (interior_shape[1] + 1))) ** 2 / (dy * dy)
    eigenvalues_z = 4.0 * np.sin(np.pi * modes_z / (2.0 * (interior_shape[2] + 1))) ** 2 / (dz * dz)
    inverse_eigenvalues = (
        coefficient_x * eigenvalues_x[:, None, None]
        + coefficient_y * eigenvalues_y[None, :, None]
        + coefficient_z * eigenvalues_z[None, None, :]
    )
    floor = max(1.0e-12, 1.0e-12 * float(np.max(inverse_eigenvalues)))
    safe_inverse_eigenvalues = jnp.asarray(1.0 / np.maximum(inverse_eigenvalues, floor), dtype=jnp.float64)

    @jax.jit
    def _apply(interior_residual):
        field = jnp.reshape(interior_residual, interior_shape)
        spectral = jnp.einsum("ia,jb,kc,abc->ijk", sine_x, sine_y, sine_z, field)
        solved = spectral * safe_inverse_eigenvalues
        restored = jnp.einsum("ai,bj,ck,ijk->abc", sine_x, sine_y, sine_z, solved)
        return jnp.reshape(restored, (-1,))

    return MonitorPoissonJaxSeparablePreconditionerContext(
        interior_shape=interior_shape,
        coefficient_x=float(coefficient_x),
        coefficient_y=float(coefficient_y),
        coefficient_z=float(coefficient_z),
        apply=_apply,
    )


def _get_monitor_open_boundary_separable_preconditioner_context_jax(
    *,
    grid_geometry: MonitorGridGeometry,
    use_cached_operator: bool,
) -> tuple[MonitorPoissonJaxSeparablePreconditionerContext, bool]:
    if not use_cached_operator:
        return (
            _build_monitor_open_boundary_separable_preconditioner_context_jax(
                grid_geometry=grid_geometry,
            ),
            False,
        )

    cache_key = _build_monitor_poisson_jax_cache_key(grid_geometry)
    cached = _MONITOR_POISSON_JAX_SEPARABLE_PRECONDITIONER_CACHE.get(cache_key)
    if cached is not None:
        return cached, False

    cached = _build_monitor_open_boundary_separable_preconditioner_context_jax(
        grid_geometry=grid_geometry,
    )
    _MONITOR_POISSON_JAX_SEPARABLE_PRECONDITIONER_CACHE[cache_key] = cached
    return cached, True


def _build_monitor_open_boundary_line_preconditioner_context_jax(
    *,
    grid_geometry: MonitorGridGeometry,
) -> MonitorPoissonJaxLinePreconditionerContext:
    setup_start = perf_counter()
    jax = get_configured_jax()
    jnp = jax.numpy
    lax_linalg = jax.lax.linalg
    jacobian = np.asarray(grid_geometry.jacobian, dtype=np.float64)
    inverse_metric = np.asarray(grid_geometry.inverse_metric_tensor, dtype=np.float64)
    dx = _logical_spacing(grid_geometry.logical_x)
    dy = _logical_spacing(grid_geometry.logical_y)
    dz = _logical_spacing(grid_geometry.logical_z)
    spacings = (dx, dy, dz)
    diagonal_flux_weights = (
        jacobian * inverse_metric[..., 0, 0],
        jacobian * inverse_metric[..., 1, 1],
        jacobian * inverse_metric[..., 2, 2],
    )
    stiffness_coefficients = tuple(
        _axis_stiffness_from_monitor_operator_diagonal(
            jacobian=jacobian,
            diagonal_flux_weight=diagonal_flux_weights[axis],
            spacing=spacings[axis],
            axis=axis,
        )
        for axis in range(3)
    )
    stiffness_scores = tuple(
        2.0 * stiffness_coefficients[axis] / (spacings[axis] * spacings[axis])
        for axis in range(3)
    )
    axis = int(np.argmax(np.asarray(stiffness_scores, dtype=np.float64)))
    axis_label = "xyz"[axis]
    spacing = spacings[axis]
    diagonal_flux_weight = diagonal_flux_weights[axis]
    jacobian_center = jacobian[1:-1, 1:-1, 1:-1]
    axis_lower_coefficients: list[np.ndarray] = []
    axis_upper_coefficients: list[np.ndarray] = []
    diagonal_contributions: list[np.ndarray] = []
    for axis_index, (axis_spacing, axis_diagonal_flux_weight) in enumerate(
        zip(spacings, diagonal_flux_weights, strict=False)
    ):
        lower_slice = [slice(1, -1), slice(1, -1), slice(1, -1)]
        upper_slice = [slice(1, -1), slice(1, -1), slice(1, -1)]
        lower_slice[axis_index] = slice(0, -2)
        upper_slice[axis_index] = slice(2, None)
        center = axis_diagonal_flux_weight[1:-1, 1:-1, 1:-1]
        lower = axis_diagonal_flux_weight[tuple(lower_slice)]
        upper = axis_diagonal_flux_weight[tuple(upper_slice)]
        lower_coefficient = 0.5 * (center + lower) / (
            jacobian_center * axis_spacing * axis_spacing
        )
        upper_coefficient = 0.5 * (center + upper) / (
            jacobian_center * axis_spacing * axis_spacing
        )
        axis_lower_coefficients.append(lower_coefficient)
        axis_upper_coefficients.append(upper_coefficient)
        diagonal_contributions.append(lower_coefficient + upper_coefficient)

    lower_coefficient = axis_lower_coefficients[axis]
    upper_coefficient = axis_upper_coefficients[axis]
    full_diagonal = sum(diagonal_contributions)

    lower_diagonal = np.zeros_like(full_diagonal, dtype=np.float64)
    upper_diagonal = np.zeros_like(full_diagonal, dtype=np.float64)
    if axis == 0:
        lower_diagonal[1:, :, :] = -lower_coefficient[1:, :, :]
        upper_diagonal[:-1, :, :] = -upper_coefficient[:-1, :, :]
    elif axis == 1:
        lower_diagonal[:, 1:, :] = -lower_coefficient[:, 1:, :]
        upper_diagonal[:, :-1, :] = -upper_coefficient[:, :-1, :]
    else:
        lower_diagonal[:, :, 1:] = -lower_coefficient[:, :, 1:]
        upper_diagonal[:, :, :-1] = -upper_coefficient[:, :, :-1]

    diagonal_lines = np.moveaxis(full_diagonal, axis, -1)
    lower_lines = np.moveaxis(lower_diagonal, axis, -1)
    upper_lines = np.moveaxis(upper_diagonal, axis, -1)
    line_length = diagonal_lines.shape[-1]
    diagonal_lines = jnp.asarray(diagonal_lines, dtype=jnp.float64)
    lower_lines = jnp.asarray(lower_lines, dtype=jnp.float64)
    upper_lines = jnp.asarray(upper_lines, dtype=jnp.float64)
    interior_shape = tuple(int(length - 2) for length in grid_geometry.spec.shape)

    @jax.jit
    def _reorder_to_lines(interior_residual):
        field = jnp.reshape(interior_residual, interior_shape)
        return jnp.moveaxis(field, axis, -1)

    @jax.jit
    def _tridiagonal_only(rhs_lines):
        return lax_linalg.tridiagonal_solve(
            lower_lines,
            diagonal_lines,
            upper_lines,
            rhs_lines[..., None],
        )[..., 0]

    @jax.jit
    def _restore_from_lines(solved_lines):
        restored = jnp.moveaxis(solved_lines, -1, axis)
        return jnp.reshape(restored, (-1,))

    @jax.jit
    def _apply(interior_residual):
        rhs = _reorder_to_lines(interior_residual)
        solved = _tridiagonal_only(rhs)
        return _restore_from_lines(solved)

    @jax.jit
    def _axis_reorder_probe(interior_residual):
        rhs = _reorder_to_lines(interior_residual)
        return _restore_from_lines(rhs)

    setup_elapsed = perf_counter() - setup_start

    return MonitorPoissonJaxLinePreconditionerContext(
        axis=axis,
        axis_label=axis_label,
        line_length=int(line_length),
        average_line_diagonal_shift=float(np.mean(full_diagonal - (lower_coefficient + upper_coefficient))),
        setup_wall_time_seconds=float(setup_elapsed),
        apply=_apply,
        apply_probe=_apply,
        reorder_to_lines_probe=_reorder_to_lines,
        axis_reorder_probe=_axis_reorder_probe,
        tridiagonal_solve_probe=_tridiagonal_only,
    )


def _get_monitor_open_boundary_line_preconditioner_context_jax(
    *,
    grid_geometry: MonitorGridGeometry,
    use_cached_operator: bool,
) -> tuple[MonitorPoissonJaxLinePreconditionerContext, bool]:
    if not use_cached_operator:
        return (
            _build_monitor_open_boundary_line_preconditioner_context_jax(
                grid_geometry=grid_geometry,
            ),
            False,
        )

    cache_key = _build_monitor_poisson_jax_cache_key(grid_geometry)
    cached = _MONITOR_POISSON_JAX_LINE_PRECONDITIONER_CACHE.get(cache_key)
    if cached is not None:
        return cached, False

    cached = _build_monitor_open_boundary_line_preconditioner_context_jax(
        grid_geometry=grid_geometry,
    )
    _MONITOR_POISSON_JAX_LINE_PRECONDITIONER_CACHE[cache_key] = cached
    return cached, True


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
        inverse_preconditioner_diagonal = _build_monitor_open_boundary_inverse_preconditioner_diagonal_jax(
            grid_geometry=grid_geometry,
            gather=gather,
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
                inverse_preconditioner_diagonal=inverse_preconditioner_diagonal,
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
    inverse_preconditioner_diagonal = _build_monitor_open_boundary_inverse_preconditioner_diagonal_jax(
        grid_geometry=grid_geometry,
        gather=gather,
    )
    cached = MonitorPoissonJaxCachedSolveKernels(
        interior_mask=interior_mask,
        interior_size=interior_size,
        scatter=scatter,
        gather=gather,
        boundary_laplacian=boundary_laplacian,
        matvec=matvec,
        matvec_probe=matvec_probe,
        inverse_preconditioner_diagonal=inverse_preconditioner_diagonal,
    )
    _MONITOR_POISSON_JAX_KERNEL_CACHE[cache_key] = cached
    return cached, True


def _run_jax_cg(
    *,
    matvec,
    inverse_preconditioner_diagonal,
    separable_preconditioner_apply,
    line_preconditioner_apply,
    rhs,
    tolerance: float,
    max_iterations: int,
    cg_preconditioner: str,
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

    def _apply_preconditioner(residual):
        if cg_preconditioner == "diag":
            return inverse_preconditioner_diagonal * residual
        if cg_preconditioner == "separable":
            return separable_preconditioner_apply(residual)
        if cg_preconditioner == "line":
            return line_preconditioner_apply(residual)
        return residual

    rhs_values = jnp.asarray(rhs, dtype=jnp.float64)
    x = jnp.zeros_like(rhs_values)
    r = rhs_values - _timed_matvec(x)
    z = _apply_preconditioner(r)
    p = z
    rz = jnp.vdot(r, z)

    converged = False
    iteration_count = 0
    residual_max = float(jnp.max(jnp.abs(r)))
    for iteration in range(1, max_iterations + 1):
        ap = _timed_matvec(p)
        denominator = jnp.vdot(p, ap)
        if float(jnp.abs(denominator)) <= 1.0e-20:
            break
        alpha = rz / denominator
        x = x + alpha * p
        r = r - alpha * ap
        residual_max = float(jnp.max(jnp.abs(r)))
        iteration_count = iteration
        if residual_max < tolerance:
            converged = True
            break
        z = _apply_preconditioner(r)
        rz_new = jnp.vdot(r, z)
        if float(jnp.abs(rz)) <= 1.0e-30:
            break
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return np.asarray(x, dtype=np.float64), MonitorPoissonJaxSolveDiagnostics(
        solver_method="jax_cg_monitor",
        iteration_count=iteration_count,
        residual_max=residual_max,
        converged=converged,
        cg_impl="baseline",
        cg_preconditioner=cg_preconditioner,
        cg_other_overhead_wall_time_seconds=0.0,
        matvec_call_count=int(matvec_call_count),
        matvec_wall_time_seconds=float(matvec_wall_time_seconds),
        matvec_timing_is_estimated=False,
    )


def _build_jax_loop_cg_solver(
    *,
    matvec,
    inverse_preconditioner_diagonal,
    separable_preconditioner_apply,
    line_preconditioner_apply,
    tolerance: float,
    max_iterations: int,
    cg_preconditioner: str,
):
    jax = get_configured_jax()
    jnp = jax.numpy
    lax = jax.lax
    tolerance_value = float(tolerance)
    max_iterations_value = int(max_iterations)
    normalized_preconditioner = _normalize_monitor_poisson_jax_cg_preconditioner(cg_preconditioner)

    def _apply_preconditioner(residual):
        if normalized_preconditioner == "diag":
            return inverse_preconditioner_diagonal * residual
        if normalized_preconditioner == "separable":
            return separable_preconditioner_apply(residual)
        if normalized_preconditioner == "line":
            return line_preconditioner_apply(residual)
        return residual

    @jax.jit
    def _solve(rhs_values):
        rhs_values = jnp.asarray(rhs_values, dtype=jnp.float64)
        x0 = jnp.zeros_like(rhs_values)
        r0 = rhs_values - matvec(x0)
        z0 = _apply_preconditioner(r0)
        p0 = z0
        rz0 = jnp.vdot(r0, z0)
        residual0 = jnp.max(jnp.abs(r0))

        def _body(_, state):
            active, iteration_count, x, r, z, p, rz, residual_max = state

            def _inactive(_state):
                return _state

            def _active(_state):
                _, iteration_count, x, r, z, p, rz, residual_max = _state
                ap = matvec(p)
                denominator = jnp.vdot(p, ap)
                denominator_ok = jnp.abs(denominator) > 1.0e-20

                def _denominator_fail(__):
                    return (
                        jnp.asarray(False),
                        iteration_count,
                        x,
                        r,
                        z,
                        p,
                        rz,
                        residual_max,
                    )

                def _denominator_ok(__):
                    alpha = rz / denominator
                    x_next = x + alpha * p
                    r_next = r - alpha * ap
                    residual_next = jnp.max(jnp.abs(r_next))
                    iteration_next = iteration_count + 1
                    converged_next = residual_next < tolerance_value

                    def _converged(___):
                        return (
                            jnp.asarray(False),
                            iteration_next,
                            x_next,
                            r_next,
                            z,
                            p,
                            rz,
                            residual_next,
                        )

                    def _not_converged(___):
                        z_next = _apply_preconditioner(r_next)
                        rz_next = jnp.vdot(r_next, z_next)
                        rz_ok = jnp.abs(rz) > 1.0e-30

                        def _rr_fail(____):
                            return (
                                jnp.asarray(False),
                                iteration_next,
                                x_next,
                                r_next,
                                z_next,
                                p,
                                rz_next,
                                residual_next,
                            )

                        def _rr_ok(____):
                            beta = rz_next / rz
                            p_next = z_next + beta * p
                            still_active = iteration_next < max_iterations_value
                            return (
                                still_active,
                                iteration_next,
                                x_next,
                                r_next,
                                z_next,
                                p_next,
                                rz_next,
                                residual_next,
                            )

                        return lax.cond(rz_ok, _rr_ok, _rr_fail, operand=None)

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
                z0,
                p0,
                rz0,
                residual0,
            ),
        )
        active, iteration_count, x, r, _, _, _, residual_max = final_state
        converged = jnp.logical_and(
            jnp.logical_not(active),
            residual_max < tolerance_value,
        )
        return x, iteration_count, residual_max, converged

    return _solve


def _build_monitor_poisson_cg_loop_cache_key(
    grid_geometry: MonitorGridGeometry,
    *,
    tolerance: float,
    max_iterations: int,
    cg_preconditioner: str,
) -> tuple[int, tuple[int, int, int], int, int, str]:
    tolerance_key = int(round(float(tolerance) * 1.0e16))
    return (
        id(grid_geometry),
        tuple(grid_geometry.spec.shape),
        int(max_iterations),
        tolerance_key,
        cg_preconditioner,
    )


def _get_monitor_poisson_cg_loop_solver(
    *,
    grid_geometry: MonitorGridGeometry,
    matvec,
    inverse_preconditioner_diagonal,
    separable_preconditioner_apply,
    line_preconditioner_apply,
    tolerance: float,
    max_iterations: int,
    use_cached_operator: bool,
    cg_preconditioner: str,
):
    if not use_cached_operator:
        return _build_jax_loop_cg_solver(
            matvec=matvec,
            inverse_preconditioner_diagonal=inverse_preconditioner_diagonal,
            separable_preconditioner_apply=separable_preconditioner_apply,
            line_preconditioner_apply=line_preconditioner_apply,
            tolerance=tolerance,
            max_iterations=max_iterations,
            cg_preconditioner=cg_preconditioner,
        )

    cache_key = _build_monitor_poisson_cg_loop_cache_key(
        grid_geometry,
        tolerance=tolerance,
        max_iterations=max_iterations,
        cg_preconditioner=cg_preconditioner,
    )
    cached_solver = _MONITOR_POISSON_JAX_CG_LOOP_SOLVER_CACHE.get(cache_key)
    if cached_solver is not None:
        return cached_solver

    cached_solver = _build_jax_loop_cg_solver(
        matvec=matvec,
        inverse_preconditioner_diagonal=inverse_preconditioner_diagonal,
        separable_preconditioner_apply=separable_preconditioner_apply,
        line_preconditioner_apply=line_preconditioner_apply,
        tolerance=tolerance,
        max_iterations=max_iterations,
        cg_preconditioner=cg_preconditioner,
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


def _estimate_monitor_poisson_jax_preconditioner_apply_count(
    *,
    iteration_count: int,
    converged: bool,
    cg_preconditioner: str,
) -> int:
    if cg_preconditioner == "none":
        return 0
    if iteration_count <= 0:
        return 0
    return int(iteration_count if converged else iteration_count + 1)


def _time_jax_probe_call(*, probe, values) -> float:
    if probe is None:
        return 0.0
    warmup = probe(values)
    if hasattr(warmup, "block_until_ready"):
        warmup.block_until_ready()
    else:
        np.asarray(warmup, dtype=np.float64)
    start = perf_counter()
    output = probe(values)
    if hasattr(output, "block_until_ready"):
        output.block_until_ready()
    else:
        np.asarray(output, dtype=np.float64)
    return float(perf_counter() - start)


def _estimate_line_preconditioner_wall_times(
    *,
    line_context: MonitorPoissonJaxLinePreconditionerContext | None,
    interior_values: np.ndarray,
    apply_count: int,
) -> tuple[float, float, float, float]:
    if line_context is None or apply_count <= 0:
        return 0.0, 0.0, 0.0, 0.0

    jax = get_configured_jax()
    jnp = jax.numpy
    values = jnp.asarray(interior_values, dtype=jnp.float64)
    rhs_lines = line_context.reorder_to_lines_probe(values)
    if hasattr(rhs_lines, "block_until_ready"):
        rhs_lines.block_until_ready()
    else:
        np.asarray(rhs_lines, dtype=np.float64)
    rhs_lines = line_context.reorder_to_lines_probe(values)
    if hasattr(rhs_lines, "block_until_ready"):
        rhs_lines.block_until_ready()
    else:
        np.asarray(rhs_lines, dtype=np.float64)
    total_single = _time_jax_probe_call(probe=line_context.apply_probe, values=values)
    reorder_single = _time_jax_probe_call(probe=line_context.axis_reorder_probe, values=values)
    tridiagonal_single = _time_jax_probe_call(
        probe=line_context.tridiagonal_solve_probe,
        values=rhs_lines,
    )
    total = float(total_single * apply_count)
    reorder = float(reorder_single * apply_count)
    tridiagonal = float(tridiagonal_single * apply_count)
    other = max(0.0, total - reorder - tridiagonal)
    return total, reorder, tridiagonal, other


def _run_jax_cg_loop(
    *,
    grid_geometry: MonitorGridGeometry,
    matvec,
    matvec_probe,
    inverse_preconditioner_diagonal,
    separable_preconditioner_apply,
    line_preconditioner_context: MonitorPoissonJaxLinePreconditionerContext | None,
    line_preconditioner_apply,
    rhs,
    tolerance: float,
    max_iterations: int,
    use_cached_operator: bool,
    cg_preconditioner: str,
):
    solver = _get_monitor_poisson_cg_loop_solver(
        grid_geometry=grid_geometry,
        matvec=matvec,
        inverse_preconditioner_diagonal=inverse_preconditioner_diagonal,
        separable_preconditioner_apply=separable_preconditioner_apply,
        line_preconditioner_apply=line_preconditioner_apply,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_cached_operator=use_cached_operator,
        cg_preconditioner=cg_preconditioner,
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
    preconditioner_apply_count = _estimate_monitor_poisson_jax_preconditioner_apply_count(
        iteration_count=iteration_count_value,
        converged=bool(converged),
        cg_preconditioner=cg_preconditioner,
    )
    (
        estimated_preconditioner_apply_wall_time_seconds,
        estimated_preconditioner_axis_reorder_wall_time_seconds,
        estimated_preconditioner_tridiagonal_solve_wall_time_seconds,
        estimated_preconditioner_other_overhead_wall_time_seconds,
    ) = _estimate_line_preconditioner_wall_times(
        line_context=line_preconditioner_context,
        interior_values=np.asarray(rhs_values, dtype=np.float64),
        apply_count=preconditioner_apply_count,
    )
    return np.asarray(solution, dtype=np.float64), MonitorPoissonJaxSolveDiagnostics(
        solver_method="jax_cg_monitor",
        iteration_count=iteration_count_value,
        residual_max=float(residual_max),
        converged=bool(converged),
        cg_impl="jax_loop",
        cg_preconditioner=cg_preconditioner,
        cg_wall_time_seconds=float(solve_elapsed),
        cg_other_overhead_wall_time_seconds=max(
            0.0,
            float(solve_elapsed) - float(estimated_matvec_wall_time_seconds),
        ),
        matvec_call_count=matvec_call_count,
        matvec_wall_time_seconds=float(estimated_matvec_wall_time_seconds),
        matvec_timing_is_estimated=True,
        preconditioner_apply_count=int(preconditioner_apply_count),
        preconditioner_apply_wall_time_seconds=float(
            estimated_preconditioner_apply_wall_time_seconds
        ),
        preconditioner_axis_reorder_wall_time_seconds=float(
            estimated_preconditioner_axis_reorder_wall_time_seconds
        ),
        preconditioner_tridiagonal_solve_wall_time_seconds=float(
            estimated_preconditioner_tridiagonal_solve_wall_time_seconds
        ),
        preconditioner_other_overhead_wall_time_seconds=float(
            estimated_preconditioner_other_overhead_wall_time_seconds
        ),
        preconditioner_timing_is_estimated=bool(preconditioner_apply_count > 0),
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
    cg_preconditioner: str = "none",
) -> tuple[OpenBoundaryPoissonResult, MonitorPoissonJaxSolveDiagnostics]:
    """Solve the monitor-grid open-boundary Poisson problem with JAX CG."""

    global _LAST_MONITOR_POISSON_JAX_DIAGNOSTICS

    solve_start = perf_counter()
    density = validate_density_field(rho, grid_geometry=grid_geometry)
    normalized_cg_impl = _normalize_monitor_poisson_jax_cg_impl(cg_impl)
    normalized_cg_preconditioner = _normalize_monitor_poisson_jax_cg_preconditioner(
        cg_preconditioner
    )
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
    separable_preconditioner_apply = None
    if normalized_cg_preconditioner == "separable":
        separable_context, _ = _get_monitor_open_boundary_separable_preconditioner_context_jax(
            grid_geometry=grid_geometry,
            use_cached_operator=use_cached_operator,
        )
        separable_preconditioner_apply = separable_context.apply
    line_context = None
    line_context_built_this_solve = False
    line_preconditioner_apply = None
    if normalized_cg_preconditioner == "line":
        line_context, line_context_built_this_solve = _get_monitor_open_boundary_line_preconditioner_context_jax(
            grid_geometry=grid_geometry,
            use_cached_operator=use_cached_operator,
        )
        line_preconditioner_apply = line_context.apply
    build_elapsed = perf_counter() - build_start
    interior_mask = kernels.interior_mask
    scatter = kernels.scatter
    gather = kernels.gather
    boundary_laplacian = kernels.boundary_laplacian
    matvec = kernels.matvec
    matvec_probe = kernels.matvec_probe
    inverse_preconditioner_diagonal = kernels.inverse_preconditioner_diagonal
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
            inverse_preconditioner_diagonal=inverse_preconditioner_diagonal,
            separable_preconditioner_apply=separable_preconditioner_apply,
            line_preconditioner_context=line_context,
            line_preconditioner_apply=line_preconditioner_apply,
            rhs=rhs_interior,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_cached_operator=use_cached_operator,
            cg_preconditioner=normalized_cg_preconditioner,
        )
        cg_elapsed = diagnostics.cg_wall_time_seconds
    else:
        cg_start = perf_counter()
        interior_solution, diagnostics = _run_jax_cg(
            matvec=matvec,
            inverse_preconditioner_diagonal=inverse_preconditioner_diagonal,
            separable_preconditioner_apply=separable_preconditioner_apply,
            line_preconditioner_apply=line_preconditioner_apply,
            rhs=rhs_interior,
            tolerance=tolerance,
            max_iterations=max_iterations,
            cg_preconditioner=normalized_cg_preconditioner,
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
        cg_preconditioner=normalized_cg_preconditioner,
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
        preconditioner_apply_count=int(diagnostics.preconditioner_apply_count),
        preconditioner_setup_wall_time_seconds=(
            0.0
            if line_context is None or not line_context_built_this_solve
            else float(line_context.setup_wall_time_seconds)
        ),
        preconditioner_apply_wall_time_seconds=float(diagnostics.preconditioner_apply_wall_time_seconds),
        preconditioner_axis_reorder_wall_time_seconds=float(
            diagnostics.preconditioner_axis_reorder_wall_time_seconds
        ),
        preconditioner_tridiagonal_solve_wall_time_seconds=float(
            diagnostics.preconditioner_tridiagonal_solve_wall_time_seconds
        ),
        preconditioner_other_overhead_wall_time_seconds=float(
            diagnostics.preconditioner_other_overhead_wall_time_seconds
        ),
        preconditioner_timing_is_estimated=bool(diagnostics.preconditioner_timing_is_estimated),
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
