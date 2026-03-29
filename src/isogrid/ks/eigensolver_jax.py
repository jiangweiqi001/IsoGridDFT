"""JAX-native fixed-potential eigensolver helpers for the A-grid local-only path."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from isogrid.config.runtime_jax import get_configured_jax
from isogrid.ks.eigensolver_jax_cache import get_fixed_potential_static_local_block_kernel_cached
from isogrid.ops.reductions_jax import flatten_orbital_block_jax
from isogrid.ops.reductions_jax import reshape_orbital_columns_jax


@dataclass(frozen=True)
class JaxFixedPotentialSubspaceIterationResult:
    """Resolved JAX-native fixed-potential eigensolver result."""

    target_orbitals: int
    solver_method: str
    solver_note: str
    converged: bool
    iteration_count: int
    tolerance: float
    eigenvalues: np.ndarray
    orbitals: np.ndarray
    weighted_overlap: np.ndarray
    max_orthogonality_error: float
    residual_norms: np.ndarray
    residual_history: np.ndarray
    ritz_value_history: np.ndarray
    subspace_dimensions: tuple[int, ...]
    ritz_matrix: np.ndarray
    initial_guess_orbitals: np.ndarray
    final_basis_orbitals: np.ndarray
    wall_time_seconds: float


def solve_fixed_potential_static_local_eigenproblem_jax(
    *,
    operator_context,
    k: int,
    initial_guess_orbitals: np.ndarray | None,
    max_iterations: int,
    tolerance: float,
    initial_subspace_size: int | None,
    ncv: int | None,
) -> JaxFixedPotentialSubspaceIterationResult:
    """Solve the local-only fixed-potential problem with a JAX-native block iteration."""

    if k <= 0:
        raise ValueError(f"k must be positive; received {k}.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive; received {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive; received {tolerance}.")

    from isogrid.ks.eigensolver import _build_initial_guess_block
    from isogrid.ks.eigensolver import validate_orbital_block

    grid_geometry = operator_context.grid_geometry
    block_size = max(
        2 * k + 2,
        4,
        0 if initial_subspace_size is None else int(initial_subspace_size),
        0 if ncv is None else int(ncv),
    )
    initial_guess = _build_initial_guess_block(
        operator_context=operator_context,
        k=k,
        basis_size=block_size,
        initial_guess_orbitals=initial_guess_orbitals,
        use_jax_block_kernels=True,
    )
    initial_guess = validate_orbital_block(initial_guess, grid_geometry=grid_geometry)[:block_size]

    jax = get_configured_jax()
    jnp = jax.numpy
    weights_flat = jnp.asarray(grid_geometry.cell_volumes.reshape(-1), dtype=jnp.float64)
    sqrt_weights = jnp.sqrt(weights_flat)
    inverse_sqrt_weights = 1.0 / sqrt_weights
    block_shape = tuple(initial_guess.shape[1:])
    block_apply = get_fixed_potential_static_local_block_kernel_cached(operator_context)

    def _weighted_overlap_columns(left_columns, right_columns):
        return jnp.conjugate(left_columns).T @ (weights_flat[:, None] * right_columns)

    def _weighted_qr_columns(columns):
        weighted_columns = sqrt_weights[:, None] * columns
        q_weighted, _ = jnp.linalg.qr(weighted_columns, mode="reduced")
        return inverse_sqrt_weights[:, None] * q_weighted

    def _columns_to_block(columns):
        return reshape_orbital_columns_jax(columns, block_shape)

    def _weighted_column_norms(columns):
        norms_squared = jnp.sum(jnp.abs(columns) ** 2 * weights_flat[:, None], axis=0)
        return jnp.sqrt(jnp.clip(norms_squared, 0.0, None))

    @jax.jit
    def _run(initial_block):
        initial_columns = flatten_orbital_block_jax(initial_block)
        initial_basis = _weighted_qr_columns(initial_columns)
        residual_history = jnp.zeros((max_iterations, k), dtype=jnp.float64)
        ritz_value_history = jnp.zeros((max_iterations, k), dtype=jnp.float64)
        subspace_dimensions = jnp.zeros((max_iterations,), dtype=jnp.int32)
        initial_eigenvalues = jnp.zeros((k,), dtype=jnp.float64)
        initial_residuals = jnp.full((k,), jnp.inf, dtype=jnp.float64)
        initial_orbitals = initial_basis[:, :k]
        initial_projected = jnp.zeros((k, k), dtype=jnp.float64)

        state = (
            jnp.int32(0),
            initial_basis,
            jnp.bool_(False),
            initial_eigenvalues,
            initial_residuals,
            residual_history,
            ritz_value_history,
            subspace_dimensions,
            initial_orbitals,
            initial_basis,
            initial_projected,
        )

        def _cond(loop_state):
            iteration, _, converged, *_ = loop_state
            return jnp.logical_and(iteration < max_iterations, jnp.logical_not(converged))

        def _body(loop_state):
            (
                iteration,
                basis_columns,
                _converged,
                _last_eigenvalues,
                _last_residuals,
                residual_history,
                ritz_value_history,
                subspace_dimensions,
                _last_orbital_columns,
                _last_basis_columns,
                _last_projected_matrix,
            ) = loop_state

            basis_columns = _weighted_qr_columns(basis_columns)
            basis_block = _columns_to_block(basis_columns)
            action_block = block_apply(basis_block)
            action_columns = flatten_orbital_block_jax(action_block)
            projected_matrix = _weighted_overlap_columns(basis_columns, action_columns)
            projected_matrix = 0.5 * (projected_matrix + jnp.conjugate(projected_matrix).T)
            subspace_eigenvalues, subspace_vectors = jnp.linalg.eigh(projected_matrix)
            order = jnp.argsort(subspace_eigenvalues)
            subspace_eigenvalues = subspace_eigenvalues[order]
            subspace_vectors = subspace_vectors[:, order]
            coeffs = subspace_vectors[:, :k]
            orbital_columns = basis_columns @ coeffs
            orbital_action_columns = action_columns @ coeffs
            residual_columns = orbital_action_columns - orbital_columns * subspace_eigenvalues[None, :k]
            residual_norms = _weighted_column_norms(residual_columns)
            converged = jnp.max(residual_norms) <= tolerance

            residual_history = residual_history.at[iteration].set(residual_norms)
            ritz_value_history = ritz_value_history.at[iteration].set(subspace_eigenvalues[:k])
            subspace_dimensions = subspace_dimensions.at[iteration].set(basis_columns.shape[1])

            rotated_columns = basis_columns @ subspace_vectors
            candidate_columns = jnp.concatenate(
                [
                    orbital_columns,
                    residual_columns,
                    rotated_columns[:, k:],
                ],
                axis=1,
            )[:, : basis_columns.shape[1]]
            next_basis_columns = _weighted_qr_columns(candidate_columns)
            basis_columns = jax.lax.select(converged, basis_columns, next_basis_columns)

            return (
                iteration + jnp.int32(1),
                basis_columns,
                converged,
                subspace_eigenvalues[:k],
                residual_norms,
                residual_history,
                ritz_value_history,
                subspace_dimensions,
                orbital_columns,
                basis_columns,
                projected_matrix[:k, :k],
            )

        return jax.lax.while_loop(_cond, _body, state)

    solve_start = time.perf_counter()
    final_state = _run(jnp.asarray(initial_guess, dtype=jnp.float64))
    wall_time_seconds = time.perf_counter() - solve_start

    (
        iteration_count,
        _basis_columns,
        converged,
        eigenvalues,
        residual_norms,
        residual_history,
        ritz_value_history,
        subspace_dimensions,
        orbital_columns,
        final_basis_columns,
        ritz_matrix,
    ) = final_state
    jax.block_until_ready(eigenvalues)

    iteration_count_int = int(iteration_count)
    orbitals = np.asarray(_columns_to_block(orbital_columns), dtype=np.float64)
    final_basis_orbitals = np.asarray(_columns_to_block(final_basis_columns), dtype=np.float64)
    weighted_overlap = np.asarray(
        _weighted_overlap_columns(orbital_columns, orbital_columns),
        dtype=np.float64,
    )
    max_orthogonality_error = float(
        np.max(np.abs(weighted_overlap - np.eye(k, dtype=np.float64)))
    )

    return JaxFixedPotentialSubspaceIterationResult(
        target_orbitals=k,
        solver_method="jax_block_subspace_iteration",
        solver_note=(
            "JAX-native fixed-size block subspace iteration on the cached local-only block Hamiltonian. "
            "The outer eigensolver loop, orthogonalization, projected solve, and residual update stay in JAX, "
            "and `ncv` is reinterpreted as the fixed subspace size on this route."
        ),
        converged=bool(converged),
        iteration_count=iteration_count_int,
        tolerance=float(tolerance),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        orbitals=orbitals,
        weighted_overlap=weighted_overlap,
        max_orthogonality_error=max_orthogonality_error,
        residual_norms=np.asarray(residual_norms, dtype=np.float64),
        residual_history=np.asarray(residual_history[:iteration_count_int], dtype=np.float64),
        ritz_value_history=np.asarray(ritz_value_history[:iteration_count_int], dtype=np.float64),
        subspace_dimensions=tuple(
            int(value) for value in np.asarray(subspace_dimensions[:iteration_count_int], dtype=np.int32)
        ),
        ritz_matrix=np.asarray(ritz_matrix, dtype=np.float64),
        initial_guess_orbitals=np.asarray(initial_guess, dtype=np.float64),
        final_basis_orbitals=final_basis_orbitals,
        wall_time_seconds=float(wall_time_seconds),
    )


__all__ = [
    "JaxFixedPotentialSubspaceIterationResult",
    "solve_fixed_potential_static_local_eigenproblem_jax",
]
