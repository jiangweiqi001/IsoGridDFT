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
    internal_profile: JaxFixedPotentialInternalProfile | None = None


@dataclass(frozen=True)
class JaxFixedPotentialInternalProfile:
    """Very small in-loop timing profile for the JAX-native eigensolver."""

    subspace_iteration_wall_time_seconds: float
    orthogonalization_wall_time_seconds: float
    residual_expansion_wall_time_seconds: float
    rayleigh_ritz_wall_time_seconds: float
    hamiltonian_apply_wall_time_seconds: float
    projected_matrix_build_wall_time_seconds: float


def solve_fixed_potential_static_local_eigenproblem_jax(
    *,
    operator_context,
    k: int,
    initial_guess_orbitals: np.ndarray | None,
    max_iterations: int,
    tolerance: float,
    initial_subspace_size: int | None,
    ncv: int | None,
    profile_internals: bool = False,
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
    # Keep a materially larger working subspace for small near-degenerate blocks.
    # For k=2 on the current H2 local-only route, a tiny basis tends to collapse
    # the second Ritz vector; we therefore reinterpret `ncv` as a minimum working
    # subspace size and enforce a larger floor for multi-state solves.
    block_size = max(
        2 * k + 2,
        4,
        12 * k + 8,
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

    def _project_out_weighted(columns, against):
        projection = _weighted_overlap_columns(against, columns)
        return columns - against @ projection

    def _build_trial_projected_from_parts(
        basis_projected,
        basis_columns,
        basis_action_columns,
        residual_columns,
        residual_action_columns,
    ):
        basis_residual = _weighted_overlap_columns(basis_columns, residual_action_columns)
        residual_basis = _weighted_overlap_columns(residual_columns, basis_action_columns)
        residual_residual = _weighted_overlap_columns(residual_columns, residual_action_columns)
        top = jnp.concatenate([basis_projected, basis_residual], axis=1)
        bottom = jnp.concatenate([residual_basis, residual_residual], axis=1)
        trial_projected = jnp.concatenate([top, bottom], axis=0)
        return 0.5 * (trial_projected + jnp.conjugate(trial_projected).T)

    def _compress_basis_from_parts(basis_columns, residual_columns, trial_vectors):
        basis_size = basis_columns.shape[1]
        basis_coefficients = trial_vectors[:basis_size, :basis_size]
        residual_coefficients = trial_vectors[basis_size:, :basis_size]
        return basis_columns @ basis_coefficients + residual_columns @ residual_coefficients

    def _block_until_ready_tree(value):
        leaves = jax.tree_util.tree_leaves(value)
        for leaf in leaves:
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        return value

    def _timed_call(bucket: str, fn, *args):
        start = time.perf_counter()
        value = fn(*args)
        _block_until_ready_tree(value)
        elapsed = time.perf_counter() - start
        profile_totals[bucket] += elapsed
        return value

    @jax.jit
    def _weighted_qr_columns_jit(columns):
        return _weighted_qr_columns(columns)

    @jax.jit
    def _apply_block_columns_jit(columns):
        block = _columns_to_block(columns)
        return flatten_orbital_block_jax(block_apply(block))

    @jax.jit
    def _build_projected_matrix_jit(columns, action_columns):
        projected_matrix = _weighted_overlap_columns(columns, action_columns)
        return 0.5 * (projected_matrix + jnp.conjugate(projected_matrix).T)

    @jax.jit
    def _rayleigh_ritz_jit(projected_matrix):
        subspace_eigenvalues, subspace_vectors = jnp.linalg.eigh(projected_matrix)
        order = jnp.argsort(subspace_eigenvalues)
        return subspace_eigenvalues[order], subspace_vectors[:, order]

    @jax.jit
    def _compute_residual_data_jit(
        basis_columns,
        action_columns,
        subspace_eigenvalues,
        subspace_vectors,
    ):
        coeffs = subspace_vectors[:, :k]
        orbital_columns = basis_columns @ coeffs
        orbital_action_columns = action_columns @ coeffs
        residual_columns = (
            orbital_action_columns
            - orbital_columns * subspace_eigenvalues[None, :k]
        )
        residual_norms = _weighted_column_norms(residual_columns)
        return orbital_columns, residual_columns, residual_norms

    @jax.jit
    def _project_residual_columns_jit(residual_columns, basis_columns):
        return _project_out_weighted(residual_columns, basis_columns)

    @jax.jit
    def _normalize_residual_columns_jit(residual_columns):
        return _weighted_qr_columns(residual_columns)

    @jax.jit
    def _build_trial_projected_from_parts_jit(
        basis_projected,
        basis_columns,
        basis_action_columns,
        residual_columns,
        residual_action_columns,
    ):
        return _build_trial_projected_from_parts(
            basis_projected,
            basis_columns,
            basis_action_columns,
            residual_columns,
            residual_action_columns,
        )

    def _run_profiled(initial_block):
        initial_columns = flatten_orbital_block_jax(initial_block)
        _block_until_ready_tree(initial_columns)
        basis_columns = _timed_call(
            "orthogonalization",
            _weighted_qr_columns_jit,
            initial_columns,
        )
        residual_history = np.zeros((max_iterations, k), dtype=np.float64)
        ritz_value_history = np.zeros((max_iterations, k), dtype=np.float64)
        subspace_dimensions = np.zeros((max_iterations,), dtype=np.int32)
        basis_size = int(basis_columns.shape[1])

        @jax.jit
        def _compress_basis_jit_local(basis_columns, residual_columns, trial_vectors):
            next_basis_columns = _compress_basis_from_parts(
                basis_columns,
                residual_columns,
                trial_vectors,
            )
            return _weighted_qr_columns(next_basis_columns)

        orbital_columns = basis_columns[:, :k]
        residual_norms = np.full((k,), np.inf, dtype=np.float64)
        eigenvalues = np.zeros((k,), dtype=np.float64)
        projected_matrix = np.zeros((k, k), dtype=np.float64)
        loop_start = time.perf_counter()
        converged = False
        iteration_count_int = 0

        for iteration in range(max_iterations):
            action_columns = _timed_call(
                "hamiltonian_apply",
                _apply_block_columns_jit,
                basis_columns,
            )
            projected_matrix_jax = _timed_call(
                "projected_matrix_build",
                _build_projected_matrix_jit,
                basis_columns,
                action_columns,
            )
            subspace_eigenvalues, subspace_vectors = _timed_call(
                "rayleigh_ritz",
                _rayleigh_ritz_jit,
                projected_matrix_jax,
            )
            orbital_columns, residual_columns, residual_norms_jax = _timed_call(
                "residual_expansion",
                _compute_residual_data_jit,
                basis_columns,
                action_columns,
                subspace_eigenvalues[:k],
                subspace_vectors[:, :k],
            )
            residual_norms = np.asarray(residual_norms_jax, dtype=np.float64)
            eigenvalues = np.asarray(subspace_eigenvalues[:k], dtype=np.float64)
            residual_history[iteration] = residual_norms
            ritz_value_history[iteration] = eigenvalues
            subspace_dimensions[iteration] = basis_size
            projected_matrix = np.asarray(projected_matrix_jax[:k, :k], dtype=np.float64)
            iteration_count_int = iteration + 1
            if float(np.max(residual_norms)) <= tolerance:
                converged = True
                break

            residual_columns = _timed_call(
                "residual_expansion",
                _project_residual_columns_jit,
                residual_columns,
                basis_columns,
            )
            residual_columns = _timed_call(
                "orthogonalization",
                _normalize_residual_columns_jit,
                residual_columns,
            )
            residual_action_columns = _timed_call(
                "hamiltonian_apply",
                _apply_block_columns_jit,
                residual_columns,
            )
            trial_projected_matrix = _timed_call(
                "projected_matrix_build",
                _build_trial_projected_from_parts_jit,
                projected_matrix_jax,
                basis_columns,
                action_columns,
                residual_columns,
                residual_action_columns,
            )
            trial_eigenvalues, trial_vectors = _timed_call(
                "rayleigh_ritz",
                _rayleigh_ritz_jit,
                trial_projected_matrix,
            )
            del trial_eigenvalues
            basis_columns = _timed_call(
                "orthogonalization",
                _compress_basis_jit_local,
                basis_columns,
                residual_columns,
                trial_vectors,
            )

        subspace_iteration_wall_time_seconds = time.perf_counter() - loop_start
        return (
            iteration_count_int,
            converged,
            eigenvalues,
            residual_norms,
            residual_history[:iteration_count_int],
            ritz_value_history[:iteration_count_int],
            tuple(int(value) for value in subspace_dimensions[:iteration_count_int]),
            np.asarray(_columns_to_block(orbital_columns), dtype=np.float64),
            np.asarray(_columns_to_block(basis_columns), dtype=np.float64),
            projected_matrix,
            JaxFixedPotentialInternalProfile(
                subspace_iteration_wall_time_seconds=float(subspace_iteration_wall_time_seconds),
                orthogonalization_wall_time_seconds=float(profile_totals["orthogonalization"]),
                residual_expansion_wall_time_seconds=float(profile_totals["residual_expansion"]),
                rayleigh_ritz_wall_time_seconds=float(profile_totals["rayleigh_ritz"]),
                hamiltonian_apply_wall_time_seconds=float(profile_totals["hamiltonian_apply"]),
                projected_matrix_build_wall_time_seconds=float(
                    profile_totals["projected_matrix_build"]
                ),
            ),
        )

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

            residual_columns = _project_out_weighted(residual_columns, basis_columns)
            residual_columns = _weighted_qr_columns(residual_columns)
            residual_action_block = block_apply(_columns_to_block(residual_columns))
            residual_action_columns = flatten_orbital_block_jax(residual_action_block)

            # Reuse H@B from the current Ritz step and only apply H to the new
            # residual block before the Rayleigh-Ritz restart.
            trial_projected = _build_trial_projected_from_parts(
                projected_matrix,
                basis_columns,
                action_columns,
                residual_columns,
                residual_action_columns,
            )
            trial_eigenvalues, trial_vectors = jnp.linalg.eigh(trial_projected)
            trial_order = jnp.argsort(trial_eigenvalues)
            trial_vectors = trial_vectors[:, trial_order]
            next_basis_columns = _compress_basis_from_parts(
                basis_columns,
                residual_columns,
                trial_vectors,
            )
            next_basis_columns = _weighted_qr_columns(next_basis_columns)
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

    profile_totals = {
        "orthogonalization": 0.0,
        "residual_expansion": 0.0,
        "rayleigh_ritz": 0.0,
        "hamiltonian_apply": 0.0,
        "projected_matrix_build": 0.0,
    }
    solve_start = time.perf_counter()
    internal_profile = None
    if profile_internals:
        (
            iteration_count_int,
            converged,
            eigenvalues,
            residual_norms,
            residual_history_np,
            ritz_value_history_np,
            subspace_dimensions_tuple,
            orbitals,
            final_basis_orbitals,
            ritz_matrix_np,
            internal_profile,
        ) = _run_profiled(jnp.asarray(initial_guess, dtype=jnp.float64))
        weighted_overlap = np.asarray(
            _weighted_overlap_columns(
                flatten_orbital_block_jax(jnp.asarray(orbitals, dtype=jnp.float64)),
                flatten_orbital_block_jax(jnp.asarray(orbitals, dtype=jnp.float64)),
            ),
            dtype=np.float64,
        )
    else:
        final_state = _run(jnp.asarray(initial_guess, dtype=jnp.float64))
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
        residual_history_np = np.asarray(
            residual_history[:iteration_count_int],
            dtype=np.float64,
        )
        ritz_value_history_np = np.asarray(
            ritz_value_history[:iteration_count_int],
            dtype=np.float64,
        )
        subspace_dimensions_tuple = tuple(
            int(value) for value in np.asarray(subspace_dimensions[:iteration_count_int], dtype=np.int32)
        )
        ritz_matrix_np = np.asarray(ritz_matrix, dtype=np.float64)
    wall_time_seconds = time.perf_counter() - solve_start

    max_orthogonality_error = float(
        np.max(np.abs(weighted_overlap - np.eye(k, dtype=np.float64)))
    )

    return JaxFixedPotentialSubspaceIterationResult(
        target_orbitals=k,
        solver_method="jax_block_subspace_iteration",
        solver_note=(
            "JAX-native fixed-size block subspace iteration on the cached local-only block Hamiltonian. "
            "Each step expands the current basis with weighted-orthogonalized residual directions and "
            "compresses the enlarged trial space back through a Rayleigh-Ritz restart in JAX. "
            "The outer eigensolver loop, orthogonalization, projected solve, and residual update stay in JAX, "
            "and `ncv` is reinterpreted as a minimum working subspace size on this route."
        ),
        converged=bool(converged),
        iteration_count=iteration_count_int,
        tolerance=float(tolerance),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        orbitals=orbitals,
        weighted_overlap=weighted_overlap,
        max_orthogonality_error=max_orthogonality_error,
        residual_norms=np.asarray(residual_norms, dtype=np.float64),
        residual_history=residual_history_np,
        ritz_value_history=ritz_value_history_np,
        subspace_dimensions=subspace_dimensions_tuple,
        ritz_matrix=ritz_matrix_np,
        initial_guess_orbitals=np.asarray(initial_guess, dtype=np.float64),
        final_basis_orbitals=final_basis_orbitals,
        wall_time_seconds=float(wall_time_seconds),
        internal_profile=internal_profile,
    )


__all__ = [
    "JaxFixedPotentialInternalProfile",
    "JaxFixedPotentialSubspaceIterationResult",
    "solve_fixed_potential_static_local_eigenproblem_jax",
]
