"""JAX reductions and weighted linear-algebra kernels for the scientific path."""

from __future__ import annotations

import numpy as np

from isogrid.config.runtime_jax import get_configured_jax
from isogrid.config.runtime_jax import get_configured_jax_numpy


def _as_weighted_array(values, *, name: str):
    jnp = get_configured_jax_numpy()
    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError(f"{name} must not be scalar.")
    return array


def weighted_inner_product_jax(left, right, weights):
    """Return the weighted inner product `<left|right>_W`."""

    jnp = get_configured_jax_numpy()
    left_array = _as_weighted_array(left, name="left")
    right_array = _as_weighted_array(right, name="right")
    weights_array = _as_weighted_array(weights, name="weights")
    if left_array.shape != right_array.shape or left_array.shape != weights_array.shape:
        raise ValueError("left, right, and weights must share the same shape.")
    return jnp.sum(jnp.conjugate(left_array) * right_array * weights_array)


def weighted_l2_norm_jax(field, weights):
    """Return the weighted L2 norm `sqrt(<field|field>_W)`."""

    jnp = get_configured_jax_numpy()
    values = _as_weighted_array(field, name="field")
    weights_array = _as_weighted_array(weights, name="weights")
    if values.shape != weights_array.shape:
        raise ValueError("field and weights must share the same shape.")
    norm_squared = jnp.sum(jnp.abs(values) ** 2 * weights_array)
    return jnp.sqrt(norm_squared)


def accumulate_density_from_orbitals_jax(
    orbitals,
    occupations=None,
):
    """Accumulate `rho = sum_i occ_i |psi_i|^2` from one orbital block."""

    jnp = get_configured_jax_numpy()
    block = jnp.asarray(orbitals)
    if block.ndim == 3:
        block = block[jnp.newaxis, ...]
    if block.ndim != 4:
        raise ValueError("orbitals must be one 3D orbital or a 4D orbital block.")
    if occupations is None:
        occupations_array = jnp.ones((block.shape[0],), dtype=block.dtype)
    else:
        occupations_array = jnp.asarray(occupations, dtype=block.real.dtype)
        if occupations_array.shape != (block.shape[0],):
            raise ValueError("occupations must match the orbital count.")
    return jnp.sum(occupations_array[:, None, None, None] * jnp.abs(block) ** 2, axis=0)


def flatten_orbital_block_jax(orbitals) -> np.ndarray:
    """Flatten a `(k, nx, ny, nz)` block into `(n_grid, k)` columns."""

    jnp = get_configured_jax_numpy()
    block = jnp.asarray(orbitals)
    if block.ndim == 3:
        block = block[jnp.newaxis, ...]
    if block.ndim != 4:
        raise ValueError("orbitals must be one 3D orbital or a 4D orbital block.")
    return block.reshape((block.shape[0], -1)).T


def reshape_orbital_columns_jax(orbital_columns, shape: tuple[int, int, int]):
    """Reshape `(n_grid, k)` columns back to `(k, nx, ny, nz)`."""

    jnp = get_configured_jax_numpy()
    columns = jnp.asarray(orbital_columns)
    if columns.ndim == 1:
        columns = columns[:, jnp.newaxis]
    expected = int(np.prod(shape))
    if columns.ndim != 2 or columns.shape[0] != expected:
        raise ValueError("orbital_columns must have shape (n_grid, k) for the requested shape.")
    return columns.T.reshape((columns.shape[1],) + shape)


def weighted_overlap_matrix_jax(orbitals, weights, other=None):
    """Return the weighted overlap/Gram matrix under the cell-volume metric."""

    jnp = get_configured_jax_numpy()
    left_columns = flatten_orbital_block_jax(orbitals)
    right_columns = left_columns if other is None else flatten_orbital_block_jax(other)
    weights_flat = jnp.asarray(weights).reshape(-1)
    if left_columns.shape[0] != weights_flat.shape[0]:
        raise ValueError("weights must match the orbital grid size.")
    return jnp.conjugate(left_columns).T @ (weights_flat[:, None] * right_columns)


def weighted_gram_matrix_jax(orbitals, weights):
    """Alias for the weighted overlap matrix used by block methods."""

    return weighted_overlap_matrix_jax(orbitals, weights)


def weighted_orthonormalize_orbitals_jax(
    orbitals,
    weights,
    *,
    rank_tolerance: float = 1.0e-12,
    require_full_rank: bool = True,
):
    """Weighted-orthonormalize one orbital block using an overlap eigensolve."""

    jnp = get_configured_jax_numpy()
    block = jnp.asarray(orbitals)
    if block.ndim == 3:
        block = block[jnp.newaxis, ...]
    if block.ndim != 4:
        raise ValueError("orbitals must be one 3D orbital or a 4D orbital block.")
    if block.shape[0] == 0:
        return block

    overlap = weighted_overlap_matrix_jax(block, weights)
    evals, evecs = jnp.linalg.eigh(overlap)
    evals_np = np.asarray(evals, dtype=np.float64)
    max_eval = float(np.max(evals_np))
    cutoff = rank_tolerance * max(block.shape[0], 1) * max(max_eval, 1.0)
    keep_mask = evals_np > cutoff
    rank = int(np.count_nonzero(keep_mask))
    if rank == 0:
        if require_full_rank:
            raise ValueError("The orbital block is numerically rank-deficient.")
        return jnp.zeros((0,) + tuple(block.shape[1:]), dtype=block.dtype)
    if require_full_rank and rank < block.shape[0]:
        raise ValueError(
            "The orbital block is rank-deficient under the weighted metric: "
            f"rank={rank}, requested={block.shape[0]}."
        )

    kept_evals = evals[keep_mask]
    kept_evecs = evecs[:, keep_mask]
    inverse_sqrt = 1.0 / jnp.sqrt(kept_evals)
    columns = flatten_orbital_block_jax(block)
    orthonormal_columns = columns @ (kept_evecs * inverse_sqrt[None, :])
    return reshape_orbital_columns_jax(orthonormal_columns, tuple(block.shape[1:]))


def build_weighted_overlap_kernel_jax(weights):
    """Return a small jitted weighted overlap kernel for one fixed weight field."""

    jax = get_configured_jax()
    jnp = jax.numpy
    weights_flat = jnp.asarray(weights).reshape(-1)

    @jax.jit
    def _kernel(left_columns, right_columns):
        return jnp.conjugate(left_columns).T @ (weights_flat[:, None] * right_columns)

    return _kernel


__all__ = [
    "accumulate_density_from_orbitals_jax",
    "build_weighted_overlap_kernel_jax",
    "flatten_orbital_block_jax",
    "reshape_orbital_columns_jax",
    "weighted_gram_matrix_jax",
    "weighted_inner_product_jax",
    "weighted_l2_norm_jax",
    "weighted_orthonormalize_orbitals_jax",
    "weighted_overlap_matrix_jax",
]
