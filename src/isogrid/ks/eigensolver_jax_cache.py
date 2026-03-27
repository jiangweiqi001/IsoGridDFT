"""Very small cache/reuse helpers for the JAX eigensolver hot path."""

from __future__ import annotations

from typing import Callable

import numpy as np

from isogrid.config.runtime_jax import get_configured_jax
from isogrid.grid import MonitorGridGeometry
from isogrid.ops.kinetic_jax import build_monitor_grid_kinetic_operator_jax
from isogrid.ops.reductions_jax import build_weighted_overlap_kernel_jax
from isogrid.ops.reductions_jax import flatten_orbital_block_jax
from isogrid.ops.reductions_jax import reshape_orbital_columns_jax

_BLOCK_OPERATOR_CACHE: dict[tuple[object, ...], Callable] = {}
_WEIGHTED_OVERLAP_CACHE: dict[tuple[object, ...], Callable] = {}


def _require_monitor_geometry(grid_geometry) -> MonitorGridGeometry:
    if not isinstance(grid_geometry, MonitorGridGeometry):
        raise TypeError("The JAX eigensolver hot-path cache currently supports only MonitorGridGeometry.")
    return grid_geometry


def _block_operator_cache_key(operator_context) -> tuple[object, ...]:
    grid_geometry = _require_monitor_geometry(operator_context.grid_geometry)
    potential = np.asarray(operator_context.effective_local_potential)
    return (
        id(grid_geometry),
        id(operator_context.effective_local_potential),
        potential.shape,
        str(potential.dtype),
        operator_context.kinetic_version,
    )


def get_fixed_potential_static_local_block_kernel_cached(operator_context):
    """Return one cached compiled block-Hamiltonian callable for the current context."""

    key = _block_operator_cache_key(operator_context)
    kernel = _BLOCK_OPERATOR_CACHE.get(key)
    if kernel is not None:
        return kernel

    grid_geometry = _require_monitor_geometry(operator_context.grid_geometry)
    jax = get_configured_jax()
    jnp = jax.numpy
    kinetic = build_monitor_grid_kinetic_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=(operator_context.kinetic_version == "trial_fix"),
    )
    effective_local_potential = jnp.asarray(
        operator_context.effective_local_potential,
        dtype=jnp.float64,
    )

    def _apply_one(field):
        return kinetic(field) + effective_local_potential * field

    kernel = jax.jit(jax.vmap(_apply_one))
    _BLOCK_OPERATOR_CACHE[key] = kernel
    return kernel


def apply_fixed_potential_static_local_block_cached_jax(
    orbitals,
    *,
    operator_context,
):
    """Apply the cached JAX block-Hamiltonian callable to one orbital block."""

    grid_geometry = _require_monitor_geometry(operator_context.grid_geometry)
    block = np.asarray(orbitals)
    if block.ndim == 3:
        block = block[np.newaxis, ...]
    if block.ndim != 4 or tuple(block.shape[1:]) != grid_geometry.spec.shape:
        raise ValueError(
            "orbitals must be one 3D orbital or a 4D orbital block compatible with the monitor-grid shape."
        )
    kernel = get_fixed_potential_static_local_block_kernel_cached(operator_context)
    jax = get_configured_jax()
    return kernel(jax.numpy.asarray(block, dtype=jax.numpy.float64))


def _weighted_overlap_cache_key(weights) -> tuple[object, ...]:
    weights_array = np.asarray(weights)
    return (
        id(weights),
        weights_array.shape,
        str(weights_array.dtype),
    )


def get_weighted_overlap_kernel_cached(weights):
    """Return one cached compiled weighted-overlap kernel for a fixed weight field."""

    key = _weighted_overlap_cache_key(weights)
    kernel = _WEIGHTED_OVERLAP_CACHE.get(key)
    if kernel is not None:
        return kernel
    kernel = build_weighted_overlap_kernel_jax(weights)
    _WEIGHTED_OVERLAP_CACHE[key] = kernel
    return kernel


def weighted_overlap_matrix_cached_jax(orbitals, weights, other=None):
    """Return the cached weighted overlap/Gram matrix for one fixed weight field."""

    left_columns = flatten_orbital_block_jax(orbitals)
    right_columns = left_columns if other is None else flatten_orbital_block_jax(other)
    kernel = get_weighted_overlap_kernel_cached(weights)
    return kernel(left_columns, right_columns)


def weighted_orthonormalize_orbitals_cached_jax(
    orbitals,
    weights,
    *,
    rank_tolerance: float = 1.0e-12,
    require_full_rank: bool = True,
):
    """Weighted-orthonormalize one block while reusing the cached overlap kernel."""

    jax = get_configured_jax()
    jnp = jax.numpy
    block = jnp.asarray(orbitals)
    if block.ndim == 3:
        block = block[jnp.newaxis, ...]
    if block.ndim != 4:
        raise ValueError("orbitals must be one 3D orbital or a 4D orbital block.")
    if block.shape[0] == 0:
        return block

    overlap = weighted_overlap_matrix_cached_jax(block, weights)
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


__all__ = [
    "apply_fixed_potential_static_local_block_cached_jax",
    "get_fixed_potential_static_local_block_kernel_cached",
    "get_weighted_overlap_kernel_cached",
    "weighted_overlap_matrix_cached_jax",
    "weighted_orthonormalize_orbitals_cached_jax",
]
