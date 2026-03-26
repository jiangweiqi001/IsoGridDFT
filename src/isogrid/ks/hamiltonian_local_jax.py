"""JAX local-only Hamiltonian apply kernels for the stable A-grid hot path."""

from __future__ import annotations

import numpy as np

from isogrid.config.runtime_jax import get_configured_jax
from isogrid.grid import MonitorGridGeometry
from isogrid.ks.eigensolver import FixedPotentialStaticLocalOperatorContext
from isogrid.ks.eigensolver import validate_orbital_block
from isogrid.ops.kinetic import validate_orbital_field
from isogrid.ops.kinetic_jax import build_monitor_grid_kinetic_operator_jax


def _require_monitor_static_local_context(
    operator_context: FixedPotentialStaticLocalOperatorContext,
) -> MonitorGridGeometry:
    grid_geometry = operator_context.grid_geometry
    if not isinstance(grid_geometry, MonitorGridGeometry):
        raise TypeError(
            "The first JAX local-Hamiltonian migration supports only the monitor-grid "
            "static-local operator context."
        )
    return grid_geometry


def apply_fixed_potential_static_local_operator_jax(
    psi: np.ndarray,
    *,
    operator_context: FixedPotentialStaticLocalOperatorContext,
):
    """Apply the frozen local-only Hamiltonian with JAX on the monitor grid."""

    grid_geometry = _require_monitor_static_local_context(operator_context)
    field = validate_orbital_field(psi, grid_geometry=grid_geometry, name="psi")
    jax = get_configured_jax()
    jnp = jax.numpy
    field_jax = jnp.asarray(field, dtype=jnp.float64)
    effective_local_potential = jnp.asarray(
        operator_context.effective_local_potential,
        dtype=jnp.float64,
    )
    kinetic = build_monitor_grid_kinetic_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=(operator_context.kinetic_version == "trial_fix"),
    )
    kinetic_action = kinetic(field_jax)
    return kinetic_action + effective_local_potential * field_jax


def apply_fixed_potential_static_local_block_jax(
    orbitals: np.ndarray,
    *,
    operator_context: FixedPotentialStaticLocalOperatorContext,
):
    """Apply the frozen local-only Hamiltonian to one orbital block with JAX."""

    _require_monitor_static_local_context(operator_context)
    block = validate_orbital_block(orbitals, grid_geometry=operator_context.grid_geometry)
    jax = get_configured_jax()
    jnp = jax.numpy
    block_jax = jnp.asarray(block, dtype=jnp.float64)
    kinetic = build_monitor_grid_kinetic_operator_jax(
        operator_context.grid_geometry,
        use_trial_boundary_fix=(operator_context.kinetic_version == "trial_fix"),
    )
    effective_local_potential = jnp.asarray(
        operator_context.effective_local_potential,
        dtype=jnp.float64,
    )

    def _apply_one(field):
        return kinetic(field) + effective_local_potential * field

    return jax.vmap(_apply_one)(block_jax)


def build_fixed_potential_static_local_operator_jax(
    operator_context: FixedPotentialStaticLocalOperatorContext,
):
    """Return a callable JAX local-only matvec for the current monitor-grid context."""

    def _operator(psi: np.ndarray):
        return apply_fixed_potential_static_local_operator_jax(
            psi,
            operator_context=operator_context,
        )

    return _operator


__all__ = [
    "apply_fixed_potential_static_local_block_jax",
    "apply_fixed_potential_static_local_operator_jax",
    "build_fixed_potential_static_local_operator_jax",
]
