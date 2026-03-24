"""Sanity checks for the first fixed-potential eigensolver slice."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import numpy as np

from isogrid.audit.local_hamiltonian_h2_trial_audit import build_symmetric_h2_trial_orbital
from isogrid.ks import build_singlet_like_spin_densities
from isogrid.ks import solve_fixed_potential_eigenproblem
from isogrid.ks import weighted_orbital_norms
from isogrid.ops import weighted_l2_norm


@lru_cache(maxsize=None)
def _solve_default_h2(k: int):
    trial_orbital, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(
        trial_orbital,
        grid_geometry=grid_geometry,
    )
    result = solve_fixed_potential_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=k,
        tolerance=1.0e-3,
        ncv=20,
    )
    return result, grid_geometry


def test_eigensolver_module_imports() -> None:
    eigensolver_module = import_module("isogrid.ks.eigensolver")
    audit_module = import_module("isogrid.audit.fixed_potential_h2_eigensolver_audit")

    assert hasattr(eigensolver_module, "solve_fixed_potential_eigenproblem")
    assert hasattr(audit_module, "main")


def test_default_h2_lowest_orbital_can_be_solved() -> None:
    result, grid_geometry = _solve_default_h2(1)

    assert result.converged
    assert result.orbitals.shape == (1,) + grid_geometry.spec.shape
    assert result.eigenvalues.shape == (1,)
    assert np.all(np.isfinite(result.orbitals))
    assert np.all(np.isfinite(result.eigenvalues))
    assert np.all(np.isfinite(result.residual_norms))
    assert result.residual_norms[0] < 1.0e-3


def test_fixed_potential_orbitals_are_weighted_normalized() -> None:
    result, grid_geometry = _solve_default_h2(1)
    norms = weighted_orbital_norms(result.orbitals, grid_geometry=grid_geometry)

    assert np.allclose(norms, 1.0, atol=1.0e-8)
    assert result.max_orthogonality_error < 1.0e-8


def test_two_fixed_potential_orbitals_remain_weighted_orthogonal_and_sorted() -> None:
    result, grid_geometry = _solve_default_h2(2)

    assert result.converged
    assert result.orbitals.shape == (2,) + grid_geometry.spec.shape
    assert np.all(np.isfinite(result.orbitals))
    assert np.all(np.isfinite(result.eigenvalues))
    assert np.all(np.diff(result.eigenvalues) >= -1.0e-10)
    assert np.max(result.residual_norms) < 1.0e-3
    assert result.max_orthogonality_error < 1.0e-8

    overlap = result.weighted_overlap - np.eye(2)
    assert np.max(np.abs(overlap)) < 1.0e-8


def test_ground_orbital_preserves_basic_h2_mirror_symmetry() -> None:
    result, grid_geometry = _solve_default_h2(1)
    orbital = result.orbitals[0]
    flipped = orbital[:, :, ::-1]
    even_error = weighted_l2_norm(orbital - flipped, grid_geometry=grid_geometry)
    odd_error = weighted_l2_norm(orbital + flipped, grid_geometry=grid_geometry)

    assert min(even_error, odd_error) < 1.0e-3
