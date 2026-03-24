"""Sanity checks for the first local-Hamiltonian slice."""

from __future__ import annotations

from importlib import import_module

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.ks import apply_local_hamiltonian
from isogrid.ks import evaluate_local_hamiltonian_terms
from isogrid.ops import apply_kinetic_operator
from isogrid.ops import weighted_l2_norm
from isogrid.pseudo import build_default_h2_local_ionic_potential


def _build_symmetric_trial_orbital(grid_geometry, exponent: float = 0.8) -> np.ndarray:
    psi = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    for atom in H2_BENCHMARK_CASE.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        psi += np.exp(-exponent * (dx * dx + dy * dy + dz * dz))
    psi /= weighted_l2_norm(psi, grid_geometry=grid_geometry)
    return psi


def test_ops_and_local_hamiltonian_audit_modules_import() -> None:
    ops_module = import_module("isogrid.ops")
    audit_module = import_module("isogrid.audit.local_hamiltonian_h2_trial_audit")

    assert hasattr(ops_module, "apply_kinetic_operator")
    assert hasattr(audit_module, "main")


def test_kinetic_operator_runs_on_default_h2_trial_orbital() -> None:
    grid_geometry = build_default_h2_grid_geometry()
    psi = _build_symmetric_trial_orbital(grid_geometry)
    kinetic_action = apply_kinetic_operator(psi=psi, grid_geometry=grid_geometry)

    assert kinetic_action.shape == grid_geometry.spec.shape


def test_apply_local_hamiltonian_matches_input_shape_and_is_finite() -> None:
    grid_geometry = build_default_h2_grid_geometry()
    psi = _build_symmetric_trial_orbital(grid_geometry)
    local_ionic_potential = build_default_h2_local_ionic_potential(grid_geometry=grid_geometry)
    action = apply_local_hamiltonian(
        psi=psi,
        grid_geometry=grid_geometry,
        local_ionic_potential=local_ionic_potential,
    )

    assert action.shape == psi.shape
    assert np.all(np.isfinite(action))


def test_constant_field_has_zero_interior_kinetic_action() -> None:
    grid_geometry = build_default_h2_grid_geometry()
    psi = np.ones(grid_geometry.spec.shape, dtype=np.float64)
    kinetic_action = apply_kinetic_operator(psi=psi, grid_geometry=grid_geometry)

    assert np.allclose(kinetic_action[1:-1, 1:-1, 1:-1], 0.0)


def test_local_hamiltonian_preserves_basic_h2_mirror_symmetry() -> None:
    grid_geometry = build_default_h2_grid_geometry()
    psi = _build_symmetric_trial_orbital(grid_geometry)
    local_ionic_potential = build_default_h2_local_ionic_potential(grid_geometry=grid_geometry)
    terms = evaluate_local_hamiltonian_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        local_ionic_potential=local_ionic_potential,
    )

    assert np.allclose(terms.total_action, terms.total_action[:, :, ::-1])
