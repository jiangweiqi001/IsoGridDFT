"""Sanity checks for the first static KS Hamiltonian slice."""

from __future__ import annotations

from importlib import import_module

import numpy as np

from isogrid.grid import build_default_h2_grid_geometry
from isogrid.ks import apply_static_ks_hamiltonian
from isogrid.ks import build_singlet_like_spin_densities
from isogrid.ks import evaluate_static_ks_terms
from isogrid.pseudo import evaluate_atomic_nonlocal_action
from isogrid.pseudo import load_gth_pseudo_data
from isogrid.xc import evaluate_lsda_terms

from isogrid.audit.local_hamiltonian_h2_trial_audit import build_symmetric_h2_trial_orbital


def test_nonlocal_lsda_and_static_ks_modules_import() -> None:
    nonlocal_module = import_module("isogrid.pseudo.nonlocal")
    lsda_module = import_module("isogrid.xc.lsda")
    audit_module = import_module("isogrid.audit.static_ks_h2_trial_audit")

    assert hasattr(nonlocal_module, "evaluate_nonlocal_ionic_action")
    assert hasattr(lsda_module, "evaluate_lsda_terms")
    assert hasattr(audit_module, "main")


def test_nonlocal_action_runs_on_default_grid_for_supported_element() -> None:
    grid_geometry = build_default_h2_grid_geometry()
    pseudo_data = load_gth_pseudo_data("O")
    psi = np.exp(
        -0.8
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + grid_geometry.z_points**2
        )
    )
    contribution = evaluate_atomic_nonlocal_action(
        position=(0.0, 0.0, 0.0),
        grid_geometry=grid_geometry,
        psi=psi,
        pseudo_data=pseudo_data,
        atom_index=0,
        element="O",
    )

    assert contribution.nonlocal_action.shape == grid_geometry.spec.shape
    assert np.all(np.isfinite(contribution.nonlocal_action))
    assert len(contribution.projector_fields) > 0
    assert np.max(np.abs(contribution.nonlocal_action)) > 0.0


def test_lsda_terms_are_finite_for_positive_spin_density() -> None:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    lsda = evaluate_lsda_terms(rho_up=rho_up, rho_down=0.5 * rho_down)

    assert lsda.eps_xc.shape == psi.shape
    assert np.all(np.isfinite(lsda.eps_xc))
    assert np.all(np.isfinite(lsda.v_xc_up))
    assert np.all(np.isfinite(lsda.v_xc_down))


def test_apply_static_ks_hamiltonian_matches_input_shape_and_is_finite() -> None:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    action = apply_static_ks_hamiltonian(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
    )

    assert action.shape == psi.shape
    assert np.all(np.isfinite(action))


def test_static_ks_preserves_basic_h2_mirror_symmetry() -> None:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    terms = evaluate_static_ks_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
    )

    assert np.allclose(terms.total_action, terms.total_action[:, :, ::-1])
    assert np.allclose(terms.xc_action, terms.xc_action[:, :, ::-1])
