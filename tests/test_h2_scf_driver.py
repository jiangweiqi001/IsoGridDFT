"""Sanity checks for the first minimal H2 SCF single-point driver."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import numpy as np

from isogrid.ops import integrate_field
from isogrid.scf import run_h2_minimal_scf


@lru_cache(maxsize=None)
def _run_h2_scf(spin_label: str):
    return run_h2_minimal_scf(spin_label)


def test_scf_driver_modules_import() -> None:
    scf_module = import_module("isogrid.scf.driver")
    audit_module = import_module("isogrid.audit.h2_scf_single_point_audit")

    assert hasattr(scf_module, "run_h2_minimal_scf")
    assert hasattr(audit_module, "main")


def test_h2_singlet_minimal_scf_runs() -> None:
    result = _run_h2_scf("singlet")

    assert result.converged
    assert result.spin_state_label == "singlet"
    assert np.isfinite(result.energy.total)
    assert np.all(np.isfinite(result.rho_up))
    assert np.all(np.isfinite(result.rho_down))


def test_h2_triplet_minimal_scf_runs() -> None:
    result = _run_h2_scf("triplet")

    assert result.converged
    assert result.spin_state_label == "triplet"
    assert np.isfinite(result.energy.total)
    assert np.all(np.isfinite(result.rho_up))
    assert np.all(np.isfinite(result.rho_down))


def test_h2_scf_energy_components_are_finite() -> None:
    result = _run_h2_scf("singlet")
    components = result.energy

    assert np.isfinite(components.kinetic)
    assert np.isfinite(components.local_ionic)
    assert np.isfinite(components.nonlocal_ionic)
    assert np.isfinite(components.hartree)
    assert np.isfinite(components.xc)
    assert np.isfinite(components.ion_ion_repulsion)
    assert np.isfinite(components.total)


def test_h2_scf_density_shapes_and_electron_counts_are_consistent() -> None:
    for spin_label in ("singlet", "triplet"):
        result = _run_h2_scf(spin_label)
        grid_geometry = (
            result.solve_up.operator_context.grid_geometry
            if result.solve_up is not None
            else result.solve_down.operator_context.grid_geometry
        )

        assert result.rho_up.shape == grid_geometry.spec.shape
        assert result.rho_down.shape == grid_geometry.spec.shape
        assert np.all(np.isfinite(result.rho_up))
        assert np.all(np.isfinite(result.rho_down))

        n_up = float(integrate_field(result.rho_up, grid_geometry=grid_geometry))
        n_down = float(integrate_field(result.rho_down, grid_geometry=grid_geometry))
        assert abs(n_up - result.occupations.n_alpha) < 1.0e-6
        assert abs(n_down - result.occupations.n_beta) < 1.0e-6


def test_h2_scf_singlet_and_triplet_energies_are_finite() -> None:
    singlet = _run_h2_scf("singlet")
    triplet = _run_h2_scf("triplet")

    assert np.isfinite(singlet.energy.total)
    assert np.isfinite(triplet.energy.total)
