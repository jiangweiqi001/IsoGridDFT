"""Audit script for the first H2 static KS slice with Hartree included."""

from __future__ import annotations

import numpy as np

from isogrid.ks import apply_static_ks_hamiltonian
from isogrid.ks import build_singlet_like_spin_densities
from isogrid.ks import build_total_density
from isogrid.ks import evaluate_static_ks_terms
from isogrid.ops import integrate_field
from isogrid.ops import weighted_l2_norm
from isogrid.poisson import solve_hartree_potential

from .local_hamiltonian_h2_trial_audit import build_symmetric_h2_trial_orbital


def _summary(name: str, values: np.ndarray, grid_geometry) -> list[str]:
    return [
        f"{name} min: {float(np.min(values)):.12f}",
        f"{name} max: {float(np.max(values)):.12f}",
        f"{name} weighted norm: {weighted_l2_norm(values, grid_geometry=grid_geometry):.12f}",
    ]


def _density_summary(name: str, rho: np.ndarray, grid_geometry) -> list[str]:
    return [
        f"{name} min: {float(np.min(rho)):.12f}",
        f"{name} max: {float(np.max(rho)):.12f}",
        f"{name} integral: {float(integrate_field(rho, grid_geometry=grid_geometry)):.12f}",
    ]


def _centerline_samples(grid_geometry, psi, hartree_action, total_action) -> list[str]:
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    indices = (
        0,
        len(grid_geometry.z_coordinates) // 4,
        len(grid_geometry.z_coordinates) // 2,
        3 * len(grid_geometry.z_coordinates) // 4,
        len(grid_geometry.z_coordinates) - 1,
    )
    rows = []
    for index in indices:
        rows.append(
            f"  z[{index:02d}] = {grid_geometry.z_coordinates[index]: .6f} Bohr -> "
            f"psi={psi[center_ix, center_iy, index]: .12f}, "
            f"Vhpsi={hartree_action[center_ix, center_iy, index]: .12f}, "
            f"Hpsi={total_action[center_ix, center_iy, index]: .12f}"
        )
    return rows


def main() -> int:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    rho_total = build_total_density(rho_up=rho_up, rho_down=rho_down, grid_geometry=grid_geometry)
    hartree_result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho_total,
    )
    terms = evaluate_static_ks_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        hartree_potential=hartree_result,
    )
    applied = apply_static_ks_hamiltonian(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        hartree_potential=hartree_result,
    )

    print("IsoGridDFT static KS H2 Hartree audit")
    print(f"grid shape: {grid_geometry.spec.shape}")
    print(f"trial orbital weighted norm: {weighted_l2_norm(psi, grid_geometry=grid_geometry):.12f}")
    print(f"open-boundary route: {hartree_result.description}")
    print(f"boundary approximation: {hartree_result.boundary_condition.description}")
    print(f"poisson solver method: {hartree_result.solver_method}")
    print(f"poisson iterations: {hartree_result.solver_iterations}")
    print(f"poisson residual max: {hartree_result.residual_max:.12e}")
    for line in _density_summary("rho_up", rho_up, grid_geometry):
        print(line)
    for line in _density_summary("rho_down", rho_down, grid_geometry):
        print(line)
    for line in _density_summary("rho_total", rho_total, grid_geometry):
        print(line)
    for line in _summary("v_h", hartree_result.potential, grid_geometry):
        print(line)
    for line in _summary("v_h psi", terms.hartree_action, grid_geometry):
        print(line)
    for line in _summary("H_ks_static psi", terms.total_action, grid_geometry):
        print(line)
    print(f"hartree energy: {terms.hartree_energy:.12f}")
    print(f"apply_static_ks_hamiltonian agrees with resolved terms: {bool(np.allclose(applied, terms.total_action))}")
    print("centerline samples on the x=0, y=0 slice:")
    for line in _centerline_samples(grid_geometry, psi, terms.hartree_action, terms.total_action):
        print(line)
    print(f"v_h mirror symmetry: {bool(np.allclose(hartree_result.potential, hartree_result.potential[:, :, ::-1]))}")
    print(f"H_ks_static psi mirror symmetry: {bool(np.allclose(terms.total_action, terms.total_action[:, :, ::-1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
