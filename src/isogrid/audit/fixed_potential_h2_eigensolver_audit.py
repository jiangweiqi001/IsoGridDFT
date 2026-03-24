"""Audit script for the first fixed-potential H2 eigensolver slice."""

from __future__ import annotations

import numpy as np

from isogrid.ks import build_singlet_like_spin_densities
from isogrid.ks import solve_fixed_potential_eigenproblem
from isogrid.ks import weighted_orbital_norms
from isogrid.ops import integrate_field
from isogrid.ops import weighted_l2_norm

from .local_hamiltonian_h2_trial_audit import build_symmetric_h2_trial_orbital


def _density_summary(name: str, rho: np.ndarray, grid_geometry) -> list[str]:
    return [
        f"{name} min: {float(np.min(rho)):.12f}",
        f"{name} max: {float(np.max(rho)):.12f}",
        f"{name} integral: {float(integrate_field(rho, grid_geometry=grid_geometry)):.12f}",
    ]


def _orbital_mirror_error(orbital: np.ndarray, grid_geometry) -> float:
    flipped = orbital[:, :, ::-1]
    even_error = weighted_l2_norm(orbital - flipped, grid_geometry=grid_geometry)
    odd_error = weighted_l2_norm(orbital + flipped, grid_geometry=grid_geometry)
    return float(min(even_error, odd_error))


def _centerline_samples(orbitals: np.ndarray, grid_geometry) -> list[str]:
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
    for orbital_index, orbital in enumerate(orbitals):
        rows.append(f"  orbital[{orbital_index}] centerline:")
        for index in indices:
            rows.append(
                f"    z[{index:02d}] = {grid_geometry.z_coordinates[index]: .6f} Bohr -> "
                f"psi={orbital[center_ix, center_iy, index]: .12f}"
            )
    return rows


def main() -> int:
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
        k=2,
        tolerance=1.0e-3,
        ncv=20,
    )

    orbital_norms = weighted_orbital_norms(result.orbitals, grid_geometry=grid_geometry)
    rho_total = rho_up + rho_down
    mirror_errors = [
        _orbital_mirror_error(orbital, grid_geometry=grid_geometry)
        for orbital in result.orbitals
    ]

    print("IsoGridDFT fixed-potential H2 eigensolver audit")
    print("note: this is a fixed-potential eigensolver audit, not an SCF result")
    print(f"grid shape: {grid_geometry.spec.shape}")
    print(f"frozen trial orbital weighted norm: {weighted_l2_norm(trial_orbital, grid_geometry=grid_geometry):.12f}")
    for line in _density_summary("rho_up", rho_up, grid_geometry):
        print(line)
    for line in _density_summary("rho_down", rho_down, grid_geometry):
        print(line)
    for line in _density_summary("rho_total", rho_total, grid_geometry):
        print(line)
    print(f"target orbital count: {result.target_orbitals}")
    print(f"solver method: {result.solver_method}")
    print(f"solver note: {result.solver_note}")
    print(f"converged: {result.converged}")
    print(f"iterations: {'unavailable from eigsh' if result.iteration_count < 0 else result.iteration_count}")
    print(f"tolerance: {result.tolerance:.6e}")
    print(f"eigenvalues (Ha): {result.eigenvalues.tolist()}")
    print(f"weighted orbital norms: {orbital_norms.tolist()}")
    print(f"max orthogonality error: {result.max_orthogonality_error:.6e}")
    print(f"residual norms: {result.residual_norms.tolist()}")
    print(f"subspace dimensions: {list(result.subspace_dimensions)}")
    print(f"orbital mirror errors: {mirror_errors}")
    print("centerline samples on the x=0, y=0 slice:")
    for line in _centerline_samples(result.orbitals, grid_geometry):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
