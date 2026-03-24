"""Audit script for the first local-Hamiltonian H2 trial-orbital slice."""

from __future__ import annotations

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.ks import evaluate_local_hamiltonian_terms
from isogrid.ops import weighted_l2_norm
from isogrid.pseudo import build_default_h2_local_ionic_potential


def build_symmetric_h2_trial_orbital(
    exponent: float = 0.8,
) -> tuple[np.ndarray, object]:
    """Build a symmetric two-center Gaussian trial orbital for the default H2 case."""

    case = H2_BENCHMARK_CASE
    grid_geometry = build_default_h2_grid_geometry(case=case)
    psi = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    for atom in case.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        radius_squared = dx * dx + dy * dy + dz * dz
        psi += np.exp(-exponent * radius_squared)

    psi /= weighted_l2_norm(psi, grid_geometry=grid_geometry)
    return psi, grid_geometry


def _summary(name: str, values: np.ndarray, grid_geometry) -> list[str]:
    return [
        f"{name} min: {float(np.min(values)):.12f}",
        f"{name} max: {float(np.max(values)):.12f}",
        f"{name} weighted norm: {weighted_l2_norm(values, grid_geometry=grid_geometry):.12f}",
    ]


def _centerline_samples(grid_geometry, psi, kinetic, potential_action, total_action) -> list[str]:
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    indices = (0, len(grid_geometry.z_coordinates) // 4, len(grid_geometry.z_coordinates) // 2, 3 * len(grid_geometry.z_coordinates) // 4, len(grid_geometry.z_coordinates) - 1)
    rows = []
    for index in indices:
        rows.append(
            f"  z[{index:02d}] = {grid_geometry.z_coordinates[index]: .6f} Bohr -> "
            f"psi={psi[center_ix, center_iy, index]: .12f}, "
            f"Tpsi={kinetic[center_ix, center_iy, index]: .12f}, "
            f"Vpsi={potential_action[center_ix, center_iy, index]: .12f}, "
            f"Hpsi={total_action[center_ix, center_iy, index]: .12f}"
        )
    return rows


def main() -> int:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    local_ionic_potential = build_default_h2_local_ionic_potential(grid_geometry=grid_geometry)
    terms = evaluate_local_hamiltonian_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        local_ionic_potential=local_ionic_potential,
    )

    psi_norm = weighted_l2_norm(psi, grid_geometry=grid_geometry)
    mirror_psi = bool(np.allclose(psi, psi[:, :, ::-1]))
    mirror_hpsi = bool(np.allclose(terms.total_action, terms.total_action[:, :, ::-1]))

    print("IsoGridDFT local Hamiltonian H2 trial audit")
    print(f"grid shape: {grid_geometry.spec.shape}")
    print(f"trial orbital shape: {psi.shape}")
    print(f"trial orbital weighted norm: {psi_norm:.12f}")
    for line in _summary("T psi", terms.kinetic_action, grid_geometry):
        print(line)
    for line in _summary("V_local psi", terms.local_potential_action, grid_geometry):
        print(line)
    for line in _summary("H_local psi", terms.total_action, grid_geometry):
        print(line)
    print("centerline samples on the x=0, y=0 slice:")
    for line in _centerline_samples(
        grid_geometry,
        terms.psi,
        terms.kinetic_action,
        terms.local_potential_action,
        terms.total_action,
    ):
        print(line)
    print(f"trial orbital mirror symmetry: {mirror_psi}")
    print(f"H_local psi mirror symmetry: {mirror_hpsi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
