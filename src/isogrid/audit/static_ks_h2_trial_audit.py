"""Audit script for the first static KS H2 trial-orbital slice."""

from __future__ import annotations

import numpy as np

from isogrid.ks import build_singlet_like_spin_densities
from isogrid.ks import evaluate_static_ks_terms
from isogrid.ops import integrate_field
from isogrid.ops import weighted_l2_norm
from isogrid.pseudo import build_default_h2_nonlocal_ionic_action

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


def _centerline_samples(grid_geometry, psi, terms) -> list[str]:
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
            f"Tpsi={terms.kinetic_action[center_ix, center_iy, index]: .12f}, "
            f"Vlocpsi={terms.local_ionic_action[center_ix, center_iy, index]: .12f}, "
            f"Vnlpsi={terms.nonlocal_ionic_action[center_ix, center_iy, index]: .12f}, "
            f"Vxcpsi={terms.xc_action[center_ix, center_iy, index]: .12f}, "
            f"Hpsi={terms.total_action[center_ix, center_iy, index]: .12f}"
        )
    return rows


def main() -> int:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    nonlocal_evaluation = build_default_h2_nonlocal_ionic_action(
        psi=psi,
        grid_geometry=grid_geometry,
    )
    terms = evaluate_static_ks_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        nonlocal_ionic_action=nonlocal_evaluation,
    )

    projector_field_count = sum(
        len(atom.projector_fields)
        for atom in nonlocal_evaluation.atom_contributions
    )
    psi_norm = weighted_l2_norm(psi, grid_geometry=grid_geometry)
    mirror_hpsi = bool(np.allclose(terms.total_action, terms.total_action[:, :, ::-1]))

    print("IsoGridDFT static KS H2 trial audit")
    print(f"grid shape: {grid_geometry.spec.shape}")
    print(f"trial orbital shape: {psi.shape}")
    print(f"trial orbital weighted norm: {psi_norm:.12f}")
    print(f"nonlocal projector field count: {projector_field_count}")
    for line in _density_summary("rho_up", rho_up, grid_geometry):
        print(line)
    for line in _density_summary("rho_down", rho_down, grid_geometry):
        print(line)
    for line in _summary("T psi", terms.kinetic_action, grid_geometry):
        print(line)
    for line in _summary("V_loc,ion psi", terms.local_ionic_action, grid_geometry):
        print(line)
    for line in _summary("V_nl,ion psi", terms.nonlocal_ionic_action, grid_geometry):
        print(line)
    for line in _summary("V_xc psi", terms.xc_action, grid_geometry):
        print(line)
    for line in _summary("H_ks_static psi", terms.total_action, grid_geometry):
        print(line)
    print("centerline samples on the x=0, y=0 slice:")
    for line in _centerline_samples(grid_geometry, psi, terms):
        print(line)
    print(f"H_ks_static psi mirror symmetry: {mirror_hpsi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
