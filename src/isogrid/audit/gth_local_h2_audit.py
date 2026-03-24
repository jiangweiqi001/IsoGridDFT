"""Audit script for the first H2 local GTH pseudopotential slice."""

from __future__ import annotations

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.pseudo import SUPPORTED_GTH_ELEMENTS
from isogrid.pseudo import build_default_h2_local_ionic_potential
from isogrid.pseudo import load_case_gth_pseudo_data


def _format_loaded_pseudo_summary(case) -> list[str]:
    data_by_element = load_case_gth_pseudo_data(case)
    lines = []
    for element, pseudo_data in data_by_element.items():
        lines.append(
            f"  {element}: ionic_charge={pseudo_data.ionic_charge}, "
            f"rloc={pseudo_data.local.rloc:.8f}, "
            f"local_coefficients={pseudo_data.local.coefficients}, "
            f"nonlocal_channels={len(pseudo_data.nonlocal_channels)}"
        )
    return lines


def _centerline_sample_rows(z_coordinates: np.ndarray, values: np.ndarray) -> list[str]:
    sample_indices = (0, len(z_coordinates) // 4, len(z_coordinates) // 2, 3 * len(z_coordinates) // 4, len(z_coordinates) - 1)
    rows = []
    for index in sample_indices:
        rows.append(
            f"  z[{index:02d}] = {z_coordinates[index]: .6f} Bohr -> V = {values[index]: .12f} Ha"
        )
    return rows


def main() -> int:
    case = H2_BENCHMARK_CASE
    grid_geometry = build_default_h2_grid_geometry(case=case)
    evaluation = build_default_h2_local_ionic_potential(case=case, grid_geometry=grid_geometry)

    total_potential = evaluation.total_local_potential
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    centerline = total_potential[center_ix, center_iy, :]
    symmetry_ok = bool(np.allclose(total_potential, total_potential[:, :, ::-1]))

    print("IsoGridDFT GTH local H2 audit")
    print(f"benchmark: {case.name}")
    print(f"pseudo family: {evaluation.pseudo_family}")
    print(f"supported elements in current layer: {', '.join(SUPPORTED_GTH_ELEMENTS)}")
    print("loaded pseudopotential summary:")
    for line in _format_loaded_pseudo_summary(case):
        print(line)
    print(f"grid shape: {grid_geometry.spec.shape}")
    print(f"potential shape: {total_potential.shape}")
    print(f"potential minimum [Ha]: {float(np.min(total_potential)):.12f}")
    print(f"potential maximum [Ha]: {float(np.max(total_potential)):.12f}")
    print("centerline samples on the x=0, y=0 slice:")
    for line in _centerline_sample_rows(grid_geometry.z_coordinates, centerline):
        print(line)
    print(f"mirror symmetry about the molecular center: {symmetry_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
