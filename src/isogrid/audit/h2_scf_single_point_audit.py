"""Audit script for the first minimal H2 SCF single-point closed loop."""

from __future__ import annotations

import numpy as np

from isogrid.ops import integrate_field
from isogrid.scf import run_h2_minimal_scf


def _density_summary(name: str, density: np.ndarray, grid_geometry) -> list[str]:
    return [
        f"{name} min: {float(np.min(density)):.12f}",
        f"{name} max: {float(np.max(density)):.12f}",
        f"{name} integral: {float(integrate_field(density, grid_geometry=grid_geometry)):.12f}",
    ]


def _lowest_eigenvalue(result) -> float | None:
    candidates = []
    if result.eigenvalues_up.size:
        candidates.append(float(result.eigenvalues_up[0]))
    if result.eigenvalues_down.size:
        candidates.append(float(result.eigenvalues_down[0]))
    return min(candidates) if candidates else None


def _print_result(result) -> None:
    print(f"spin state: {result.spin_state_label}")
    print(f"converged: {result.converged}")
    print(f"iterations: {result.iteration_count}")
    print(f"total energy: {result.energy.total:.12f} Ha")
    print(f"kinetic: {result.energy.kinetic:.12f} Ha")
    print(f"local ionic: {result.energy.local_ionic:.12f} Ha")
    print(f"nonlocal ionic: {result.energy.nonlocal_ionic:.12f} Ha")
    print(f"hartree: {result.energy.hartree:.12f} Ha")
    print(f"xc: {result.energy.xc:.12f} Ha")
    print(f"ion-ion repulsion: {result.energy.ion_ion_repulsion:.12f} Ha")
    lowest = _lowest_eigenvalue(result)
    print(f"lowest eigenvalue: {lowest:.12f} Ha" if lowest is not None else "lowest eigenvalue: n/a")
    print(
        f"final density residual: {result.history[-1].density_residual:.6e}"
        if result.history
        else "final density residual: n/a"
    )
    print(
        f"final energy change: {result.history[-1].energy_change:.6e}"
        if result.history and result.history[-1].energy_change is not None
        else "final energy change: n/a"
    )
    for line in _density_summary("rho_up", result.rho_up, result.solve_up.operator_context.grid_geometry if result.solve_up is not None else result.solve_down.operator_context.grid_geometry):
        print(line)
    for line in _density_summary("rho_down", result.rho_down, result.solve_up.operator_context.grid_geometry if result.solve_up is not None else result.solve_down.operator_context.grid_geometry):
        print(line)


def main() -> int:
    print("IsoGridDFT H2 minimal SCF single-point audit")
    print("note: this is the first minimal SCF single-point audit, not the final high-accuracy production result")
    singlet = run_h2_minimal_scf("singlet")
    triplet = run_h2_minimal_scf("triplet")

    print("")
    _print_result(singlet)
    print("")
    _print_result(triplet)
    print("")
    lower_label = "singlet" if singlet.energy.total < triplet.energy.total else "triplet"
    print(f"lower-energy spin state in this minimal SCF audit: {lower_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
