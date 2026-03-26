"""Minimal correctness audit for the first JAX hot-kernel migration."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.config.runtime_jax import get_jax_runtime_configuration
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ks import apply_fixed_potential_static_local_operator
from isogrid.ks import prepare_fixed_potential_static_local_operator
from isogrid.ks import weighted_overlap_matrix
from isogrid.ks.hamiltonian_local_jax import apply_fixed_potential_static_local_operator_jax
from isogrid.ops import integrate_field
from isogrid.ops import weighted_l2_norm
from isogrid.ops.reductions_jax import accumulate_density_from_orbitals_jax
from isogrid.ops.reductions_jax import weighted_inner_product_jax
from isogrid.ops.reductions_jax import weighted_l2_norm_jax
from isogrid.ops.reductions_jax import weighted_orthonormalize_orbitals_jax
from isogrid.ops.reductions_jax import weighted_overlap_matrix_jax
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential
from isogrid.poisson.poisson_jax import solve_open_boundary_poisson_monitor_jax


@dataclass(frozen=True)
class JaxReductionConsistencyResult:
    """Compact reductions audit summary."""

    weighted_inner_product_abs_diff: float
    weighted_norm_abs_diff: float
    density_accumulation_max_abs_diff: float
    overlap_matrix_max_abs_diff: float
    orthonormalization_overlap_max_abs_diff: float


@dataclass(frozen=True)
class JaxPoissonConsistencyResult:
    """Compact JAX-vs-current Poisson audit summary."""

    solver_method: str
    iteration_count: int
    residual_max: float
    potential_max_abs_diff: float
    hartree_energy_abs_diff_ha: float
    elapsed_seconds: float


@dataclass(frozen=True)
class JaxLocalHamiltonianConsistencyResult:
    """Compact JAX-vs-current local Hamiltonian audit summary."""

    action_max_abs_diff: float
    action_weighted_norm_diff: float
    elapsed_seconds: float


@dataclass(frozen=True)
class H2JaxKernelConsistencyAuditResult:
    """Top-level first-batch JAX migration audit result."""

    runtime_summary: str
    monitor_shape: tuple[int, int, int]
    reductions: JaxReductionConsistencyResult
    poisson: JaxPoissonConsistencyResult
    local_hamiltonian: JaxLocalHamiltonianConsistencyResult
    note: str


def _build_h2_bonding_trial_orbital(grid_geometry) -> np.ndarray:
    atom_fields = []
    for atom in H2_BENCHMARK_CASE.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        atom_fields.append(np.exp(-0.8 * (dx * dx + dy * dy + dz * dz)))
    orbital = np.asarray(atom_fields[0] + atom_fields[1], dtype=np.float64)
    norm = float(np.sqrt(integrate_field(orbital * orbital, grid_geometry=grid_geometry)))
    return orbital / norm


def _build_probe_orbital_block(grid_geometry) -> np.ndarray:
    bonding = _build_h2_bonding_trial_orbital(grid_geometry)
    center_z = 0.5 * (
        H2_BENCHMARK_CASE.geometry.atoms[0].position[2]
        + H2_BENCHMARK_CASE.geometry.atoms[1].position[2]
    )
    z_shift = grid_geometry.z_points - center_z
    p_like = z_shift * np.exp(-0.5 * ((grid_geometry.x_points**2) + (grid_geometry.y_points**2) + (z_shift**2)))
    p_norm = float(np.sqrt(integrate_field(p_like * p_like, grid_geometry=grid_geometry)))
    return np.asarray([bonding, p_like / p_norm], dtype=np.float64)


def run_h2_jax_kernel_consistency_audit() -> H2JaxKernelConsistencyAuditResult:
    """Run one minimal correctness audit for the first JAX hot kernels."""

    grid_geometry = build_h2_local_patch_development_monitor_grid()
    weights = grid_geometry.cell_volumes
    orbital_block = _build_probe_orbital_block(grid_geometry)
    trial_orbital = orbital_block[0]
    rho_total = 2.0 * np.abs(trial_orbital) ** 2

    inner_numpy = float(np.real_if_close(np.sum(np.conjugate(trial_orbital) * trial_orbital * weights)))
    inner_jax = float(np.real_if_close(weighted_inner_product_jax(trial_orbital, trial_orbital, weights)))
    norm_numpy = float(weighted_l2_norm(trial_orbital, grid_geometry=grid_geometry))
    norm_jax = float(weighted_l2_norm_jax(trial_orbital, weights))
    density_numpy = np.sum(np.abs(orbital_block) ** 2, axis=0)
    density_jax = np.asarray(accumulate_density_from_orbitals_jax(orbital_block), dtype=np.float64)
    overlap_numpy = weighted_overlap_matrix(orbital_block, grid_geometry=grid_geometry)
    overlap_jax = np.asarray(weighted_overlap_matrix_jax(orbital_block, weights), dtype=np.float64)
    orthonormal_jax = np.asarray(
        weighted_orthonormalize_orbitals_jax(orbital_block, weights),
        dtype=np.float64,
    )
    orthonormal_overlap = weighted_overlap_matrix(orthonormal_jax, grid_geometry=grid_geometry)

    reductions_result = JaxReductionConsistencyResult(
        weighted_inner_product_abs_diff=float(abs(inner_numpy - inner_jax)),
        weighted_norm_abs_diff=float(abs(norm_numpy - norm_jax)),
        density_accumulation_max_abs_diff=float(np.max(np.abs(density_numpy - density_jax))),
        overlap_matrix_max_abs_diff=float(np.max(np.abs(overlap_numpy - overlap_jax))),
        orthonormalization_overlap_max_abs_diff=float(
            np.max(np.abs(orthonormal_overlap - np.eye(orthonormal_overlap.shape[0])))
        ),
    )

    start = time.perf_counter()
    poisson_numpy = solve_hartree_potential(grid_geometry=grid_geometry, rho=rho_total)
    poisson_jax, poisson_diagnostics = solve_open_boundary_poisson_monitor_jax(
        grid_geometry=grid_geometry,
        rho=rho_total,
    )
    poisson_elapsed = time.perf_counter() - start
    hartree_numpy = evaluate_hartree_energy(
        rho=rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=poisson_numpy,
    )
    hartree_jax = evaluate_hartree_energy(
        rho=rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=poisson_jax,
    )
    poisson_result = JaxPoissonConsistencyResult(
        solver_method=poisson_diagnostics.solver_method,
        iteration_count=poisson_diagnostics.iteration_count,
        residual_max=poisson_diagnostics.residual_max,
        potential_max_abs_diff=float(
            np.max(np.abs(poisson_numpy.potential - poisson_jax.potential))
        ),
        hartree_energy_abs_diff_ha=float(abs(hartree_numpy - hartree_jax)),
        elapsed_seconds=float(poisson_elapsed),
    )

    operator_context = prepare_fixed_potential_static_local_operator(
        grid_geometry=grid_geometry,
        rho_up=np.abs(trial_orbital) ** 2,
        rho_down=np.abs(trial_orbital) ** 2,
        spin_channel="up",
        use_monitor_patch=True,
        kinetic_version="trial_fix",
    )
    start = time.perf_counter()
    action_numpy = apply_fixed_potential_static_local_operator(
        trial_orbital,
        operator_context=operator_context,
    )
    action_jax = np.asarray(
        apply_fixed_potential_static_local_operator_jax(
            trial_orbital,
            operator_context=operator_context,
        ),
        dtype=np.float64,
    )
    local_elapsed = time.perf_counter() - start
    local_result = JaxLocalHamiltonianConsistencyResult(
        action_max_abs_diff=float(np.max(np.abs(action_numpy - action_jax))),
        action_weighted_norm_diff=float(
            abs(
                weighted_l2_norm(action_numpy, grid_geometry=grid_geometry)
                - weighted_l2_norm(action_jax, grid_geometry=grid_geometry)
            )
        ),
        elapsed_seconds=float(local_elapsed),
    )

    runtime = get_jax_runtime_configuration()
    return H2JaxKernelConsistencyAuditResult(
        runtime_summary=(
            f"x64={runtime.enable_x64}, disable_jit={runtime.disable_jit}, "
            f"platform={runtime.platform_name or 'default'}"
        ),
        monitor_shape=grid_geometry.spec.shape,
        reductions=reductions_result,
        poisson=poisson_result,
        local_hamiltonian=local_result,
        note=(
            "This is a first-batch JAX correctness audit only. It checks stable hot kernels "
            "on the repaired local-only A-grid path and keeps the Python/SciPy control flow "
            "and audit path in place."
        ),
    )


def print_h2_jax_kernel_consistency_summary(result: H2JaxKernelConsistencyAuditResult) -> None:
    """Print the compact first-batch JAX consistency summary."""

    print("IsoGridDFT first-batch JAX kernel consistency audit")
    print(f"runtime: {result.runtime_summary}")
    print(f"monitor shape: {result.monitor_shape}")
    print(result.note)
    print()
    print("reductions:")
    print(f"  weighted inner-product abs diff: {result.reductions.weighted_inner_product_abs_diff:.6e}")
    print(f"  weighted norm abs diff: {result.reductions.weighted_norm_abs_diff:.6e}")
    print(f"  density accumulation max abs diff: {result.reductions.density_accumulation_max_abs_diff:.6e}")
    print(f"  overlap matrix max abs diff: {result.reductions.overlap_matrix_max_abs_diff:.6e}")
    print(
        "  orthonormalization overlap max abs diff: "
        f"{result.reductions.orthonormalization_overlap_max_abs_diff:.6e}"
    )
    print()
    print("poisson:")
    print(f"  solver method: {result.poisson.solver_method}")
    print(f"  iterations: {result.poisson.iteration_count}")
    print(f"  residual max: {result.poisson.residual_max:.6e}")
    print(f"  potential max abs diff: {result.poisson.potential_max_abs_diff:.6e}")
    print(f"  Hartree energy abs diff [Ha]: {result.poisson.hartree_energy_abs_diff_ha:.6e}")
    print(f"  elapsed [s]: {result.poisson.elapsed_seconds:.3f}")
    print()
    print("local hamiltonian:")
    print(f"  action max abs diff: {result.local_hamiltonian.action_max_abs_diff:.6e}")
    print(f"  action weighted norm diff: {result.local_hamiltonian.action_weighted_norm_diff:.6e}")
    print(f"  elapsed [s]: {result.local_hamiltonian.elapsed_seconds:.3f}")


def main() -> int:
    result = run_h2_jax_kernel_consistency_audit()
    print_h2_jax_kernel_consistency_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
