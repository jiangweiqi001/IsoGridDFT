"""First fixed-potential iterative eigensolver scaffold for the static KS backbone.

This module solves the lowest few orbitals of the current frozen-potential
static KS Hamiltonian without introducing SCF. The current first-stage solver
uses an iterative Lanczos solve on the weighted-similarity-transformed operator

    A = W^(1/2) H W^(-1/2)

where W is the diagonal cell-volume metric on the structured adaptive grid.
This keeps the physical orbitals orthonormal under the weighted inner product

    <phi|psi>_W = sum_r conj(phi[r]) psi[r] w[r]

and lets the eigensolver operate on a standard Euclidean symmetric problem.

This is a first formal fixed-potential eigensolver skeleton. It is not yet a
production SCF eigensolver and does not update the density.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.ks.static_hamiltonian import apply_static_ks_hamiltonian
from isogrid.ops import apply_kinetic_operator
from isogrid.ops import validate_orbital_field
from isogrid.ops import weighted_l2_norm
from isogrid.ops.kinetic import apply_monitor_grid_kinetic_operator_trial_boundary_fix
from isogrid.poisson import OpenBoundaryPoissonResult
from isogrid.poisson import solve_hartree_potential
from isogrid.pseudo import FrozenPatchLocalPotentialEmbedding
from isogrid.pseudo import LocalIonicPotentialEvaluation
from isogrid.pseudo import LocalPotentialPatchParameters
from isogrid.pseudo import evaluate_local_ionic_potential
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential_with_frozen_patch_field
from isogrid.xc import LSDAEvaluation
from isogrid.xc import evaluate_lsda_terms

_VALID_SPIN_CHANNELS = {"up", "down"}
_VALID_STATIC_LOCAL_KINETIC_VERSIONS = {"production", "trial_fix"}
GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class FixedPotentialOperatorContext:
    """Frozen static-KS potential data used by the fixed-potential solver."""

    grid_geometry: GridGeometryLike
    case: BenchmarkCase
    spin_channel: str
    rho_up: np.ndarray
    rho_down: np.ndarray
    rho_total: np.ndarray
    local_ionic_potential: np.ndarray
    hartree_potential: np.ndarray
    xc_potential: np.ndarray
    effective_local_potential: np.ndarray
    local_ionic_evaluation: LocalIonicPotentialEvaluation | None
    hartree_poisson_result: OpenBoundaryPoissonResult | None
    lsda_evaluation: LSDAEvaluation | None


@dataclass(frozen=True)
class FixedPotentialStaticLocalOperatorContext:
    """Frozen static-local potential data used by the A-grid eigensolver path.

    The current operator contains only

        T + V_loc,ion + V_H + V_xc

    with the density frozen externally. For monitor-grid patch work, the local
    ionic potential may include a frozen patch embedding that reproduces the
    current near-core local-GTH patch energy on the chosen frozen density.
    """

    grid_geometry: GridGeometryLike
    case: BenchmarkCase
    spin_channel: str
    rho_up: np.ndarray
    rho_down: np.ndarray
    rho_total: np.ndarray
    local_ionic_potential: np.ndarray
    hartree_potential: np.ndarray
    xc_potential: np.ndarray
    effective_local_potential: np.ndarray
    local_ionic_evaluation: LocalIonicPotentialEvaluation | None
    frozen_patch_local_embedding: FrozenPatchLocalPotentialEmbedding | None
    hartree_poisson_result: OpenBoundaryPoissonResult | None
    lsda_evaluation: LSDAEvaluation | None
    kinetic_version: str = "production"


@dataclass(frozen=True)
class FixedPotentialStaticLocalPreparationProfile:
    """Very small timing/profile summary for static-local context preparation."""

    density_build_wall_time_seconds: float
    local_ionic_resolve_wall_time_seconds: float
    hartree_resolve_wall_time_seconds: float
    xc_resolve_wall_time_seconds: float
    total_wall_time_seconds: float


@dataclass(frozen=True)
class FixedPotentialEigensolverResult:
    """Audit-facing result of the first fixed-potential eigensolver."""

    target_orbitals: int
    solver_method: str
    solver_note: str
    converged: bool
    iteration_count: int
    tolerance: float
    eigenvalues: np.ndarray
    orbitals: np.ndarray
    weighted_overlap: np.ndarray
    max_orthogonality_error: float
    residual_norms: np.ndarray
    residual_history: np.ndarray
    ritz_value_history: np.ndarray
    subspace_dimensions: tuple[int, ...]
    ritz_matrix: np.ndarray
    initial_guess_orbitals: np.ndarray
    final_basis_orbitals: np.ndarray
    operator_context: FixedPotentialOperatorContext | FixedPotentialStaticLocalOperatorContext
    static_local_preparation_profile: FixedPotentialStaticLocalPreparationProfile | None = None
    use_jax_block_kernels: bool = False
    use_jax_cached_kernels: bool = False
    wall_time_seconds: float | None = None


def _normalize_spin_channel(spin_channel: str) -> str:
    normalized = spin_channel.strip().lower()
    if normalized not in _VALID_SPIN_CHANNELS:
        raise ValueError(
            "spin_channel must be `up` or `down`; "
            f"received `{spin_channel}`."
        )
    return normalized


def _build_total_density_on_grid(
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    rho_up_field = validate_orbital_field(rho_up, grid_geometry=grid_geometry, name="rho_up")
    rho_down_field = validate_orbital_field(rho_down, grid_geometry=grid_geometry, name="rho_down")
    return np.asarray(rho_up_field + rho_down_field, dtype=np.float64)


def _normalize_static_local_kinetic_version(kinetic_version: str) -> str:
    normalized = kinetic_version.strip().lower()
    if normalized not in _VALID_STATIC_LOCAL_KINETIC_VERSIONS:
        raise ValueError(
            "kinetic_version must be `production` or `trial_fix`; "
            f"received `{kinetic_version}`."
        )
    return normalized


def validate_orbital_block(
    orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
    name: str = "orbitals",
) -> np.ndarray:
    """Validate a block of orbitals stored as (k, nx, ny, nz)."""

    block = np.asarray(orbitals)
    if block.ndim == 3:
        block = block[np.newaxis, ...]
    if block.ndim != 4:
        raise ValueError(f"{name} must be a 3D orbital or a 4D orbital block.")
    if block.shape[1:] != grid_geometry.spec.shape:
        raise ValueError(
            f"{name} must match the grid shape {grid_geometry.spec.shape}; "
            f"received {block.shape[1:]}."
        )
    if not np.issubdtype(block.dtype, np.inexact):
        block = block.astype(np.float64)
    return block


def flatten_orbital_block(
    orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    """Flatten a block of orbitals into column-major (n_grid, k) form."""

    block = validate_orbital_block(orbitals, grid_geometry=grid_geometry)
    return np.asarray(block.reshape(block.shape[0], -1).T)


def reshape_orbital_columns(
    orbital_columns: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    """Reshape (n_grid, k) orbital columns back to (k, nx, ny, nz)."""

    columns = np.asarray(orbital_columns)
    if columns.ndim == 1:
        columns = columns[:, np.newaxis]
    n_grid = int(np.prod(grid_geometry.spec.shape))
    if columns.ndim != 2 or columns.shape[0] != n_grid:
        raise ValueError(
            "orbital_columns must have shape (n_grid, k) compatible with the current grid."
        )
    return np.asarray(columns.T.reshape((columns.shape[1],) + grid_geometry.spec.shape))


def weighted_overlap_matrix(
    orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
    other: np.ndarray | None = None,
    *,
    use_jax_block_kernels: bool = False,
) -> np.ndarray:
    """Return the weighted block overlap matrix under the cell-volume metric."""

    if use_jax_block_kernels and isinstance(grid_geometry, MonitorGridGeometry):
        from isogrid.ks.eigensolver_jax_cache import weighted_overlap_matrix_cached_jax

        overlap = weighted_overlap_matrix_cached_jax(
            orbitals,
            grid_geometry.cell_volumes,
            other=other,
        )
        return np.asarray(overlap, dtype=np.float64)

    left_columns = flatten_orbital_block(orbitals, grid_geometry=grid_geometry)
    if other is None:
        right_columns = left_columns
    else:
        right_columns = flatten_orbital_block(other, grid_geometry=grid_geometry)
    weights = grid_geometry.cell_volumes.reshape(-1)
    overlap = np.conjugate(left_columns).T @ (weights[:, None] * right_columns)
    return np.asarray(overlap)


def weighted_orbital_norms(
    orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
    *,
    use_jax_block_kernels: bool = False,
) -> np.ndarray:
    """Return the weighted norms of a block of orbitals."""

    overlap = weighted_overlap_matrix(
        orbitals,
        grid_geometry=grid_geometry,
        use_jax_block_kernels=use_jax_block_kernels,
    )
    diagonal = np.real_if_close(np.diag(overlap))
    return np.sqrt(np.clip(diagonal.astype(np.float64), 0.0, None))


def weighted_orthonormalize_orbitals(
    orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
    *,
    require_full_rank: bool = True,
    rank_tolerance: float = 1.0e-12,
    use_jax_block_kernels: bool = False,
) -> np.ndarray:
    """Weighted-orthonormalize a block by SVD of sqrt(W) * Psi."""

    block = validate_orbital_block(orbitals, grid_geometry=grid_geometry)
    if block.shape[0] == 0:
        return block.copy()

    if use_jax_block_kernels and isinstance(grid_geometry, MonitorGridGeometry):
        from isogrid.ks.eigensolver_jax_cache import weighted_orthonormalize_orbitals_cached_jax

        orthonormal = weighted_orthonormalize_orbitals_cached_jax(
            block,
            grid_geometry.cell_volumes,
            require_full_rank=require_full_rank,
            rank_tolerance=rank_tolerance,
        )
        return np.asarray(orthonormal, dtype=np.float64)

    columns = flatten_orbital_block(block, grid_geometry=grid_geometry)
    sqrt_weights = np.sqrt(grid_geometry.cell_volumes.reshape(-1))[:, None]
    weighted_columns = sqrt_weights * columns
    _, singular_values, right_vectors = np.linalg.svd(weighted_columns, full_matrices=False)

    if singular_values.size == 0:
        if require_full_rank:
            raise ValueError("No nonzero orbital directions were provided for orthonormalization.")
        return np.zeros((0,) + grid_geometry.spec.shape, dtype=block.dtype)

    cutoff = rank_tolerance * max(weighted_columns.shape) * singular_values[0]
    rank = int(np.sum(singular_values > cutoff))
    if rank == 0:
        if require_full_rank:
            raise ValueError("The orbital block is numerically rank-deficient in the weighted metric.")
        return np.zeros((0,) + grid_geometry.spec.shape, dtype=block.dtype)
    if require_full_rank and rank < block.shape[0]:
        raise ValueError(
            "The orbital block is rank-deficient under weighted orthonormalization: "
            f"rank={rank}, requested={block.shape[0]}."
        )

    transform = right_vectors[:rank].conj().T / singular_values[:rank]
    orthonormal_columns = columns @ transform
    return reshape_orbital_columns(orthonormal_columns, grid_geometry=grid_geometry)


def _resolve_local_ionic_potential(
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None,
) -> tuple[np.ndarray, LocalIonicPotentialEvaluation | None]:
    if local_ionic_potential is None:
        evaluation = evaluate_local_ionic_potential(case=case, grid_geometry=grid_geometry)
        return evaluation.total_local_potential, evaluation
    if isinstance(local_ionic_potential, LocalIonicPotentialEvaluation):
        return local_ionic_potential.total_local_potential, local_ionic_potential
    return (
        validate_orbital_field(
            local_ionic_potential,
            grid_geometry=grid_geometry,
            name="local_ionic_potential",
        ),
        None,
    )


def _resolve_hartree_potential(
    grid_geometry: GridGeometryLike,
    rho_total: np.ndarray,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray | None,
    *,
    hartree_backend: str = "python",
    use_jax_hartree_cached_operator: bool = False,
    jax_hartree_cg_impl: str = "baseline",
    jax_hartree_cg_preconditioner: str = "none",
) -> tuple[np.ndarray, OpenBoundaryPoissonResult | None]:
    if hartree_potential is None:
        poisson_result = solve_hartree_potential(
            grid_geometry=grid_geometry,
            rho=rho_total,
            backend=hartree_backend,
            use_jax_cached_operator=use_jax_hartree_cached_operator,
            cg_impl=jax_hartree_cg_impl,
            cg_preconditioner=jax_hartree_cg_preconditioner,
        )
        return poisson_result.potential, poisson_result
    if isinstance(hartree_potential, OpenBoundaryPoissonResult):
        return hartree_potential.potential, hartree_potential
    return (
        validate_orbital_field(
            hartree_potential,
            grid_geometry=grid_geometry,
            name="hartree_potential",
        ),
        None,
    )


def _resolve_xc_potential(
    grid_geometry: GridGeometryLike,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    case: BenchmarkCase,
    xc_potential: np.ndarray | None,
    xc_functional: str | None,
) -> tuple[np.ndarray, LSDAEvaluation | None]:
    if xc_potential is not None:
        return (
            validate_orbital_field(
                xc_potential,
                grid_geometry=grid_geometry,
                name="xc_potential",
            ),
            None,
        )

    lsda_evaluation = evaluate_lsda_terms(
        rho_up=rho_up,
        rho_down=rho_down,
        functional=case.reference_model.xc if xc_functional is None else xc_functional,
    )
    potential = lsda_evaluation.v_xc_up if spin_channel == "up" else lsda_evaluation.v_xc_down
    return potential, lsda_evaluation


def _resolve_static_local_ionic_potential(
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase,
    rho_total: np.ndarray,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None,
    *,
    use_monitor_patch: bool,
    patch_parameters: LocalPotentialPatchParameters | None,
    base_local_ionic_evaluation: LocalIonicPotentialEvaluation | None = None,
) -> tuple[
    np.ndarray,
    LocalIonicPotentialEvaluation | None,
    FrozenPatchLocalPotentialEmbedding | None,
]:
    if local_ionic_potential is not None:
        if isinstance(local_ionic_potential, LocalIonicPotentialEvaluation):
            return local_ionic_potential.total_local_potential, local_ionic_potential, None
        return (
            validate_orbital_field(
                local_ionic_potential,
                grid_geometry=grid_geometry,
                name="local_ionic_potential",
            ),
            None,
            None,
        )

    if isinstance(grid_geometry, MonitorGridGeometry) and use_monitor_patch:
        embedding = evaluate_monitor_grid_local_ionic_potential_with_frozen_patch_field(
            case=case,
            grid_geometry=grid_geometry,
            density_field=rho_total,
            patch_parameters=patch_parameters,
            base_evaluation=base_local_ionic_evaluation,
        )
        return (
            embedding.corrected_total_local_potential,
            embedding.base_evaluation,
            embedding,
        )

    local_field, local_evaluation = _resolve_local_ionic_potential(
        grid_geometry=grid_geometry,
        case=case,
        local_ionic_potential=local_ionic_potential,
    )
    return local_field, local_evaluation, None


def prepare_fixed_potential_static_ks_operator(
    grid_geometry: StructuredGridGeometry,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray | None = None,
    xc_potential: np.ndarray | None = None,
    xc_functional: str | None = None,
) -> FixedPotentialOperatorContext:
    """Freeze the current static-KS local terms for a fixed-density solve."""

    rho_up_field = validate_orbital_field(rho_up, grid_geometry=grid_geometry, name="rho_up")
    rho_down_field = validate_orbital_field(rho_down, grid_geometry=grid_geometry, name="rho_down")
    normalized_spin = _normalize_spin_channel(spin_channel)
    rho_total = _build_total_density_on_grid(
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        grid_geometry=grid_geometry,
    )
    local_field, local_evaluation = _resolve_local_ionic_potential(
        grid_geometry=grid_geometry,
        case=case,
        local_ionic_potential=local_ionic_potential,
    )
    hartree_field, hartree_result = _resolve_hartree_potential(
        grid_geometry=grid_geometry,
        rho_total=rho_total,
        hartree_potential=hartree_potential,
    )
    xc_field, lsda_evaluation = _resolve_xc_potential(
        grid_geometry=grid_geometry,
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        spin_channel=normalized_spin,
        case=case,
        xc_potential=xc_potential,
        xc_functional=xc_functional,
    )
    return FixedPotentialOperatorContext(
        grid_geometry=grid_geometry,
        case=case,
        spin_channel=normalized_spin,
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        rho_total=rho_total,
        local_ionic_potential=local_field,
        hartree_potential=hartree_field,
        xc_potential=xc_field,
        effective_local_potential=np.asarray(local_field + hartree_field + xc_field, dtype=np.float64),
        local_ionic_evaluation=local_evaluation,
        hartree_poisson_result=hartree_result,
        lsda_evaluation=lsda_evaluation,
    )


def apply_fixed_potential_static_ks_operator(
    psi: np.ndarray,
    operator_context: FixedPotentialOperatorContext,
) -> np.ndarray:
    """Apply the frozen static-KS Hamiltonian to one orbital field.

    This is a thin wrapper over the current static-KS Hamiltonian apply path with
    the density-derived local terms frozen once in the operator context.
    """

    return apply_static_ks_hamiltonian(
        psi=psi,
        grid_geometry=operator_context.grid_geometry,
        rho_up=operator_context.rho_up,
        rho_down=operator_context.rho_down,
        spin_channel=operator_context.spin_channel,
        case=operator_context.case,
        local_ionic_potential=operator_context.local_ionic_potential,
        hartree_potential=operator_context.hartree_potential,
        xc_potential=operator_context.xc_potential,
    )


def build_fixed_potential_static_ks_operator(
    operator_context: FixedPotentialOperatorContext,
):
    """Return a callable frozen-potential operator wrapper for the eigensolver."""

    def _operator(psi: np.ndarray) -> np.ndarray:
        return apply_fixed_potential_static_ks_operator(
            psi=psi,
            operator_context=operator_context,
        )

    return _operator


def prepare_fixed_potential_static_local_operator_profiled(
    grid_geometry: GridGeometryLike,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray | None = None,
    xc_potential: np.ndarray | None = None,
    xc_functional: str | None = None,
    *,
    use_monitor_patch: bool = False,
    patch_parameters: LocalPotentialPatchParameters | None = None,
    kinetic_version: str = "production",
    base_local_ionic_evaluation: LocalIonicPotentialEvaluation | None = None,
    hartree_backend: str = "python",
    use_jax_hartree_cached_operator: bool = False,
    jax_hartree_cg_impl: str = "baseline",
    jax_hartree_cg_preconditioner: str = "none",
) -> tuple[
    FixedPotentialStaticLocalOperatorContext,
    FixedPotentialStaticLocalPreparationProfile,
]:
    """Freeze the static local chain `T + V_loc + V_H + V_xc` on one grid.

    This operator intentionally excludes nonlocal ionic action and any SCF
    update. When `use_monitor_patch=True` on the monitor grid, the current
    near-core local-GTH patch correction is embedded into a frozen local
    potential field matched to the chosen frozen density.
    """

    total_start = time.perf_counter()
    rho_up_field = validate_orbital_field(rho_up, grid_geometry=grid_geometry, name="rho_up")
    rho_down_field = validate_orbital_field(rho_down, grid_geometry=grid_geometry, name="rho_down")
    normalized_spin = _normalize_spin_channel(spin_channel)
    normalized_kinetic_version = _normalize_static_local_kinetic_version(kinetic_version)
    density_start = time.perf_counter()
    rho_total = _build_total_density_on_grid(
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        grid_geometry=grid_geometry,
    )
    density_elapsed = time.perf_counter() - density_start
    local_start = time.perf_counter()
    local_field, local_evaluation, patch_embedding = _resolve_static_local_ionic_potential(
        grid_geometry=grid_geometry,
        case=case,
        rho_total=rho_total,
        local_ionic_potential=local_ionic_potential,
        use_monitor_patch=use_monitor_patch,
        patch_parameters=patch_parameters,
        base_local_ionic_evaluation=base_local_ionic_evaluation,
    )
    local_elapsed = time.perf_counter() - local_start
    hartree_start = time.perf_counter()
    hartree_field, hartree_result = _resolve_hartree_potential(
        grid_geometry=grid_geometry,
        rho_total=rho_total,
        hartree_potential=hartree_potential,
        hartree_backend=hartree_backend,
        use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
        jax_hartree_cg_impl=jax_hartree_cg_impl,
        jax_hartree_cg_preconditioner=jax_hartree_cg_preconditioner,
    )
    hartree_elapsed = time.perf_counter() - hartree_start
    xc_start = time.perf_counter()
    xc_field, lsda_evaluation = _resolve_xc_potential(
        grid_geometry=grid_geometry,
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        spin_channel=normalized_spin,
        case=case,
        xc_potential=xc_potential,
        xc_functional=xc_functional,
    )
    xc_elapsed = time.perf_counter() - xc_start
    context = FixedPotentialStaticLocalOperatorContext(
        grid_geometry=grid_geometry,
        case=case,
        spin_channel=normalized_spin,
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        rho_total=rho_total,
        local_ionic_potential=local_field,
        hartree_potential=hartree_field,
        xc_potential=xc_field,
        effective_local_potential=np.asarray(
            local_field + hartree_field + xc_field,
            dtype=np.float64,
        ),
        local_ionic_evaluation=local_evaluation,
        frozen_patch_local_embedding=patch_embedding,
        hartree_poisson_result=hartree_result,
        lsda_evaluation=lsda_evaluation,
        kinetic_version=normalized_kinetic_version,
    )
    profile = FixedPotentialStaticLocalPreparationProfile(
        density_build_wall_time_seconds=float(density_elapsed),
        local_ionic_resolve_wall_time_seconds=float(local_elapsed),
        hartree_resolve_wall_time_seconds=float(hartree_elapsed),
        xc_resolve_wall_time_seconds=float(xc_elapsed),
        total_wall_time_seconds=float(time.perf_counter() - total_start),
    )
    return context, profile


def prepare_fixed_potential_static_local_operator(
    grid_geometry: GridGeometryLike,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray | None = None,
    xc_potential: np.ndarray | None = None,
    xc_functional: str | None = None,
    *,
    use_monitor_patch: bool = False,
    patch_parameters: LocalPotentialPatchParameters | None = None,
    kinetic_version: str = "production",
    base_local_ionic_evaluation: LocalIonicPotentialEvaluation | None = None,
    hartree_backend: str = "python",
    use_jax_hartree_cached_operator: bool = False,
    jax_hartree_cg_impl: str = "baseline",
    jax_hartree_cg_preconditioner: str = "none",
) -> FixedPotentialStaticLocalOperatorContext:
    """Freeze the static local chain `T + V_loc + V_H + V_xc` on one grid."""

    context, _ = prepare_fixed_potential_static_local_operator_profiled(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel=spin_channel,
        case=case,
        local_ionic_potential=local_ionic_potential,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        xc_functional=xc_functional,
        use_monitor_patch=use_monitor_patch,
        patch_parameters=patch_parameters,
        kinetic_version=kinetic_version,
        base_local_ionic_evaluation=base_local_ionic_evaluation,
        hartree_backend=hartree_backend,
        use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
        jax_hartree_cg_impl=jax_hartree_cg_impl,
        jax_hartree_cg_preconditioner=jax_hartree_cg_preconditioner,
    )
    return context


def apply_fixed_potential_static_local_operator(
    psi: np.ndarray,
    operator_context: FixedPotentialStaticLocalOperatorContext,
) -> np.ndarray:
    """Apply the frozen static local Hamiltonian to one orbital field."""

    field = validate_orbital_field(
        psi,
        grid_geometry=operator_context.grid_geometry,
        name="psi",
    )
    if (
        operator_context.kinetic_version == "trial_fix"
        and isinstance(operator_context.grid_geometry, MonitorGridGeometry)
    ):
        kinetic_action = apply_monitor_grid_kinetic_operator_trial_boundary_fix(
            psi=field,
            grid_geometry=operator_context.grid_geometry,
        )
    else:
        kinetic_action = apply_kinetic_operator(
            psi=field,
            grid_geometry=operator_context.grid_geometry,
        )
    return np.asarray(
        kinetic_action + operator_context.effective_local_potential * field,
        dtype=np.float64,
    )


def build_fixed_potential_static_local_operator(
    operator_context: FixedPotentialStaticLocalOperatorContext,
):
    """Return a callable frozen static-local operator wrapper."""

    def _operator(psi: np.ndarray) -> np.ndarray:
        return apply_fixed_potential_static_local_operator(
            psi=psi,
            operator_context=operator_context,
        )

    return _operator


def apply_fixed_potential_static_local_block(
    orbitals: np.ndarray,
    operator_context: FixedPotentialStaticLocalOperatorContext,
) -> np.ndarray:
    """Apply the frozen static-local Hamiltonian to one orbital block."""

    block = validate_orbital_block(orbitals, grid_geometry=operator_context.grid_geometry)
    actions = [
        apply_fixed_potential_static_local_operator(psi=orbital, operator_context=operator_context)
        for orbital in block
    ]
    return np.asarray(actions)


def apply_fixed_potential_static_local_block_jax_hotpath(
    orbitals: np.ndarray,
    operator_context: FixedPotentialStaticLocalOperatorContext,
) -> np.ndarray:
    """Apply the frozen static-local Hamiltonian block through the JAX hot path."""

    from isogrid.ks.eigensolver_jax_cache import apply_fixed_potential_static_local_block_cached_jax

    return np.asarray(
        apply_fixed_potential_static_local_block_cached_jax(
            orbitals,
            operator_context=operator_context,
        ),
        dtype=np.float64,
    )


def apply_fixed_potential_static_ks_block(
    orbitals: np.ndarray,
    operator_context: FixedPotentialOperatorContext,
) -> np.ndarray:
    """Apply the frozen static-KS Hamiltonian to a block of orbitals."""

    block = validate_orbital_block(orbitals, grid_geometry=operator_context.grid_geometry)
    actions = [
        apply_fixed_potential_static_ks_operator(psi=orbital, operator_context=operator_context)
        for orbital in block
    ]
    return np.asarray(actions)


def _build_default_guess_orbitals(
    operator_context: FixedPotentialOperatorContext | FixedPotentialStaticLocalOperatorContext,
    count: int,
) -> np.ndarray:
    """Build deterministic, symmetry-friendly first-stage guess orbitals."""

    grid_geometry = operator_context.grid_geometry
    if isinstance(grid_geometry, MonitorGridGeometry):
        center = np.asarray(
            (
                0.5 * (grid_geometry.spec.box_bounds[0][0] + grid_geometry.spec.box_bounds[0][1]),
                0.5 * (grid_geometry.spec.box_bounds[1][0] + grid_geometry.spec.box_bounds[1][1]),
                0.5 * (grid_geometry.spec.box_bounds[2][0] + grid_geometry.spec.box_bounds[2][1]),
            ),
            dtype=np.float64,
        )
    else:
        center = np.asarray(grid_geometry.spec.reference_center, dtype=np.float64)
    x_shift = grid_geometry.x_points - center[0]
    y_shift = grid_geometry.y_points - center[1]
    z_shift = grid_geometry.z_points - center[2]
    radius_squared = x_shift * x_shift + y_shift * y_shift + z_shift * z_shift

    guesses: list[np.ndarray] = []
    atom_gaussians = []
    for atom in operator_context.case.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        atom_gaussians.append(np.exp(-0.85 * (dx * dx + dy * dy + dz * dz)))
    if atom_gaussians:
        guesses.append(np.sum(atom_gaussians, axis=0))
        if len(atom_gaussians) >= 2:
            alternating = np.zeros_like(atom_gaussians[0])
            for index, atom_gaussian in enumerate(atom_gaussians):
                alternating += atom_gaussian if index % 2 == 0 else -atom_gaussian
            guesses.append(alternating)

    base = np.exp(-0.65 * radius_squared)
    guesses.append(base)
    guesses.append(z_shift * np.exp(-0.55 * radius_squared))
    guesses.append(x_shift * np.exp(-0.55 * radius_squared))
    guesses.append(y_shift * np.exp(-0.55 * radius_squared))
    guesses.append(np.exp(-1.10 * radius_squared))
    guesses.append((3.0 * z_shift * z_shift - radius_squared) * np.exp(-0.45 * radius_squared))

    rng = np.random.default_rng(12345)
    while len(guesses) < max(count, 8):
        random_field = rng.normal(size=grid_geometry.spec.shape)
        guesses.append(base * random_field)

    guess_block = np.asarray(guesses[: max(count, 8)], dtype=np.float64)
    orthonormal_guess = weighted_orthonormalize_orbitals(
        guess_block,
        grid_geometry=grid_geometry,
        require_full_rank=False,
    )
    if orthonormal_guess.shape[0] < count:
        raise ValueError(
            "Unable to build enough independent default guess orbitals for the fixed-potential solve."
        )
    return orthonormal_guess[:count]


def _build_initial_guess_block(
    operator_context: FixedPotentialOperatorContext | FixedPotentialStaticLocalOperatorContext,
    k: int,
    basis_size: int,
    initial_guess_orbitals: np.ndarray | None,
    *,
    use_jax_block_kernels: bool = False,
) -> np.ndarray:
    target_size = max(k, basis_size)
    default_guesses = _build_default_guess_orbitals(
        operator_context=operator_context,
        count=target_size,
    )
    if initial_guess_orbitals is None:
        return default_guesses[:target_size]

    user_guess = validate_orbital_block(
        initial_guess_orbitals,
        grid_geometry=operator_context.grid_geometry,
    )
    combined_guess = np.concatenate([user_guess, default_guesses], axis=0)
    orthonormal_guess = weighted_orthonormalize_orbitals(
        combined_guess,
        grid_geometry=operator_context.grid_geometry,
        require_full_rank=False,
        use_jax_block_kernels=use_jax_block_kernels,
    )
    if orthonormal_guess.shape[0] < k:
        raise ValueError(
            "The provided initial guesses do not span enough weighted-independent directions."
        )
    return orthonormal_guess[:target_size]


def _residual_norms(
    orbitals: np.ndarray,
    actions: np.ndarray,
    eigenvalues: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    """Return weighted residual norms ||H psi - eps psi||_W for one orbital block."""

    block = validate_orbital_block(orbitals, grid_geometry=grid_geometry)
    action_block = validate_orbital_block(actions, grid_geometry=grid_geometry, name="actions")
    if action_block.shape[0] != block.shape[0]:
        raise ValueError("actions must contain one field per orbital.")
    residuals = []
    for index, eigenvalue in enumerate(np.asarray(eigenvalues, dtype=np.float64)):
        residual = action_block[index] - eigenvalue * block[index]
        residuals.append(weighted_l2_norm(residual, grid_geometry=grid_geometry))
    return np.asarray(residuals, dtype=np.float64)


def _require_scipy_iterative_solver():
    try:
        from scipy.sparse.linalg import ArpackNoConvergence
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import eigsh
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required to run the first fixed-potential eigensolver scaffold."
        ) from exc
    return LinearOperator, eigsh, ArpackNoConvergence


def _build_weighted_euclidean_operator(
    *,
    operator_context: FixedPotentialOperatorContext | FixedPotentialStaticLocalOperatorContext,
    block_apply,
    use_jax_block_kernels: bool = False,
):
    LinearOperator, _, _ = _require_scipy_iterative_solver()
    grid_geometry = operator_context.grid_geometry
    weights = grid_geometry.cell_volumes.reshape(-1)
    sqrt_weights = np.sqrt(weights)
    inverse_sqrt_weights = 1.0 / sqrt_weights
    n_grid = weights.size

    def _apply_columns(columns: np.ndarray):
        values = np.asarray(columns, dtype=np.float64)
        original_vector = values.ndim == 1
        if original_vector:
            values = values[:, np.newaxis]
        orbital_block = reshape_orbital_columns(
            inverse_sqrt_weights[:, None] * values,
            grid_geometry=grid_geometry,
        )
        if (
            use_jax_block_kernels
            and isinstance(operator_context, FixedPotentialStaticLocalOperatorContext)
            and isinstance(grid_geometry, MonitorGridGeometry)
        ):
            action_block = apply_fixed_potential_static_local_block_jax_hotpath(
                orbital_block,
                operator_context=operator_context,
            )
        else:
            action_block = block_apply(orbital_block, operator_context=operator_context)
        action_columns = flatten_orbital_block(action_block, grid_geometry=grid_geometry)
        transformed = sqrt_weights[:, None] * action_columns
        return transformed[:, 0] if original_vector else transformed

    operator = LinearOperator(
        shape=(n_grid, n_grid),
        matvec=_apply_columns,
        matmat=_apply_columns,
        dtype=np.float64,
    )
    return operator, sqrt_weights, inverse_sqrt_weights


def _solve_weighted_fixed_potential_problem(
    *,
    operator_context: FixedPotentialOperatorContext | FixedPotentialStaticLocalOperatorContext,
    block_apply,
    k: int,
    initial_guess_orbitals: np.ndarray | None,
    max_iterations: int,
    tolerance: float,
    initial_subspace_size: int | None,
    ncv: int | None,
    use_jax_block_kernels: bool = False,
) -> FixedPotentialEigensolverResult:
    if k <= 0:
        raise ValueError(f"k must be positive; received {k}.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive; received {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive; received {tolerance}.")
    if initial_subspace_size is None:
        initial_subspace_size = max(2 * k, 4)
    if initial_subspace_size < k:
        raise ValueError("initial_subspace_size must be at least k.")

    grid_geometry = operator_context.grid_geometry
    initial_guess = _build_initial_guess_block(
        operator_context=operator_context,
        k=k,
        basis_size=initial_subspace_size,
        initial_guess_orbitals=initial_guess_orbitals,
        use_jax_block_kernels=use_jax_block_kernels,
    )

    LinearOperator, eigsh, ArpackNoConvergence = _require_scipy_iterative_solver()
    del LinearOperator
    weighted_operator, sqrt_weights, inverse_sqrt_weights = _build_weighted_euclidean_operator(
        operator_context=operator_context,
        block_apply=block_apply,
        use_jax_block_kernels=use_jax_block_kernels,
    )

    guess_columns = flatten_orbital_block(initial_guess, grid_geometry=grid_geometry)
    v0 = sqrt_weights * guess_columns[:, 0]
    v0 = np.asarray(v0 / np.linalg.norm(v0), dtype=np.float64)

    n_grid = int(np.prod(grid_geometry.spec.shape))
    if ncv is None:
        ncv = max(20, 2 * k + 1)
    ncv = int(min(max(ncv, k + 2), n_grid - 1))

    solver_note = "SciPy eigsh does not expose exact iteration counts; iteration_count is -1."
    converged = True
    solve_start = time.perf_counter()
    try:
        eigenvalues, eigenvectors = eigsh(
            weighted_operator,
            k=k,
            which="SA",
            v0=v0,
            tol=tolerance,
            maxiter=max_iterations,
            ncv=ncv,
        )
    except ArpackNoConvergence as exc:
        converged = False
        eigenvalues = exc.eigenvalues
        eigenvectors = exc.eigenvectors
        solver_note = (
            "SciPy eigsh reached the iteration limit before full convergence; "
            "partial Ritz vectors are returned."
        )
    wall_time_seconds = time.perf_counter() - solve_start

    if eigenvalues is None or eigenvectors is None or len(eigenvalues) < k:
        raise RuntimeError(
            "The fixed-potential eigensolver did not return enough Ritz pairs for the requested k."
        )

    order = np.argsort(np.asarray(eigenvalues, dtype=np.float64))[:k]
    selected_eigenvalues = np.asarray(np.asarray(eigenvalues)[order], dtype=np.float64)
    selected_vectors = np.asarray(eigenvectors[:, order], dtype=np.float64)

    orbitals = reshape_orbital_columns(
        inverse_sqrt_weights[:, None] * selected_vectors,
        grid_geometry=grid_geometry,
    )
    actions = block_apply(orbitals, operator_context=operator_context)
    residual_norms = _residual_norms(
        orbitals,
        actions,
        selected_eigenvalues,
        grid_geometry=grid_geometry,
    )
    weighted_overlap = np.asarray(
        weighted_overlap_matrix(
            orbitals,
            grid_geometry=grid_geometry,
            use_jax_block_kernels=use_jax_block_kernels,
        ),
        dtype=np.float64,
    )
    ritz_matrix = np.asarray(
        weighted_overlap_matrix(
            orbitals,
            grid_geometry=grid_geometry,
            other=actions,
            use_jax_block_kernels=use_jax_block_kernels,
        ),
        dtype=np.float64,
    )
    ritz_matrix = 0.5 * (ritz_matrix + ritz_matrix.T)
    max_orthogonality_error = float(
        np.max(np.abs(weighted_overlap - np.eye(k, dtype=np.float64)))
    )
    converged = bool(converged and float(np.max(residual_norms)) <= tolerance)

    return FixedPotentialEigensolverResult(
        target_orbitals=k,
        solver_method="scipy_eigsh_lanczos",
        solver_note=solver_note,
        converged=converged,
        iteration_count=-1,
        tolerance=float(tolerance),
        eigenvalues=selected_eigenvalues,
        orbitals=np.asarray(orbitals, dtype=np.float64),
        weighted_overlap=weighted_overlap,
        max_orthogonality_error=max_orthogonality_error,
        residual_norms=residual_norms,
        residual_history=residual_norms[np.newaxis, :],
        ritz_value_history=selected_eigenvalues[np.newaxis, :],
        subspace_dimensions=(ncv,),
        ritz_matrix=ritz_matrix,
        initial_guess_orbitals=np.asarray(initial_guess, dtype=np.float64),
        final_basis_orbitals=np.asarray(orbitals, dtype=np.float64),
        operator_context=operator_context,
        static_local_preparation_profile=None,
        use_jax_block_kernels=use_jax_block_kernels,
        use_jax_cached_kernels=bool(use_jax_block_kernels),
        wall_time_seconds=float(wall_time_seconds),
    )


def solve_fixed_potential_eigenproblem(
    grid_geometry: StructuredGridGeometry,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    k: int,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    initial_guess_orbitals: np.ndarray | None = None,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray | None = None,
    xc_potential: np.ndarray | None = None,
    xc_functional: str | None = None,
    max_iterations: int = 400,
    tolerance: float = 1.0e-3,
    initial_subspace_size: int | None = None,
    ncv: int | None = None,
) -> FixedPotentialEigensolverResult:
    """Solve the lowest few frozen-potential static-KS orbitals.

    The current first-stage default route is SciPy's iterative symmetric Lanczos
    solver `eigsh` applied to the weighted-similarity-transformed operator. This
    is a formal fixed-potential scaffold, not the final JAX-native SCF solver.
    """

    operator_context = prepare_fixed_potential_static_ks_operator(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel=spin_channel,
        case=case,
        local_ionic_potential=local_ionic_potential,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        xc_functional=xc_functional,
    )
    return _solve_weighted_fixed_potential_problem(
        operator_context=operator_context,
        block_apply=apply_fixed_potential_static_ks_block,
        k=k,
        initial_guess_orbitals=initial_guess_orbitals,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_subspace_size=initial_subspace_size,
        ncv=ncv,
    )


def solve_fixed_potential_static_local_eigenproblem(
    grid_geometry: GridGeometryLike,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    k: int,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    initial_guess_orbitals: np.ndarray | None = None,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray | None = None,
    xc_potential: np.ndarray | None = None,
    xc_functional: str | None = None,
    operator_context: FixedPotentialStaticLocalOperatorContext | None = None,
    max_iterations: int = 400,
    tolerance: float = 1.0e-3,
    initial_subspace_size: int | None = None,
    ncv: int | None = None,
    *,
    use_monitor_patch: bool = False,
    patch_parameters: LocalPotentialPatchParameters | None = None,
    kinetic_version: str = "production",
    use_jax_block_kernels: bool = False,
    operator_preparation_profile: FixedPotentialStaticLocalPreparationProfile | None = None,
    base_local_ionic_evaluation: LocalIonicPotentialEvaluation | None = None,
    hartree_backend: str = "python",
    use_jax_hartree_cached_operator: bool = False,
    jax_hartree_cg_impl: str = "baseline",
    jax_hartree_cg_preconditioner: str = "none",
) -> FixedPotentialEigensolverResult:
    """Solve the lowest few frozen-potential orbitals of the static local chain.

    This route intentionally contains only

        T + V_loc,ion + V_H + V_xc

    with frozen density-derived local terms and no nonlocal ionic action.
    """

    if operator_context is None:
        operator_context, operator_preparation_profile = (
            prepare_fixed_potential_static_local_operator_profiled(
                grid_geometry=grid_geometry,
                rho_up=rho_up,
                rho_down=rho_down,
                spin_channel=spin_channel,
                case=case,
                local_ionic_potential=local_ionic_potential,
                hartree_potential=hartree_potential,
                xc_potential=xc_potential,
                xc_functional=xc_functional,
                use_monitor_patch=use_monitor_patch,
                patch_parameters=patch_parameters,
                kinetic_version=kinetic_version,
                base_local_ionic_evaluation=base_local_ionic_evaluation,
                hartree_backend=hartree_backend,
                use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
                jax_hartree_cg_impl=jax_hartree_cg_impl,
                jax_hartree_cg_preconditioner=jax_hartree_cg_preconditioner,
            )
        )
    result = _solve_weighted_fixed_potential_problem(
        operator_context=operator_context,
        block_apply=apply_fixed_potential_static_local_block,
        k=k,
        initial_guess_orbitals=initial_guess_orbitals,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_subspace_size=initial_subspace_size,
        ncv=ncv,
        use_jax_block_kernels=use_jax_block_kernels,
    )
    return FixedPotentialEigensolverResult(
        target_orbitals=result.target_orbitals,
        solver_method=result.solver_method,
        solver_note=result.solver_note,
        converged=result.converged,
        iteration_count=result.iteration_count,
        tolerance=result.tolerance,
        eigenvalues=result.eigenvalues,
        orbitals=result.orbitals,
        weighted_overlap=result.weighted_overlap,
        max_orthogonality_error=result.max_orthogonality_error,
        residual_norms=result.residual_norms,
        residual_history=result.residual_history,
        ritz_value_history=result.ritz_value_history,
        subspace_dimensions=result.subspace_dimensions,
        ritz_matrix=result.ritz_matrix,
        initial_guess_orbitals=result.initial_guess_orbitals,
        final_basis_orbitals=result.final_basis_orbitals,
        operator_context=result.operator_context,
        static_local_preparation_profile=operator_preparation_profile,
        use_jax_block_kernels=result.use_jax_block_kernels,
        use_jax_cached_kernels=result.use_jax_cached_kernels,
        wall_time_seconds=result.wall_time_seconds,
    )
