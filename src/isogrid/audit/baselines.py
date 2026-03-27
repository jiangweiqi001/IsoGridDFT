"""Lightweight regression baselines for the current H2 audit path.

These values are not acceptance targets. They are the first recorded audit
baseline for the current minimal H2 SCF closed loop against the PySCF
reference under the shared nominal model:

- H2 at R = 1.4 Bohr
- UKS / gth-pade / gth-dzvp / lda,vwn

The numeric values below are intentionally stored at the same precision used by
the first formal audit report so that later numerical changes can be compared
against one clear reference point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class H2PySCFRegressionBaseline:
    """Recorded H2 error baseline for the current minimal SCF implementation."""

    benchmark_name: str
    geometry_label: str
    reference_model_summary: str
    singlet_total_error_ha: float
    singlet_total_error_mha: float
    triplet_total_error_ha: float
    triplet_total_error_mha: float
    singlet_triplet_gap_error_ha: float
    singlet_triplet_gap_error_mha: float
    lower_spin_state_isogrid: str
    lower_spin_state_pyscf: str
    grid_shape: tuple[int, int, int]
    min_cell_widths_bohr: tuple[float, float, float]
    box_half_extents_bohr: tuple[float, float, float]
    density_tolerance: float
    eigensolver_tolerance: float
    mixing: float
    max_iterations: int
    eigensolver_ncv: int


@dataclass(frozen=True)
class H2MonitorPoissonShapeRegressionPoint:
    """Recorded A-grid shape-scan summary for the operator audit."""

    shape: tuple[int, int, int]
    hartree_energy_ha: float
    residual_rms: float
    negative_interior_fraction: float
    center_potential_ha: float
    delta_vs_baseline_mha: float


@dataclass(frozen=True)
class H2MonitorPoissonRegressionBaseline:
    """Recorded H2 monitor-grid Poisson audit baseline after the split fix."""

    benchmark_name: str
    density_label: str
    box_half_extents_bohr: tuple[float, float, float]
    monitor_shape: tuple[int, int, int]
    legacy_hartree_energy_ha: float
    monitor_hartree_energy_ha: float
    monitor_vs_legacy_delta_mha: float
    monitor_negative_interior_fraction: float
    monitor_full_residual_rms: float
    legacy_far_field_centerline_v_ha: float
    monitor_far_field_centerline_v_ha: float
    shape_scan: tuple[H2MonitorPoissonShapeRegressionPoint, ...]
    note: str


@dataclass(frozen=True)
class H2StaticLocalChainRouteBaseline:
    """Recorded static local-chain components for one H2 audit route."""

    path_type: str
    kinetic_energy_ha: float
    local_ionic_energy_ha: float
    hartree_energy_ha: float
    xc_energy_ha: float
    static_local_sum_ha: float


@dataclass(frozen=True)
class H2StaticLocalChainRegressionBaseline:
    """Recorded post-fix H2 static local-chain audit on legacy and A-grid routes."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_route: H2StaticLocalChainRouteBaseline
    monitor_route: H2StaticLocalChainRouteBaseline
    monitor_patch_route: H2StaticLocalChainRouteBaseline
    monitor_vs_legacy_delta_mha: float
    monitor_patch_vs_legacy_delta_mha: float
    monitor_patch_improvement_vs_monitor_mha: float
    note: str


@dataclass(frozen=True)
class H2HartreeTailRecheckPointBaseline:
    """Recorded H2 Hartree tail-recheck point on the A-grid."""

    point_label: str
    shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    hartree_energy_ha: float
    hartree_delta_vs_legacy_mha: float
    hartree_delta_vs_baseline_mha: float
    residual_rms: float
    negative_interior_fraction: float
    far_field_potential_mean_ha: float
    far_field_residual_rms: float
    centerline_far_field_potential_mean_ha: float


@dataclass(frozen=True)
class H2HartreeTailRecheckRegressionBaseline:
    """Recorded very small H2 Hartree tail-recheck after the split fix."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_hartree_energy_ha: float
    baseline_point: H2HartreeTailRecheckPointBaseline
    finer_shape_point: H2HartreeTailRecheckPointBaseline
    larger_box_point: H2HartreeTailRecheckPointBaseline
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2FixedPotentialEigensolverRouteBaseline:
    """Recorded fixed-potential eigensolver result for one audit route."""

    path_type: str
    target_orbitals: int
    eigenvalues_ha: tuple[float, ...]
    max_residual_norm: float
    max_orthogonality_error: float
    converged: bool
    kinetic_version: str = "production"
    use_jax_block_kernels: bool = False
    use_jax_cached_kernels: bool = False
    wall_time_seconds: float | None = None


@dataclass(frozen=True)
class H2FixedPotentialEigensolverRegressionBaseline:
    """Recorded fixed-potential eigensolver audit on legacy and A-grid+patch."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_k1_route: H2FixedPotentialEigensolverRouteBaseline
    monitor_patch_k1_route: H2FixedPotentialEigensolverRouteBaseline
    legacy_k2_route: H2FixedPotentialEigensolverRouteBaseline
    monitor_patch_k2_route: H2FixedPotentialEigensolverRouteBaseline
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2FixedPotentialOperatorRouteBaseline:
    """Recorded operator-level audit result for one fixed-potential route."""

    path_type: str
    eigenvalue_ha: float
    weighted_residual_norm: float
    converged: bool
    trial_rayleigh_ha: float
    trial_kinetic_ha: float
    trial_local_ionic_ha: float
    trial_hartree_ha: float
    trial_xc_ha: float
    eigen_rayleigh_ha: float
    eigen_kinetic_ha: float
    eigen_local_ionic_ha: float
    eigen_hartree_ha: float
    eigen_xc_ha: float
    self_adjoint_total_relative_difference: float
    self_adjoint_kinetic_relative_difference: float
    self_adjoint_local_relative_difference: float
    patch_embedded_correction_mha: float | None
    patch_embedding_energy_mismatch_ha: float | None
    kinetic_version: str = "production"


@dataclass(frozen=True)
class H2FixedPotentialTrialFixOperatorRegressionBaseline:
    """Recorded operator-level comparison after wiring in the kinetic trial fix."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_route: H2FixedPotentialOperatorRouteBaseline
    monitor_patch_production_route: H2FixedPotentialOperatorRouteBaseline
    monitor_patch_trial_fix_route: H2FixedPotentialOperatorRouteBaseline
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2FixedPotentialTrialFixEigensolverRegressionBaseline:
    """Recorded fixed-potential eigensolver comparison after the kinetic trial fix."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_k1_route: H2FixedPotentialEigensolverRouteBaseline
    monitor_patch_production_k1_route: H2FixedPotentialEigensolverRouteBaseline
    monitor_patch_trial_fix_k1_route: H2FixedPotentialEigensolverRouteBaseline
    legacy_k2_route: H2FixedPotentialEigensolverRouteBaseline
    monitor_patch_production_k2_route: H2FixedPotentialEigensolverRouteBaseline
    monitor_patch_trial_fix_k2_route: H2FixedPotentialEigensolverRouteBaseline
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2FixedPotentialOperatorRegressionBaseline:
    """Recorded operator-level failure baseline for the A-grid static-local path."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_route: H2FixedPotentialOperatorRouteBaseline
    monitor_unpatched_route: H2FixedPotentialOperatorRouteBaseline
    monitor_patch_route: H2FixedPotentialOperatorRouteBaseline
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2KineticOperatorRouteBaseline:
    """Recorded kinetic-operator audit result for one route and shape label."""

    path_type: str
    shape_label: str
    frozen_kinetic_ha: float
    eigen_kinetic_ha: float
    eigenvalue_ha: float
    eigensolver_residual_norm: float
    kinetic_self_adjoint_relative_difference: float
    eigen_negative_indicator_fraction: float
    eigen_far_field_contribution_ha: float
    eigen_center_contribution_ha: float
    converged: bool


@dataclass(frozen=True)
class H2KineticOperatorSmoothFieldBaseline:
    """Recorded kinetic probe result for one simple smooth field."""

    field_label: str
    legacy_kinetic_ha: float
    monitor_kinetic_ha: float


@dataclass(frozen=True)
class H2KineticOperatorRegressionBaseline:
    """Recorded kinetic-operator failure baseline on legacy and A-grid routes."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    finer_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_route: H2KineticOperatorRouteBaseline
    monitor_unpatched_baseline_route: H2KineticOperatorRouteBaseline
    monitor_patch_baseline_route: H2KineticOperatorRouteBaseline
    monitor_patch_finer_shape_route: H2KineticOperatorRouteBaseline
    smooth_fields: tuple[H2KineticOperatorSmoothFieldBaseline, ...]
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2KineticFormRouteBaseline:
    """Recorded production-vs-reference kinetic-form result for one field."""

    shape_label: str
    orbital_label: str
    production_kinetic_ha: float
    reference_kinetic_ha: float
    delta_kinetic_mha: float
    production_tpsi_rms: float
    reference_tpsi_rms: float
    delta_tpsi_rms: float
    centerline_midpoint_delta: float
    far_field_delta_weighted_rms: float
    eigensolver_eigenvalue_ha: float | None
    eigensolver_residual_norm: float | None
    eigensolver_converged: bool | None


@dataclass(frozen=True)
class H2KineticFormSmoothFieldBaseline:
    """Recorded kinetic-form comparison for one smooth probe field."""

    field_label: str
    production_kinetic_ha: float
    reference_kinetic_ha: float
    delta_kinetic_mha: float


@dataclass(frozen=True)
class H2KineticFormRegressionBaseline:
    """Recorded production-vs-reference kinetic-form failure baseline."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    finer_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    frozen_trial_baseline: H2KineticFormRouteBaseline
    bad_eigen_baseline: H2KineticFormRouteBaseline
    bad_eigen_finer_shape: H2KineticFormRouteBaseline
    smooth_fields: tuple[H2KineticFormSmoothFieldBaseline, ...]
    production_self_adjoint_baseline_rel: float
    reference_self_adjoint_baseline_rel: float
    production_self_adjoint_finer_rel: float
    reference_self_adjoint_finer_rel: float
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2GeometryConsistencyFieldBaseline:
    """Recorded operator-vs-gradient kinetic identity for one field."""

    field_label: str
    shape_label: str
    operator_kinetic_ha: float
    gradient_reference_ha: float
    delta_kinetic_mha: float
    operator_tpsi_rms: float
    gradient_indicator_rms: float
    delta_indicator_rms: float
    far_field_operator_contribution_ha: float
    far_field_gradient_contribution_ha: float
    far_field_delta_mha: float
    source_eigenvalue_ha: float | None
    source_residual_norm: float | None
    source_converged: bool | None


@dataclass(frozen=True)
class H2GeometryConsistencySmoothFieldBaseline:
    """Recorded geometry-consistency probe for one smooth field."""

    field_label: str
    operator_kinetic_ha: float
    gradient_reference_ha: float
    delta_kinetic_mha: float


@dataclass(frozen=True)
class H2GeometryConsistencyRegressionBaseline:
    """Recorded geometry-consistency failure baseline on the A-grid."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    jacobian_reconstruction_relative_rms: float
    inverse_metric_reconstruction_relative_rms: float
    cell_volume_vs_jacobian_relative_rms: float
    sqrt_det_metric_vs_jacobian_relative_rms: float
    metric_inverse_identity_rms: float
    legacy_frozen_kinetic_reference_ha: float
    legacy_bad_eigen_kinetic_reference_ha: float
    frozen_trial_baseline: H2GeometryConsistencyFieldBaseline
    bad_eigen_baseline: H2GeometryConsistencyFieldBaseline
    smooth_fields: tuple[H2GeometryConsistencySmoothFieldBaseline, ...]
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2KineticGreenIdentityFieldBaseline:
    """Recorded discrete Green-identity audit result for one field."""

    shape_label: str
    field_label: str
    operator_kinetic_ha: float
    gradient_kinetic_ha: float
    delta_kinetic_mha: float
    boundary_term_ha: float
    closure_mismatch_mha: float
    far_field_delta_mha: float
    far_field_boundary_ha: float
    far_field_closure_mha: float
    source_eigenvalue_ha: float | None
    source_residual_norm: float | None
    source_converged: bool | None


@dataclass(frozen=True)
class H2KineticGreenIdentitySmoothFieldBaseline:
    """Recorded discrete Green-identity audit result for one smooth field."""

    field_label: str
    operator_kinetic_ha: float
    gradient_kinetic_ha: float
    delta_kinetic_mha: float
    boundary_term_ha: float
    closure_mismatch_mha: float


@dataclass(frozen=True)
class H2KineticGreenIdentityRegressionBaseline:
    """Recorded discrete Green-identity / boundary-mismatch failure baseline."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    finer_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    frozen_trial_baseline: H2KineticGreenIdentityFieldBaseline
    bad_eigen_baseline: H2KineticGreenIdentityFieldBaseline
    bad_eigen_finer_shape: H2KineticGreenIdentityFieldBaseline
    smooth_fields: tuple[H2KineticGreenIdentitySmoothFieldBaseline, ...]
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2OrbitalShapeOrbitalBaseline:
    """Recorded shape/symmetry summary for one fixed-potential orbital."""

    path_type: str
    kinetic_version: str
    orbital_label: str
    eigenvalue_ha: float
    residual_norm: float
    converged: bool
    inversion_best_parity: str
    inversion_best_mismatch: float
    z_mirror_best_parity: str
    z_mirror_best_mismatch: float
    centerline_sign_changes: int
    far_field_sign_changes: int
    node_positions_bohr: tuple[float, ...]
    far_field_norm_fraction: float
    boundary_layer_norm_fraction: float
    far_field_max_abs_value: float
    boundary_layer_max_abs_value: float


@dataclass(frozen=True)
class H2OrbitalShapeRegressionBaseline:
    """Recorded fixed-potential orbital-shape baseline for H2."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_k1_orbital: H2OrbitalShapeOrbitalBaseline
    legacy_k2_low_orbital: H2OrbitalShapeOrbitalBaseline
    legacy_k2_high_orbital: H2OrbitalShapeOrbitalBaseline
    monitor_trial_fix_k1_orbital: H2OrbitalShapeOrbitalBaseline
    monitor_trial_fix_k2_first_orbital: H2OrbitalShapeOrbitalBaseline
    monitor_trial_fix_k2_second_orbital: H2OrbitalShapeOrbitalBaseline
    legacy_k2_gap_ha: float
    monitor_trial_fix_k2_gap_ha: float
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2K2SubspaceOrbitalBaseline:
    """Recorded k=2 subspace orbital summary."""

    path_type: str
    kinetic_version: str
    orbital_label: str
    eigenvalue_ha: float
    residual_norm: float
    z_mirror_best_mismatch: float
    centerline_sign_changes: int
    far_field_sign_changes: int
    center_value: float
    far_field_norm_fraction: float
    boundary_layer_norm_fraction: float


@dataclass(frozen=True)
class H2K2SubspaceMatrixBaseline:
    """Recorded 2x2 subspace matrix summary."""

    label: str
    matrix: tuple[tuple[float, float], tuple[float, float]]
    eigenvalues: tuple[float, float]


@dataclass(frozen=True)
class H2K2SubspaceRotationBaseline:
    """Recorded very small k=2 subspace rotation summary."""

    rotation_label: str
    rotation_matrix: tuple[tuple[float, float], tuple[float, float]]
    raw_bonding_overlaps: tuple[float, float]
    rotated_bonding_overlaps: tuple[float, float]
    rotated_first_orbital: H2K2SubspaceOrbitalBaseline
    rotated_second_orbital: H2K2SubspaceOrbitalBaseline
    note: str


@dataclass(frozen=True)
class H2K2SubspaceRegressionBaseline:
    """Recorded H2 k=2 subspace audit baseline."""

    benchmark_name: str
    density_label: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    legacy_raw_eigenvalues_ha: tuple[float, float]
    legacy_raw_orbitals: tuple[H2K2SubspaceOrbitalBaseline, H2K2SubspaceOrbitalBaseline]
    monitor_raw_eigenvalues_ha: tuple[float, float]
    monitor_raw_orbitals: tuple[H2K2SubspaceOrbitalBaseline, H2K2SubspaceOrbitalBaseline]
    monitor_inversion_matrix: H2K2SubspaceMatrixBaseline
    monitor_z_mirror_matrix: H2K2SubspaceMatrixBaseline
    monitor_bonding_rotation: H2K2SubspaceRotationBaseline
    legacy_k2_gap_ha: float
    monitor_k2_gap_ha: float
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2ScfDryRunRouteBaseline:
    """Recorded H2 SCF dry-run result for one route and one spin state."""

    path_type: str
    spin_state_label: str
    kinetic_version: str
    includes_nonlocal: bool
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    lowest_eigenvalue_ha: float
    final_density_residual: float
    final_energy_change_ha: float


@dataclass(frozen=True)
class H2ScfDryRunRegressionBaseline:
    """Recorded H2 SCF dry-run baseline for legacy and A-grid routes."""

    benchmark_name: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    legacy_singlet_route: H2ScfDryRunRouteBaseline
    monitor_singlet_route: H2ScfDryRunRouteBaseline
    legacy_triplet_route: H2ScfDryRunRouteBaseline | None
    monitor_triplet_route: H2ScfDryRunRouteBaseline | None
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2SingletStabilityRouteBaseline:
    """Recorded singlet stability result for one conservative mixing scheme."""

    scheme_label: str
    kinetic_version: str
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str
    diis_used_iterations: tuple[int, ...]
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    final_energy_change_ha: float | None
    detected_two_cycle: bool
    two_cycle_verdict: str
    even_odd_energy_gap_ha: float | None
    even_odd_residual_gap: float | None


@dataclass(frozen=True)
class H2SingletStabilityRegressionBaseline:
    """Recorded very small A-grid singlet stability audit baseline."""

    benchmark_name: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    baseline_route: H2SingletStabilityRouteBaseline
    smaller_mixing_route: H2SingletStabilityRouteBaseline
    diis_prototype_route: H2SingletStabilityRouteBaseline
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2DiisScfRouteBaseline:
    """Recorded A-grid DIIS SCF result for one spin state and one scheme."""

    spin_state_label: str
    scheme_label: str
    kinetic_version: str
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str
    diis_used_iterations: tuple[int, ...]
    diis_fallback_iterations: tuple[int, ...]
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    final_energy_change_ha: float | None
    trajectory_verdict: str


@dataclass(frozen=True)
class H2DiisScfSpinBaseline:
    """Recorded three-scheme A-grid DIIS SCF baseline for one spin state."""

    spin_state_label: str
    baseline_route: H2DiisScfRouteBaseline
    smaller_mixing_route: H2DiisScfRouteBaseline
    diis_prototype_route: H2DiisScfRouteBaseline


@dataclass(frozen=True)
class H2DiisScfRegressionBaseline:
    """Recorded A-grid DIIS SCF audit baseline on singlet and triplet."""

    benchmark_name: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    singlet: H2DiisScfSpinBaseline
    triplet: H2DiisScfSpinBaseline
    diagnosis: str
    note: str


H2_DEFAULT_PYSCF_REGRESSION_BASELINE = H2PySCFRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    geometry_label="H2, R = 1.4 Bohr",
    reference_model_summary="UKS / gth-pade / gth-dzvp / lda,vwn",
    singlet_total_error_ha=-0.015231356907,
    singlet_total_error_mha=-15.231,
    triplet_total_error_ha=-0.037934329867,
    triplet_total_error_mha=-37.934,
    singlet_triplet_gap_error_ha=-0.022702972959,
    singlet_triplet_gap_error_mha=-22.703,
    lower_spin_state_isogrid="singlet",
    lower_spin_state_pyscf="singlet",
    grid_shape=(51, 51, 51),
    min_cell_widths_bohr=(0.132447424326, 0.132447424326, 0.137006128136),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    density_tolerance=2.5e-3,
    eigensolver_tolerance=5.0e-3,
    mixing=0.6,
    max_iterations=8,
    eigensolver_ncv=20,
)


H2_MONITOR_POISSON_REGRESSION_BASELINE = H2MonitorPoissonRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    monitor_shape=(67, 67, 81),
    legacy_hartree_energy_ha=1.748670071699,
    monitor_hartree_energy_ha=1.765755769546,
    monitor_vs_legacy_delta_mha=17.086,
    monitor_negative_interior_fraction=0.0,
    monitor_full_residual_rms=5.649e-09,
    legacy_far_field_centerline_v_ha=0.224739241566,
    monitor_far_field_centerline_v_ha=0.223365547675,
    shape_scan=(
        H2MonitorPoissonShapeRegressionPoint(
            shape=(59, 59, 71),
            hartree_energy_ha=1.769172021332,
            residual_rms=2.922e-09,
            negative_interior_fraction=0.0,
            center_potential_ha=2.457154825450,
            delta_vs_baseline_mha=3.416,
        ),
        H2MonitorPoissonShapeRegressionPoint(
            shape=(67, 67, 81),
            hartree_energy_ha=1.765755769546,
            residual_rms=5.649e-09,
            negative_interior_fraction=0.0,
            center_potential_ha=2.455055393670,
            delta_vs_baseline_mha=0.0,
        ),
        H2MonitorPoissonShapeRegressionPoint(
            shape=(75, 75, 91),
            hartree_energy_ha=1.763347405954,
            residual_rms=8.028e-09,
            negative_interior_fraction=0.0,
            center_potential_ha=2.454044482845,
            delta_vs_baseline_mha=-2.408,
        ),
    ),
    note=(
        "Post-fix monitor-grid Poisson operator baseline after repairing the monitor "
        "boundary-split / RHS sign consistency in open_boundary.py."
    ),
)


H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE = H2StaticLocalChainRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_route=H2StaticLocalChainRouteBaseline(
        path_type="legacy",
        kinetic_energy_ha=1.995064629672,
        local_ionic_energy_ha=-4.347356525602,
        hartree_energy_ha=1.748670071699,
        xc_energy_ha=-0.869107874980,
        static_local_sum_ha=-1.472729699211,
    ),
    monitor_route=H2StaticLocalChainRouteBaseline(
        path_type="monitor_a_grid",
        kinetic_energy_ha=1.934245385147,
        local_ionic_energy_ha=-4.364949512790,
        hartree_energy_ha=1.765755769546,
        xc_energy_ha=-0.868937109053,
        static_local_sum_ha=-1.533885467150,
    ),
    monitor_patch_route=H2StaticLocalChainRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        kinetic_energy_ha=1.934245385147,
        local_ionic_energy_ha=-4.287134389497,
        hartree_energy_ha=1.765755769546,
        xc_energy_ha=-0.868937109053,
        static_local_sum_ha=-1.456070343858,
    ),
    monitor_vs_legacy_delta_mha=-61.156,
    monitor_patch_vs_legacy_delta_mha=16.659,
    monitor_patch_improvement_vs_monitor_mha=77.815,
    note=(
        "Post-fix H2 static local-chain baseline after the monitor Poisson split repair. "
        "This baseline still excludes nonlocal, eigensolver, and SCF."
    ),
)


H2_HARTREE_TAIL_RECHECK_BASELINE = H2HartreeTailRecheckRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_hartree_energy_ha=1.748670071699,
    baseline_point=H2HartreeTailRecheckPointBaseline(
        point_label="baseline",
        shape=(67, 67, 81),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        hartree_energy_ha=1.765755769546,
        hartree_delta_vs_legacy_mha=17.086,
        hartree_delta_vs_baseline_mha=0.0,
        residual_rms=5.64945879475746e-09,
        negative_interior_fraction=0.0,
        far_field_potential_mean_ha=0.203771260565,
        far_field_residual_rms=1.4250923970591246e-09,
        centerline_far_field_potential_mean_ha=0.281858261947,
    ),
    finer_shape_point=H2HartreeTailRecheckPointBaseline(
        point_label="finer-shape",
        shape=(75, 75, 91),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        hartree_energy_ha=1.763347405954,
        hartree_delta_vs_legacy_mha=14.677,
        hartree_delta_vs_baseline_mha=-2.408,
        residual_rms=8.02802837283354e-09,
        negative_interior_fraction=0.0,
        far_field_potential_mean_ha=0.201446656961,
        far_field_residual_rms=2.5096069859814528e-09,
        centerline_far_field_potential_mean_ha=0.280005875175,
    ),
    larger_box_point=H2HartreeTailRecheckPointBaseline(
        point_label="larger-box",
        shape=(67, 67, 81),
        box_half_extents_bohr=(9.0, 9.0, 11.0),
        hartree_energy_ha=1.768774788860,
        hartree_delta_vs_legacy_mha=20.105,
        hartree_delta_vs_baseline_mha=3.019,
        residual_rms=4.691294017224396e-09,
        negative_interior_fraction=0.0,
        far_field_potential_mean_ha=0.179902839022,
        far_field_residual_rms=5.841016840151627e-10,
        centerline_far_field_potential_mean_ha=0.279733456838,
    ),
    diagnosis=(
        "The remaining +17 mHa Hartree offset looks more like an A-grid geometry / "
        "resolution tail than a surviving monitor-Poisson split bug: a slightly finer "
        "shape moves E_H in the right direction, while a slightly larger box does not."
    ),
    note=(
        "Very small post-fix Hartree tail-recheck for the H2 singlet frozen density on the "
        "A-grid+patch development baseline. Patch parameters stay frozen but patch does not "
        "directly modify Hartree."
    ),
)


H2_FIXED_POTENTIAL_EIGENSOLVER_BASELINE = H2FixedPotentialEigensolverRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="legacy",
        target_orbitals=1,
        eigenvalues_ha=(-0.20527465416922755,),
        max_residual_norm=2.8929876944581593e-04,
        max_orthogonality_error=1.1102230246251565e-15,
        converged=True,
    ),
    monitor_patch_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=1,
        eigenvalues_ha=(-6.574031909859388,),
        max_residual_norm=3.3342921454967853,
        max_orthogonality_error=1.1102230246251565e-15,
        converged=False,
    ),
    legacy_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="legacy",
        target_orbitals=2,
        eigenvalues_ha=(-0.20529016296596161, 0.06286688164303063),
        max_residual_norm=6.54810468883838e-04,
        max_orthogonality_error=2.220446049250313e-15,
        converged=True,
    ),
    monitor_patch_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=2,
        eigenvalues_ha=(-6.634541712761564, -6.219685689382547),
        max_residual_norm=7.833230053900307,
        max_orthogonality_error=6.661338147750939e-16,
        converged=False,
    ),
    diagnosis=(
        "The current A-grid+patch fixed-potential eigensolver path is not yet stable enough "
        "to replace the legacy route. Patch embedding reproduces the frozen local-GTH energy "
        "correction, but the monitor-grid static-local operator still yields overly deep "
        "eigenvalues and O(1) residuals."
    ),
    note=(
        "First fixed-potential eigensolver baseline on the repaired A-grid static local chain. "
        "This baseline still excludes nonlocal ionic action and SCF."
    ),
)


H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE = H2FixedPotentialOperatorRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_route=H2FixedPotentialOperatorRouteBaseline(
        path_type="legacy",
        eigenvalue_ha=-0.205274654169,
        weighted_residual_norm=2.892987694458e-04,
        converged=True,
        trial_rayleigh_ha=0.003887274419,
        trial_kinetic_ha=0.997532314836,
        trial_local_ionic_ha=-2.173678262801,
        trial_hartree_ha=1.748670071699,
        trial_xc_ha=-0.568636849315,
        eigen_rayleigh_ha=-0.205308969426,
        eigen_kinetic_ha=0.468898008999,
        eigen_local_ionic_ha=-1.683453804287,
        eigen_hartree_ha=1.407843104057,
        eigen_xc_ha=-0.398596278195,
        self_adjoint_total_relative_difference=7.775e-16,
        self_adjoint_kinetic_relative_difference=0.0,
        self_adjoint_local_relative_difference=0.0,
        patch_embedded_correction_mha=None,
        patch_embedding_energy_mismatch_ha=None,
    ),
    monitor_unpatched_route=H2FixedPotentialOperatorRouteBaseline(
        path_type="monitor_a_grid",
        eigenvalue_ha=-2.844168300574,
        weighted_residual_norm=1.237350032968,
        converged=False,
        trial_rayleigh_ha=-0.018116838274,
        trial_kinetic_ha=0.967122692573,
        trial_local_ionic_ha=-2.182474756395,
        trial_hartree_ha=1.765755769546,
        trial_xc_ha=-0.568520543999,
        eigen_rayleigh_ha=-1.642746017159,
        eigen_kinetic_ha=-1.642701348259,
        eigen_local_ionic_ha=-0.139440636607,
        eigen_hartree_ha=0.139412025095,
        eigen_xc_ha=-0.000016057387,
        self_adjoint_total_relative_difference=8.906e-16,
        self_adjoint_kinetic_relative_difference=1.596e-16,
        self_adjoint_local_relative_difference=0.0,
        patch_embedded_correction_mha=None,
        patch_embedding_energy_mismatch_ha=None,
    ),
    monitor_patch_route=H2FixedPotentialOperatorRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        eigenvalue_ha=-6.574031909859,
        weighted_residual_norm=3.334292145497,
        converged=False,
        trial_rayleigh_ha=0.020790723372,
        trial_kinetic_ha=0.967122692573,
        trial_local_ionic_ha=-2.143567194749,
        trial_hartree_ha=1.765755769546,
        trial_xc_ha=-0.568520543999,
        eigen_rayleigh_ha=-3.375004168561,
        eigen_kinetic_ha=-3.374908449316,
        eigen_local_ionic_ha=-0.137471599383,
        eigen_hartree_ha=0.137437670912,
        eigen_xc_ha=-0.000061790774,
        self_adjoint_total_relative_difference=1.448e-15,
        self_adjoint_kinetic_relative_difference=1.596e-16,
        self_adjoint_local_relative_difference=0.0,
        patch_embedded_correction_mha=77.815,
        patch_embedding_energy_mismatch_ha=0.0,
    ),
    diagnosis=(
        "The failure does not look like a weighted self-adjointness bug. The dominant anomaly "
        "appears in the A-grid kinetic contribution on the eigensolver-selected orbital, while "
        "patch on/off only changes the already-bad eigenvalue moderately."
    ),
    note=(
        "Failure regression baseline for the H2 fixed-potential static-local operator audit on "
        "legacy, A-grid, and A-grid+patch routes. This baseline is for diagnosis, not acceptance."
    ),
)


H2_FIXED_POTENTIAL_OPERATOR_TRIAL_FIX_BASELINE = H2FixedPotentialTrialFixOperatorRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_route=H2FixedPotentialOperatorRouteBaseline(
        path_type="legacy",
        eigenvalue_ha=-0.205274654169,
        weighted_residual_norm=2.892987694458e-04,
        converged=True,
        trial_rayleigh_ha=0.003887274419,
        trial_kinetic_ha=0.997532314836,
        trial_local_ionic_ha=-2.173678262801,
        trial_hartree_ha=1.748670071699,
        trial_xc_ha=-0.568636849315,
        eigen_rayleigh_ha=-0.205308969426,
        eigen_kinetic_ha=0.468898008999,
        eigen_local_ionic_ha=-1.683453804287,
        eigen_hartree_ha=1.407843104057,
        eigen_xc_ha=-0.398596278195,
        self_adjoint_total_relative_difference=7.775e-16,
        self_adjoint_kinetic_relative_difference=0.0,
        self_adjoint_local_relative_difference=0.0,
        patch_embedded_correction_mha=None,
        patch_embedding_energy_mismatch_ha=None,
        kinetic_version="production",
    ),
    monitor_patch_production_route=H2FixedPotentialOperatorRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        eigenvalue_ha=-6.574031909859388,
        weighted_residual_norm=3.3342921454967853,
        converged=False,
        trial_rayleigh_ha=0.020790723371890127,
        trial_kinetic_ha=0.9671226925733339,
        trial_local_ionic_ha=-2.143567194749,
        trial_hartree_ha=1.765755769546,
        trial_xc_ha=-0.568520543999,
        eigen_rayleigh_ha=-3.375004168561428,
        eigen_kinetic_ha=-3.374908449316184,
        eigen_local_ionic_ha=-0.137471599383,
        eigen_hartree_ha=0.137437670912,
        eigen_xc_ha=-0.000061790774,
        self_adjoint_total_relative_difference=1.448e-15,
        self_adjoint_kinetic_relative_difference=1.596e-16,
        self_adjoint_local_relative_difference=0.0,
        patch_embedded_correction_mha=77.815,
        patch_embedding_energy_mismatch_ha=0.0,
        kinetic_version="production",
    ),
    monitor_patch_trial_fix_route=H2FixedPotentialOperatorRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        eigenvalue_ha=-0.18662718689698515,
        weighted_residual_norm=0.0001483108390547803,
        converged=True,
        trial_rayleigh_ha=0.020790723371890127,
        trial_kinetic_ha=0.9671226925733339,
        trial_local_ionic_ha=-2.143567194749,
        trial_hartree_ha=1.765755769546,
        trial_xc_ha=-0.568520543999,
        eigen_rayleigh_ha=-0.18662718689697988,
        eigen_kinetic_ha=0.4469647941700376,
        eigen_local_ionic_ha=-1.6440610304184193,
        eigen_hartree_ha=1.4006881504210158,
        eigen_xc_ha=-0.3902191010696139,
        self_adjoint_total_relative_difference=2.068854834241115e-16,
        self_adjoint_kinetic_relative_difference=4.78838460645473e-16,
        self_adjoint_local_relative_difference=0.0,
        patch_embedded_correction_mha=77.815,
        patch_embedding_energy_mismatch_ha=0.0,
        kinetic_version="trial_fix",
    ),
    diagnosis=(
        "The kinetic boundary/ghost trial-fix pulls the A-grid+patch operator-level audit back "
        "toward the legacy regime: the k=1 eigenvalue rises from -6.57 Ha to -0.19 Ha and the "
        "weighted residual collapses from O(1) to 1.48e-4. The dominant change is the kinetic "
        "expectation on the eigensolver-selected orbital, which flips from strongly negative to "
        "a physically plausible positive value."
    ),
    note=(
        "Operator-level trial-fix regression baseline for the H2 frozen-density static-local "
        "audit. This baseline is still diagnostic only and does not imply that the full A-grid "
        "mainline is ready."
    ),
)


H2_FIXED_POTENTIAL_EIGENSOLVER_TRIAL_FIX_BASELINE = H2FixedPotentialTrialFixEigensolverRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="legacy",
        target_orbitals=1,
        eigenvalues_ha=(-0.20527465416922755,),
        max_residual_norm=2.8929876944581593e-04,
        max_orthogonality_error=1.1102230246251565e-15,
        converged=True,
        kinetic_version="production",
    ),
    monitor_patch_production_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=1,
        eigenvalues_ha=(-6.574031909859388,),
        max_residual_norm=3.3342921454967853,
        max_orthogonality_error=1.1102230246251565e-15,
        converged=False,
        kinetic_version="production",
    ),
    monitor_patch_trial_fix_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=1,
        eigenvalues_ha=(-0.18662718689698515,),
        max_residual_norm=0.0001483108390547803,
        max_orthogonality_error=3.3306690738754696e-16,
        converged=True,
        kinetic_version="trial_fix",
    ),
    legacy_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="legacy",
        target_orbitals=2,
        eigenvalues_ha=(-0.20529016296596161, 0.06286688164303063),
        max_residual_norm=6.54810468883838e-04,
        max_orthogonality_error=2.220446049250313e-15,
        converged=True,
        kinetic_version="production",
    ),
    monitor_patch_production_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=2,
        eigenvalues_ha=(-6.634541712761564, -6.219685689382547),
        max_residual_norm=7.833230053900307,
        max_orthogonality_error=6.661338147750939e-16,
        converged=False,
        kinetic_version="production",
    ),
    monitor_patch_trial_fix_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=2,
        eigenvalues_ha=(-0.1866584331584699, -0.1865957329712637),
        max_residual_norm=0.00017589071376628463,
        max_orthogonality_error=1.4254871966453894e-15,
        converged=True,
        kinetic_version="trial_fix",
    ),
    diagnosis=(
        "The kinetic trial-fix branch is the first A-grid+patch fixed-potential route that "
        "actually converges in this audit. k=1 rises from -6.57 Ha to -0.1866 Ha and the "
        "residual collapses by roughly 4.45e-5 relative to the production branch. The "
        "remaining caution is that the k=2 pair is nearly degenerate and still needs follow-up "
        "before this route can be promoted beyond fixed-potential auditing."
    ),
    note=(
        "Fixed-potential eigensolver trial-fix regression baseline for the H2 A-grid+patch "
        "static-local chain after wiring in the kinetic boundary/ghost closure prototype."
    ),
)


H2_KINETIC_OPERATOR_AUDIT_BASELINE = H2KineticOperatorRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    finer_shape=(75, 75, 91),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_route=H2KineticOperatorRouteBaseline(
        path_type="legacy",
        shape_label="baseline",
        frozen_kinetic_ha=0.997532314836,
        eigen_kinetic_ha=0.468898008999,
        eigenvalue_ha=-0.205274654169,
        eigensolver_residual_norm=2.892987694458e-04,
        kinetic_self_adjoint_relative_difference=0.0,
        eigen_negative_indicator_fraction=0.941764,
        eigen_far_field_contribution_ha=-0.000036930131,
        eigen_center_contribution_ha=0.005278427204,
        converged=True,
    ),
    monitor_unpatched_baseline_route=H2KineticOperatorRouteBaseline(
        path_type="monitor_a_grid",
        shape_label="baseline",
        frozen_kinetic_ha=0.967122692573,
        eigen_kinetic_ha=-1.642701348259,
        eigenvalue_ha=-2.844168300574,
        eigensolver_residual_norm=1.237350032968,
        kinetic_self_adjoint_relative_difference=1.596e-16,
        eigen_negative_indicator_fraction=0.524445,
        eigen_far_field_contribution_ha=-1.643335183718,
        eigen_center_contribution_ha=0.000000449782,
        converged=False,
    ),
    monitor_patch_baseline_route=H2KineticOperatorRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        shape_label="baseline",
        frozen_kinetic_ha=0.967122692573,
        eigen_kinetic_ha=-3.374908449316,
        eigenvalue_ha=-6.574031909859,
        eigensolver_residual_norm=3.334292145497,
        kinetic_self_adjoint_relative_difference=1.596e-16,
        eigen_negative_indicator_fraction=0.370689,
        eigen_far_field_contribution_ha=-3.377973017068,
        eigen_center_contribution_ha=0.000063639777,
        converged=False,
    ),
    monitor_patch_finer_shape_route=H2KineticOperatorRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        shape_label="finer-shape",
        frozen_kinetic_ha=0.974023111229,
        eigen_kinetic_ha=-3.893655956329,
        eigenvalue_ha=-7.556869846560,
        eigensolver_residual_norm=3.762909029926,
        kinetic_self_adjoint_relative_difference=3.177e-16,
        eigen_negative_indicator_fraction=0.348648,
        eigen_far_field_contribution_ha=-3.896350755891,
        eigen_center_contribution_ha=0.000015145796,
        converged=False,
    ),
    smooth_fields=(
        H2KineticOperatorSmoothFieldBaseline(
            field_label="gaussian",
            legacy_kinetic_ha=0.671972140862,
            monitor_kinetic_ha=0.659748142599,
        ),
        H2KineticOperatorSmoothFieldBaseline(
            field_label="cosine",
            legacy_kinetic_ha=0.050828543196,
            monitor_kinetic_ha=0.050856046166,
        ),
    ),
    diagnosis=(
        "The kinetic failure does not look like a weighted self-adjointness bug. Smooth test "
        "fields and the frozen trial orbital still give positive kinetic quotients, but the "
        "A-grid eigensolver orbitals drive <psi|T|psi> strongly negative. The finer-shape "
        "recheck makes the bad kinetic mode even deeper, so the dominant problem looks more "
        "like an operator-form or geometry/kinetic consistency defect than a simple resolution tail."
    ),
    note=(
        "Kinetic failure regression baseline for the H2 fixed-potential A-grid audit. This "
        "baseline isolates T only and is intended for later operator repairs, not as an "
        "acceptance target."
    ),
)


H2_KINETIC_FORM_AUDIT_BASELINE = H2KineticFormRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    finer_shape=(75, 75, 91),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    frozen_trial_baseline=H2KineticFormRouteBaseline(
        shape_label="baseline",
        orbital_label="frozen_trial_orbital",
        production_kinetic_ha=0.9671226925733339,
        reference_kinetic_ha=0.972715190398377,
        delta_kinetic_mha=-5.592497825043008,
        production_tpsi_rms=0.037373265866581344,
        reference_tpsi_rms=0.03784024467923909,
        delta_tpsi_rms=0.0010572327875104722,
        centerline_midpoint_delta=0.00615301727798101,
        far_field_delta_weighted_rms=1.2632054416391796e-17,
        eigensolver_eigenvalue_ha=None,
        eigensolver_residual_norm=None,
        eigensolver_converged=None,
    ),
    bad_eigen_baseline=H2KineticFormRouteBaseline(
        shape_label="baseline",
        orbital_label="bad_eigensolver_orbital_k1",
        production_kinetic_ha=-3.374908449316184,
        reference_kinetic_ha=-3.3748778659558227,
        delta_kinetic_mha=-0.030583360361102763,
        production_tpsi_rms=0.0480384363184959,
        reference_tpsi_rms=0.048063093485328424,
        delta_tpsi_rms=0.0012378229975185762,
        centerline_midpoint_delta=0.004732665534982555,
        far_field_delta_weighted_rms=2.8891658859320897e-07,
        eigensolver_eigenvalue_ha=-6.574031909859388,
        eigensolver_residual_norm=3.3342921454967853,
        eigensolver_converged=False,
    ),
    bad_eigen_finer_shape=H2KineticFormRouteBaseline(
        shape_label="finer-shape",
        orbital_label="bad_eigensolver_orbital_k1",
        production_kinetic_ha=-3.8936559563292596,
        reference_kinetic_ha=-3.893653222200653,
        delta_kinetic_mha=-0.0027341286066295822,
        production_tpsi_rms=0.054723869131267505,
        reference_tpsi_rms=0.054725680216826465,
        delta_tpsi_rms=0.000394249654047961,
        centerline_midpoint_delta=-0.0028285644728183595,
        far_field_delta_weighted_rms=1.4755344898350383e-07,
        eigensolver_eigenvalue_ha=-7.556869846559655,
        eigensolver_residual_norm=3.762909029925938,
        eigensolver_converged=False,
    ),
    smooth_fields=(
        H2KineticFormSmoothFieldBaseline(
            field_label="smooth_gaussian",
            production_kinetic_ha=0.659748142599247,
            reference_kinetic_ha=0.6638084614995698,
            delta_kinetic_mha=-4.060318900322879,
        ),
        H2KineticFormSmoothFieldBaseline(
            field_label="smooth_cosine",
            production_kinetic_ha=0.05085604616554839,
            reference_kinetic_ha=0.05085911164759871,
            delta_kinetic_mha=-0.003065482050319812,
        ),
    ),
    production_self_adjoint_baseline_rel=1.596128202151577e-16,
    reference_self_adjoint_baseline_rel=9.576290326531951e-04,
    production_self_adjoint_finer_rel=3.176772806680616e-16,
    reference_self_adjoint_finer_rel=7.255873359351101e-04,
    diagnosis=(
        "Production and reference kinetic forms stay close on the same A-grid geometry: the "
        "frozen trial orbital and smooth fields remain positive, while the bad eigensolver "
        "orbital is strongly negative in both discretizations with only sub-mHa production/"
        "reference differences. The finer-shape recheck makes that shared negative kinetic "
        "mode deeper. This points away from a production-only flux/divergence bug and more "
        "toward a geometry/metric consistency defect that both discretizations inherit on the "
        "same monitor-grid geometry."
    ),
    note=(
        "Kinetic-form failure baseline for production-vs-reference monitor-grid kinetic audits "
        "on the fixed H2 singlet frozen density. This baseline is diagnostic only."
    ),
)


H2_GEOMETRY_CONSISTENCY_AUDIT_BASELINE = H2GeometryConsistencyRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    jacobian_reconstruction_relative_rms=0.0,
    inverse_metric_reconstruction_relative_rms=0.0,
    cell_volume_vs_jacobian_relative_rms=8.898863206300374e-17,
    sqrt_det_metric_vs_jacobian_relative_rms=8.512930925289784e-16,
    metric_inverse_identity_rms=1.9351025908322702e-16,
    legacy_frozen_kinetic_reference_ha=0.997532314836,
    legacy_bad_eigen_kinetic_reference_ha=0.468898008999,
    frozen_trial_baseline=H2GeometryConsistencyFieldBaseline(
        field_label="frozen_trial_orbital",
        shape_label="baseline",
        operator_kinetic_ha=0.9671226925733339,
        gradient_reference_ha=0.9671226925733338,
        delta_kinetic_mha=1.1102230246251565e-13,
        operator_tpsi_rms=0.037373265866581344,
        gradient_indicator_rms=0.0038090118454572794,
        delta_indicator_rms=0.012211935797465987,
        far_field_operator_contribution_ha=-3.922194791252705e-28,
        far_field_gradient_contribution_ha=4.335193096252534e-28,
        far_field_delta_mha=-8.257387887505238e-25,
        source_eigenvalue_ha=None,
        source_residual_norm=None,
        source_converged=None,
    ),
    bad_eigen_baseline=H2GeometryConsistencyFieldBaseline(
        field_label="bad_eigensolver_orbital_k1",
        shape_label="baseline",
        operator_kinetic_ha=-3.374908449316184,
        gradient_reference_ha=3.8531390008473174,
        delta_kinetic_mha=-7228.047450163502,
        operator_tpsi_rms=0.0480384363184959,
        gradient_indicator_rms=0.03220564363391905,
        delta_indicator_rms=0.0604694331445142,
        far_field_operator_contribution_ha=-3.3779730170681517,
        far_field_gradient_contribution_ha=3.849174389245284,
        far_field_delta_mha=-7227.1474063134365,
        source_eigenvalue_ha=-6.574031909859388,
        source_residual_norm=3.3342921454967853,
        source_converged=False,
    ),
    smooth_fields=(
        H2GeometryConsistencySmoothFieldBaseline(
            field_label="smooth_gaussian",
            operator_kinetic_ha=0.659748142599247,
            gradient_reference_ha=0.6597481425992471,
            delta_kinetic_mha=-1.1102230246251565e-13,
        ),
        H2GeometryConsistencySmoothFieldBaseline(
            field_label="smooth_cosine",
            operator_kinetic_ha=0.05085604616554839,
            gradient_reference_ha=0.05233941382379163,
            delta_kinetic_mha=-1.4833676582432445,
        ),
    ),
    diagnosis=(
        "The stored monitor-grid geometry closes internally to machine precision, but the "
        "bad eigensolver orbital breaks the kinetic-energy identity badly: operator kinetic "
        "becomes strongly negative while the gradient-based reference stays positive, and the "
        "mismatch is almost entirely a far-field contribution. This points more toward a "
        "geometry/operator/boundary coupling defect than to a raw jacobian or inverse-metric "
        "storage bug."
    ),
    note=(
        "Geometry-consistency failure baseline for the H2 singlet frozen-density A-grid audit. "
        "This baseline is diagnostic only and does not imply any geometry or kinetic fix yet."
    ),
)


H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE = H2KineticGreenIdentityRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    finer_shape=(75, 75, 91),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    frozen_trial_baseline=H2KineticGreenIdentityFieldBaseline(
        shape_label="baseline",
        field_label="frozen_trial_orbital",
        operator_kinetic_ha=0.9671226925733339,
        gradient_kinetic_ha=0.9671226925733338,
        delta_kinetic_mha=1.1102230246251565e-13,
        boundary_term_ha=-3.28956539048495e-42,
        closure_mismatch_mha=2.1684043449710082e-14,
        far_field_delta_mha=-8.257387887505238e-25,
        far_field_boundary_ha=-3.28956539048495e-42,
        far_field_closure_mha=-8.257387887505206e-25,
        source_eigenvalue_ha=None,
        source_residual_norm=None,
        source_converged=None,
    ),
    bad_eigen_baseline=H2KineticGreenIdentityFieldBaseline(
        shape_label="baseline",
        field_label="bad_eigensolver_orbital_k1",
        operator_kinetic_ha=-3.374908449316184,
        gradient_kinetic_ha=3.8531390008473174,
        delta_kinetic_mha=-7228.047450163502,
        boundary_term_ha=-1.0962748848744346,
        closure_mismatch_mha=-6131.772565289067,
        far_field_delta_mha=-7227.1474063134365,
        far_field_boundary_ha=-1.0962748848744344,
        far_field_closure_mha=-6130.8725214390015,
        source_eigenvalue_ha=-6.574031909859388,
        source_residual_norm=3.3342921454967853,
        source_converged=False,
    ),
    bad_eigen_finer_shape=H2KineticGreenIdentityFieldBaseline(
        shape_label="finer-shape",
        field_label="bad_eigensolver_orbital_k1",
        operator_kinetic_ha=-3.8936559563292596,
        gradient_kinetic_ha=4.347050665433583,
        delta_kinetic_mha=-8240.706621762843,
        boundary_term_ha=-1.367375216225131,
        closure_mismatch_mha=-6873.331405537713,
        far_field_delta_mha=-8239.608233541221,
        far_field_boundary_ha=-1.3673752162251307,
        far_field_closure_mha=-6872.23301731609,
        source_eigenvalue_ha=-7.556869846559655,
        source_residual_norm=3.762909029925938,
        source_converged=False,
    ),
    smooth_fields=(
        H2KineticGreenIdentitySmoothFieldBaseline(
            field_label="smooth_gaussian",
            operator_kinetic_ha=0.659748142599247,
            gradient_kinetic_ha=0.6597481425992472,
            delta_kinetic_mha=-2.220446049250313e-13,
            boundary_term_ha=-2.2004378638003802e-24,
            closure_mismatch_mha=-7.849623728795053e-14,
        ),
        H2KineticGreenIdentitySmoothFieldBaseline(
            field_label="smooth_cosine",
            operator_kinetic_ha=0.05085604616554839,
            gradient_kinetic_ha=0.05233941382407794,
            delta_kinetic_mha=-1.4833676585295497,
            boundary_term_ha=3.97033888625584e-18,
            closure_mismatch_mha=-1.4833676585295597,
        ),
    ),
    diagnosis=(
        "The H2 bad eigensolver orbital supports a boundary-handling diagnosis. Frozen and "
        "smooth fields satisfy the discrete Green identity almost exactly, but the bad orbital "
        "develops a large negative Delta K that is dominated by the far-field region. An "
        "explicit boundary-flux term explains only part of that gap, while the remaining "
        "closure defect is also far-field dominated. The finer-shape recheck makes both the "
        "Delta K and the closure gap worse, which points more toward a discrete boundary/ghost "
        "closure defect than to a simple resolution tail."
    ),
    note=(
        "Discrete Green-identity failure baseline for the H2 singlet frozen-density A-grid "
        "kinetic audit. This baseline is diagnostic only and does not imply a kinetic fix."
    ),
)


H2_KINETIC_GREEN_IDENTITY_TRIAL_FIX_BASELINE = H2KineticGreenIdentityRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    finer_shape=(75, 75, 91),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    frozen_trial_baseline=H2KineticGreenIdentityFieldBaseline(
        shape_label="baseline",
        field_label="frozen_trial_orbital",
        operator_kinetic_ha=0.9671226925733343,
        gradient_kinetic_ha=0.9671226925733338,
        delta_kinetic_mha=4.440892098500626e-13,
        boundary_term_ha=2.0911931031685546e-43,
        closure_mismatch_mha=3.1485231088979044e-13,
        far_field_delta_mha=-8.257387887505129e-25,
        far_field_boundary_ha=2.0911931031685546e-43,
        far_field_closure_mha=-8.257387887505129e-25,
        source_eigenvalue_ha=None,
        source_residual_norm=None,
        source_converged=None,
    ),
    bad_eigen_baseline=H2KineticGreenIdentityFieldBaseline(
        shape_label="baseline",
        field_label="bad_eigensolver_orbital_k1",
        operator_kinetic_ha=3.394498206320347,
        gradient_kinetic_ha=3.8531390008473174,
        delta_kinetic_mha=-458.64079452697035,
        boundary_term_ha=0.9283910649165813,
        closure_mismatch_mha=-1387.0318594435507,
        far_field_delta_mha=-457.7407506769053,
        far_field_boundary_ha=0.9283910649165814,
        far_field_closure_mha=-1386.1318155934866,
        source_eigenvalue_ha=-6.574031909859388,
        source_residual_norm=3.3342921454967853,
        source_converged=False,
    ),
    bad_eigen_finer_shape=H2KineticGreenIdentityFieldBaseline(
        shape_label="finer-shape",
        field_label="bad_eigensolver_orbital_k1",
        operator_kinetic_ha=4.135089743442165,
        gradient_kinetic_ha=4.347050665433583,
        delta_kinetic_mha=-211.96092199141782,
        boundary_term_ha=1.2641434744933913,
        closure_mismatch_mha=-1476.1043964848081,
        far_field_delta_mha=-210.86253376979448,
        far_field_boundary_ha=1.2641434744933908,
        far_field_closure_mha=-1475.0060082631855,
        source_eigenvalue_ha=-7.556869846559655,
        source_residual_norm=3.762909029925938,
        source_converged=False,
    ),
    smooth_fields=(
        H2KineticGreenIdentitySmoothFieldBaseline(
            field_label="smooth_gaussian",
            operator_kinetic_ha=0.6597481425992471,
            gradient_kinetic_ha=0.6597481425992472,
            delta_kinetic_mha=-1.1102230246251565e-13,
            boundary_term_ha=1.1802469575905652e-24,
            closure_mismatch_mha=3.252606517456514e-14,
        ),
        H2KineticGreenIdentitySmoothFieldBaseline(
            field_label="smooth_cosine",
            operator_kinetic_ha=0.05011510590928958,
            gradient_kinetic_ha=0.05233941382407794,
            delta_kinetic_mha=-2.224307914788358,
            boundary_term_ha=1.9832026149313038e-18,
            closure_mismatch_mha=-2.2243079147883624,
        ),
    ),
    diagnosis=(
        "The kinetic boundary/ghost trial fix materially reduces the bad-orbital Green-identity "
        "failure: the baseline bad-eigen Delta K drops from -7228 mHa to -459 mHa and K_op "
        "flips from strongly negative to positive. The remaining mismatch is still far-field "
        "dominated, and the explicit boundary term only explains part of it, so the prototype "
        "hits the main symptom but does not yet fully close the discrete boundary identity."
    ),
    note=(
        "Discrete Green-identity trial-fix baseline for the H2 singlet frozen-density A-grid "
        "kinetic audit after switching the monitor-grid boundary/ghost handling to a centered "
        "zero-ghost prototype in kinetic.py."
    ),
)


H2_ORBITAL_SHAPE_AUDIT_BASELINE = H2OrbitalShapeRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_k1_orbital=H2OrbitalShapeOrbitalBaseline(
        path_type="legacy",
        kinetic_version="production",
        orbital_label="k1",
        eigenvalue_ha=-0.20527465416922755,
        residual_norm=2.8929876944581593e-04,
        converged=True,
        inversion_best_parity="even",
        inversion_best_mismatch=6.895105654464947e-08,
        z_mirror_best_parity="even",
        z_mirror_best_mismatch=5.4184637833479585e-08,
        centerline_sign_changes=0,
        far_field_sign_changes=0,
        node_positions_bohr=(),
        far_field_norm_fraction=0.00018165736444526356,
        boundary_layer_norm_fraction=5.3152289305315985e-05,
        far_field_max_abs_value=0.0018771107352923485,
        boundary_layer_max_abs_value=0.0009420164827175654,
    ),
    legacy_k2_low_orbital=H2OrbitalShapeOrbitalBaseline(
        path_type="legacy",
        kinetic_version="production",
        orbital_label="k2_orbital_0",
        eigenvalue_ha=-0.20529016296596161,
        residual_norm=6.094926935830485e-05,
        converged=True,
        inversion_best_parity="even",
        inversion_best_mismatch=2.882950060275491e-12,
        z_mirror_best_parity="even",
        z_mirror_best_mismatch=1.2026511261227441e-13,
        centerline_sign_changes=0,
        far_field_sign_changes=0,
        node_positions_bohr=(),
        far_field_norm_fraction=0.0001782105598979954,
        boundary_layer_norm_fraction=5.203449064227384e-05,
        far_field_max_abs_value=0.0018325927985254838,
        boundary_layer_max_abs_value=0.0009196093858996899,
    ),
    legacy_k2_high_orbital=H2OrbitalShapeOrbitalBaseline(
        path_type="legacy",
        kinetic_version="production",
        orbital_label="k2_orbital_1",
        eigenvalue_ha=0.06286688164303063,
        residual_norm=6.54810468883838e-04,
        converged=True,
        inversion_best_parity="even",
        inversion_best_mismatch=1.1253274631577421e-07,
        z_mirror_best_parity="even",
        z_mirror_best_mismatch=1.3767992212459453e-08,
        centerline_sign_changes=2,
        far_field_sign_changes=0,
        node_positions_bohr=(-2.3282559031219483, 2.3282559031219483),
        far_field_norm_fraction=0.1637710972490747,
        boundary_layer_norm_fraction=0.06142466633151523,
        far_field_max_abs_value=0.1670974972830774,
        boundary_layer_max_abs_value=0.08656378070422546,
    ),
    monitor_trial_fix_k1_orbital=H2OrbitalShapeOrbitalBaseline(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        orbital_label="k1",
        eigenvalue_ha=-0.18662718689698515,
        residual_norm=1.483108390547803e-04,
        converged=True,
        inversion_best_parity="even",
        inversion_best_mismatch=2.2848751569444183e-13,
        z_mirror_best_parity="even",
        z_mirror_best_mismatch=1.8778668416252935e-13,
        centerline_sign_changes=0,
        far_field_sign_changes=0,
        node_positions_bohr=(),
        far_field_norm_fraction=0.0002568543229815991,
        boundary_layer_norm_fraction=0.0001339988515724429,
        far_field_max_abs_value=0.0035255929885303545,
        boundary_layer_max_abs_value=0.0015654296457986632,
    ),
    monitor_trial_fix_k2_first_orbital=H2OrbitalShapeOrbitalBaseline(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        orbital_label="k2_orbital_0",
        eigenvalue_ha=-0.1866584331584699,
        residual_norm=1.7486394700554094e-04,
        converged=True,
        inversion_best_parity="even",
        inversion_best_mismatch=5.529166721212679e-10,
        z_mirror_best_parity="even",
        z_mirror_best_mismatch=5.505569527583639e-10,
        centerline_sign_changes=0,
        far_field_sign_changes=0,
        node_positions_bohr=(),
        far_field_norm_fraction=0.0002572071606617207,
        boundary_layer_norm_fraction=0.0001343530814508898,
        far_field_max_abs_value=0.0031744864671881393,
        boundary_layer_max_abs_value=0.0014094329653375726,
    ),
    monitor_trial_fix_k2_second_orbital=H2OrbitalShapeOrbitalBaseline(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        orbital_label="k2_orbital_1",
        eigenvalue_ha=-0.1865957329712637,
        residual_norm=1.7589071376628463e-04,
        converged=True,
        inversion_best_parity="even",
        inversion_best_mismatch=5.562350291804711e-10,
        z_mirror_best_parity="even",
        z_mirror_best_mismatch=5.538464428381761e-10,
        centerline_sign_changes=0,
        far_field_sign_changes=0,
        node_positions_bohr=(),
        far_field_norm_fraction=0.000256500309599944,
        boundary_layer_norm_fraction=0.00013364462236141998,
        far_field_max_abs_value=0.003876227869692682,
        boundary_layer_max_abs_value=0.0017213598983194028,
    ),
    legacy_k2_gap_ha=0.26815704460899224,
    monitor_trial_fix_k2_gap_ha=6.270018720619117e-05,
    diagnosis=(
        "The repaired A-grid fixed-potential path now produces a healthy-looking k=1 bonding-like "
        "orbital with tiny symmetry mismatch and negligible boundary leakage, but its k=2 solve "
        "returns a nearly degenerate even/even pair instead of a clear legacy-like antibonding "
        "state. The immediate risk before any SCF dry-run is subspace mixing inside that k=2 pair, "
        "not obvious boundary pollution."
    ),
    note=(
        "Very small orbital-shape baseline for the H2 fixed-potential legacy route versus the "
        "A-grid+patch+kinetic-trial-fix route. This is a morphology/symmetry audit only, not a "
        "full physical validation."
    ),
)


H2_K2_SUBSPACE_AUDIT_BASELINE = H2K2SubspaceRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    density_label="h2_singlet_frozen_density",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    legacy_raw_eigenvalues_ha=(-0.20529016296596161, 0.06286688164303063),
    legacy_raw_orbitals=(
        H2K2SubspaceOrbitalBaseline(
            path_type="legacy",
            kinetic_version="production",
            orbital_label="raw_k2_orbital_0",
            eigenvalue_ha=-0.20529016296596161,
            residual_norm=6.0949266778088785e-05,
            z_mirror_best_mismatch=1.2026507023916232e-13,
            centerline_sign_changes=0,
            far_field_sign_changes=0,
            center_value=0.32599942912178453,
            far_field_norm_fraction=0.00017807090829986114,
            boundary_layer_norm_fraction=5.24272545370966e-05,
        ),
        H2K2SubspaceOrbitalBaseline(
            path_type="legacy",
            kinetic_version="production",
            orbital_label="raw_k2_orbital_1",
            eigenvalue_ha=0.06286688164303063,
            residual_norm=0.000654810468883838,
            z_mirror_best_mismatch=1.376798826073179e-08,
            centerline_sign_changes=2,
            far_field_sign_changes=0,
            center_value=-0.07934285278634354,
            far_field_norm_fraction=0.16377129456312522,
            boundary_layer_norm_fraction=0.06142525766232676,
        ),
    ),
    monitor_raw_eigenvalues_ha=(-0.1866584331584699, -0.1865957329712637),
    monitor_raw_orbitals=(
        H2K2SubspaceOrbitalBaseline(
            path_type="monitor_a_grid_plus_patch",
            kinetic_version="trial_fix",
            orbital_label="raw_k2_orbital_0",
            eigenvalue_ha=-0.1866584331584699,
            residual_norm=0.00017486394700554094,
            z_mirror_best_mismatch=5.505570380930582e-10,
            centerline_sign_changes=0,
            far_field_sign_changes=0,
            center_value=-2.0764397553265664e-05,
            far_field_norm_fraction=0.000256552436838866,
            boundary_layer_norm_fraction=0.00013429772500605329,
        ),
        H2K2SubspaceOrbitalBaseline(
            path_type="monitor_a_grid_plus_patch",
            kinetic_version="trial_fix",
            orbital_label="raw_k2_orbital_1",
            eigenvalue_ha=-0.1865957329712637,
            residual_norm=0.00017589071376628463,
            z_mirror_best_mismatch=5.538463649706932e-10,
            centerline_sign_changes=0,
            far_field_sign_changes=0,
            center_value=-0.4439333144909637,
            far_field_norm_fraction=0.000256703747460823,
            boundary_layer_norm_fraction=0.00013434470345349386,
        ),
    ),
    monitor_inversion_matrix=H2K2SubspaceMatrixBaseline(
        label="inversion",
        matrix=((1.0000000000000018, 1.4256416159176753e-15), (1.4256439018312981e-15, 1.0000000000000016)),
        eigenvalues=(1.0000000000000002, 1.000000000000003),
    ),
    monitor_z_mirror_matrix=H2K2SubspaceMatrixBaseline(
        label="z_mirror",
        matrix=((1.0000000000000018, 1.425638914759385e-15), (1.4256386831488135e-15, 1.0000000000000016)),
        eigenvalues=(1.0000000000000002, 1.000000000000003),
    ),
    monitor_bonding_rotation=H2K2SubspaceRotationBaseline(
        rotation_label="bonding_overlap_rotation",
        rotation_matrix=(
            (-0.7071744179213354, 0.7070391379814985),
            (-0.7070391379814985, -0.7071744179213354),
        ),
        raw_bonding_overlaps=(-0.617933556193693, -0.6178153477119004),
        rotated_bonding_overlaps=(0.8738064337932976, -3.058479094175603e-16),
        rotated_first_orbital=H2K2SubspaceOrbitalBaseline(
            path_type="monitor_a_grid_plus_patch",
            kinetic_version="trial_fix",
            orbital_label="rotated_k2_orbital_0",
            eigenvalue_ha=-0.1866584331584699,
            residual_norm=0.00017486394700554094,
            z_mirror_best_mismatch=2.257931484599113e-12,
            centerline_sign_changes=0,
            far_field_sign_changes=0,
            center_value=0.3138929120497134,
            far_field_norm_fraction=0.00025666264565709355,
            boundary_layer_norm_fraction=0.00013434535905920513,
        ),
        rotated_second_orbital=H2K2SubspaceOrbitalBaseline(
            path_type="monitor_a_grid_plus_patch",
            kinetic_version="trial_fix",
            orbital_label="rotated_k2_orbital_1",
            eigenvalue_ha=-0.1865957329712637,
            residual_norm=0.00017589071376628463,
            z_mirror_best_mismatch=7.809313367782012e-10,
            centerline_sign_changes=68,
            far_field_sign_changes=12,
            center_value=0.3139236020292898,
            far_field_norm_fraction=0.0002565935386425954,
            boundary_layer_norm_fraction=0.00013429706940034205,
        ),
        note=(
            "The chosen very small rotation maximizes frozen-bonding overlap inside the raw "
            "A-grid k=2 subspace. It cleanly isolates one bonding-like state, but its orthogonal "
            "complement becomes a strongly oscillatory even mode rather than a legacy-like "
            "antibonding orbital."
        ),
    ),
    legacy_k2_gap_ha=0.26815704460899226,
    monitor_k2_gap_ha=6.270018720619386e-05,
    diagnosis=(
        "The A-grid+patch+trial-fix k=2 pair is still best interpreted as a near-degenerate mixed "
        "subspace. Symmetry projection does not split it, and the simplest bonding-overlap rotation "
        "yields one bonding-like state plus a highly oscillatory orthogonal complement, not a clean "
        "legacy-like antibonding orbital. This is a subspace-shape warning, not a boundary-pollution "
        "warning."
    ),
    note=(
        "Very small H2 k=2 subspace baseline for the repaired A-grid+patch+trial-fix fixed-potential "
        "route versus the legacy fixed-potential reference."
    ),
)


H2_SCF_DRY_RUN_BASELINE = H2ScfDryRunRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    legacy_singlet_route=H2ScfDryRunRouteBaseline(
        path_type="legacy",
        spin_state_label="singlet",
        kinetic_version="production",
        includes_nonlocal=True,
        converged=True,
        iteration_count=5,
        final_total_energy_ha=-1.1461203815144159,
        lowest_eigenvalue_ha=-0.3804835585677118,
        final_density_residual=0.001750856831863596,
        final_energy_change_ha=-1.7328473695954472e-06,
    ),
    monitor_singlet_route=H2ScfDryRunRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        spin_state_label="singlet",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        converged=False,
        iteration_count=20,
        final_total_energy_ha=-0.13034787232113343,
        lowest_eigenvalue_ha=-0.4530723625192544,
        final_density_residual=0.337104348281785,
        final_energy_change_ha=0.010836526571194716,
    ),
    legacy_triplet_route=H2ScfDryRunRouteBaseline(
        path_type="legacy",
        spin_state_label="triplet",
        kinetic_version="production",
        includes_nonlocal=True,
        converged=True,
        iteration_count=6,
        final_total_energy_ha=-0.7660687190478731,
        lowest_eigenvalue_ha=-0.6255731151927522,
        final_density_residual=0.0022480823759300898,
        final_energy_change_ha=-3.974051383526245e-06,
    ),
    monitor_triplet_route=H2ScfDryRunRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        spin_state_label="triplet",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        converged=True,
        iteration_count=18,
        final_total_energy_ha=-1.2214418066604806,
        lowest_eigenvalue_ha=-0.4168423341628571,
        final_density_residual=0.004552787297010315,
        final_energy_change_ha=6.047076691828579e-06,
    ),
    diagnosis=(
        "The repaired A-grid+patch+trial-fix line is strong enough to sustain a genuine H2 "
        "SCF dry-run, but it is not yet uniformly healthy across spin channels. The singlet "
        "enters multi-step iteration and remains finite, yet settles into a persistent "
        "two-cycle with O(10^-1) density residual instead of converging. The triplet route, "
        "by contrast, converges in 18 steps under a very small protective damping change "
        "(mixing=0.20, max_iterations=20). This is therefore a mixed dry-run baseline: the "
        "new A-grid line no longer catastrophically blows up, but singlet stability is not "
        "yet sufficient for an SCF handoff."
    ),
    note=(
        "H2 SCF dry-run baseline for the repaired A-grid+patch+kinetic-trial-fix route. "
        "The A-grid dry-run includes only T + V_loc,ion + V_H + V_xc, while the legacy route "
        "still includes nonlocal ionic action. These numbers are therefore dry-run regression "
        "markers, not final acceptance benchmarks."
    ),
)


H2_SINGLET_STABILITY_BASELINE = H2SingletStabilityRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    baseline_route=H2SingletStabilityRouteBaseline(
        scheme_label="baseline",
        kinetic_version="trial_fix",
        diis_enabled=False,
        diis_warmup_iterations=3,
        diis_history_length=4,
        diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
        diis_used_iterations=(),
        converged=False,
        iteration_count=10,
        final_total_energy_ha=-0.1408486512266819,
        final_lowest_eigenvalue_ha=-0.4361423023175884,
        final_density_residual=0.3336218796626913,
        final_energy_change_ha=0.002219282558758362,
        detected_two_cycle=False,
        two_cycle_verdict="stable_not_converged",
        even_odd_energy_gap_ha=0.0018213009296351168,
        even_odd_residual_gap=0.005781240287303924,
    ),
    smaller_mixing_route=H2SingletStabilityRouteBaseline(
        scheme_label="smaller-mixing",
        kinetic_version="trial_fix",
        diis_enabled=False,
        diis_warmup_iterations=3,
        diis_history_length=4,
        diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
        diis_used_iterations=(),
        converged=False,
        iteration_count=10,
        final_total_energy_ha=-0.19604042867532245,
        final_lowest_eigenvalue_ha=-0.3505684854591779,
        final_density_residual=0.2993033291514288,
        final_energy_change_ha=-0.0018690642042323846,
        detected_two_cycle=False,
        two_cycle_verdict="slow_monotone_or_damped",
        even_odd_energy_gap_ha=0.004581969684613829,
        even_odd_residual_gap=0.01157349178976752,
    ),
    diis_prototype_route=H2SingletStabilityRouteBaseline(
        scheme_label="diis-prototype",
        kinetic_version="trial_fix",
        diis_enabled=True,
        diis_warmup_iterations=3,
        diis_history_length=4,
        diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
        diis_used_iterations=(3, 4, 5, 7, 8, 9, 10),
        converged=False,
        iteration_count=10,
        final_total_energy_ha=-0.07906163475219974,
        final_lowest_eigenvalue_ha=-0.5474502295844333,
        final_density_residual=0.3585728293427899,
        final_energy_change_ha=0.11667317822978585,
        detected_two_cycle=False,
        two_cycle_verdict="stable_not_converged",
        even_odd_energy_gap_ha=0.01829133687234341,
        even_odd_residual_gap=0.0008350592949901148,
    ),
    diagnosis=(
        "Within a deliberately small 10-step audit budget, the repaired A-grid singlet route no "
        "longer blows up under either conservative scheme, but it also does not converge. The "
        "baseline mixing=0.20 path remains stalled near residual ~0.334 and is best described as "
        "stable-but-not-converged over this short window. Reducing the mixing to 0.10 suppresses "
        "the sharper oscillatory behavior seen in the earlier 20-step dry-run baseline and turns "
        "the update into a visibly more damped trajectory with residual ~0.299 after 10 steps. "
        "The minimal DIIS prototype, however, is not the missing silver bullet here: after a 3-step "
        "warmup it activates repeatedly, but the final residual rebounds to ~0.359, the final energy "
        "jumps back upward to about -0.079 Ha, and the lowest eigenvalue becomes overly deep again. "
        "So the current singlet route is not simply 'one standard DIIS away' from convergence. "
        "This remains a narrow stability baseline, not a full SCF fix."
    ),
    note=(
        "Very small singlet-only A-grid stability baseline on the current "
        "A-grid+patch+kinetic-trial-fix dry-run path. It compares baseline mixing=0.20, "
        "smaller mixing=0.10, and one minimal DIIS prototype on top of mixing=0.10. "
        "The iteration budget is kept at 10 steps so this remains a narrow stability "
        "audit rather than a wider SCF tuning pass."
    ),
)


H2_DIIS_SCF_BASELINE = H2DiisScfRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    singlet=H2DiisScfSpinBaseline(
        spin_state_label="singlet",
        baseline_route=H2DiisScfRouteBaseline(
            spin_state_label="singlet",
            scheme_label="baseline",
            kinetic_version="trial_fix",
            diis_enabled=False,
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            diis_used_iterations=(),
            diis_fallback_iterations=(),
            converged=False,
            iteration_count=20,
            final_total_energy_ha=-0.13034787232113343,
            final_lowest_eigenvalue_ha=-0.4530723625192544,
            final_density_residual=0.337104348281785,
            final_energy_change_ha=0.010836526571194716,
            trajectory_verdict="stable_not_converged",
        ),
        smaller_mixing_route=H2DiisScfRouteBaseline(
            spin_state_label="singlet",
            scheme_label="smaller-mixing",
            kinetic_version="trial_fix",
            diis_enabled=False,
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            diis_used_iterations=(),
            diis_fallback_iterations=(),
            converged=False,
            iteration_count=20,
            final_total_energy_ha=-0.16491247803199938,
            final_lowest_eigenvalue_ha=-0.3936762340394962,
            final_density_residual=0.30825645760616305,
            final_energy_change_ha=0.008438151037082342,
            trajectory_verdict="stable_not_converged",
        ),
        diis_prototype_route=H2DiisScfRouteBaseline(
            spin_state_label="singlet",
            scheme_label="diis-prototype",
            kinetic_version="trial_fix",
            diis_enabled=True,
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            diis_used_iterations=(3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 19, 20),
            diis_fallback_iterations=(6, 13, 17, 18),
            converged=False,
            iteration_count=20,
            final_total_energy_ha=-0.1731403750781818,
            final_lowest_eigenvalue_ha=-0.3793841527894403,
            final_density_residual=0.29358610500934057,
            final_energy_change_ha=0.004076261207450749,
            trajectory_verdict="slow_monotone_or_damped",
        ),
    ),
    triplet=H2DiisScfSpinBaseline(
        spin_state_label="triplet",
        baseline_route=H2DiisScfRouteBaseline(
            spin_state_label="triplet",
            scheme_label="baseline",
            kinetic_version="trial_fix",
            diis_enabled=False,
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            diis_used_iterations=(),
            diis_fallback_iterations=(),
            converged=True,
            iteration_count=18,
            final_total_energy_ha=-1.2214418066604806,
            final_lowest_eigenvalue_ha=-0.4168423341628571,
            final_density_residual=0.004552787297010315,
            final_energy_change_ha=6.047076691828579e-06,
            trajectory_verdict="converged",
        ),
        smaller_mixing_route=H2DiisScfRouteBaseline(
            spin_state_label="triplet",
            scheme_label="smaller-mixing",
            kinetic_version="trial_fix",
            diis_enabled=False,
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            diis_used_iterations=(),
            diis_fallback_iterations=(),
            converged=False,
            iteration_count=20,
            final_total_energy_ha=-1.221256217234775,
            final_lowest_eigenvalue_ha=-0.4090704203001049,
            final_density_residual=0.026957478433928376,
            final_energy_change_ha=-3.205091858449194e-05,
            trajectory_verdict="slow_monotone_or_damped",
        ),
        diis_prototype_route=H2DiisScfRouteBaseline(
            spin_state_label="triplet",
            scheme_label="diis-prototype",
            kinetic_version="trial_fix",
            diis_enabled=True,
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            diis_used_iterations=(3, 4, 6, 8),
            diis_fallback_iterations=(5, 7),
            converged=True,
            iteration_count=8,
            final_total_energy_ha=-1.2215069883120075,
            final_lowest_eigenvalue_ha=-0.41931346686692206,
            final_density_residual=0.0038475723808240677,
            final_energy_change_ha=1.3193541439804335e-05,
            trajectory_verdict="converged",
        ),
    ),
    diagnosis=(
        "On the repaired A-grid+patch+kinetic-trial-fix path, a small but formal Pulay/DIIS "
        "prototype is enough to help, but not enough to finish the singlet job. For singlet, "
        "DIIS improves the 20-step outcome relative to pure mixing=0.10: the final density "
        "residual drops from about 0.308 to 0.294, the final energy becomes more negative, and "
        "the trajectory looks more damped than the plain linear routes. But it still does not "
        "meet the dry-run convergence thresholds. For triplet, DIIS is clearly healthy: it keeps "
        "the route convergent and cuts the iteration count from 18 to 8 while reaching a slightly "
        "lower final residual. So the current A-grid H2 SCF line is no longer blocked by gross "
        "triplet instability, but singlet still cannot be called fully converged."
    ),
    note=(
        "A-grid-only H2 DIIS SCF baseline on the repaired A-grid+patch+kinetic-trial-fix dry-run "
        "path. The comparison is limited to baseline linear mixing=0.20, smaller linear mixing=0.10, "
        "and a small warmup+DIIS prototype with history length 4. Nonlocal remains absent."
    ),
)


@dataclass(frozen=True)
class H2JaxKernelConsistencyReductionsBaseline:
    """Recorded first-batch JAX reductions consistency summary."""

    weighted_inner_product_abs_diff: float
    weighted_norm_abs_diff: float
    density_accumulation_max_abs_diff: float
    overlap_matrix_max_abs_diff: float
    orthonormalization_overlap_max_abs_diff: float


@dataclass(frozen=True)
class H2JaxKernelConsistencyPoissonBaseline:
    """Recorded first-batch JAX Poisson consistency summary."""

    solver_method: str
    iteration_count: int
    residual_max: float
    potential_max_abs_diff: float
    hartree_energy_abs_diff_ha: float


@dataclass(frozen=True)
class H2JaxKernelConsistencyLocalHamiltonianBaseline:
    """Recorded first-batch JAX local-Hamiltonian consistency summary."""

    action_max_abs_diff: float
    action_weighted_norm_diff: float


@dataclass(frozen=True)
class H2JaxKernelConsistencyRegressionBaseline:
    """Recorded first-batch JAX kernel migration baseline on the H2 A-grid path."""

    benchmark_name: str
    runtime_summary: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    reductions: H2JaxKernelConsistencyReductionsBaseline
    poisson: H2JaxKernelConsistencyPoissonBaseline
    local_hamiltonian: H2JaxKernelConsistencyLocalHamiltonianBaseline
    note: str


@dataclass(frozen=True)
class H2JaxEigensolverHotpathRegressionBaseline:
    """Recorded old-vs-JAX block-hot-path comparison on the H2 fixed-potential line."""

    benchmark_name: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    old_k1_route: H2FixedPotentialEigensolverRouteBaseline
    jax_k1_route: H2FixedPotentialEigensolverRouteBaseline
    old_k2_route: H2FixedPotentialEigensolverRouteBaseline | None
    jax_k2_route: H2FixedPotentialEigensolverRouteBaseline | None
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2JaxEigensolverHotpathReuseRegressionBaseline:
    """Recorded compiled-kernel reuse/caching comparison on the JAX eigensolver hot path."""

    benchmark_name: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    old_k1_route: H2FixedPotentialEigensolverRouteBaseline
    preoptimization_jax_k1_wall_time_seconds: float
    preoptimization_jax_k2_wall_time_seconds: float
    optimized_jax_k1_route: H2FixedPotentialEigensolverRouteBaseline
    optimized_jax_k2_route: H2FixedPotentialEigensolverRouteBaseline | None
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2JaxScfHotpathRouteBaseline:
    """Recorded A-grid SCF hot-path profiling result for one spin state and route."""

    spin_state_label: str
    use_jax_block_kernels: bool
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    eigensolver_wall_time_seconds: float
    energy_evaluation_wall_time_seconds: float
    density_update_wall_time_seconds: float
    bookkeeping_wall_time_seconds: float


@dataclass(frozen=True)
class H2JaxScfHotpathRegressionBaseline:
    """Recorded A-grid SCF hot-path profiling baseline after JAX handoff."""

    benchmark_name: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    triplet_old_route: H2JaxScfHotpathRouteBaseline
    triplet_jax_route: H2JaxScfHotpathRouteBaseline
    singlet_old_route: H2JaxScfHotpathRouteBaseline | None
    singlet_jax_route: H2JaxScfHotpathRouteBaseline | None
    diagnosis: str
    note: str


@dataclass(frozen=True)
class H2JaxTripletHartreeEnergyRouteBaseline:
    """Recorded triplet-only SCF profiling route for Hartree/energy optimization."""

    path_label: str
    hartree_backend: str
    cg_impl: str
    cg_preconditioner: str
    use_jax_hartree_cached_operator: bool
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    hartree_solve_call_count: int
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    average_hartree_solve_wall_time_seconds: float | None
    first_hartree_solve_wall_time_seconds: float | None
    repeated_hartree_solve_average_wall_time_seconds: float | None
    repeated_hartree_solve_min_wall_time_seconds: float | None
    repeated_hartree_solve_max_wall_time_seconds: float | None
    average_hartree_cg_iterations: float | None
    first_hartree_cg_iterations: int | None
    repeated_hartree_cg_iteration_average: float | None
    average_hartree_boundary_condition_wall_time_seconds: float | None
    average_hartree_build_wall_time_seconds: float | None
    average_hartree_rhs_assembly_wall_time_seconds: float | None
    average_hartree_cg_wall_time_seconds: float | None
    average_hartree_cg_other_overhead_wall_time_seconds: float | None
    average_hartree_matvec_call_count: float | None
    average_hartree_matvec_wall_time_seconds: float | None
    average_hartree_matvec_wall_time_per_call_seconds: float | None
    average_hartree_cg_iteration_wall_time_seconds: float | None
    average_hartree_matvec_wall_time_per_iteration_seconds: float | None
    average_hartree_other_cg_overhead_wall_time_per_iteration_seconds: float | None
    first_hartree_matvec_call_count: int | None
    repeated_hartree_matvec_call_count_average: float | None
    first_hartree_matvec_wall_time_seconds: float | None
    repeated_hartree_matvec_average_wall_time_seconds: float | None
    first_hartree_matvec_wall_time_per_call_seconds: float | None
    repeated_hartree_matvec_wall_time_per_call_seconds: float | None
    hartree_cached_operator_usage_count: int
    hartree_cached_operator_first_solve_count: int
    eigensolver_wall_time_seconds: float
    static_local_prepare_wall_time_seconds: float
    hartree_solve_wall_time_seconds: float
    local_ionic_resolve_wall_time_seconds: float
    xc_resolve_wall_time_seconds: float
    energy_evaluation_wall_time_seconds: float
    kinetic_energy_wall_time_seconds: float
    local_ionic_energy_wall_time_seconds: float
    hartree_energy_wall_time_seconds: float
    xc_energy_wall_time_seconds: float
    ion_ion_energy_wall_time_seconds: float
    density_update_wall_time_seconds: float
    bookkeeping_wall_time_seconds: float


@dataclass(frozen=True)
class H2JaxTripletHartreeEnergyRegressionBaseline:
    """Recorded triplet-only SCF profiling baseline for stronger JAX Hartree PCG checks."""

    benchmark_name: str
    monitor_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    jax_hartree_cgloop_route: H2JaxTripletHartreeEnergyRouteBaseline
    jax_hartree_pcg_stronger_route: H2JaxTripletHartreeEnergyRouteBaseline
    diagnosis: str
    note: str


H2_JAX_KERNEL_CONSISTENCY_BASELINE = H2JaxKernelConsistencyRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    runtime_summary="x64=True, disable_jit=False, platform=default",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    reductions=H2JaxKernelConsistencyReductionsBaseline(
        weighted_inner_product_abs_diff=0.0,
        weighted_norm_abs_diff=0.0,
        density_accumulation_max_abs_diff=0.0,
        overlap_matrix_max_abs_diff=8.881784197001252e-16,
        orthonormalization_overlap_max_abs_diff=1.3322676295501878e-15,
    ),
    poisson=H2JaxKernelConsistencyPoissonBaseline(
        solver_method="jax_cg_monitor",
        iteration_count=400,
        residual_max=2.1363030857485987e-07,
        potential_max_abs_diff=2.6137809472359663e-07,
        hartree_energy_abs_diff_ha=7.741737029220985e-09,
    ),
    local_hamiltonian=H2JaxKernelConsistencyLocalHamiltonianBaseline(
        action_max_abs_diff=4.440892098500626e-16,
        action_weighted_norm_diff=0.0,
    ),
    note=(
        "First-batch JAX migration baseline for the current stable H2 monitor-grid hot kernels. "
        "It records reductions, monitor Poisson CG, and local-only Hamiltonian apply on the "
        "A-grid+patch+kinetic-trial-fix line without migrating nonlocal or the SCF/eigensolver "
        "outer control flow."
    ),
)


H2_JAX_EIGENSOLVER_HOTPATH_BASELINE = H2JaxEigensolverHotpathRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    old_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=1,
        eigenvalues_ha=(-0.18662718689698515,),
        max_residual_norm=0.0001483108390547803,
        max_orthogonality_error=3.3306690738754696e-16,
        converged=True,
        kinetic_version="trial_fix",
        use_jax_block_kernels=False,
        wall_time_seconds=9.949421839788556,
    ),
    jax_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=1,
        eigenvalues_ha=(-0.18662718689697616,),
        max_residual_norm=0.0001483108390478155,
        max_orthogonality_error=2.220446049250313e-15,
        converged=True,
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
        wall_time_seconds=80.35037644766271,
    ),
    old_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=2,
        eigenvalues_ha=(-0.1866584331584699, -0.1865957329712637),
        max_residual_norm=0.00017589071376628463,
        max_orthogonality_error=1.4254871126234903e-15,
        converged=True,
        kinetic_version="trial_fix",
        use_jax_block_kernels=False,
        wall_time_seconds=14.538639488164335,
    ),
    jax_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=2,
        eigenvalues_ha=(-0.18665843315846384, -0.18659573297125773),
        max_residual_norm=0.00017589071378107762,
        max_orthogonality_error=2.6645352591003757e-15,
        converged=True,
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
        wall_time_seconds=192.0402395427227,
    ),
    diagnosis=(
        "The first JAX handoff into the fixed-potential eigensolver block hot path is numerically "
        "clean but not yet faster. On the repaired A-grid+patch+trial-fix route, the JAX block "
        "kernels reproduce the old hot-path eigenvalues, residuals, and orthogonality to machine "
        "precision for both k=1 and k=2. But on the current CPU-first audit environment the JAX "
        "route is still substantially slower, even after a warmup call, which means the current "
        "win is correctness and kernel placement rather than immediate throughput."
    ),
    note=(
        "Very small H2 fixed-potential regression baseline for the first JAX block-hot-path "
        "handoff inside the eigensolver. The outer eigensolver iteration, Ritz solve, and "
        "convergence control remain in Python/SciPy; only block Hamiltonian apply and weighted "
        "block linear algebra are switched."
    ),
)


H2_JAX_EIGENSOLVER_HOTPATH_REUSE_BASELINE = H2JaxEigensolverHotpathReuseRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    old_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=1,
        eigenvalues_ha=(-0.18662718689698515,),
        max_residual_norm=0.0001483108390547803,
        max_orthogonality_error=3.3306690738754696e-16,
        converged=True,
        kinetic_version="trial_fix",
        use_jax_block_kernels=False,
        use_jax_cached_kernels=False,
        wall_time_seconds=5.467907413840294,
    ),
    preoptimization_jax_k1_wall_time_seconds=80.35037644766271,
    preoptimization_jax_k2_wall_time_seconds=192.0402395427227,
    optimized_jax_k1_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=1,
        eigenvalues_ha=(-0.18662718689697616,),
        max_residual_norm=0.0001483108390478155,
        max_orthogonality_error=2.3314683517128287e-15,
        converged=True,
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
        use_jax_cached_kernels=True,
        wall_time_seconds=3.5797004560008645,
    ),
    optimized_jax_k2_route=H2FixedPotentialEigensolverRouteBaseline(
        path_type="monitor_a_grid_plus_patch",
        target_orbitals=2,
        eigenvalues_ha=(-0.18665843315846384, -0.18659573297125773),
        max_residual_norm=0.00017589071378107762,
        max_orthogonality_error=2.6645352591003757e-15,
        converged=True,
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
        use_jax_cached_kernels=True,
        wall_time_seconds=6.926582344807684,
    ),
    diagnosis=(
        "The compiled-kernel reuse/caching pass hits the real bottleneck that remained after the "
        "first JAX handoff. The JAX route still matches the old hot path to machine precision, but "
        "the repeated closure construction and re-binding overhead are now collapsed enough that the "
        "monitor-grid fixed-potential case turns from clearly slower to clearly faster. On the same "
        "H2 trial-fix route, post-warmup k=1 timing drops from about 80.35 s before the reuse pass "
        "to about 3.58 s after it, and k=2 drops from about 192.04 s to about 6.93 s. At this point "
        "the dominant overhead is no longer JAX block-kernel setup; it has shifted back toward the "
        "remaining Python/SciPy outer eigensolver work and the old-path matvec cost."
    ),
    note=(
        "Regression baseline for the compiled-kernel reuse/caching pass on the H2 fixed-potential "
        "JAX eigensolver hot path. It records the old hot path, the pre-optimization JAX timing "
        "summary from the first handoff baseline, and the optimized JAX k=1/k=2 results."
    ),
)


H2_JAX_SCF_HOTPATH_BASELINE = H2JaxScfHotpathRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    triplet_old_route=H2JaxScfHotpathRouteBaseline(
        spin_state_label="triplet",
        use_jax_block_kernels=False,
        converged=True,
        iteration_count=18,
        final_total_energy_ha=-1.2214418066604806,
        final_lowest_eigenvalue_ha=-0.4168423341628571,
        final_density_residual=0.004552787297010315,
        total_wall_time_seconds=724.5313918269967,
        average_iteration_wall_time_seconds=40.251743990388704,
        eigensolver_wall_time_seconds=434.59398937699007,
        energy_evaluation_wall_time_seconds=289.75772428499477,
        density_update_wall_time_seconds=0.09753895502217347,
        bookkeeping_wall_time_seconds=0.08213920998969115,
    ),
    triplet_jax_route=H2JaxScfHotpathRouteBaseline(
        spin_state_label="triplet",
        use_jax_block_kernels=True,
        converged=True,
        iteration_count=18,
        final_total_energy_ha=-1.2214447513523696,
        final_lowest_eigenvalue_ha=-0.4168390115357246,
        final_density_residual=0.004951321275927076,
        total_wall_time_seconds=613.7521552780017,
        average_iteration_wall_time_seconds=34.097341959888986,
        eigensolver_wall_time_seconds=338.9236460610264,
        energy_evaluation_wall_time_seconds=274.67114807197504,
        density_update_wall_time_seconds=0.08512964200781425,
        bookkeeping_wall_time_seconds=0.07223150299250847,
    ),
    singlet_old_route=H2JaxScfHotpathRouteBaseline(
        spin_state_label="singlet",
        use_jax_block_kernels=False,
        converged=False,
        iteration_count=10,
        final_total_energy_ha=-0.1408486512266819,
        final_lowest_eigenvalue_ha=-0.4361423023175884,
        final_density_residual=0.3336218796629846,
        total_wall_time_seconds=430.8440534900001,
        average_iteration_wall_time_seconds=43.08440534900001,
        eigensolver_wall_time_seconds=219.9329558509853,
        energy_evaluation_wall_time_seconds=210.7553523260067,
        density_update_wall_time_seconds=0.1056408599979477,
        bookkeeping_wall_time_seconds=0.05010445301013533,
    ),
    singlet_jax_route=H2JaxScfHotpathRouteBaseline(
        spin_state_label="singlet",
        use_jax_block_kernels=True,
        converged=False,
        iteration_count=10,
        final_total_energy_ha=-0.14689331535515016,
        final_lowest_eigenvalue_ha=-0.4440994383951656,
        final_density_residual=0.3336247606043798,
        total_wall_time_seconds=419.7974512669971,
        average_iteration_wall_time_seconds=41.97974512669971,
        eigensolver_wall_time_seconds=211.97360892900178,
        energy_evaluation_wall_time_seconds=207.70101472899842,
        density_update_wall_time_seconds=0.07198377999884542,
        bookkeeping_wall_time_seconds=0.05084382899804041,
    ),
    diagnosis=(
        "After wiring the optimized JAX fixed-potential hot path into the A-grid SCF dry-run loop, "
        "the H2 triplet route becomes measurably faster end-to-end without changing the outer SCF "
        "algorithm or the local-only physics. The triplet case still converges and its total wall "
        "time drops from about 724.53 s on the old hot path to about 613.75 s on the JAX hot path. "
        "The rough breakdown shows that the dominant cost is no longer tiny block linear algebra; it "
        "is split mainly between repeated fixed-potential eigensolver work and repeated single-point "
        "energy evaluation, which currently includes the monitor-grid Hartree/Poisson path. The "
        "singlet auxiliary 10-step reference also gets slightly faster, but it remains unconverged, "
        "so the profiling conclusion should still be anchored to triplet."
    ),
    note=(
        "Very rough H2 A-grid SCF hot-path profiling baseline after wiring the already-correct JAX "
        "block kernels into the dry-run loop. Triplet is the main converged profiling case; singlet "
        "is only a 10-step auxiliary reference. Nonlocal remains absent from the A-grid path."
    ),
)


H2_JAX_TRIPLET_HARTREE_ENERGY_BASELINE = H2JaxTripletHartreeEnergyRegressionBaseline(
    benchmark_name="h2_r1p4_bohr",
    monitor_shape=(67, 67, 81),
    box_half_extents_bohr=(8.0, 8.0, 10.0),
    patch_radius_scale=0.75,
    patch_grid_shape=(25, 25, 25),
    correction_strength=1.30,
    interpolation_neighbors=8,
    kinetic_version="trial_fix",
    jax_hartree_cgloop_route=H2JaxTripletHartreeEnergyRouteBaseline(
        path_label="jax-hartree-cgloop",
        hartree_backend="jax",
        cg_impl="jax_loop",
        cg_preconditioner="none",
        use_jax_hartree_cached_operator=True,
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
        converged=True,
        iteration_count=18,
        final_total_energy_ha=-1.2214497987316029,
        final_lowest_eigenvalue_ha=-0.41687445715911675,
        hartree_solve_call_count=37,
        total_wall_time_seconds=109.34411338799691,
        average_iteration_wall_time_seconds=6.074672965999828,
        average_hartree_solve_wall_time_seconds=0.3713480588111624,
        first_hartree_solve_wall_time_seconds=3.7895925769989844,
        repeated_hartree_solve_average_wall_time_seconds=0.27639682219483397,
        repeated_hartree_solve_min_wall_time_seconds=0.12893978499778314,
        repeated_hartree_solve_max_wall_time_seconds=0.49277640099899145,
        average_hartree_cg_iterations=400.0,
        first_hartree_cg_iterations=400,
        repeated_hartree_cg_iteration_average=400.0,
        average_hartree_boundary_condition_wall_time_seconds=0.011885553972820777,
        average_hartree_build_wall_time_seconds=0.02935757443275007,
        average_hartree_rhs_assembly_wall_time_seconds=0.02732031140594454,
        average_hartree_cg_wall_time_seconds=0.27019251491907886,
        average_hartree_cg_other_overhead_wall_time_seconds=0.020603655345562525,
        average_hartree_matvec_call_count=401.0,
        average_hartree_matvec_wall_time_seconds=0.550537072512344,
        average_hartree_matvec_wall_time_per_call_seconds=0.0013729104052676907,
        average_hartree_cg_iteration_wall_time_seconds=0.0006754812872976972,
        average_hartree_matvec_wall_time_per_iteration_seconds=0.0013763426812808599,
        average_hartree_other_cg_overhead_wall_time_per_iteration_seconds=0.0,
        first_hartree_matvec_call_count=401,
        repeated_hartree_matvec_call_count_average=401.0,
        first_hartree_matvec_wall_time_seconds=0.5377947342167317,
        repeated_hartree_matvec_average_wall_time_seconds=0.5508910263538888,
        first_hartree_matvec_wall_time_per_call_seconds=0.0013411340005404782,
        repeated_hartree_matvec_wall_time_per_call_seconds=0.00137379308317678,
        hartree_cached_operator_usage_count=37,
        hartree_cached_operator_first_solve_count=1,
        eigensolver_wall_time_seconds=85.37543347799874,
        static_local_prepare_wall_time_seconds=22.852133282998693,
        hartree_solve_wall_time_seconds=13.76030497199099,
        local_ionic_resolve_wall_time_seconds=7.595771613006946,
        xc_resolve_wall_time_seconds=1.4446051410050131,
        energy_evaluation_wall_time_seconds=12.258049055995798,
        kinetic_energy_wall_time_seconds=0.8978722129977541,
        local_ionic_energy_wall_time_seconds=3.6489997000899166e-05,
        hartree_energy_wall_time_seconds=0.0354724949975207,
        xc_energy_wall_time_seconds=0.010911275010585086,
        ion_ion_energy_wall_time_seconds=0.001247325009899214,
        density_update_wall_time_seconds=0.09542860399960773,
        bookkeeping_wall_time_seconds=11.615202250002767,
    ),
    jax_hartree_pcg_stronger_route=H2JaxTripletHartreeEnergyRouteBaseline(
        path_label="jax-hartree-pcg-stronger",
        hartree_backend="jax",
        cg_impl="jax_loop",
        cg_preconditioner="line",
        use_jax_hartree_cached_operator=True,
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
        converged=True,
        iteration_count=20,
        final_total_energy_ha=-1.221469804365308,
        final_lowest_eigenvalue_ha=-0.4177845481788924,
        hartree_solve_call_count=41,
        total_wall_time_seconds=140.58545625699844,
        average_iteration_wall_time_seconds=7.029272812849922,
        average_hartree_solve_wall_time_seconds=0.5569952675366772,
        first_hartree_solve_wall_time_seconds=3.2063673329976154,
        repeated_hartree_solve_average_wall_time_seconds=0.49076096590015367,
        repeated_hartree_solve_min_wall_time_seconds=0.32941070300148567,
        repeated_hartree_solve_max_wall_time_seconds=0.7263267850030388,
        average_hartree_cg_iterations=269.8536585365854,
        first_hartree_cg_iterations=283,
        repeated_hartree_cg_iteration_average=269.525,
        average_hartree_boundary_condition_wall_time_seconds=0.011878401121687535,
        average_hartree_build_wall_time_seconds=0.0007198030731914758,
        average_hartree_rhs_assembly_wall_time_seconds=0.020327769975510154,
        average_hartree_cg_wall_time_seconds=0.49401404970744695,
        average_hartree_cg_other_overhead_wall_time_seconds=0.14551695256434358,
        average_hartree_matvec_call_count=270.8536585365854,
        average_hartree_matvec_wall_time_seconds=0.35771564945921785,
        average_hartree_matvec_wall_time_per_call_seconds=0.0013206971299259732,
        average_hartree_cg_iteration_wall_time_seconds=0.0018306739007597004,
        average_hartree_matvec_wall_time_per_iteration_seconds=0.0013255912534190105,
        average_hartree_other_cg_overhead_wall_time_per_iteration_seconds=0.00050508264734069,
        first_hartree_matvec_call_count=284,
        repeated_hartree_matvec_call_count_average=270.525,
        first_hartree_matvec_wall_time_seconds=0.378574556147214,
        repeated_hartree_matvec_average_wall_time_seconds=0.357194176792018,
        first_hartree_matvec_wall_time_per_call_seconds=0.0013330090005183592,
        repeated_hartree_matvec_wall_time_per_call_seconds=0.0013203740016339267,
        hartree_cached_operator_usage_count=41,
        hartree_cached_operator_first_solve_count=1,
        eigensolver_wall_time_seconds=106.59518006299913,
        static_local_prepare_wall_time_seconds=32.734559642984095,
        hartree_solve_wall_time_seconds=22.85945238999193,
        local_ionic_resolve_wall_time_seconds=8.312332736993994,
        xc_resolve_wall_time_seconds=1.5099134020092606,
        energy_evaluation_wall_time_seconds=17.577609923999262,
        kinetic_energy_wall_time_seconds=1.012964835994353,
        local_ionic_energy_wall_time_seconds=9.74060021690093e-05,
        hartree_energy_wall_time_seconds=0.04294327599563985,
        xc_energy_wall_time_seconds=0.012593028994160704,
        ion_ion_energy_wall_time_seconds=0.0017679369993857108,
        density_update_wall_time_seconds=0.10290262100534164,
        bookkeeping_wall_time_seconds=16.309763648994704,
    ),
    diagnosis=(
        "This baseline freezes the first stronger-but-still-small JAX Hartree PCG experiment on the "
        "cached triplet SCF route. Both main routes keep the JAX eigensolver hot path, step-local "
        "static-local reuse, cached Hartree operator reuse, and `cg_impl='jax_loop'`; the only "
        "intended difference is `cg_preconditioner='none'` versus the stronger metric-aware line "
        "preconditioner. On H2 triplet, that stronger route finally lowers the average Hartree "
        "iteration count materially, from about 400.0 to about 269.85. However, the line "
        "preconditioner apply is still expensive enough that end-to-end SCF wall time gets worse, "
        "and the route also takes 20 SCF steps instead of 18. The conclusion is that iteration count "
        "is now the right target, but this stronger preconditioner is not yet a net timing win."
    ),
    note=(
        "Very rough triplet-only SCF profiling baseline for the stronger JAX Hartree PCG prototype "
        "on the repaired A-grid+patch+kinetic-trial-fix route. Both routes use "
        "hartree_backend='jax', use_jax_hartree_cached_operator=True, and the JAX eigensolver hot "
        "path; they differ only in `cg_preconditioner`. Timing values come from the latest audit run "
        "and should be interpreted as rough wall-time references, not formal benchmarks."
    ),
)


__all__ = [
    "H2JaxEigensolverHotpathReuseRegressionBaseline",
    "H2JaxEigensolverHotpathRegressionBaseline",
    "H2JaxScfHotpathRegressionBaseline",
    "H2JaxScfHotpathRouteBaseline",
    "H2JaxTripletHartreeEnergyRegressionBaseline",
    "H2JaxTripletHartreeEnergyRouteBaseline",
    "H2JaxKernelConsistencyLocalHamiltonianBaseline",
    "H2JaxKernelConsistencyPoissonBaseline",
    "H2JaxKernelConsistencyReductionsBaseline",
    "H2JaxKernelConsistencyRegressionBaseline",
    "H2DiisScfRegressionBaseline",
    "H2DiisScfRouteBaseline",
    "H2DiisScfSpinBaseline",
    "H2GeometryConsistencyFieldBaseline",
    "H2GeometryConsistencyRegressionBaseline",
    "H2GeometryConsistencySmoothFieldBaseline",
    "H2KineticGreenIdentityFieldBaseline",
    "H2KineticGreenIdentityRegressionBaseline",
    "H2KineticGreenIdentitySmoothFieldBaseline",
    "H2KineticFormRegressionBaseline",
    "H2KineticFormRouteBaseline",
    "H2KineticFormSmoothFieldBaseline",
    "H2KineticOperatorRegressionBaseline",
    "H2KineticOperatorRouteBaseline",
    "H2KineticOperatorSmoothFieldBaseline",
    "H2K2SubspaceMatrixBaseline",
    "H2K2SubspaceOrbitalBaseline",
    "H2K2SubspaceRegressionBaseline",
    "H2K2SubspaceRotationBaseline",
    "H2OrbitalShapeOrbitalBaseline",
    "H2OrbitalShapeRegressionBaseline",
    "H2ScfDryRunRegressionBaseline",
    "H2ScfDryRunRouteBaseline",
    "H2SingletStabilityRegressionBaseline",
    "H2SingletStabilityRouteBaseline",
    "H2FixedPotentialOperatorRegressionBaseline",
    "H2FixedPotentialOperatorRouteBaseline",
    "H2FixedPotentialEigensolverRegressionBaseline",
    "H2FixedPotentialEigensolverRouteBaseline",
    "H2HartreeTailRecheckPointBaseline",
    "H2HartreeTailRecheckRegressionBaseline",
    "H2MonitorPoissonRegressionBaseline",
    "H2MonitorPoissonShapeRegressionPoint",
    "H2StaticLocalChainRegressionBaseline",
    "H2StaticLocalChainRouteBaseline",
    "H2PySCFRegressionBaseline",
    "H2_DEFAULT_PYSCF_REGRESSION_BASELINE",
    "H2_FIXED_POTENTIAL_EIGENSOLVER_BASELINE",
    "H2_FIXED_POTENTIAL_EIGENSOLVER_TRIAL_FIX_BASELINE",
    "H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE",
    "H2_FIXED_POTENTIAL_OPERATOR_TRIAL_FIX_BASELINE",
    "H2_GEOMETRY_CONSISTENCY_AUDIT_BASELINE",
    "H2_HARTREE_TAIL_RECHECK_BASELINE",
    "H2_DIIS_SCF_BASELINE",
    "H2_JAX_EIGENSOLVER_HOTPATH_REUSE_BASELINE",
    "H2_JAX_EIGENSOLVER_HOTPATH_BASELINE",
    "H2_JAX_KERNEL_CONSISTENCY_BASELINE",
    "H2_JAX_SCF_HOTPATH_BASELINE",
    "H2_JAX_TRIPLET_HARTREE_ENERGY_BASELINE",
    "H2_K2_SUBSPACE_AUDIT_BASELINE",
    "H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE",
    "H2_KINETIC_GREEN_IDENTITY_TRIAL_FIX_BASELINE",
    "H2_KINETIC_FORM_AUDIT_BASELINE",
    "H2_KINETIC_OPERATOR_AUDIT_BASELINE",
    "H2_MONITOR_POISSON_REGRESSION_BASELINE",
    "H2_ORBITAL_SHAPE_AUDIT_BASELINE",
    "H2_SCF_DRY_RUN_BASELINE",
    "H2_SINGLET_STABILITY_BASELINE",
    "H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE",
]
