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


__all__ = [
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
    "H2OrbitalShapeOrbitalBaseline",
    "H2OrbitalShapeRegressionBaseline",
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
    "H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE",
    "H2_KINETIC_GREEN_IDENTITY_TRIAL_FIX_BASELINE",
    "H2_KINETIC_FORM_AUDIT_BASELINE",
    "H2_KINETIC_OPERATOR_AUDIT_BASELINE",
    "H2_MONITOR_POISSON_REGRESSION_BASELINE",
    "H2_ORBITAL_SHAPE_AUDIT_BASELINE",
    "H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE",
]
