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


__all__ = [
    "H2MonitorPoissonRegressionBaseline",
    "H2MonitorPoissonShapeRegressionPoint",
    "H2StaticLocalChainRegressionBaseline",
    "H2StaticLocalChainRouteBaseline",
    "H2PySCFRegressionBaseline",
    "H2_DEFAULT_PYSCF_REGRESSION_BASELINE",
    "H2_MONITOR_POISSON_REGRESSION_BASELINE",
    "H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE",
]
