# CODEBASE

Generated repository guide covering root files, production modules under `src/`, and tests under `tests/`.

Conventions:
- Only `.py` source files are indexed; `__pycache__` entries are ignored.
- Function/class descriptions prefer docstrings when present.
- Test descriptions prefer docstrings; otherwise they are derived from the test name.

## Root Files

- [README.md](README.md): Project overview, current status, milestones, and run commands.
- [AGENTS.md](AGENTS.md): Project-specific scientific goals, engineering rules, and working constraints.
- [pyproject.toml](pyproject.toml): Packaging metadata, dependencies, and tool configuration.
- [.gitignore](.gitignore): Git ignore rules.
- [docs/GRID_REDESIGN_PLAN.md](docs/GRID_REDESIGN_PLAN.md): Grid redesign planning/design notes.
- [docs/.gitkeep](docs/.gitkeep): Keeps the docs directory in git.

## Source Tree (`src/`)

### `src/isogrid/__init__.py`

- File role: IsoGridDFT package skeleton.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/api/__init__.py`

- File role: Public API placeholders for IsoGridDFT.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/audit/__init__.py`

- File role: Audit and reference utilities.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/audit/baselines.py`

- File role: Lightweight regression baselines for the current H2 audit path. These values are not acceptance targets. They are the first recorded audit baseline for the current minimal H2 SCF closed loop against the PySCF reference under the shared nominal model: - H2 at R = 1.4 Bohr - UKS / gth-pade / gth-dzvp / lda,vwn The numeric values below are intentionally stored at the same precision used by the first formal audit report so that later numerical changes can be compared against one clear reference point.
- Classes:
  - `H2PySCFRegressionBaseline`: Recorded H2 error baseline for the current minimal SCF implementation.
  - `H2MonitorPoissonShapeRegressionPoint`: Recorded A-grid shape-scan summary for the operator audit.
  - `H2MonitorPoissonRegressionBaseline`: Recorded H2 monitor-grid Poisson audit baseline after the split fix.
  - `H2StaticLocalChainRouteBaseline`: Recorded static local-chain components for one H2 audit route.
  - `H2StaticLocalChainRegressionBaseline`: Recorded post-fix H2 static local-chain audit on legacy and A-grid routes.
  - `H2HartreeTailRecheckPointBaseline`: Recorded H2 Hartree tail-recheck point on the A-grid.
  - `H2HartreeTailRecheckRegressionBaseline`: Recorded very small H2 Hartree tail-recheck after the split fix.
  - `H2FixedPotentialEigensolverRouteBaseline`: Recorded fixed-potential eigensolver result for one audit route.
  - `H2JaxNativeFixedPotentialEigensolverRegressionBaseline`: Recorded JAX-native fixed-potential eigensolver validation on the H2 A-grid path.
  - `H2FixedPotentialEigensolverRegressionBaseline`: Recorded fixed-potential eigensolver audit on legacy and A-grid+patch.
  - `H2FixedPotentialOperatorRouteBaseline`: Recorded operator-level audit result for one fixed-potential route.
  - `H2FixedPotentialTrialFixOperatorRegressionBaseline`: Recorded operator-level comparison after wiring in the kinetic trial fix.
  - `H2FixedPotentialTrialFixEigensolverRegressionBaseline`: Recorded fixed-potential eigensolver comparison after the kinetic trial fix.
  - `H2FixedPotentialOperatorRegressionBaseline`: Recorded operator-level failure baseline for the A-grid static-local path.
  - `H2KineticOperatorRouteBaseline`: Recorded kinetic-operator audit result for one route and shape label.
  - `H2KineticOperatorSmoothFieldBaseline`: Recorded kinetic probe result for one simple smooth field.
  - `H2KineticOperatorRegressionBaseline`: Recorded kinetic-operator failure baseline on legacy and A-grid routes.
  - `H2KineticFormRouteBaseline`: Recorded production-vs-reference kinetic-form result for one field.
  - `H2KineticFormSmoothFieldBaseline`: Recorded kinetic-form comparison for one smooth probe field.
  - `H2KineticFormRegressionBaseline`: Recorded production-vs-reference kinetic-form failure baseline.
  - `H2GeometryConsistencyFieldBaseline`: Recorded operator-vs-gradient kinetic identity for one field.
  - `H2GeometryConsistencySmoothFieldBaseline`: Recorded geometry-consistency probe for one smooth field.
  - `H2GeometryConsistencyRegressionBaseline`: Recorded geometry-consistency failure baseline on the A-grid.
  - `H2KineticGreenIdentityFieldBaseline`: Recorded discrete Green-identity audit result for one field.
  - `H2KineticGreenIdentitySmoothFieldBaseline`: Recorded discrete Green-identity audit result for one smooth field.
  - `H2KineticGreenIdentityRegressionBaseline`: Recorded discrete Green-identity / boundary-mismatch failure baseline.
  - `H2OrbitalShapeOrbitalBaseline`: Recorded shape/symmetry summary for one fixed-potential orbital.
  - `H2OrbitalShapeRegressionBaseline`: Recorded fixed-potential orbital-shape baseline for H2.
  - `H2K2SubspaceOrbitalBaseline`: Recorded k=2 subspace orbital summary.
  - `H2K2SubspaceMatrixBaseline`: Recorded 2x2 subspace matrix summary.
  - `H2K2SubspaceRotationBaseline`: Recorded very small k=2 subspace rotation summary.
  - `H2K2SubspaceRegressionBaseline`: Recorded H2 k=2 subspace audit baseline.
  - `H2ScfDryRunRouteBaseline`: Recorded H2 SCF dry-run result for one route and one spin state.
  - `H2ScfDryRunRegressionBaseline`: Recorded H2 SCF dry-run baseline for legacy and A-grid routes.
  - `H2SingletStabilityRouteBaseline`: Recorded singlet stability result for one conservative mixing scheme.
  - `H2SingletStabilityRegressionBaseline`: Recorded very small A-grid singlet stability audit baseline.
  - `H2DiisScfRouteBaseline`: Recorded A-grid DIIS SCF result for one spin state and one scheme.
  - `H2DiisScfSpinBaseline`: Recorded three-scheme A-grid DIIS SCF baseline for one spin state.
  - `H2DiisScfRegressionBaseline`: Recorded A-grid DIIS SCF audit baseline on singlet and triplet.
  - `H2JaxKernelConsistencyReductionsBaseline`: Recorded first-batch JAX reductions consistency summary.
  - `H2JaxKernelConsistencyPoissonBaseline`: Recorded first-batch JAX Poisson consistency summary.
  - `H2JaxKernelConsistencyLocalHamiltonianBaseline`: Recorded first-batch JAX local-Hamiltonian consistency summary.
  - `H2JaxKernelConsistencyRegressionBaseline`: Recorded first-batch JAX kernel migration baseline on the H2 A-grid path.
  - `H2JaxEigensolverHotpathRegressionBaseline`: Recorded old-vs-JAX block-hot-path comparison on the H2 fixed-potential line.
  - `H2JaxEigensolverHotpathReuseRegressionBaseline`: Recorded compiled-kernel reuse/caching comparison on the JAX eigensolver hot path.
  - `H2JaxScfHotpathRouteBaseline`: Recorded A-grid SCF hot-path profiling result for one spin state and route.
  - `H2JaxScfHotpathRegressionBaseline`: Recorded A-grid SCF hot-path profiling baseline after JAX handoff.
  - `H2JaxSingletResponseChannelDifficultyBaseline`: Recorded channel-wise tail difficulty proxy for one singlet route.
  - `H2JaxSingletMainlineRouteBaseline`: Recorded one H2 singlet mixing route on the frozen JAX A-grid mainline.
  - `H2JaxSingletMainlineRegressionBaseline`: Recorded H2 singlet formal-mixer audit on the frozen JAX A-grid mainline.
  - `H2JaxSingletAcceptanceRegressionBaseline`: Recorded single-route H2 singlet acceptance result on the latest JAX mainline.
  - `H2JaxSingletHartreeTailGuardRegressionBaseline`: Recorded experimental Hartree-tail guard audit on the latest JAX singlet mainline.
  - `H2JaxSingletStructuralStabilizerRegressionBaseline`: Recorded experimental structural stabilizer audit on the latest JAX singlet mainline.
  - `H2JaxTripletHartreeEnergyRouteBaseline`: Recorded triplet-only SCF profiling route for Hartree/energy optimization.
  - `H2JaxTripletHartreeEnergyRegressionBaseline`: Recorded triplet-only SCF profiling baseline for stronger JAX Hartree PCG checks.
  - `H2JaxTripletReintegrationSmokeRouteBaseline`: Recorded H2 triplet reintegration smoke result for the JAX mainline.
  - `H2JaxTripletReintegrationSmokeRegressionBaseline`: Recorded H2 triplet smoke baseline after JAX-native eigensolver reintegration.
  - `H2JaxTripletMicroProfileStepBaseline`: Recorded one step from the triplet end-to-end micro-profile audit.
  - `H2JaxTripletEigensolverInternalBucketBaseline`: Recorded aggregate in-loop eigensolver bucket summary.
  - `H2JaxTripletEndToEndMicroProfileBaseline`: Recorded triplet end-to-end micro-profile on the JAX mainline.

### `src/isogrid/audit/fixed_potential_h2_eigensolver_audit.py`

- File role: Audit script for the first fixed-potential H2 eigensolver slice.
- Functions:
  - `_density_summary()`: Function without docstring.
  - `_orbital_mirror_error()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `main()`: Function without docstring.

### `src/isogrid/audit/gth_local_h2_audit.py`

- File role: Audit script for the first H2 local GTH pseudopotential slice.
- Functions:
  - `_format_loaded_pseudo_summary()`: Function without docstring.
  - `_centerline_sample_rows()`: Function without docstring.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_grid_convergence_audit.py`

- File role: Grid/box convergence audit for the current H2 singlet SCF path. This module is intentionally narrow. It does not change the physical model or the SCF workflow. It only scans a small set of geometry-discretization choices around the current H2 singlet default point so we can separate: - grid-resolution sensitivity - finite-domain / open-boundary sensitivity The two scan families are organized as follows: 1. grid-shape scan: keep the physical box fixed and vary the logical grid shape 2. box-half-extent scan: vary the physical box and adjust the companion grid shape so the center spacing stays close to the current baseline; this keeps the box scan focused on domain/open-boundary effects instead of folding in a large resolution jump This is a first geometric-discretization audit, not a final convergence claim.
- Classes:
  - `EnergyComponentDrift`: Per-component drift relative to the current baseline point.
  - `H2GridScanParameters`: Geometry-discretization parameters for one audit point.
  - `H2GridConvergenceScanPoint`: One H2 singlet geometry-discretization audit point.
  - `H2GridConvergenceAuditResult`: Top-level H2 singlet grid/domain audit result.
- Functions:
  - `_find_singlet_spin_state()`: Function without docstring.
  - `_build_grid_geometry_variant()`: Function without docstring.
  - `_build_parameters()`: Function without docstring.
  - `_energy_drift()`: Function without docstring.
  - `_dominant_component_drifts()`: Function without docstring.
  - `_run_singlet_point()`: Function without docstring.
  - `_assemble_scan_point()`: Function without docstring.
  - `run_h2_grid_convergence_audit()`: Run the first H2 singlet grid/domain convergence audit.
  - `_format_one_mha_status()`: Function without docstring.
  - `_print_component_drift()`: Function without docstring.
  - `_print_scan_point()`: Function without docstring.
  - `print_h2_grid_convergence_summary()`: Print the compact H2 singlet grid/domain convergence summary.
  - `main()`: Run the H2 singlet grid/domain convergence audit.

### `src/isogrid/audit/h2_hartree_boundary_diagnosis_audit.py`

- File role: Fixed-density Hartree/open-boundary diagnosis audit for the H2 A-grid path.
- Classes:
  - `FixedDensityHartreeRouteSummary`: Compact Hartree summary for one fixed-density route.
  - `FixedDensityDifferenceSummary`: Difference summary between legacy and A-grid routes for one density.
  - `MonitorBoxSensitivitySummary`: Sensitivity of the A-grid Hartree route to monitor-box enlargement.
  - `GaussianShiftSensitivitySummary`: Translation-reasonableness smoke check on the same A-grid box.
  - `MonitorVolumeConsistencySummary`: Consistency of the monitor integration measure against the physical box volume.
  - `GaussianRepresentationConsistencySummary`: Centered-Gaussian moments under uniform-box and mapped-monitor measures.
  - `MonitorInversionSymmetrySummary`: Very local inversion-pairing diagnostics for the centered-Gaussian monitor route.
  - `H2HartreeBoundaryDiagnosisAuditResult`: Top-level diagnosis result for fixed-density Hartree/open-boundary behavior.
  - `HartreeBoundaryShapeSweepPoint`: One fixed-density Hartree diagnosis sample at a selected monitor shape.
  - `HartreeBoundaryShapeSweepAuditResult`: Resolution sweep summary for fixed-density Hartree/open-boundary behavior.
  - `MeasureLedgerIntegralSummary`: One test-function integral under one discrete measure.
  - `MeasureLedgerPathSummary`: One audit ledger row for a concrete path/measure pairing.
  - `HartreeMeasureLedgerAuditResult`: Ledger audit for monitor-grid measures used by Hartree-related paths.
  - `CellVolumeConstructionSummary`: Trace how monitor cell volumes are constructed from the mapping Jacobian.
  - `PolynomialExactnessRow`: Integral biases for one polynomial under several discrete measures.
  - `SecondOrderRegionRow`: Regional mapping-distortion summary for one second-order polynomial.
  - `GeometryRepresentationErrorRegionSummary`: Where mapping and weight distortions are largest on the monitor grid.
  - `MappingZStretchSummary`: Local z-direction stretch diagnostics for the monitor mapping.
  - `AxisMappingRow`: Per-axis first/second-derivative summary for the monitor mapping.
  - `MonitorStrengthProxySummary`: Very-light proxy summary relating monitor strength to second-order distortion.
  - `HartreeGeometryRepresentationAuditResult`: Geometry/measure representation audit for the monitor grid.
  - `MappingStageAttributionRow`: One stage row in the monitor-mapping attribution ledger.
  - `H2HartreeMappingStageAttributionAuditResult`: Stage-by-stage attribution audit for monitor mapping error amplification.
  - `MappingSolveStageAttributionRow`: Deeper mapping-solve stage row with derivative-based geometry diagnostics.
  - `H2HartreeMappingSolveStageAttributionAuditResult`: Deeper attribution audit focused on the mapping-solve chain itself.
  - `ReferenceQuadratureIntegralRow`: Production nodal measure versus audit-only reference quadrature for one function.
  - `H2HartreeReferenceQuadratureAuditResult`: Audit whether a higher-quality local reference quadrature improves mapped-grid exactness.
  - `InsideCellRepresentationCellSummary`: One representative mapped cell compared under analytic vs trilinear Gaussian representation.
  - `H2HartreeInsideCellRepresentationAuditResult`: Audit whether fake quadrupole is already formed inside mapped-cell Gaussian representation.
  - `InsideCellReconstructionSummary`: One local field reconstruction compared against analytic Gaussian on a mapped cell.
  - `InsideCellReconstructionComparisonCellSummary`: One representative cell with multiple reconstruction-only comparisons.
  - `H2HartreeInsideCellReconstructionComparisonAuditResult`: Audit-only comparison of local Gaussian field reconstructions on mapped cells.
- Functions:
  - `_box_half_extents()`: Function without docstring.
  - `_build_h2_frozen_density()`: Function without docstring.
  - `_build_gaussian_density()`: Function without docstring.
  - `_trapezoidal_logical_weights()`: Function without docstring.
  - `_analytic_box_integrals()`: Function without docstring.
  - `_measure_ledger_integrals()`: Function without docstring.
  - `_logical_cell_volume()`: Function without docstring.
  - `_polynomial_reference_values()`: Function without docstring.
  - `_polynomial_fields()`: Function without docstring.
  - `_polynomial_exactness_rows()`: Function without docstring.
  - `_second_order_region_rows()`: Function without docstring.
  - `_mapping_z_stretch_summary()`: Function without docstring.
  - `_axis_mapping_rows()`: Function without docstring.
  - `_monitor_strength_proxy_summary()`: Function without docstring.
  - `_geometry_representation_error_region_summary()`: Function without docstring.
  - `_quadrupole_tensor()`: Function without docstring.
  - `_dipole_vector()`: Function without docstring.
  - `_build_uniform_box_measure_geometry()`: Function without docstring.
  - `_build_mapped_coordinate_measure_geometry()`: Function without docstring.
  - `_monitor_volume_consistency_summary()`: Function without docstring.
  - `_gaussian_representation_consistency_summary()`: Function without docstring.
  - `_monitor_inversion_symmetry_summary()`: Function without docstring.
  - `_far_field_diagnostic()`: Function without docstring.
  - `_centerline_band_means()`: Function without docstring.
  - `evaluate_fixed_density_hartree_route()`: Evaluate one fixed-density Hartree route on a selected grid.
  - `_classify_difference_pattern()`: Function without docstring.
  - `_difference_summary()`: Function without docstring.
  - `_box_sensitivity_summary()`: Function without docstring.
  - `_gaussian_shift_sensitivity_summary()`: Function without docstring.
  - `_build_monitor_geometry()`: Function without docstring.
  - `_run_operator_route()`: Function without docstring.
  - `_diagnose()`: Function without docstring.
  - `_is_nearly_nonincreasing()`: Function without docstring.
  - `_shape_sweep_diagnosis()`: Function without docstring.
  - `run_h2_hartree_boundary_diagnosis_audit()`: Run a fixed-density Hartree/open-boundary diagnosis audit for H2.
  - `run_h2_hartree_boundary_shape_sweep_audit()`: Run a small monitor-shape sweep for fixed-density Hartree/open-boundary diagnosis.
  - `run_h2_hartree_measure_ledger_audit()`: Audit the discrete measures used by monitor-grid Hartree-related paths.
  - `run_h2_hartree_geometry_representation_audit()`: Audit the monitor-grid cell-volume construction and polynomial exactness.
  - `_geometry_namespace()`: Function without docstring.
  - `_stage_polynomial_biases()`: Function without docstring.
  - `_stage_high_jacobian_contribution_distortion()`: Function without docstring.
  - `_stage_row_from_geometry()`: Function without docstring.
  - `_mapping_stage_worsened()`: Function without docstring.
  - `_stage_rows_with_worsening_flags()`: Function without docstring.
  - `_solve_weighted_harmonic_coordinates_trace()`: Audit-only trace of the production harmonic solve.
  - `_backtracking_update_trace()`: Audit-only trace of production backtracking candidates.
  - `_second_derivative_fields_from_coordinates()`: Function without docstring.
  - `_stage_coordinate_masks()`: Function without docstring.
  - `_mapping_solve_monitor_stage_row()`: Function without docstring.
  - `_mapping_solve_coordinate_stage_row()`: Function without docstring.
  - `_mapping_solve_stage_worsened()`: Function without docstring.
  - `_mapping_solve_rows_with_worsening_flags()`: Function without docstring.
  - `run_h2_hartree_mapping_solve_stage_attribution_audit()`: Deeper stage-by-stage attribution focused on the first mapping outer update.
  - `run_h2_hartree_mapping_stage_attribution_audit()`: Attribute where mapping-chain stages first amplify z-directed second-moment error.
  - `_trilinear_sample()`: Function without docstring.
  - `_trilinear_cell_value()`: Function without docstring.
  - `_cell_average_from_nodal_field()`: Function without docstring.
  - `_representative_cell_indices()`: Function without docstring.
  - `_inside_cell_profile_errors()`: Function without docstring.
  - `_inside_cell_moment_errors()`: Function without docstring.
  - `_analytic_gaussian_value()`: Function without docstring.
  - `_local_quadratic_fit_coefficients()`: Function without docstring.
  - `_evaluate_quadratic_fit()`: Function without docstring.
  - `_cell_and_neighbor_slices()`: Function without docstring.
  - `_inside_cell_profile_errors_from_reconstruction()`: Function without docstring.
  - `_inside_cell_moment_errors_from_reconstruction()`: Function without docstring.
  - `_inside_cell_reconstruction_summary()`: Function without docstring.
  - `run_h2_hartree_inside_cell_representation_audit()`: Audit Gaussian inside-cell representation error on representative mapped cells.
  - `run_h2_hartree_inside_cell_reconstruction_comparison_audit()`: Compare audit-only local Gaussian reconstructions on representative mapped cells.
  - `_reference_quadrature_summary()`: Function without docstring.
  - `run_h2_hartree_reference_quadrature_audit()`: Compare the production nodal monitor measure against an audit-only subcell quadrature.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_hartree_poisson_comparison_audit.py`

- File role: H2 singlet frozen-density Hartree / Poisson comparison audit.
- Classes:
  - `HartreeBoundarySummary`: Compact open-boundary summary for one Hartree solve.
  - `HartreeCenterLineSample`: One center-line Hartree potential sample on the molecular axis.
  - `H2HartreeRouteResult`: Resolved Hartree / Poisson result for one grid family.
  - `H2HartreeDifferenceSummary`: Difference summary between the legacy and A-grid Hartree routes.
  - `HartreeBoundaryScanPoint`: One multipole-order scan point for one grid family.
  - `HartreeToleranceScanPoint`: One Poisson tolerance scan point on the A-grid.
  - `H2HartreePoissonComparisonAuditResult`: Top-level H2 singlet Hartree / Poisson comparison audit result.
- Functions:
  - `_grid_parameter_summary()`: Function without docstring.
  - `_build_frozen_density()`: Function without docstring.
  - `_boundary_mask()`: Function without docstring.
  - `_sample_at_point()`: Function without docstring.
  - `_build_boundary_summary()`: Function without docstring.
  - `_build_centerline_samples()`: Function without docstring.
  - `evaluate_h2_singlet_hartree_route()`: Evaluate one Hartree / Poisson route for the fixed H2 singlet density.
  - `_classify_difference_pattern()`: Function without docstring.
  - `compare_h2_hartree_routes()`: Compare the resolved legacy and A-grid Hartree routes.
  - `_run_boundary_scan()`: Function without docstring.
  - `_run_tolerance_scan()`: Function without docstring.
  - `_build_diagnosis()`: Function without docstring.
  - `run_h2_hartree_poisson_comparison_audit()`: Run the H2 singlet frozen-density Hartree / Poisson comparison audit.
  - `_print_route_result()`: Function without docstring.
  - `print_h2_hartree_poisson_comparison_summary()`: Print the compact Hartree / Poisson comparison summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_hartree_tail_recheck_audit.py`

- File role: Very small H2 Hartree tail recheck on the repaired monitor Poisson path.
- Classes:
  - `H2HartreeTailRecheckPoint`: One very small A-grid Hartree tail-recheck point.
  - `H2HartreeTailRecheckAuditResult`: Top-level H2 Hartree tail-recheck result on the repaired monitor path.
- Functions:
  - `_build_h2_frozen_density()`: Function without docstring.
  - `_centerline_far_field_potential_mean()`: Function without docstring.
  - `_far_field_region()`: Function without docstring.
  - `_build_point()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_hartree_tail_recheck_audit()`: Run a very small H2 singlet Hartree tail recheck on the repaired A-grid path.
  - `_print_point()`: Function without docstring.
  - `print_h2_hartree_tail_recheck_summary()`: Print the compact H2 Hartree tail-recheck summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_jax_eigensolver_hotpath_audit.py`

- File role: Very small old-vs-JAX hot-path audit for the H2 fixed-potential eigensolver.
- Classes:
  - `H2JaxEigensolverHotpathComparison`: Old-vs-JAX comparison for one fixed-potential target-orbital count.
  - `H2JaxEigensolverHotpathAuditResult`: Very small correctness/timing audit for the JAX eigensolver hot path.
- Functions:
  - `_build_comparison()`: Function without docstring.
  - `run_h2_jax_eigensolver_hotpath_audit()`: Compare the old and JAX block hot paths on the H2 fixed-potential route.
  - `print_h2_jax_eigensolver_hotpath_summary()`: Print the compact old-vs-JAX hot-path summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_jax_kernel_consistency_audit.py`

- File role: Minimal correctness audit for the first JAX hot-kernel migration.
- Classes:
  - `JaxReductionConsistencyResult`: Compact reductions audit summary.
  - `JaxPoissonConsistencyResult`: Compact JAX-vs-current Poisson audit summary.
  - `JaxLocalHamiltonianConsistencyResult`: Compact JAX-vs-current local Hamiltonian audit summary.
  - `H2JaxKernelConsistencyAuditResult`: Top-level first-batch JAX migration audit result.
- Functions:
  - `_build_h2_bonding_trial_orbital()`: Function without docstring.
  - `_build_probe_orbital_block()`: Function without docstring.
  - `run_h2_jax_kernel_consistency_audit()`: Run one minimal correctness audit for the first JAX hot kernels.
  - `print_h2_jax_kernel_consistency_summary()`: Print the compact first-batch JAX consistency summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_jax_scf_hotpath_audit.py`

- File role: Very rough old-vs-JAX SCF hot-path audit for the H2 monitor-grid dry-run. This audit keeps the A-grid dry-run Hamiltonian restricted to T + V_loc,ion + V_H + V_xc with the repaired monitor-grid Hartree/Poisson path, patch-assisted local ionic slice, and the kinetic trial-fix branch. Nonlocal remains excluded here, so the timing/profiling results only measure the current local-only monitor-grid SCF dry-run line.
- Classes:
  - `H2JaxScfTimingBreakdown`: Very rough aggregated timing breakdown for one SCF dry-run route.
  - `H2JaxScfHotpathRouteResult`: Compact SCF hot-path profiling summary for one route and one spin state.
  - `H2JaxScfHotpathAuditResult`: Top-level H2 SCF hot-path audit for old and JAX monitor-grid routes.
- Functions:
  - `_monitor_parameter_summary()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `_run_route()`: Function without docstring.
  - `run_h2_jax_scf_hotpath_audit()`: Run the very rough H2 SCF hot-path audit on the monitor-grid route.
  - `_print_route()`: Function without docstring.
  - `print_h2_jax_scf_hotpath_summary()`: Print the compact H2 SCF hot-path profiling summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_jax_singlet_mainline_audit.py`

- File role: Singlet fixed-point local-difficulty audit on the frozen JAX A-grid mainline.
- Classes:
  - `H2JaxSingletMainlineTimingBreakdown`: Very rough timing buckets for one frozen singlet mainline route.
  - `H2JaxSingletMainlineBehavior`: Very small convergence-behavior summary for one singlet route.
  - `H2JaxSingletFixedPointLocalDifficulty`: Very small local fixed-point difficulty summary near the singlet tail.
  - `H2JaxSingletResponseChannelDifficulty`: Very small channel-wise tail difficulty summary near the singlet fixed-point tail.
  - `H2JaxSingletMainlineParameterSummary`: Frozen parameter summary for one singlet mainline route.
  - `H2JaxSingletMainlineRouteResult`: Compact audit result for one frozen singlet mainline route.
  - `H2JaxSingletMainlineAuditResult`: Singlet fixed-point local-difficulty audit on the frozen JAX A-grid mainline.
  - `H2JaxSingletAcceptanceAuditResult`: Single-route H2 singlet acceptance result on the latest frozen JAX mainline.
  - `H2JaxSingletHartreeTailGuardAuditResult`: Two-route experimental Hartree-tail guard audit on the latest JAX singlet mainline.
  - `H2JaxSingletHartreeTailGuardV2AuditResult`: Two-route experimental Hartree-tail guard 2.0 audit on the latest JAX singlet mainline.
  - `H2JaxSingletStructuralStabilizerAuditResult`: Two-route structural stabilizer audit on the latest JAX singlet mainline.
- Functions:
  - `_safe_mean()`: Function without docstring.
  - `_safe_std()`: Function without docstring.
  - `_tail_energy_change_history()`: Function without docstring.
  - `_tail_residual_ratios()`: Function without docstring.
  - `_spin_density_residual_vector()`: Function without docstring.
  - `_tail_plateau_window_length()`: Function without docstring.
  - `_estimate_secant_subspace_condition()`: Function without docstring.
  - `_estimate_secant_collinearity()`: Function without docstring.
  - `_build_fixed_point_local_difficulty()`: Function without docstring.
  - `_default_patch_parameters()`: Function without docstring.
  - `_stack_spin_fields()`: Function without docstring.
  - `_safe_norm_ratio()`: Function without docstring.
  - `_channel_contribution_share()`: Function without docstring.
  - `_build_spin_contexts()`: Function without docstring.
  - `_build_response_channel_difficulty()`: Function without docstring.
  - `_build_behavior()`: Function without docstring.
  - `_build_parameter_summary()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `_run_route()`: Function without docstring.
  - `_select_best_anderson_route()`: Function without docstring.
  - `_route_clearly_beats_baseline()`: Function without docstring.
  - `_build_diagnosis()`: Function without docstring.
  - `_route_is_close_enough_for_longer_view()`: Function without docstring.
  - `_build_acceptance_diagnosis()`: Function without docstring.
  - `run_h2_jax_singlet_mainline_audit()`: Run the frozen H2 singlet Hartree-tail mitigation audit.
  - `run_h2_jax_singlet_acceptance_audit()`: Run the single-route H2 singlet acceptance test on the latest JAX mainline.
  - `_build_guard_diagnosis()`: Function without docstring.
  - `_build_guard_v2_diagnosis()`: Function without docstring.
  - `_build_structural_stabilizer_diagnosis()`: Function without docstring.
  - `run_h2_jax_singlet_hartree_tail_guard_audit()`: Run the narrow singlet baseline-vs-guard audit on the latest JAX mainline.
  - `run_h2_jax_singlet_hartree_tail_guard_v2_audit()`: Run the narrow singlet baseline-vs-guard-2.0 audit on the latest JAX mainline.
  - `run_h2_jax_singlet_structural_stabilizer_audit()`: Run the narrow singlet baseline-vs-structural-stabilizer audit on the latest JAX mainline.
  - `_print_route()`: Function without docstring.
  - `print_h2_jax_singlet_mainline_summary()`: Print the compact H2 singlet Hartree-tail mitigation audit summary.
  - `print_h2_jax_singlet_acceptance_summary()`: Print the compact H2 singlet acceptance summary.
  - `print_h2_jax_singlet_hartree_tail_guard_summary()`: Print the compact experimental Hartree-tail guard summary.
  - `print_h2_jax_singlet_hartree_tail_guard_v2_summary()`: Print the compact experimental Hartree-tail guard 2.0 summary.
  - `print_h2_jax_singlet_structural_stabilizer_summary()`: Print the compact experimental singlet structural stabilizer summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_jax_triplet_end_to_end_micro_profile_audit.py`

- File role: Very small H2 triplet end-to-end micro-profile audit on the JAX mainline.
- Classes:
  - `H2JaxTripletMicroProfileParameterSummary`: Frozen parameter summary for the triplet micro-profile route.
  - `H2JaxTripletMicroProfileStep`: Per-step timing/profile record for one triplet mainline step.
  - `H2JaxTripletEigensolverInternalBucketSummary`: Aggregate in-loop eigensolver bucket summary.
  - `H2JaxTripletEndToEndMicroProfileResult`: Compact triplet end-to-end micro-profile result.
- Functions:
  - `_resolve_solver_backend()`: Function without docstring.
  - `_build_behavior_verdict()`: Function without docstring.
  - `_build_parameter_summary()`: Function without docstring.
  - `_build_step_profiles()`: Function without docstring.
  - `_dominant_timing_bucket()`: Function without docstring.
  - `_lowest_eigenvalue()`: Function without docstring.
  - `_build_eigensolver_internal_bucket_summary()`: Function without docstring.
  - `_build_result()`: Function without docstring.
  - `run_h2_jax_triplet_end_to_end_micro_profile_audit()`: Run one triplet end-to-end micro-profile on the frozen JAX mainline.
  - `print_h2_jax_triplet_end_to_end_micro_profile_summary()`: Print a compact per-step summary for the triplet micro-profile audit.

### `src/isogrid/audit/h2_jax_triplet_hartree_energy_audit.py`

- File role: Very rough triplet-only SCF audit for the JAX Hartree line-apply optimization. This audit keeps the A-grid H2 dry-run restricted to the current local-only Hamiltonian T + V_loc,ion + V_H + V_xc with the repaired monitor-grid Hartree path, patch-assisted local ionic slice, and the kinetic trial-fix branch. It compares two otherwise identical JAX-backed SCF routes on the already-converged H2 triplet case: - jax-hartree-cgloop: cached JAX Hartree operator with `cg_impl="jax_loop"` and no preconditioner - jax-hartree-line: the same JAX-native CG inner loop plus the metric-aware line preconditioner using the baseline apply implementation - jax-hartree-line-optimized: the same line preconditioner mathematics with a more compact apply implementation The goal is not a formal benchmark. It is a small, auditable profile that answers whether optimizing line-preconditioner apply can turn the existing iteration-count win into an end-to-end wall-time win for the H2 triplet SCF dry-run.
- Classes:
  - `H2TripletHartreeEnergyTimingBreakdown`: Very rough triplet SCF timing buckets for one route.
  - `H2TripletHartreeEnergyRouteResult`: Compact triplet SCF profiling summary for one JAX Hartree route.
  - `H2TripletHartreeSingleSolveResult`: Same-density single-solve comparison for one JAX CG implementation.
  - `H2TripletHartreeEnergyAuditResult`: Top-level triplet-only SCF audit for the JAX Hartree line routes.
- Functions:
  - `_average()`: Function without docstring.
  - `_monitor_parameter_summary()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `_run_route()`: Function without docstring.
  - `_run_single_solve()`: Function without docstring.
  - `run_h2_jax_triplet_hartree_energy_audit()`: Run the triplet-only SCF profiling audit for the JAX Hartree line routes.
  - `_print_route()`: Function without docstring.
  - `_print_single_solve()`: Function without docstring.
  - `print_h2_jax_triplet_hartree_energy_summary()`: Print the compact triplet profiling summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_jax_triplet_reintegration_smoke_audit.py`

- File role: Very small H2 triplet smoke audit for the JAX-native eigensolver reintegration.
- Classes:
  - `H2JaxTripletReintegrationSmokeParameterSummary`: Frozen parameter summary for the triplet reintegration smoke route.
  - `H2JaxTripletReintegrationSmokeRouteResult`: Compact triplet reintegration smoke summary for one mainline route.
- Functions:
  - `_resolve_solver_backend()`: Function without docstring.
  - `_tail_energy_changes()`: Function without docstring.
  - `_build_behavior_summary()`: Function without docstring.
  - `_lowest_eigenvalue()`: Function without docstring.
  - `_build_parameter_summary()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `run_h2_jax_triplet_reintegration_smoke_audit()`: Run one very small triplet dry-run to verify JAX eigensolver reintegration.
  - `print_h2_jax_triplet_reintegration_smoke_summary()`: Print a compact summary for the triplet reintegration smoke result.

### `src/isogrid/audit/h2_monitor_grid_diis_scf_audit.py`

- File role: Small H2 A-grid SCF audit for linear mixing versus a minimal DIIS prototype. This audit intentionally stays inside the current repaired A-grid dry-run path: - H2 only - A-grid + patch + kinetic trial-fix only - local static chain only: T + V_loc,ion + V_H + V_xc - no nonlocal migration - no legacy changes The goal is narrow and explicit: check whether a small Pulay/DIIS prototype is enough to move the current singlet path from "more stable but not converged" into actual convergence, while confirming that triplet remains healthy.
- Classes:
  - `H2DiisScfParameterSummary`: Fixed parameter summary for one A-grid DIIS audit scheme.
  - `H2DiisScfRouteResult`: Compact result for one spin state and one mixer scheme.
  - `H2DiisScfSpinAuditResult`: Three-scheme DIIS audit summary for one spin state.
  - `H2MonitorGridDiisScfAuditResult`: Top-level H2 A-grid DIIS SCF audit result.
- Functions:
  - `_parameter_summary()`: Function without docstring.
  - `_classify_trajectory()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `_run_monitor_spin_scheme()`: Function without docstring.
  - `_run_spin_audit()`: Function without docstring.
  - `run_h2_monitor_grid_diis_scf_audit()`: Run the small A-grid DIIS SCF audit on singlet and triplet.
  - `_print_route()`: Function without docstring.
  - `print_h2_monitor_grid_diis_scf_summary()`: Print the small H2 A-grid DIIS SCF audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_fair_calibration_audit.py`

- File role: H2 singlet fair-calibration audit for the A-grid `T_s + E_loc,ion` reconnect.
- Classes:
  - `H2MonitorFairCalibrationParameters`: One A-grid calibration point for the H2 singlet fairness scan.
  - `H2MonitorFairCalibrationPoint`: Resolved H2 singlet `T_s + E_loc,ion` data for one A-grid calibration point.
  - `H2MonitorFairCalibrationAuditResult`: Top-level H2 singlet fairness audit for the A-grid calibration scan.
- Functions:
  - `_default_scan_parameters()`: Function without docstring.
  - `_scaled_element_parameters()`: Function without docstring.
  - `_build_scan_point()`: Function without docstring.
  - `_pick_best_fair_point()`: Function without docstring.
  - `run_h2_monitor_grid_fair_calibration_audit()`: Run the H2 singlet A-grid fairness calibration scan.
  - `_print_geometry_fairness_header()`: Function without docstring.
  - `_print_original_default()`: Function without docstring.
  - `_print_scan_point()`: Function without docstring.
  - `print_h2_monitor_grid_fair_calibration_summary()`: Print the compact fairness calibration summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_fixed_potential_eigensolver_audit.py`

- File role: H2 fixed-potential eigensolver audit on legacy and A-grid+patch routes. This audit migrates only the static local chain T + V_loc,ion + V_H + V_xc to the current A-grid+patch development baseline. Nonlocal ionic action and SCF updates are intentionally left on their current paths.
- Classes:
  - `H2FixedPotentialCenterlineSample`: One center-line orbital sample for the fixed-potential audit.
  - `H2FixedPotentialRouteResult`: Resolved fixed-potential eigensolver audit result for one route.
  - `H2MonitorGridFixedPotentialEigensolverAuditResult`: Top-level H2 fixed-potential audit on legacy and A-grid+patch routes.
- Functions:
  - `_grid_parameter_summary()`: Function without docstring.
  - `_default_patch_parameters()`: Function without docstring.
  - `_build_frozen_spin_densities()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `_evaluate_route()`: Function without docstring.
  - `run_h2_monitor_grid_fixed_potential_eigensolver_audit()`: Run the H2 fixed-potential eigensolver audit on legacy and A-grid+patch.
  - `_print_route_result()`: Function without docstring.
  - `print_h2_monitor_grid_fixed_potential_eigensolver_summary()`: Print the compact H2 fixed-potential eigensolver audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_geometry_consistency_audit.py`

- File role: Geometry-consistency audit for the H2 monitor-grid kinetic path. This audit does not modify the monitor geometry or the kinetic implementation. It checks whether the stored monitor-grid geometry quantities are internally consistent and whether the A-grid kinetic energy identity <psi, T psi> vs 1/2 \int g^{ab} (d_a psi) (d_b psi) J dxi closes on the same geometry for the frozen H2 trial orbital, the current bad fixed-potential eigensolver orbital, and a pair of smooth probe fields.
- Classes:
  - `GeometryMismatchSummary`: One global geometry mismatch summary field.
  - `GeometryRegionSummary`: Regionwise geometry summary on the monitor grid.
  - `GeometryConsistencySummary`: Top-level stored/reconstructed geometry consistency summary.
  - `KineticIdentityCenterlineSample`: Center-line sample for operator/reference kinetic-density comparison.
  - `KineticIdentityRegionSummary`: Regionwise operator-vs-gradient kinetic-energy diagnostics.
  - `KineticIdentityFieldResult`: Kinetic-energy identity diagnostics for one field on the A-grid.
  - `H2MonitorGridGeometryConsistencyAuditResult`: Top-level H2 monitor-grid geometry-consistency audit result.
- Functions:
  - `_default_patch_parameters()`: Function without docstring.
  - `_logical_spacing()`: Function without docstring.
  - `_relative_mismatch()`: Function without docstring.
  - `_reconstruct_geometry_from_coordinates()`: Function without docstring.
  - `_geometry_region_summaries()`: Function without docstring.
  - `_geometry_summary()`: Function without docstring.
  - `_gradient_reference_density()`: Function without docstring.
  - `_kinetic_identity_centerline_samples()`: Function without docstring.
  - `_kinetic_identity_region_summaries()`: Function without docstring.
  - `_evaluate_field()`: Function without docstring.
  - `_evaluate_bad_eigen_orbital()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_geometry_consistency_audit()`: Run the H2 monitor-grid geometry-consistency audit.
  - `_print_field_result()`: Function without docstring.
  - `print_h2_monitor_grid_geometry_consistency_audit_summary()`: Print the compact H2 monitor-grid geometry-consistency audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_k2_subspace_audit.py`

- File role: Very small H2 k=2 subspace audit on legacy and A-grid+patch+trial-fix routes. This audit does not change the eigensolver, SCF, or any operator kernel. It only asks whether the current near-degenerate A-grid k=2 subspace can be reorganized into a more interpretable basis by a tiny in-subspace analysis.
- Classes:
  - `H2K2SubspaceMatrixSummary`: Small 2x2 subspace matrix summary.
  - `H2K2SubspaceRotationSummary`: Chosen very small in-subspace rotation summary.
  - `H2K2SubspaceRouteResult`: k=2 subspace audit result for one route.
  - `H2MonitorGridK2SubspaceAuditResult`: Top-level H2 k=2 subspace audit.
- Functions:
  - `_matrix_tuple()`: Function without docstring.
  - `_subspace_matrix()`: Function without docstring.
  - `_bonding_rotation_summary()`: Function without docstring.
  - `_route_result()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_k2_subspace_audit()`: Run the very small H2 k=2 subspace audit.
  - `_print_orbital()`: Function without docstring.
  - `_print_route()`: Function without docstring.
  - `print_h2_monitor_grid_k2_subspace_audit_summary()`: Function without docstring.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_kinetic_form_audit.py`

- File role: Production-vs-reference audit for the A-grid kinetic operator form. This audit does not modify the production kinetic implementation. It compares the current monitor-grid kinetic operator against a more direct reference discretization on the same monitor-grid geometry and the same H2 singlet frozen density / fixed-potential orbitals.
- Classes:
  - `KineticFormCenterlineSample`: Center-line sample for production/reference kinetic comparison.
  - `KineticFormRegionSummary`: Regionwise summary for one kinetic field or difference field.
  - `KineticFormComparisonResult`: Production-vs-reference kinetic comparison for one field/orbital.
  - `KineticFormSelfAdjointnessComparison`: Self-adjointness comparison for production and reference kinetic paths.
  - `H2MonitorGridKineticFormAuditResult`: Top-level A-grid kinetic production-vs-reference audit result.
- Functions:
  - `apply_monitor_grid_reference_kinetic_operator()`: Apply a more direct reference kinetic discretization on the A-grid. Production path: T_prod psi = -1/2 * (1/J) * d_a [ J g^{ab} d_b psi ] Reference path used here: T_ref psi = -1/2 * [ g^{ab} d_ab psi + b^b d_b psi ] b^b = (1/J) d_a (J g^{ab}) The reference form expands the divergence explicitly, uses symmetrized mixed second derivatives, and therefore avoids the production flux-divergence assembly as a direct audit-side comparison.
  - `_region_summaries()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `_kinetic_quotient()`: Function without docstring.
  - `_build_comparison()`: Function without docstring.
  - `_self_adjoint_probe()`: Function without docstring.
  - `_evaluate_bad_eigen_orbital()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_kinetic_form_audit()`: Run the H2 A-grid production-vs-reference kinetic form audit.
  - `_print_comparison()`: Function without docstring.
  - `print_h2_monitor_grid_kinetic_form_audit_summary()`: Print the compact H2 kinetic-form audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_kinetic_green_identity_audit.py`

- File role: Discrete Green-identity audit for the H2 monitor-grid kinetic path. This audit checks a trial-fix branch for the monitor-grid kinetic path. The production path is left intact; the audit only swaps the A-grid boundary/ghost handling to a centered zero-ghost closure and then rechecks the logical-cube Green identity T psi = -1/2 * (1/J) d_a [ F^a ] F^a = J g^{ab} d_b psi For a real field psi, the continuous identity is K_op = <psi, T psi> K_grad = 1/2 \int (d_a psi) F^a d\xi K_bdry = -1/2 \oint psi F^a n_a dS_\xi K_op = K_grad + K_bdry The audit computes the operator energy, a gradient-form reference energy, and an explicit boundary-flux proxy on the same A-grid geometry in order to check whether the current bad eigensolver orbital violates that identity mainly through a boundary-term mismatch.
- Classes:
  - `BoundaryFaceContribution`: One face contribution to the Green-identity boundary term.
  - `GreenIdentityCenterlineSample`: Center-line sample for operator/gradient/boundary identity diagnostics.
  - `GreenIdentityRegionSummary`: Regionwise Green-identity diagnostics for one field.
  - `GreenIdentityFieldResult`: Discrete Green-identity diagnostics for one field.
  - `H2MonitorGridKineticGreenIdentityAuditResult`: Top-level H2 discrete Green-identity audit result.
- Functions:
  - `_default_patch_parameters()`: Function without docstring.
  - `_logical_spacing()`: Function without docstring.
  - `_flux_components()`: Function without docstring.
  - `_gradient_reference_density()`: Function without docstring.
  - `_boundary_term_proxy()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `_region_summaries()`: Function without docstring.
  - `_evaluate_field()`: Function without docstring.
  - `_evaluate_bad_eigen_orbital()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_kinetic_green_identity_audit()`: Run the H2 monitor-grid discrete Green-identity audit.
  - `_print_field_result()`: Function without docstring.
  - `print_h2_monitor_grid_kinetic_green_identity_audit_summary()`: Print the compact H2 monitor-grid Green-identity audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_kinetic_operator_audit.py`

- File role: Kinetic-operator audit for the H2 frozen-density A-grid failure mode. This audit isolates only the kinetic sub-operator T = -1/2 nabla^2 on the legacy grid, the raw A-grid, and the A-grid+patch fixed-potential path. Patch does not directly modify kinetic; it only changes which orbital the fixed-potential eigensolver selects on the frozen static-local operator.
- Classes:
  - `KineticCenterlineSample`: One center-line kinetic sample for one orbital.
  - `KineticRegionSummary`: Regionwise kinetic diagnostics for one orbital.
  - `KineticOrbitalSummary`: Kinetic diagnostics for one orbital on one route.
  - `KineticRouteAuditResult`: Kinetic audit result for one route and one monitor shape label.
  - `SmoothFieldKineticResult`: Kinetic audit result for one simple smooth test field.
  - `H2MonitorGridKineticOperatorAuditResult`: Top-level H2 kinetic-operator audit result.
- Functions:
  - `_build_monitor_grid()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `_kinetic_region_summaries()`: Function without docstring.
  - `_summarize_kinetic_orbital()`: Function without docstring.
  - `_self_adjoint_kinetic_probe()`: Function without docstring.
  - `_build_smooth_field()`: Function without docstring.
  - `_evaluate_smooth_field()`: Function without docstring.
  - `_build_route_geometry()`: Function without docstring.
  - `_evaluate_route()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_kinetic_operator_audit()`: Run the H2 kinetic-operator audit on legacy and A-grid routes.
  - `_print_orbital_summary()`: Function without docstring.
  - `_print_route_result()`: Function without docstring.
  - `print_h2_monitor_grid_kinetic_operator_audit_summary()`: Print the compact H2 kinetic-operator audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_operator_audit.py`

- File role: Operator-level audit for the H2 frozen-density static-local Hamiltonian. This audit does not modify the eigensolver or SCF. It isolates the current fixed-potential static-local operator H_local = T + V_loc,ion + V_H + V_xc on the legacy grid, the raw A-grid, and the A-grid plus frozen patch embedding. The goal is to diagnose why the A-grid path can look favorable at the static local-chain energy level but still fail badly once the fixed-potential eigensolver acts on it.
- Classes:
  - `ScalarFieldSummary`: Compact scalar-field summary for one 3D field.
  - `WeightedExpectationSummary`: Weighted Rayleigh quotient and component expectations for one orbital.
  - `ResidualRegionSummary`: Regionwise residual diagnostics for one operator residual field.
  - `SelfAdjointnessProbe`: Weighted self-adjointness probe for one operator or sub-operator.
  - `OperatorCenterlineSample`: One center-line sample for orbital or residual diagnostics.
  - `H2StaticLocalOperatorRouteResult`: Operator-level audit result for one grid/path type.
  - `H2MonitorGridOperatorAuditResult`: Top-level operator-level audit for legacy, A-grid, and A-grid+patch.
- Functions:
  - `_default_patch_parameters()`: Function without docstring.
  - `_grid_parameter_summary()`: Function without docstring.
  - `_build_frozen_density()`: Function without docstring.
  - `_weighted_inner_product()`: Function without docstring.
  - `_field_summary()`: Function without docstring.
  - `_compute_region_masks()`: Function without docstring.
  - `_residual_region_summaries()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `_component_actions()`: Function without docstring.
  - `_expectation_summary()`: Function without docstring.
  - `_build_probe_field()`: Function without docstring.
  - `_self_adjoint_probe()`: Function without docstring.
  - `_prepare_operator_context()`: Function without docstring.
  - `_evaluate_route()`: Function without docstring.
  - `_grid_parameter_summary()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_operator_audit()`: Run the H2 static-local operator-level audit on legacy and A-grid routes.
  - `_print_expectation()`: Function without docstring.
  - `_print_route()`: Function without docstring.
  - `print_h2_monitor_grid_operator_audit_summary()`: Print the compact H2 static-local operator audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_orbital_shape_audit.py`

- File role: Very small H2 fixed-potential orbital-shape audit on legacy and A-grid paths. This audit does not modify the eigensolver or SCF. It inspects the spatial shape of the already-converged fixed-potential orbitals on T + V_loc,ion + V_H + V_xc for the legacy grid and the current A-grid+patch+kinetic-trial-fix route. The goal is to decide whether the fixed-potential A-grid orbitals are already physically healthy enough to justify an eventual A-grid SCF dry-run.
- Classes:
  - `OrbitalCenterlineSample`: One center-line sample for orbital-shape diagnostics.
  - `OrbitalSymmetrySummary`: Parity and inversion diagnostics for one orbital.
  - `OrbitalNodeSummary`: Center-line node summary for one orbital.
  - `OrbitalBoundarySummary`: Regionwise norm fractions for one orbital.
  - `H2OrbitalShapeResult`: Shape audit result for one orbital from one solve.
  - `H2OrbitalShapeRouteResult`: Fixed-potential orbital-shape audit result for one route.
  - `H2MonitorGridOrbitalShapeAuditResult`: Top-level H2 orbital-shape audit on legacy and A-grid+patch+trial-fix.
- Functions:
  - `_default_patch_parameters()`: Function without docstring.
  - `_grid_parameter_summary()`: Function without docstring.
  - `_build_frozen_spin_densities()`: Function without docstring.
  - `_weighted_norm()`: Function without docstring.
  - `_normalize_orbital()`: Function without docstring.
  - `_centerline_values()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `_weighted_mismatch_ratio()`: Function without docstring.
  - `_symmetry_summary()`: Function without docstring.
  - `_node_summary()`: Function without docstring.
  - `_boundary_summary()`: Function without docstring.
  - `_orbital_result()`: Function without docstring.
  - `_evaluate_route()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_orbital_shape_audit()`: Run the very small H2 orbital-shape audit on legacy and A-grid+patch+trial-fix.
  - `_print_orbital_result()`: Function without docstring.
  - `_print_route()`: Function without docstring.
  - `print_h2_monitor_grid_orbital_shape_audit_summary()`: Print the compact H2 orbital-shape audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_patch_hartree_xc_audit.py`

- File role: H2 singlet static local-chain audit for legacy, A-grid, and A-grid+patch. This audit keeps the same frozen H2 singlet trial orbital and density on all three routes and compares only the static local chain T_s + E_loc,ion + E_H + E_xc The monitor-grid patch still corrects only the near-core local-GTH energy. It does not alter the trial density, Hartree solve, or LSDA evaluation.
- Classes:
  - `H2StaticLocalRouteResult`: Resolved H2 singlet static-local audit result for one route.
  - `H2MonitorPatchHartreeXCAuditResult`: Top-level H2 singlet static local-chain audit result.
- Functions:
  - `_find_h2_singlet()`: Function without docstring.
  - `_grid_parameter_summary()`: Function without docstring.
  - `_default_patch_recheck()`: Function without docstring.
  - `_build_singlet_spin_densities()`: Function without docstring.
  - `_resolve_hartree_and_xc()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `_pick_best_patch_result()`: Function without docstring.
  - `run_h2_monitor_grid_patch_hartree_xc_audit()`: Audit `T_s + E_loc,ion + E_H + E_xc` for legacy, A-grid, and A-grid+patch.
  - `_print_route_result()`: Function without docstring.
  - `print_h2_monitor_grid_patch_hartree_xc_summary()`: Print the compact H2 singlet static local-chain audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_patch_local_audit.py`

- File role: H2 singlet audit for patch-assisted local-GTH correction on the A-grid.
- Classes:
  - `H2MonitorPatchParameterSummary`: Patch-parameter summary for one local-GTH patch scan point.
  - `H2TsElocPatchRouteResult`: Resolved H2 singlet `T_s + E_loc,ion` result for one route.
  - `H2MonitorPatchLocalAuditResult`: Top-level H2 singlet patch-local audit result.
- Functions:
  - `_find_h2_singlet()`: Function without docstring.
  - `_default_patch_scan()`: Function without docstring.
  - `_grid_parameter_summary()`: Function without docstring.
  - `_patch_summary()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `_build_patch_route_result()`: Function without docstring.
  - `_pick_best_patch_result()`: Function without docstring.
  - `run_h2_monitor_grid_patch_local_audit()`: Audit the patch-assisted local-GTH correction on the H2 singlet A-grid.
  - `_print_route_result()`: Function without docstring.
  - `print_h2_monitor_grid_patch_local_summary()`: Print the compact H2 singlet A-grid local-patch audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_poisson_operator_audit.py`

- File role: A-grid Poisson-operator audit for the H2 singlet frozen density.
- Classes:
  - `ScalarFieldSummary`: Compact scalar-field statistics on one selected mask.
  - `RegionDiagnostic`: Regional statistics for v_H, L(v_H), and the Poisson residual.
  - `CenterLineSample`: One center-line sample on the H2 molecular axis.
  - `BoundaryConditionSummary`: Compact open-boundary multipole data.
  - `SelfAdjointnessProbe`: Lightweight numerical clue for discrete self-adjointness.
  - `BoundarySplitDiagnostic`: Diagnostic for consistency between full and split Poisson residuals.
  - `PoissonOperatorRouteResult`: Resolved Poisson-operator audit result on one grid path.
  - `PoissonOperatorDifferenceSummary`: Comparison summary between legacy and A-grid routes.
  - `MonitorShapeScanPoint`: Very small A-grid shape scan point for operator-diagnostic trend checks.
  - `H2MonitorGridPoissonOperatorAuditResult`: Top-level A-grid Poisson-operator audit result.
- Functions:
  - `_boundary_mask()`: Function without docstring.
  - `_interior_mask()`: Function without docstring.
  - `_grid_parameter_summary()`: Function without docstring.
  - `_build_h2_frozen_density()`: Function without docstring.
  - `_build_gaussian_density()`: Function without docstring.
  - `_sample_at_point()`: Function without docstring.
  - `_scalar_summary()`: Function without docstring.
  - `_boundary_summary()`: Function without docstring.
  - `_region_masks()`: Function without docstring.
  - `_region_diagnostics()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `_self_adjointness_probe()`: Function without docstring.
  - `_boundary_split_diagnostic()`: Function without docstring.
  - `evaluate_poisson_operator_route()`: Evaluate one legacy or A-grid Poisson-operator route for a fixed density.
  - `_difference_summary()`: Function without docstring.
  - `_shape_scan()`: Function without docstring.
  - `_diagnosis()`: Function without docstring.
  - `run_h2_monitor_grid_poisson_operator_audit()`: Run the H2 singlet frozen-density A-grid Poisson-operator audit.
  - `_print_route_result()`: Function without docstring.
  - `print_h2_monitor_grid_poisson_operator_summary()`: Print the compact A-grid Poisson-operator audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_scf_dry_run_audit.py`

- File role: H2 minimal SCF dry-run audit on the repaired A-grid static-local path. This audit intentionally keeps the A-grid Hamiltonian limited to T + V_loc,ion + V_H + V_xc with the current patch-assisted local ionic path, repaired monitor-grid Hartree/Poisson, and the kinetic trial-fix branch. Nonlocal ionic action is still excluded on the A-grid side, so this is a dry-run of the new main-grid line rather than a final production SCF benchmark.
- Classes:
  - `H2ScfDryRunRouteResult`: Compact SCF dry-run summary for one route and one spin state.
  - `H2MonitorGridScfDryRunAuditResult`: Top-level H2 SCF dry-run audit result for legacy and A-grid routes.
- Functions:
  - `_legacy_parameter_summary()`: Function without docstring.
  - `_monitor_parameter_summary()`: Function without docstring.
  - `_lowest_eigenvalue()`: Function without docstring.
  - `_build_route_result_from_legacy()`: Function without docstring.
  - `_build_route_result_from_monitor()`: Function without docstring.
  - `run_h2_monitor_grid_scf_dry_run_audit()`: Run the H2 minimal SCF dry-run audit on legacy and A-grid routes.
  - `_print_route()`: Function without docstring.
  - `print_h2_monitor_grid_scf_dry_run_summary()`: Print the compact H2 SCF dry-run summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_singlet_stability_audit.py`

- File role: Very small singlet-only SCF stability audit on the repaired A-grid path. This audit intentionally stays narrow: - H2 singlet only - current A-grid+patch+kinetic-trial-fix baseline only - no nonlocal migration - only one minimal DIIS prototype beyond linear mixing The only comparison here is whether a more conservative linear mixing value, plus one explicit minimal DIIS branch, changes the previously observed weak singlet two-cycle behavior.
- Classes:
  - `H2SingletStabilityParameterSummary`: Fixed parameter summary for one singlet stability scheme.
  - `H2TwoCycleDiagnostics`: Lightweight tail diagnostics for the singlet oscillation pattern.
  - `H2SingletStabilityRouteResult`: Compact result for one singlet stability scheme.
  - `H2SingletStabilityAuditResult`: Top-level H2 singlet stability audit result.
- Functions:
  - `_parameter_summary()`: Function without docstring.
  - `_safe_mean()`: Function without docstring.
  - `_safe_std()`: Function without docstring.
  - `_build_two_cycle_diagnostics()`: Function without docstring.
  - `_build_route_result()`: Function without docstring.
  - `_run_monitor_singlet_scheme()`: Function without docstring.
  - `run_h2_monitor_grid_singlet_stability_audit()`: Run a very small singlet-only stability audit on the repaired A-grid path.
  - `print_h2_monitor_grid_singlet_stability_summary()`: Print the compact singlet stability audit summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_monitor_grid_ts_eloc_audit.py`

- File role: H2 singlet audit for the first A-grid `T_s + E_loc,ion` reconnect.
- Classes:
  - `H2GridEnergyGeometrySummary`: Compact geometry summary for one H2 `T_s + E_loc,ion` audit point.
  - `H2TsElocGridResult`: Resolved `T_s + E_loc,ion` audit result on one grid family.
  - `H2MonitorGridTsElocAuditResult`: Top-level H2 singlet comparison between legacy and A-grid.
- Functions:
  - `_find_h2_singlet()`: Function without docstring.
  - `_geometry_center()`: Function without docstring.
  - `_build_h2_bonding_trial_orbital()`: Function without docstring.
  - `_box_half_extents_bohr()`: Function without docstring.
  - `_spacing_measure_on_legacy_grid()`: Function without docstring.
  - `_near_far_spacing_summary()`: Function without docstring.
  - `_near_core_min_spacing()`: Function without docstring.
  - `_jacobian_range()`: Function without docstring.
  - `_min_spacing_estimate()`: Function without docstring.
  - `_center_line_local_samples()`: Function without docstring.
  - `_geometry_summary()`: Function without docstring.
  - `evaluate_h2_singlet_ts_eloc_on_legacy_grid()`: Evaluate `T_s + E_loc,ion` on the legacy H2 grid.
  - `evaluate_h2_singlet_ts_eloc_on_monitor_grid()`: Evaluate `T_s + E_loc,ion` on the new A-grid.
  - `_build_result()`: Function without docstring.
  - `run_h2_monitor_grid_ts_eloc_audit()`: Compare legacy vs A-grid `T_s + E_loc,ion` for H2 singlet.
  - `_print_grid_result()`: Function without docstring.
  - `print_h2_monitor_grid_ts_eloc_summary()`: Print the compact H2 singlet legacy-vs-A-grid `T_s + E_loc,ion` summary.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_scf_single_point_audit.py`

- File role: Audit script for the first minimal H2 SCF single-point closed loop.
- Functions:
  - `_density_summary()`: Function without docstring.
  - `_lowest_eigenvalue()`: Function without docstring.
  - `_print_result()`: Function without docstring.
  - `main()`: Function without docstring.

### `src/isogrid/audit/h2_vs_pyscf_audit.py`

- File role: Quantitative H2 error audit against the current PySCF reference baseline. This module compares the first minimal IsoGridDFT H2 SCF single-point loop to the current PySCF audit reference under the same nominal physical model: - H2 at R = 1.4 Bohr - gth-pade pseudopotential - lda,vwn LSDA - UKS reference path for both singlet and triplet The goal here is not to claim final acceptance. It is to quantify the present gap to the PySCF baseline and to provide one small parameter scan that helps separate solver-side error from grid/discretization error.
- Classes:
  - `AuditParameterSummary`: Compact summary of the key isogrid-side audit parameters.
  - `H2SpinComparisonResult`: Quantitative comparison for one H2 candidate spin state.
  - `H2GapComparison`: Singlet-triplet gap comparison against the PySCF baseline.
  - `H2ParameterScanResult`: One small audit point used to localize present error sources.
  - `H2VsPySCFAuditResult`: Top-level H2 audit result for both spin candidates and the small scan.
- Functions:
  - `_find_reference_result()`: Function without docstring.
  - `_build_parameter_summary()`: Function without docstring.
  - `_build_scan_grid_geometry()`: Function without docstring.
  - `run_spin_state_comparison()`: Run one H2 isogrid vs PySCF comparison for a fixed spin state.
  - `_build_gap_comparison()`: Function without docstring.
  - `run_minimal_parameter_scan()`: Run one small singlet-focused scan to localize dominant error sources.
  - `run_h2_vs_pyscf_audit()`: Run the quantitative H2 audit against the current PySCF baseline.
  - `_format_target_gap()`: Function without docstring.
  - `_print_spin_comparison()`: Function without docstring.
  - `print_h2_vs_pyscf_summary()`: Print the compact quantitative H2 vs PySCF audit summary.
  - `main()`: Run the H2 isogrid-vs-PySCF quantitative audit.

### `src/isogrid/audit/local_hamiltonian_h2_trial_audit.py`

- File role: Audit script for the first local-Hamiltonian H2 trial-orbital slice.
- Functions:
  - `build_symmetric_h2_trial_orbital()`: Build a symmetric two-center Gaussian trial orbital for the default H2 case.
  - `_summary()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `main()`: Function without docstring.

### `src/isogrid/audit/monitor_grid_audit.py`

- File role: Audit entrypoint for the new 3D monitor-driven main grid core.
- Classes:
  - `MonitorGridAuditResult`: Compact audit summary for one generated monitor grid.
- Functions:
  - `_summarize_geometry()`: Function without docstring.
  - `run_monitor_grid_audit()`: Generate the new monitor grid for the four audit molecules.
  - `print_monitor_grid_audit()`: Print the compact monitor-grid audit report.
  - `main()`: Function without docstring.

### `src/isogrid/audit/pyscf_h2_basis_convergence.py`

- File role: Basis-sequence audit for the default H2 PySCF reference case.
- Classes:
  - `BasisScanEntry`: Reference results for one basis in the scan sequence.
- Functions:
  - `run_basis_convergence_audit()`: Run the default PySCF basis scan for one benchmark case.
  - `print_basis_convergence_summary()`: Print a compact basis-scan table.
  - `main()`: Run and print the default H2 basis-sequence audit.

### `src/isogrid/audit/pyscf_h2_reference.py`

- File role: PySCF audit baseline for the default H2 benchmark case.
- Classes:
  - `ReferenceResult`: Single-spin-state PySCF audit result.
- Functions:
  - `_load_pyscf()`: Function without docstring.
  - `build_pyscf_molecule()`: Build the PySCF molecule object for one benchmark spin state.
  - `build_mean_field()`: Build the PySCF mean-field object for the configured audit model.
  - `run_reference_spin_state()`: Run the configured PySCF DFT reference for one candidate spin state.
  - `run_reference_case()`: Evaluate all candidate spin states for one configured benchmark case.
  - `select_lower_energy_state()`: Return the lower-energy candidate spin state.
  - `print_reference_summary()`: Print a compact summary for one benchmark reference calculation.
  - `main()`: Run the default H2 benchmark reference calculation.

### `src/isogrid/audit/static_ks_h2_hartree_audit.py`

- File role: Audit script for the first H2 static KS slice with Hartree included.
- Functions:
  - `_summary()`: Function without docstring.
  - `_density_summary()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `main()`: Function without docstring.

### `src/isogrid/audit/static_ks_h2_trial_audit.py`

- File role: Audit script for the first static KS H2 trial-orbital slice.
- Functions:
  - `_summary()`: Function without docstring.
  - `_density_summary()`: Function without docstring.
  - `_centerline_samples()`: Function without docstring.
  - `main()`: Function without docstring.

### `src/isogrid/config/__init__.py`

- File role: Configuration helpers for benchmark defaults and runtime setup.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/config/defaults.py`

- File role: Default benchmark and reference configurations for stage-1 auditing.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/config/model.py`

- File role: Lightweight configuration models for benchmark and audit settings.
- Classes:
  - `AtomSpec`: Single atom with element symbol and Cartesian position.
  - `MoleculeGeometry`: Molecular geometry for an isolated benchmark case.
  - `SpinStateSpec`: Candidate spin state using the PySCF spin convention.
  - `ReferenceModelSettings`: Reference-model controls shared by the PySCF audit scripts.
  - `ScfSettings`: Minimal SCF controls for reference-side audit calculations.
  - `BenchmarkCase`: Named benchmark case used by audit and later solver development.

### `src/isogrid/config/runtime.py`

- File role: Backward-compatible runtime imports for the scientific JAX path.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/config/runtime_jax.py`

- File role: Central JAX runtime helpers for the scientific path. This module is the single place where IsoGridDFT configures JAX runtime behavior for the scientific path. The current first-stage defaults are: - float64 enabled by default - JIT enabled by default - platform selection optional and environment-driven The goal here is not to design a full runtime framework yet. It is simply to keep the JAX configuration explicit and out of the numerical kernels.
- Classes:
  - `JaxRuntimeConfiguration`: Resolved JAX runtime settings for the scientific path.
- Functions:
  - `is_jax_available()`: Return whether JAX can be imported in the current environment.
  - `require_jax()`: Import JAX or raise a clear runtime error.
  - `_env_bool()`: Function without docstring.
  - `get_jax_runtime_configuration()`: Resolve the JAX runtime configuration for the scientific path.
  - `configure_jax_runtime()`: Apply the minimum JAX runtime settings used by the scientific path.
  - `get_configured_jax()`: Return JAX after applying the scientific-path runtime settings.
  - `get_configured_jax_numpy()`: Return `jax.numpy` after applying the scientific-path runtime settings.
  - `get_jax_scientific_dtype()`: Return the JAX float64 dtype used by the scientific path.

### `src/isogrid/driver/__init__.py`

- File role: Driver-layer placeholders for IsoGridDFT.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/grid/__init__.py`

- File role: Structured adaptive grid geometry and mapping helpers.
- Functions:
  - `compute_geometry_reference_center()`: Compute the simple geometric center of the benchmark nuclei.
  - `build_default_h2_grid_spec()`: Build the first structured grid baseline for the default H2 benchmark.
  - `build_default_h2_grid_geometry()`: Build the geometry objects for the default H2 structured grid.

### `src/isogrid/grid/geometry.py`

- File role: Geometric quantities derived from a structured grid mapping.
- Classes:
  - `StructuredGridGeometry`: Structured grid coordinates and minimal geometric weights.
- Functions:
  - `compute_cell_edges_1d()`: Build point-centered cell edges from a monotone 1D grid.
  - `compute_cell_widths_1d()`: Compute positive point-centered 1D cell widths.
  - `build_grid_geometry()`: Build structured grid coordinates and minimal geometric weights.

### `src/isogrid/grid/mapping.py`

- File role: Structured logical-to-physical mapping helpers.
- Classes:
  - `AxisMapping`: One-dimensional logical and physical coordinates for one axis.
- Functions:
  - `logical_axis_coordinates()`: Create the canonical logical axis on [-1, 1].
  - `_stretched_fraction()`: Function without docstring.
  - `_stretched_fraction_derivative()`: Function without docstring.
  - `map_logical_to_physical_1d()`: Map a logical 1D axis to physical coordinates. This first version uses a separable, piecewise-sinh stretch. It is a geometry-driven default, not a final adaptive strategy.
  - `mapping_jacobian_1d()`: Return dx/du for one separable axis mapping.
  - `build_axis_mapping()`: Build the logical and physical coordinates for one axis.
  - `build_axis_mappings()`: Build the three separable axis mappings for one grid specification.
  - `build_grid_point_coordinates()`: Combine three 1D physical axes into a structured 3D grid.

### `src/isogrid/grid/model.py`

- File role: Minimal data models for a structured adaptive grid.
- Classes:
  - `AxisStretchSpec`: One-dimensional structured stretch about a reference center.
    - `__post_init__()`: Method without docstring.
    - `physical_bounds()`: Return the physical bounds for this axis.
  - `StructuredGridSpec`: Minimal specification for a structured adaptive 3D grid.
    - `__post_init__()`: Method without docstring.
    - `shape()`: Return the logical grid shape.

### `src/isogrid/grid/monitor_builder.py`

- File role: Builders for the 3D atom-centered monitor grid.
- Functions:
  - `_geometry_center()`: Function without docstring.
  - `build_default_near_core_element_parameters()`: Build first-stage element parameters from GTH-inspired seed metadata.
  - `_default_box_half_extents()`: Function without docstring.
  - `build_monitor_grid_spec_for_case()`: Build the first formal 3D monitor-grid spec for one case.
  - `build_monitor_grid_for_case()`: Build the full 3D monitor grid geometry for one configured case.
  - `build_h2_local_patch_development_element_parameters()`: Return the current H2 A-grid development-point parameters for local-GTH patch work.
  - `build_h2_local_patch_development_monitor_grid()`: Build the current best-fair H2 A-grid baseline used for local-GTH patch audits.
  - `build_default_h2_monitor_grid()`: Function without docstring.
  - `build_default_n2_monitor_grid()`: Function without docstring.
  - `build_default_co_monitor_grid()`: Function without docstring.
  - `build_default_h2o_monitor_grid()`: Function without docstring.

### `src/isogrid/grid/monitor_geometry.py`

- File role: 3D monitor evaluation and harmonic structured-grid generation.
- Functions:
  - `_logical_axis()`: Function without docstring.
  - `build_reference_box_coordinates()`: Build the logical and initial physical coordinates for the monitor grid.
  - `evaluate_global_monitor_field()`: Evaluate the full 3D atom-centered monitor field on current coordinates.
  - `_smooth_monitor_field()`: Function without docstring.
  - `_logical_spacing()`: Function without docstring.
  - `_boundary_mask()`: Function without docstring.
  - `_subcell_midpoint_samples()`: Function without docstring.
  - `_trilinear_sample_nodal_field()`: Function without docstring.
  - `build_monitor_cell_local_quadrature()`: Build the explicit logical-cell quadrature used by monitor-grid moments.
  - `evaluate_monitor_cell_local_sample_weights()`: Evaluate all cell-local quadrature weights on the mapped monitor grid.
  - `evaluate_monitor_cell_local_field_samples()`: Evaluate one nodal field on the monitor cell-local quadrature samples.
  - `_solve_weighted_harmonic_coordinates()`: Solve div(M grad X) = 0 for the three physical coordinates.
  - `_covariant_basis()`: Function without docstring.
  - `_basic_geometry_from_coordinates()`: Function without docstring.
  - `_backtracking_update()`: Function without docstring.
  - `_build_patch_interfaces()`: Function without docstring.
  - `_quality_report()`: Function without docstring.
  - `generate_monitor_grid_geometry()`: Generate a full 3D monitor-driven harmonic structured grid.

### `src/isogrid/grid/monitor_model.py`

- File role: Data models for the next-generation 3D atom-centered monitor grid.
- Classes:
  - `NearCoreElementParameters`: Element-level near-core parameters for the 3D monitor field.
    - `__post_init__()`: Method without docstring.
  - `MonitorPatchInterface`: Minimal interface placeholder for an atom-centered auxiliary fine patch.
  - `AtomicMonitorContribution`: One atom's resolved 3D monitor contribution on the current grid.
  - `GlobalMonitorField`: Global 3D monitor field assembled from all atomic contributions.
  - `MonitorCellLocalQuadrature`: Auditably explicit logical-cell quadrature on the mapped monitor grid.
  - `MonitorGridSpec`: Specification for a full 3D monitor-driven structured grid.
    - `__post_init__()`: Method without docstring.
    - `shape()`: Method without docstring.
  - `MonitorGridQualityReport`: Basic quality indicators for the generated monitor grid.
  - `MonitorGridGeometry`: Generated 3D monitor-driven structured grid geometry.

### `src/isogrid/ks/__init__.py`

- File role: Kohn-Sham Hamiltonian helpers for the structured-grid prototype.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/ks/eigensolver.py`

- File role: First fixed-potential iterative eigensolver scaffold for the static KS backbone. This module solves the lowest few orbitals of the current frozen-potential static KS Hamiltonian without introducing SCF. On the current monitor-grid static-local route, the formal main path uses a JAX-native block subspace iteration. SciPy Lanczos remains available only as an explicit fallback / audit route on older code paths and for cross-validation. The weighted problem is formulated through the similarity-transformed operator A = W^(1/2) H W^(-1/2) where W is the diagonal cell-volume metric on the structured adaptive grid. This keeps the physical orbitals orthonormal under the weighted inner product <phi|psi>_W = sum_r conj(phi[r]) psi[r] w[r] and lets the eigensolver operate on a standard Euclidean symmetric problem. This is a first formal fixed-potential eigensolver skeleton. It is not yet a production SCF eigensolver and does not update the density.
- Classes:
  - `FixedPotentialOperatorContext`: Frozen static-KS potential data used by the fixed-potential solver.
  - `FixedPotentialStaticLocalOperatorContext`: Frozen static-local potential data used by the A-grid eigensolver path. The current operator contains only T + V_loc,ion + V_H + V_xc with the density frozen externally. For monitor-grid patch work, the local ionic potential may include a frozen patch embedding that reproduces the current near-core local-GTH patch energy on the chosen frozen density.
  - `FixedPotentialStaticLocalPreparationProfile`: Very small timing/profile summary for static-local context preparation.
  - `FixedPotentialEigensolverResult`: Audit-facing result of the first fixed-potential eigensolver.
    - `internal_profile()`: Backward-compatible alias for older audit code.
- Functions:
  - `_normalize_spin_channel()`: Function without docstring.
  - `_build_total_density_on_grid()`: Function without docstring.
  - `_normalize_static_local_kinetic_version()`: Function without docstring.
  - `_normalize_fixed_potential_solver_backend()`: Function without docstring.
  - `validate_orbital_block()`: Validate a block of orbitals stored as (k, nx, ny, nz).
  - `flatten_orbital_block()`: Flatten a block of orbitals into column-major (n_grid, k) form.
  - `reshape_orbital_columns()`: Reshape (n_grid, k) orbital columns back to (k, nx, ny, nz).
  - `weighted_overlap_matrix()`: Return the weighted block overlap matrix under the cell-volume metric.
  - `weighted_orbital_norms()`: Return the weighted norms of a block of orbitals.
  - `weighted_orthonormalize_orbitals()`: Weighted-orthonormalize a block by SVD of sqrt(W) * Psi.
  - `_resolve_local_ionic_potential()`: Function without docstring.
  - `_resolve_hartree_potential()`: Function without docstring.
  - `_resolve_xc_potential()`: Function without docstring.
  - `_resolve_static_local_ionic_potential()`: Function without docstring.
  - `prepare_fixed_potential_static_ks_operator()`: Freeze the current static-KS local terms for a fixed-density solve.
  - `apply_fixed_potential_static_ks_operator()`: Apply the frozen static-KS Hamiltonian to one orbital field. This is a thin wrapper over the current static-KS Hamiltonian apply path with the density-derived local terms frozen once in the operator context.
  - `build_fixed_potential_static_ks_operator()`: Return a callable frozen-potential operator wrapper for the eigensolver.
  - `prepare_fixed_potential_static_local_operator_profiled()`: Freeze the static local chain `T + V_loc + V_H + V_xc` on one grid. This operator intentionally excludes nonlocal ionic action and any SCF update. When `use_monitor_patch=True` on the monitor grid, the current near-core local-GTH patch correction is embedded into a frozen local potential field matched to the chosen frozen density.
  - `prepare_fixed_potential_static_local_operator()`: Freeze the static local chain `T + V_loc + V_H + V_xc` on one grid.
  - `apply_fixed_potential_static_local_operator()`: Apply the frozen static local Hamiltonian to one orbital field.
  - `build_fixed_potential_static_local_operator()`: Return a callable frozen static-local operator wrapper.
  - `apply_fixed_potential_static_local_block()`: Apply the frozen static-local Hamiltonian to one orbital block.
  - `apply_fixed_potential_static_local_block_jax_hotpath()`: Apply the frozen static-local Hamiltonian block through the JAX hot path.
  - `apply_fixed_potential_static_ks_block()`: Apply the frozen static-KS Hamiltonian to a block of orbitals.
  - `_build_default_guess_orbitals()`: Build deterministic, symmetry-friendly first-stage guess orbitals.
  - `_build_initial_guess_block()`: Function without docstring.
  - `_residual_norms()`: Return weighted residual norms ||H psi - eps psi||_W for one orbital block.
  - `_require_scipy_iterative_solver()`: Function without docstring.
  - `_build_weighted_euclidean_operator()`: Function without docstring.
  - `_solve_weighted_fixed_potential_problem_scipy_fallback()`: Function without docstring.
  - `solve_fixed_potential_eigenproblem()`: Solve the lowest few frozen-potential static-KS orbitals. The generic structured-grid scaffold still uses SciPy's iterative symmetric Lanczos fallback. The current JAX-native formal main path is implemented on the monitor-grid static-local route below.
  - `solve_fixed_potential_static_local_eigenproblem()`: Solve the lowest few frozen-potential orbitals of the static local chain. This route intentionally contains only T + V_loc,ion + V_H + V_xc with frozen density-derived local terms and no nonlocal ionic action. On the monitor-grid JAX hot path this now defaults to the JAX-native block subspace iteration; SciPy is retained only as an explicit fallback.

### `src/isogrid/ks/eigensolver_jax.py`

- File role: JAX-native fixed-potential eigensolver helpers for the A-grid local-only path.
- Classes:
  - `JaxFixedPotentialSubspaceIterationResult`: Resolved JAX-native fixed-potential eigensolver result.
  - `JaxFixedPotentialInternalProfile`: Very small in-loop timing profile for the JAX-native eigensolver.
- Functions:
  - `solve_fixed_potential_static_local_eigenproblem_jax()`: Solve the local-only fixed-potential problem with a JAX-native block iteration.

### `src/isogrid/ks/eigensolver_jax_cache.py`

- File role: Very small cache/reuse helpers for the JAX eigensolver hot path.
- Functions:
  - `_require_monitor_geometry()`: Function without docstring.
  - `_block_operator_cache_key()`: Function without docstring.
  - `get_fixed_potential_static_local_block_kernel_cached()`: Return one cached compiled block-Hamiltonian callable for the current context.
  - `apply_fixed_potential_static_local_block_cached_jax()`: Apply the cached JAX block-Hamiltonian callable to one orbital block.
  - `_weighted_overlap_cache_key()`: Function without docstring.
  - `get_weighted_overlap_kernel_cached()`: Return one cached compiled weighted-overlap kernel for a fixed weight field.
  - `weighted_overlap_matrix_cached_jax()`: Return the cached weighted overlap/Gram matrix for one fixed weight field.
  - `weighted_orthonormalize_orbitals_cached_jax()`: Weighted-orthonormalize one block while reusing the cached overlap kernel.

### `src/isogrid/ks/hamiltonian_local_jax.py`

- File role: JAX local-only Hamiltonian apply kernels for the stable A-grid hot path.
- Functions:
  - `_require_monitor_static_local_context()`: Function without docstring.
  - `apply_fixed_potential_static_local_operator_jax()`: Apply the frozen local-only Hamiltonian with JAX on the monitor grid.
  - `apply_fixed_potential_static_local_block_jax()`: Apply the frozen local-only Hamiltonian to one orbital block with JAX.
  - `build_fixed_potential_static_local_operator_jax()`: Return a callable JAX local-only matvec for the current monitor-grid context.

### `src/isogrid/ks/local_hamiltonian.py`

- File role: First local-Hamiltonian slice for the structured-grid prototype. The current local Hamiltonian includes only H_local psi = T psi + V_local psi with T given by the first-stage structured-grid kinetic operator and V_local currently restricted to the local ionic GTH pseudopotential plus an optional extra local potential term reserved for future density-derived contributions.
- Classes:
  - `LocalHamiltonianTerms`: Resolved local-Hamiltonian pieces for one orbital field.
- Functions:
  - `_resolve_local_potential_array()`: Function without docstring.
  - `evaluate_local_hamiltonian_terms()`: Resolve T psi, V_local psi, and their sum for one orbital field.
  - `apply_local_hamiltonian()`: Apply the current first-stage local Hamiltonian to one orbital field.
  - `build_default_h2_local_hamiltonian_action()`: Convenience wrapper for the default H2 benchmark configuration.

### `src/isogrid/ks/static_hamiltonian.py`

- File role: First static KS Hamiltonian backbone for the structured-grid prototype. The current static KS slice assembles H_ks_static psi = T psi + V_loc,ion psi + V_nl,ion psi + V_H psi + V_xc psi with the following scope restrictions: - the Hartree term is present through the first-stage open-boundary Poisson slice - no SCF loop yet - rho_up and rho_down are external inputs - only the current PySCF-aligned `lda,vwn` LSDA local term is supported This module is meant to be the first formal static KS backbone with Hartree, not the final production Hamiltonian path.
- Classes:
  - `StaticKSHamiltonianTerms`: Resolved static KS action pieces for one orbital and one spin channel.
- Functions:
  - `_normalize_spin_channel()`: Function without docstring.
  - `_resolve_local_ionic_potential()`: Function without docstring.
  - `_resolve_nonlocal_ionic_action()`: Function without docstring.
  - `_resolve_hartree_potential()`: Function without docstring.
  - `_resolve_xc_potential()`: Function without docstring.
  - `build_orbital_density()`: Build a simple orbital density rho = occupation * |psi|^2.
  - `build_singlet_like_spin_densities()`: Build a minimal closed-shell-like spin density pair from one orbital.
  - `build_total_density()`: Build the total density rho = rho_up + rho_down on the current grid.
  - `evaluate_static_ks_terms()`: Resolve the current static KS action and all of its explicit pieces.
  - `apply_static_ks_hamiltonian()`: Apply the first-stage static KS Hamiltonian with Hartree to one orbital field.
  - `build_default_h2_static_ks_action()`: Convenience wrapper for the default H2 benchmark configuration.

### `src/isogrid/ops/__init__.py`

- File role: Minimal operator helpers for the first local-Hamiltonian slice.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/ops/kinetic.py`

- File role: First-stage kinetic operators for both legacy and monitor-driven grids. Legacy structured grid: For the separable orthogonal mapping x = x(u), y = y(v), z = z(w), with point scale factors h_x = dx/du, h_y = dy/dv, h_z = dz/dw, the legacy path uses T psi = -1/2 [ 1/h_x d/du (1/h_x dpsi/du) + 1/h_y d/dv (1/h_y dpsi/dv) + 1/h_z d/dw (1/h_z dpsi/dw) ] The derivatives are discretized on the uniform logical grid with a second-order centered flux form and face-averaged scale factors. At the outer boundary, zero ghost cells are used. Monitor-driven A-grid: For the full 3D curvilinear mapping x = x(xi_1, xi_2, xi_3), the new path uses T psi = -1/2 * (1/J) * d/dxi_a [ J g^{ab} dpsi/dxi_b ] where J is the Jacobian and g^{ab} is the inverse metric tensor of the current monitor grid. The derivatives are evaluated on the uniform logical cube with second-order finite differences; at the outer boundary, NumPy's second-order one-sided derivatives are used. This is a first finite-domain A-grid kinetic slice, not the final production operator.
- Functions:
  - `validate_orbital_field()`: Validate that a field matches the 3D grid shape.
  - `integrate_field()`: Integrate a 3D field with the point-centered cell volumes.
  - `weighted_l2_norm()`: Return the weighted L2 norm of a 3D field on the structured grid.
  - `_logical_spacing()`: Function without docstring.
  - `_apply_axis_laplacian()`: Apply one separable axis contribution to the transformed Laplacian.
  - `apply_legacy_laplacian_operator()`: Apply the legacy separable transformed Laplacian to a 3D field.
  - `apply_legacy_kinetic_operator()`: Apply the legacy separable kinetic operator T = -1/2 Laplacian.
  - `_validate_monitor_geometry()`: Function without docstring.
  - `_axis_gradient_zero_ghost()`: Differentiate along one logical axis using a zero-ghost centered stencil. This is an audit-side trial fix for the monitor-grid kinetic path. The boundary node sees a zero ghost value beyond the physical box, while the interior still uses the standard centered difference.
  - `_monitor_grid_derivatives()`: Function without docstring.
  - `compute_monitor_grid_contravariant_flux_components()`: Build the monitor-grid contravariant fluxes J g^{ab} d_b psi. The optional trial-fix branch keeps the same curvilinear kinetic form but swaps the boundary derivative/ghost handling from NumPy's one-sided edge rule to a centered zero-ghost stencil at the physical box boundary.
  - `_monitor_grid_divergence_from_flux()`: Function without docstring.
  - `apply_monitor_grid_laplacian_operator()`: Apply the full curvilinear Laplacian on the 3D monitor grid. This path uses nabla^2 psi = (1/J) d/dxi_a [ J g^{ab} dpsi/dxi_b ] on the uniform logical cube of the monitor grid.
  - `apply_monitor_grid_laplacian_operator_trial_boundary_fix()`: Apply a trial monitor-grid Laplacian with zero-ghost boundary closure. The curvilinear operator form is unchanged. The only prototype change is that all logical-axis derivatives inside the A-grid kinetic path switch from one-sided edge gradients to a centered zero-ghost stencil at the physical box boundary.
  - `apply_monitor_grid_kinetic_operator()`: Apply the first A-grid kinetic operator T = -1/2 Laplacian.
  - `apply_monitor_grid_kinetic_operator_trial_boundary_fix()`: Apply the trial A-grid kinetic operator with zero-ghost boundary closure.
  - `apply_laplacian_operator()`: Apply the appropriate Laplacian on either the legacy or A-grid geometry.
  - `apply_kinetic_operator()`: Apply the appropriate kinetic operator on either supported grid family.

### `src/isogrid/ops/kinetic_jax.py`

- File role: JAX monitor-grid kinetic kernels for the local A-grid hot path.
- Functions:
  - `_logical_spacing()`: Function without docstring.
  - `_validate_monitor_geometry()`: Function without docstring.
  - `_validate_field()`: Function without docstring.
  - `_build_axis_gradient_kernel()`: Function without docstring.
  - `build_monitor_grid_laplacian_operator_jax()`: Return a jitted monitor-grid Laplacian kernel on one fixed geometry.
  - `apply_monitor_grid_laplacian_operator_jax()`: Apply the production monitor-grid Laplacian with JAX.
  - `apply_monitor_grid_laplacian_operator_trial_boundary_fix_jax()`: Apply the trial-fix monitor-grid Laplacian with JAX.
  - `build_monitor_grid_kinetic_operator_jax()`: Return a jitted monitor-grid kinetic kernel on one fixed geometry.
  - `apply_monitor_grid_kinetic_operator_jax()`: Apply the production monitor-grid kinetic operator with JAX.
  - `apply_monitor_grid_kinetic_operator_trial_boundary_fix_jax()`: Apply the trial-fix monitor-grid kinetic operator with JAX.

### `src/isogrid/ops/reductions_jax.py`

- File role: JAX reductions and weighted linear-algebra kernels for the scientific path.
- Functions:
  - `_as_weighted_array()`: Function without docstring.
  - `weighted_inner_product_jax()`: Return the weighted inner product `<left|right>_W`.
  - `weighted_l2_norm_jax()`: Return the weighted L2 norm `sqrt(<field|field>_W)`.
  - `accumulate_density_from_orbitals_jax()`: Accumulate `rho = sum_i occ_i |psi_i|^2` from one orbital block.
  - `flatten_orbital_block_jax()`: Flatten a `(k, nx, ny, nz)` block into `(n_grid, k)` columns.
  - `reshape_orbital_columns_jax()`: Reshape `(n_grid, k)` columns back to `(k, nx, ny, nz)`.
  - `weighted_overlap_matrix_jax()`: Return the weighted overlap/Gram matrix under the cell-volume metric.
  - `weighted_gram_matrix_jax()`: Alias for the weighted overlap matrix used by block methods.
  - `weighted_orthonormalize_orbitals_jax()`: Weighted-orthonormalize one orbital block using an overlap eigensolve.
  - `build_weighted_overlap_kernel_jax()`: Return a small jitted weighted overlap kernel for one fixed weight field.

### `src/isogrid/poisson/__init__.py`

- File role: Open-boundary Poisson and Hartree helpers for the stage-1 prototype.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/poisson/hartree.py`

- File role: Hartree helpers built on the first-stage open-boundary Poisson slice.
- Classes:
  - `HartreeEvaluation`: Resolved Hartree potential, action, and energy for one orbital field.
- Functions:
  - `validate_density_field()`: Validate a 3D density field on the current structured grid.
  - `solve_hartree_potential()`: Solve the first-stage Hartree potential from a total electron density.
  - `_resolve_hartree_potential_array()`: Function without docstring.
  - `evaluate_hartree_energy()`: Return the first-stage Hartree energy E_H = 1/2 int rho v_H.
  - `build_hartree_action()`: Return the Hartree action v_H * psi for one orbital field.
  - `evaluate_hartree_terms()`: Resolve the Hartree potential, action, and energy for one orbital field.

### `src/isogrid/poisson/open_boundary.py`

- File role: First-stage open-boundary Poisson solver on the structured adaptive grid. The current route solves the finite-domain Poisson equation nabla^2 v(r) = -4 pi rho(r) on the existing structured adaptive grid. The boundary values on the finite box are not periodic. Instead, they are approximated by a free-space multipole expansion of the density-induced potential about the grid reference center, truncated at quadrupole order by default. Inside the box, the same separable flux-form discretization family used by the current structured-grid kinetic operator is reused for the Poisson operator. The interior Dirichlet problem is solved with a SciPy BiCGSTAB linear solve when SciPy is available, and falls back to damped Jacobi iteration otherwise. This is a first formal open-boundary approximation for the Hartree path. It is meant to be explicit and auditable, not the final production free-space solver.
- Classes:
  - `OpenBoundaryMultipoleBoundary`: First-stage free-space boundary approximation data for Poisson.
  - `OpenBoundaryPoissonResult`: Resolved finite-domain Poisson result with open-boundary approximation.
- Functions:
  - `_logical_spacing()`: Function without docstring.
  - `_axis_neighbor_coefficients()`: Function without docstring.
  - `_boundary_mask()`: Function without docstring.
  - `_trilinear_cell_value()`: Function without docstring.
  - `_cell_average_from_nodal_field()`: Function without docstring.
  - `_monitor_grid_selected_reconstruction_cells()`: Function without docstring.
  - `_selected_cell_incident_node_counts()`: Function without docstring.
  - `_accumulate_multipole_moments()`: Function without docstring.
  - `_monitor_grid_nodal_region_moments()`: Function without docstring.
  - `_cell_and_neighbor_slices()`: Function without docstring.
  - `_local_quadratic_fit_coefficients()`: Function without docstring.
  - `_evaluate_quadratic_fit()`: Function without docstring.
  - `_monitor_grid_multipole_correction()`: Function without docstring.
  - `_evaluate_boundary_potential_from_moments()`: Function without docstring.
  - `_monitor_grid_boundary_value_correction()`: Function without docstring.
  - `_default_monitor_reference_center()`: Function without docstring.
  - `_compute_multipole_boundary_condition()`: Function without docstring.
  - `_interior_operator_coefficients()`: Function without docstring.
  - `_solve_monitor_with_scipy_bicgstab()`: Function without docstring.
  - `_monitor_diagonal_estimate()`: Function without docstring.
  - `_solve_monitor_with_jacobi()`: Function without docstring.
  - `_solve_open_boundary_poisson_monitor()`: Function without docstring.
  - `_build_poisson_rhs()`: Function without docstring.
  - `_apply_interior_operator()`: Function without docstring.
  - `_solve_with_scipy_bicgstab()`: Function without docstring.
  - `_solve_with_jacobi()`: Function without docstring.
  - `_solve_open_boundary_poisson_legacy()`: Solve the first-stage finite-domain Poisson problem with open-boundary data.
  - `solve_open_boundary_poisson()`: Solve the first-stage finite-domain Poisson problem with open-boundary data.

### `src/isogrid/poisson/poisson_jax.py`

- File role: JAX hot kernels for the monitor-grid open-boundary Poisson path. This module intentionally migrates only the numerically hot pieces of the current monitor-grid Poisson route: - monitor-grid Poisson operator apply - one small JAX conjugate-gradient solve for the interior unknown The physical boundary model is unchanged. Boundary values still come from the existing open-boundary multipole approximation, which remains in the Python audit/fallback layer.
- Classes:
  - `MonitorPoissonJaxSolveDiagnostics`: Compact diagnostics for the JAX monitor-grid Poisson solve.
  - `MonitorPoissonJaxCachedSolveKernels`: Cached monitor-grid JAX Poisson kernels for one geometry context.
  - `MonitorPoissonJaxSeparablePreconditionerContext`: Cached separable preconditioner data for one monitor-grid geometry.
  - `MonitorPoissonJaxLinePreconditionerContext`: Cached metric-aware line preconditioner for one monitor-grid geometry.
- Functions:
  - `apply_monitor_open_boundary_poisson_operator_jax()`: Apply the monitor-grid Poisson operator `-L(v)` with JAX.
  - `_build_monitor_interior_scatter()`: Function without docstring.
  - `build_monitor_open_boundary_poisson_matvec_jax()`: Return the interior Poisson matvec `x -> -L(x)` for zero-boundary unknowns.
  - `clear_monitor_poisson_jax_kernel_cache()`: Clear the thin monitor-grid JAX Poisson kernel cache.
  - `get_last_monitor_poisson_jax_solve_diagnostics()`: Return the most recent monitor-grid JAX Poisson solve diagnostics.
  - `_normalize_monitor_poisson_jax_cg_impl()`: Function without docstring.
  - `_normalize_monitor_poisson_jax_cg_preconditioner()`: Function without docstring.
  - `_normalize_monitor_poisson_jax_line_preconditioner_impl()`: Function without docstring.
  - `_logical_spacing()`: Function without docstring.
  - `_build_monitor_open_boundary_inverse_preconditioner_diagonal_jax()`: Function without docstring.
  - `_axis_stiffness_from_monitor_operator_diagonal()`: Function without docstring.
  - `_build_dirichlet_sine_basis_matrix()`: Function without docstring.
  - `_build_monitor_open_boundary_separable_preconditioner_context_jax()`: Function without docstring.
  - `_get_monitor_open_boundary_separable_preconditioner_context_jax()`: Function without docstring.
  - `_build_monitor_open_boundary_line_preconditioner_context_jax()`: Function without docstring.
  - `_get_monitor_open_boundary_line_preconditioner_context_jax()`: Function without docstring.
  - `_build_monitor_poisson_jax_cache_key()`: Function without docstring.
  - `_get_monitor_open_boundary_poisson_kernels_jax()`: Function without docstring.
  - `_run_jax_cg()`: Function without docstring.
  - `_build_jax_loop_cg_solver()`: Function without docstring.
  - `_build_monitor_poisson_cg_loop_cache_key()`: Function without docstring.
  - `_get_monitor_poisson_cg_loop_solver()`: Function without docstring.
  - `_estimate_matvec_wall_time_seconds()`: Function without docstring.
  - `_estimate_monitor_poisson_jax_preconditioner_apply_count()`: Function without docstring.
  - `_time_jax_probe_call()`: Function without docstring.
  - `_estimate_line_preconditioner_wall_times()`: Function without docstring.
  - `_run_jax_cg_loop()`: Function without docstring.
  - `solve_open_boundary_poisson_monitor_jax()`: Solve the monitor-grid open-boundary Poisson problem with JAX CG.

### `src/isogrid/pseudo/__init__.py`

- File role: Minimal GTH pseudopotential data and local/nonlocal potential helpers.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/pseudo/gth_data.py`

- File role: Internal GTH data loading for the current stage-1 element set. This layer intentionally supports only the GTH pseudopotentials needed by the current project scope: H, C, N, and O with the `gth-pade` family. The parameter values were transcribed from the CP2K-format GTH data exposed by PySCF via `pyscf.pbc.gto.pseudo.load("gth-pade", symbol)`.
- Functions:
  - `_normalize_family_name()`: Function without docstring.
  - `_normalize_element_symbol()`: Function without docstring.
  - `load_gth_pseudo_data()`: Load one supported internal GTH pseudopotential data object.
  - `load_gth_pseudo_data_for_elements()`: Load supported GTH data for a set of element symbols.
  - `load_case_gth_pseudo_data()`: Load the supported GTH data needed for one benchmark case.

### `src/isogrid/pseudo/local.py`

- File role: Local GTH pseudopotential evaluation on legacy and monitor-driven grids.
- Classes:
  - `AtomicLocalPotentialContribution`: One atom's local GTH pseudopotential on the structured grid.
  - `LocalIonicPotentialEvaluation`: Local ionic GTH potential assembled over all atoms in one geometry. Future nonlocal projector action should attach alongside the per-atom contributions collected here.
  - `LocalPotentialPatchParameters`: Parameters for the atom-centered local-GTH near-core patch correction.
    - `__post_init__()`: Method without docstring.
  - `AtomicLocalPotentialPatchCorrection`: Near-core patch correction for one atom's local GTH contribution.
  - `LocalIonicPotentialPatchEvaluation`: Patch-assisted local-GTH correction on the monitor grid. The pointwise main-grid potential is left unchanged. The correction acts on the near-core local-GTH energy functional by replacing the main-grid quadrature over each atom-centered patch with a finer auxiliary patch quadrature.
  - `FrozenPatchLocalPotentialEmbedding`: Frozen-density embedding of the patch correction into a local potential. The current near-core patch is defined as an energy correction, not as a general density-functional derivative. For the fixed-potential eigensolver audit we embed that correction into a frozen local potential field `delta V_patch(r)` chosen so that int rho_frozen(r) delta V_patch(r) dr = Delta E_loc^patch on the current frozen density. This is suitable for fixed-potential audit work only; it is not yet a general SCF-ready local-GTH potential correction.
- Functions:
  - `_screened_coulomb_term()`: Function without docstring.
  - `_local_polynomial_term()`: Function without docstring.
  - `_evaluate_atomic_local_potential_from_radial_distance()`: Function without docstring.
  - `_evaluate_atomic_local_potential_on_grid()`: Evaluate one atom's local GTH potential on one supported grid.
  - `evaluate_atomic_local_potential_on_legacy_grid()`: Evaluate one atom's local GTH potential on the legacy structured grid.
  - `evaluate_atomic_local_potential_on_monitor_grid()`: Evaluate one atom's local GTH potential on the new A-grid. This first A-grid local-GTH path uses only the main monitor grid. The patch layer is intentionally not used yet; it will be the first follow-up layer for near-core local-GTH corrections.
  - `evaluate_atomic_local_potential()`: Evaluate one atom's local GTH potential on either supported grid family.
  - `evaluate_legacy_local_ionic_potential()`: Evaluate the total local ionic GTH potential on the legacy grid.
  - `evaluate_monitor_grid_local_ionic_potential()`: Evaluate the total local ionic GTH potential on the new monitor-driven grid. This is the first A-grid local-GTH slice. It is intentionally main-grid only and does not yet use the auxiliary patch layer.
  - `evaluate_local_ionic_potential()`: Evaluate the total local ionic GTH potential on either supported grid family.
  - `evaluate_local_ionic_energy()`: Evaluate `E_loc,ion = ∫ rho V_loc` on the grid carried by one evaluation.
  - `_build_patch_grid()`: Function without docstring.
  - `_interpolate_field_to_patch_points()`: Function without docstring.
  - `_evaluate_atomic_local_patch_correction()`: Function without docstring.
  - `evaluate_monitor_grid_local_ionic_potential_with_patch()`: Apply a near-core patch correction to the A-grid local-GTH energy. The corrected local energy is E_loc^patch = E_loc^main + sum_A lambda [I_A^patch - I_A^main(patch)] with one atom-centered spherical patch per atom. `I_A^patch` is evaluated on a finer auxiliary Cartesian patch using density interpolated from the main grid, while `I_A^main(patch)` is the original main-grid quadrature over the same spherical support. The main-grid pointwise potential itself is not replaced.
  - `_build_atomic_frozen_patch_correction_field()`: Function without docstring.
  - `evaluate_monitor_grid_local_ionic_potential_with_frozen_patch_field()`: Embed the near-core patch correction into a frozen local potential field. This helper is intentionally narrow in scope. It constructs a corrected pointwise local potential only for fixed-density / fixed-potential audit paths, while leaving the patch math itself unchanged.
  - `build_default_h2_local_ionic_potential()`: Evaluate the default H2 local ionic GTH potential on the default grid.
  - `build_default_h2_monitor_grid_local_ionic_potential()`: Evaluate the default H2 local ionic GTH potential on the A-grid.
  - `build_h2_local_patch_development_monitor_grid_local_ionic_potential()`: Evaluate the H2 A-grid local-GTH potential on the patch-development baseline.

### `src/isogrid/pseudo/model.py`

- File role: Minimal data models for the stage-1 GTH pseudopotential route.
- Classes:
  - `GTHLocalTerm`: Local GTH pseudopotential parameters for one element.
    - `__post_init__()`: Method without docstring.
  - `GTHNonlocalChannel`: Nonlocal projector metadata for one angular-momentum channel.
  - `GTHPseudoData`: Minimal internal representation of one GTH pseudopotential.
    - `valence_electrons()`: Return the nominal valence electron count for this pseudopotential.

### `src/isogrid/pseudo/nonlocal.py`

- File role: First-stage real-space GTH nonlocal projector and action slice. The current implementation is intentionally narrow and audit-friendly: - only the internal `gth-pade` H/C/N/O data path is supported - only the real-space projector structure needed by that data is targeted - the current H/C/N/O `gth-pade` set has no active nonlocal projectors for H and one active s-channel projector for C, N, and O For one atom, the separable GTH nonlocal action is applied as V_nl psi = sum_{l,m} sum_{i,j} |p_i^{lm}> h_ij <p_j^{lm} | psi> with real-space projectors p_i^{lm}(r_vec) = p_i^l(r) Y_lm(r_hat) and the normalized radial part p_i^l(r) = sqrt(2) * r^(l + 2 i) * exp[-r^2 / (2 r_l^2)] / [r_l^(l + (4 i + 3)/2) * sqrt(Gamma(l + (4 i + 3)/2))] where `i` is zero-based in this implementation and corresponds to the literature index `i + 1`. This is a first formal nonlocal slice for the static KS backbone. It is not yet an optimized production implementation and does not claim general GTH support.
- Classes:
  - `ProjectorFieldEvaluation`: One evaluated nonlocal projector field on the structured grid.
  - `AtomicNonlocalActionContribution`: One atom's resolved nonlocal projector data and action.
  - `NonlocalIonicActionEvaluation`: Full nonlocal ionic action assembled over all atoms for one orbital.
- Functions:
  - `_scalarize()`: Function without docstring.
  - `_real_spherical_harmonic()`: Function without docstring.
  - `_magnetic_numbers()`: Function without docstring.
  - `_radial_projector()`: Function without docstring.
  - `evaluate_atomic_projector_field()`: Evaluate one atom-centered GTH nonlocal projector on the grid.
  - `evaluate_atomic_nonlocal_action()`: Apply one atom's separable GTH nonlocal term to one orbital field.
  - `evaluate_nonlocal_ionic_action()`: Apply the full ionic GTH nonlocal term to one orbital field.
  - `build_default_h2_nonlocal_ionic_action()`: Apply the default H2 nonlocal ionic term on the default structured grid.

### `src/isogrid/scf/__init__.py`

- File role: Minimal SCF driver exports for the structured-grid prototype.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/scf/driver.py`

- File role: First minimal H2 SCF driver for the structured-grid prototype. This module intentionally implements only the smallest formal SCF loop needed to close the H2 single-point path. The current stage-1 flow is: 1. choose one of the configured H2 candidate spin states 2. build a deterministic initial orbital / density guess 3. freeze the current density and solve the static KS problem for each spin channel with the existing fixed-potential eigensolver 4. rebuild rho_up and rho_down from the occupied orbitals 5. apply simple linear density mixing 6. monitor density residual and total-energy change 7. report a single-point total energy together with explicit component terms This is the first formal SCF driver slice. It is restricted on purpose: - only the current neutral H2 benchmark is supported - only the configured singlet and triplet candidates are supported - only linear density mixing is implemented - the total energy is evaluated as E = T_s + E_loc,ion + E_nl,ion + E_H + E_xc + E_II on the current discrete adaptive grid It is not yet a general SCF framework and it is not the final production path.
- Classes:
  - `SpinOccupations`: Minimal spin occupation data for the current H2 SCF driver.
  - `FixedPotentialSolveSummary`: Compact audit-facing summary of one frozen-potential solve.
  - `SinglePointEnergyComponents`: Single-point energy components for the current SCF density/orbitals.
  - `ScfIterationRecord`: Per-iteration audit record for the minimal H2 SCF driver.
  - `H2ScfResult`: Final result of the first minimal H2 single-point SCF loop.
  - `H2ScfDryRunParameterSummary`: Fixed parameter summary for the monitor-grid SCF dry-run.
  - `MonitorGridDiisHistoryEntry`: Small DIIS history item for the monitor-grid singlet dry-run.
  - `MonitorGridAndersonHistoryEntry`: Small Anderson history item for the monitor-grid singlet dry-run.
  - `MonitorGridAndersonApplyResult`: Small result bundle for one Anderson proposal attempt.
  - `MonitorGridBroydenHistoryEntry`: Small Broyden-like history item for the monitor-grid singlet dry-run.
  - `StaticLocalEnergyEvaluationProfile`: Very small timing summary for local-only single-point energy evaluation.
  - `H2StaticLocalScfDryRunResult`: Result of the first H2 A-grid static-local SCF dry-run.
- Functions:
  - `_empty_orbital_block()`: Function without docstring.
  - `_find_spin_state()`: Function without docstring.
  - `_validate_h2_case()`: Function without docstring.
  - `resolve_h2_spin_occupations()`: Resolve the explicit alpha/beta occupations for H2 singlet/triplet.
  - `_expectation_value()`: Function without docstring.
  - `_sum_weighted_expectations()`: Function without docstring.
  - `_build_density_from_occupied_orbitals()`: Function without docstring.
  - `_renormalize_density()`: Function without docstring.
  - `_density_residual()`: Function without docstring.
  - `_build_h2_trial_orbitals()`: Function without docstring.
  - `build_h2_initial_density_guess()`: Build the current minimal H2 initial density and orbital guesses.
  - `_build_solve_summary()`: Function without docstring.
  - `_is_h2_closed_shell_singlet()`: Function without docstring.
  - `_is_closed_shell_singlet()`: Function without docstring.
  - `evaluate_ion_ion_repulsion()`: Evaluate the valence-ion Coulomb repulsion using the current GTH charges.
  - `evaluate_single_point_energy()`: Evaluate the first-stage single-point total energy on the current grid.
  - `_default_monitor_patch_parameters()`: Function without docstring.
  - `_monitor_grid_scf_parameter_summary()`: Function without docstring.
  - `_detect_monitor_grid_singlet_alternation()`: Detect a weak even/odd alternation in the monitor-grid singlet dry-run.
  - `_is_hartree_tail_guard_v2()`: Function without docstring.
  - `_taper_hartree_tail_guard_release_potential()`: Function without docstring.
  - `_record_last_jax_hartree_solve_diagnostics()`: Function without docstring.
  - `_density_residual_fields()`: Function without docstring.
  - `_smooth_monitor_grid_field()`: Function without docstring.
  - `_apply_singlet_real_space_preconditioner()`: Function without docstring.
  - `_weighted_spin_density_dot()`: Function without docstring.
  - `_weighted_spin_density_norm()`: Function without docstring.
  - `_weighted_field_norm()`: Function without docstring.
  - `_estimate_singlet_hartree_tail_channel_shares()`: Function without docstring.
  - `_apply_singlet_hartree_tail_mitigation()`: Function without docstring.
  - `_apply_hartree_tail_guard()`: Function without docstring.
  - `_filter_monitor_grid_anderson_secants()`: Function without docstring.
  - `_select_monitor_grid_anderson_damping()`: Function without docstring.
  - `_apply_monitor_grid_density_diis()`: Function without docstring.
  - `_apply_monitor_grid_density_anderson()`: Function without docstring.
  - `_apply_monitor_grid_density_broyden_like()`: Function without docstring.
  - `_apply_kinetic_for_static_local_energy()`: Function without docstring.
  - `evaluate_static_local_single_point_energy()`: Evaluate the local-only single-point energy on either supported grid. This helper intentionally contains only T_s + E_loc,ion + E_H + E_xc + E_II and sets the nonlocal contribution to zero. It is meant for the current A-grid SCF dry-run and related local-only audits; it is not a replacement for the full legacy total-energy path.
  - `evaluate_static_local_single_point_energy_from_context()`: Evaluate the local-only single-point energy from one frozen static-local context. This reuses the step-local `rho_total`, local ionic slice, Hartree potential, and LSDA evaluation that were already assembled for the fixed-potential solve.
  - `_check_density_electron_count()`: Function without docstring.
  - `run_h2_minimal_scf()`: Run the first minimal H2 single-point SCF loop for one spin state.
  - `run_h2_monitor_grid_scf_dry_run()`: Run the first monitor-grid H2 SCF dry-run on the local static chain.

### `src/isogrid/xc/__init__.py`

- File role: Exchange-correlation helpers for the current stage-1 prototype.
- Top-level definitions: no public classes/functions; mainly exports or placeholders.

### `src/isogrid/xc/lsda.py`

- File role: First-stage spin-polarized LSDA kernel aligned with the PySCF audit baseline. This module deliberately implements only the `lda,vwn` path used by the current PySCF reference scripts. To keep the reference-side physics aligned while the real-space solver is still being assembled, the pointwise LSDA evaluation is forwarded to the locally installed PySCF/libxc backend at runtime. The module itself remains importable without PySCF. A clear runtime error is raised only when LSDA values are actually requested.
- Classes:
  - `LSDAEvaluation`: Resolved LSDA energy-density and potential data on a 3D grid.
- Functions:
  - `_require_pyscf_libxc()`: Function without docstring.
  - `_normalize_functional_name()`: Function without docstring.
  - `validate_density_field()`: Validate a 3D density field and clip tiny negative roundoff to zero.
  - `evaluate_lsda_terms()`: Evaluate the pointwise spin-polarized LSDA terms for one 3D density pair.
  - `evaluate_lsda_energy_density()`: Return the per-electron LSDA xc energy density eps_xc(r).
  - `evaluate_lsda_potential()`: Return the spin-channel LSDA potentials v_xc_up and v_xc_down.
  - `evaluate_lsda_energy()`: Return `E_xc = int rho(r) eps_xc(r)` on either supported grid family.

## Test Tree (`tests/`)

### `tests/test_fixed_potential_eigensolver.py`

- File role: Sanity checks for the first fixed-potential eigensolver slice.
- Test helpers:
  - `_solve_default_h2()`: Helper used by the tests in this file.
- Test cases:
  - `test_eigensolver_module_imports()`: eigensolver module imports
  - `test_default_h2_lowest_orbital_can_be_solved()`: default h2 lowest orbital can be solved
  - `test_fixed_potential_orbitals_are_weighted_normalized()`: fixed potential orbitals are weighted normalized
  - `test_two_fixed_potential_orbitals_remain_weighted_orthogonal_and_sorted()`: two fixed potential orbitals remain weighted orthogonal and sorted
  - `test_ground_orbital_preserves_basic_h2_mirror_symmetry()`: ground orbital preserves basic h2 mirror symmetry

### `tests/test_grid_geometry.py`

- File role: Sanity checks for the structured adaptive grid geometry layer.
- Test cases:
  - `test_h2_default_grid_can_be_constructed()`: h2 default grid can be constructed
  - `test_axis_mapping_is_monotone()`: axis mapping is monotone
  - `test_grid_coordinate_shapes_are_correct()`: grid coordinate shapes are correct
  - `test_geometric_weights_are_positive()`: geometric weights are positive
  - `test_default_h2_grid_preserves_basic_symmetry()`: default h2 grid preserves basic symmetry

### `tests/test_gth_local_potential.py`

- File role: Sanity checks for the first GTH local-potential slice.
- Test cases:
  - `test_pseudo_module_and_audit_imports()`: pseudo module and audit imports
  - `test_hcno_gth_data_can_be_loaded()`: hcno gth data can be loaded
  - `test_unsupported_element_raises_clear_error()`: unsupported element raises clear error
  - `test_unsupported_pseudo_family_raises_clear_error()`: unsupported pseudo family raises clear error
  - `test_default_h2_local_potential_can_be_constructed()`: default h2 local potential can be constructed
  - `test_local_potential_shape_matches_grid_shape()`: local potential shape matches grid shape
  - `test_local_potential_values_are_finite()`: local potential values are finite
  - `test_default_h2_local_potential_is_mirror_symmetric()`: default h2 local potential is mirror symmetric
  - `test_default_h2_local_potential_is_deeper_near_nuclei_than_far_field()`: default h2 local potential is deeper near nuclei than far field

### `tests/test_h2_grid_convergence_audit.py`

- File role: Minimal smoke tests for the H2 grid/domain convergence audit layer.
- Test cases:
  - `test_construct_grid_convergence_scan_point()`: construct grid convergence scan point
  - `test_import_regression_baseline_fields()`: import regression baseline fields

### `tests/test_h2_hartree_boundary_diagnosis_audit.py`

- File role: Minimal smoke tests for the fixed-density Hartree boundary diagnosis audit.
- Test cases:
  - `test_hartree_boundary_diagnosis_audit_module_imports()`: hartree boundary diagnosis audit module imports
  - `test_h2_monitor_grid_baseline_shape_reflects_current_frozen_electrostatics_baseline()`: h2 monitor grid baseline shape reflects current frozen electrostatics baseline
  - `test_small_hartree_boundary_diagnosis_audit_is_finite()`: small hartree boundary diagnosis audit is finite
  - `test_small_hartree_boundary_shape_sweep_is_finite_and_not_systematically_worse()`: small hartree boundary shape sweep is finite and not systematically worse
  - `test_small_hartree_measure_ledger_audit_is_finite()`: small hartree measure ledger audit is finite
  - `test_small_hartree_geometry_representation_audit_is_finite()`: small hartree geometry representation audit is finite
  - `test_small_hartree_mapping_stage_attribution_audit_is_finite()`: small hartree mapping stage attribution audit is finite
  - `test_small_hartree_mapping_solve_stage_attribution_audit_is_finite()`: small hartree mapping solve stage attribution audit is finite
  - `test_small_hartree_reference_quadrature_audit_is_finite()`: small hartree reference quadrature audit is finite
  - `test_small_hartree_inside_cell_representation_audit_is_finite()`: small hartree inside cell representation audit is finite
  - `test_small_hartree_inside_cell_reconstruction_comparison_audit_is_finite()`: small hartree inside cell reconstruction comparison audit is finite

### `tests/test_h2_hartree_poisson_comparison_audit.py`

- File role: Minimal smoke tests for the H2 Hartree / Poisson comparison audit.
- Test cases:
  - `test_hartree_poisson_comparison_audit_module_imports()`: hartree poisson comparison audit module imports
  - `test_small_monitor_grid_hartree_route_is_finite_and_symmetric()`: small monitor grid hartree route is finite and symmetric
  - `test_small_monitor_grid_gaussian_density_hartree_is_finite()`: small monitor grid gaussian density hartree is finite
  - `test_construct_hartree_comparison_result_objects()`: construct hartree comparison result objects

### `tests/test_h2_hartree_tail_recheck_audit.py`

- File role: Minimal smoke tests for the H2 Hartree tail-recheck audit.
- Test cases:
  - `test_h2_hartree_tail_recheck_module_imports()`: h2 hartree tail recheck module imports
  - `test_construct_h2_hartree_tail_recheck_point()`: construct h2 hartree tail recheck point

### `tests/test_h2_jax_eigensolver_hotpath_audit.py`

- File role: Minimal smoke tests for the JAX eigensolver hot-path audit.
- Test helpers:
  - `_route()`: Helper used by the tests in this file.
- Test cases:
  - `test_h2_jax_eigensolver_hotpath_audit_module_imports()`: h2 jax eigensolver hotpath audit module imports
  - `test_construct_h2_jax_eigensolver_hotpath_result()`: construct h2 jax eigensolver hotpath result

### `tests/test_h2_jax_kernel_consistency_audit.py`

- File role: Minimal smoke tests for the first-batch JAX kernel migration.
- Test helpers:
  - `_build_trial_orbital()`: Helper used by the tests in this file.
- Test cases:
  - `test_weighted_inner_product_jax_runs()`: weighted inner product jax runs
  - `test_monitor_poisson_apply_jax_runs()`: monitor poisson apply jax runs
  - `test_h2_jax_kernel_consistency_result_fields()`: h2 jax kernel consistency result fields

### `tests/test_h2_jax_scf_hotpath_audit.py`

- File role: Minimal smoke tests for the H2 JAX SCF hot-path audit.
- Test cases:
  - `test_h2_jax_scf_hotpath_module_imports()`: h2 jax scf hotpath module imports
  - `test_construct_h2_jax_scf_hotpath_result()`: construct h2 jax scf hotpath result

### `tests/test_h2_jax_singlet_mainline_audit.py`

- File role: Minimal smoke tests for the H2 singlet Hartree-tail mitigation audit.
- Test helpers:
  - `_build_route()`: Helper used by the tests in this file.
- Test cases:
  - `test_h2_jax_singlet_mainline_audit_module_imports()`: h2 jax singlet mainline audit module imports
  - `test_construct_h2_jax_singlet_mainline_result()`: construct h2 jax singlet mainline result
  - `test_construct_h2_jax_singlet_acceptance_result()`: construct h2 jax singlet acceptance result
  - `test_construct_h2_jax_singlet_hartree_tail_guard_result()`: construct h2 jax singlet hartree tail guard result
  - `test_construct_h2_jax_singlet_hartree_tail_guard_v2_result()`: construct h2 jax singlet hartree tail guard v2 result
  - `test_construct_h2_jax_singlet_structural_stabilizer_result()`: construct h2 jax singlet structural stabilizer result

### `tests/test_h2_jax_triplet_end_to_end_micro_profile_audit.py`

- File role: Test module without a module docstring.
- Test cases:
  - `test_triplet_end_to_end_micro_profile_result_fields()`: triplet end to end micro profile result fields

### `tests/test_h2_jax_triplet_hartree_energy_audit.py`

- File role: Minimal smoke tests for the H2 triplet JAX Hartree/energy audit.
- Test cases:
  - `test_h2_jax_triplet_hartree_energy_module_imports()`: h2 jax triplet hartree energy module imports
  - `test_construct_h2_jax_triplet_hartree_energy_result()`: construct h2 jax triplet hartree energy result

### `tests/test_h2_jax_triplet_reintegration_smoke_audit.py`

- File role: Minimal smoke tests for the triplet reintegration audit.
- Test cases:
  - `test_triplet_reintegration_module_imports()`: triplet reintegration module imports
  - `test_construct_triplet_reintegration_route_result()`: construct triplet reintegration route result

### `tests/test_h2_monitor_grid_diis_scf_audit.py`

- File role: Tiny regression scaffolding for the H2 A-grid DIIS SCF audit.
- Test cases:
  - `test_construct_h2_monitor_grid_diis_scf_audit_result()`: construct h2 monitor grid diis scf audit result

### `tests/test_h2_monitor_grid_fair_calibration_audit.py`

- File role: Minimal smoke tests for the H2 A-grid fairness calibration audit.
- Test cases:
  - `test_run_h2_monitor_grid_fair_calibration_audit_single_point()`: run h2 monitor grid fair calibration audit single point
  - `test_construct_fair_calibration_point_object()`: construct fair calibration point object

### `tests/test_h2_monitor_grid_fixed_potential_eigensolver_audit.py`

- File role: Minimal smoke tests for the A-grid fixed-potential eigensolver audit.
- Test cases:
  - `test_h2_monitor_grid_fixed_potential_module_imports()`: h2 monitor grid fixed potential module imports
  - `test_import_jax_native_fixed_potential_entrypoint()`: import jax native fixed potential entrypoint
  - `test_construct_h2_fixed_potential_route_result()`: construct h2 fixed potential route result

### `tests/test_h2_monitor_grid_geometry_consistency_audit.py`

- File role: Minimal object tests for the H2 monitor-grid geometry consistency audit.
- Test helpers:
  - `_summary()`: Helper used by the tests in this file.
  - `_geometry_summary()`: Helper used by the tests in this file.
  - `_field_result()`: Helper used by the tests in this file.
- Test cases:
  - `test_geometry_consistency_result_fields_exist()`: geometry consistency result fields exist

### `tests/test_h2_monitor_grid_k2_subspace_audit.py`

- File role: Smoke tests for the H2 k=2 subspace audit.
- Test cases:
  - `test_construct_h2_k2_subspace_audit_result()`: construct h2 k2 subspace audit result

### `tests/test_h2_monitor_grid_kinetic_form_audit.py`

- File role: Minimal object tests for the H2 monitor-grid kinetic form audit.
- Test helpers:
  - `_summary()`: Helper used by the tests in this file.
  - `_probe()`: Helper used by the tests in this file.
  - `_comparison()`: Helper used by the tests in this file.
- Test cases:
  - `test_kinetic_form_comparison_fields_exist()`: kinetic form comparison fields exist
  - `test_kinetic_form_audit_result_fields_exist()`: kinetic form audit result fields exist

### `tests/test_h2_monitor_grid_kinetic_green_identity_audit.py`

- File role: Minimal object tests for the H2 monitor-grid kinetic Green-identity audit.
- Test helpers:
  - `_summary()`: Helper used by the tests in this file.
  - `_field_result()`: Helper used by the tests in this file.
- Test cases:
  - `test_green_identity_field_fields_exist()`: green identity field fields exist
  - `test_green_identity_audit_result_fields_exist()`: green identity audit result fields exist

### `tests/test_h2_monitor_grid_kinetic_operator_audit.py`

- File role: Minimal object tests for the H2 monitor-grid kinetic operator audit.
- Test helpers:
  - `_summary()`: Helper used by the tests in this file.
  - `_probe()`: Helper used by the tests in this file.
  - `_orbital_summary()`: Helper used by the tests in this file.
- Test cases:
  - `test_kinetic_route_result_fields_exist()`: kinetic route result fields exist
  - `test_kinetic_audit_result_fields_exist()`: kinetic audit result fields exist

### `tests/test_h2_monitor_grid_operator_audit.py`

- File role: Minimal object tests for the H2 monitor-grid operator audit layer.
- Test helpers:
  - `_expectation()`: Helper used by the tests in this file.
  - `_summary()`: Helper used by the tests in this file.
  - `_probe()`: Helper used by the tests in this file.
- Test cases:
  - `test_operator_route_result_fields_exist()`: operator route result fields exist
  - `test_operator_audit_result_fields_exist()`: operator audit result fields exist

### `tests/test_h2_monitor_grid_orbital_shape_audit.py`

- File role: Smoke tests for the H2 fixed-potential orbital-shape audit.
- Test cases:
  - `test_construct_h2_monitor_grid_orbital_shape_audit_result()`: construct h2 monitor grid orbital shape audit result

### `tests/test_h2_monitor_grid_patch_hartree_xc_audit.py`

- File role: Minimal smoke tests for the H2 monitor-grid patch Hartree/XC audit.
- Test cases:
  - `test_monitor_grid_patch_hartree_xc_components_are_finite()`: monitor grid patch hartree xc components are finite
  - `test_construct_static_local_route_result_object()`: construct static local route result object

### `tests/test_h2_monitor_grid_patch_local_audit.py`

- File role: Minimal smoke tests for the H2 monitor-grid local-GTH patch audit.
- Test cases:
  - `test_monitor_grid_local_patch_path_runs()`: monitor grid local patch path runs
  - `test_construct_patch_route_result_object()`: construct patch route result object

### `tests/test_h2_monitor_grid_poisson_operator_audit.py`

- File role: Minimal smoke tests for the A-grid Poisson-operator audit.
- Test cases:
  - `test_monitor_grid_poisson_operator_audit_module_imports()`: monitor grid poisson operator audit module imports
  - `test_small_monitor_grid_poisson_operator_route_is_finite()`: small monitor grid poisson operator route is finite
  - `test_construct_poisson_operator_difference_object()`: construct poisson operator difference object

### `tests/test_h2_monitor_grid_scf_dry_run_audit.py`

- File role: Minimal smoke tests for the A-grid H2 SCF dry-run audit.
- Test cases:
  - `test_h2_monitor_grid_scf_dry_run_module_imports()`: h2 monitor grid scf dry run module imports
  - `test_construct_h2_monitor_grid_scf_dry_run_result()`: construct h2 monitor grid scf dry run result

### `tests/test_h2_monitor_grid_singlet_stability_audit.py`

- File role: Tiny regression scaffolding for the singlet stability audit.
- Test cases:
  - `test_construct_h2_singlet_stability_result()`: construct h2 singlet stability result

### `tests/test_h2_monitor_grid_ts_eloc_audit.py`

- File role: Minimal smoke tests for the H2 monitor-grid `T_s + E_loc,ion` audit.
- Test cases:
  - `test_monitor_grid_kinetic_and_local_paths_run()`: monitor grid kinetic and local paths run
  - `test_h2_monitor_grid_ts_eloc_values_are_finite()`: h2 monitor grid ts eloc values are finite
  - `test_construct_ts_eloc_result_object()`: construct ts eloc result object

### `tests/test_h2_reference_placeholder.py`

- File role: Placeholder checks for the H2 PySCF audit modules.
- Test cases:
  - `test_h2_reference_module_imports()`: h2 reference module imports
  - `test_h2_basis_convergence_module_imports()`: h2 basis convergence module imports
  - `test_h2_audit_scripts_exist()`: h2 audit scripts exist

### `tests/test_h2_scf_driver.py`

- File role: Sanity checks for the first minimal H2 SCF single-point driver.
- Test helpers:
  - `_run_h2_scf()`: Helper used by the tests in this file.
- Test cases:
  - `test_scf_driver_modules_import()`: scf driver modules import
  - `test_h2_singlet_minimal_scf_runs()`: h2 singlet minimal scf runs
  - `test_h2_triplet_minimal_scf_runs()`: h2 triplet minimal scf runs
  - `test_h2_scf_energy_components_are_finite()`: h2 scf energy components are finite
  - `test_h2_scf_density_shapes_and_electron_counts_are_consistent()`: h2 scf density shapes and electron counts are consistent
  - `test_h2_scf_singlet_and_triplet_energies_are_finite()`: h2 scf singlet and triplet energies are finite

### `tests/test_h2_vs_pyscf_audit.py`

- File role: Minimal smoke tests for the H2 vs PySCF audit layer.
- Test cases:
  - `test_construct_spin_comparison_result()`: construct spin comparison result
  - `test_construct_gap_and_scan_result()`: construct gap and scan result

### `tests/test_hartree_poisson.py`

- File role: Sanity checks for the first Hartree and open-boundary Poisson slice.
- Test helpers:
  - `_boundary_mask()`: Helper used by the tests in this file.
  - `_boundary_values_from_moments()`: Helper used by the tests in this file.
  - `_direct_selected_region_boundary_delta()`: Helper used by the tests in this file.
- Test cases:
  - `test_poisson_and_hartree_modules_import()`: poisson and hartree modules import
  - `test_hartree_potential_runs_for_simple_positive_density()`: hartree potential runs for simple positive density
  - `test_default_h2_trial_density_hartree_potential_shape_and_symmetry()`: default h2 trial density hartree potential shape and symmetry
  - `test_hartree_action_and_energy_are_finite()`: hartree action and energy are finite
  - `test_static_ks_with_hartree_runs_and_is_finite()`: static ks with hartree runs and is finite
  - `test_monitor_grid_multipole_boundary_reduces_centered_gaussian_fake_quadrupole()`: monitor grid multipole boundary reduces centered gaussian fake quadrupole
  - `test_monitor_grid_multipole_boundary_keeps_centered_gaussian_charge_close_to_two()`: monitor grid multipole boundary keeps centered gaussian charge close to two
  - `test_monitor_grid_boundary_values_track_corrected_source_for_shifted_gaussian()`: monitor grid boundary values track corrected source for shifted gaussian
  - `test_monitor_grid_nodal_region_moments_recover_full_nodal_moments_for_all_cells()`: monitor grid nodal region moments recover full nodal moments for all cells

### `tests/test_imports.py`

- File role: Import smoke tests for the package skeleton and direct entry points.
- Test cases:
  - `test_import_isogrid()`: import isogrid
  - `test_import_default_h2_config()`: import default h2 config
  - `test_import_grid_entrypoint()`: import grid entrypoint
  - `test_import_pseudo_entrypoint()`: import pseudo entrypoint
  - `test_import_ops_entrypoint()`: import ops entrypoint
  - `test_import_poisson_entrypoint()`: import poisson entrypoint
  - `test_import_local_hamiltonian_entrypoint()`: import local hamiltonian entrypoint
  - `test_import_eigensolver_jax_cache_entrypoint()`: import eigensolver jax cache entrypoint
  - `test_import_nonlocal_entrypoint()`: import nonlocal entrypoint
  - `test_import_lsda_entrypoint()`: import lsda entrypoint
  - `test_import_static_ks_entrypoint()`: import static ks entrypoint
  - `test_import_fixed_potential_eigensolver_entrypoint()`: import fixed potential eigensolver entrypoint
  - `test_import_fixed_potential_static_local_eigensolver_entrypoint()`: import fixed potential static local eigensolver entrypoint
  - `test_import_scf_driver_entrypoint()`: import scf driver entrypoint
  - `test_import_h2_vs_pyscf_audit_entrypoint()`: import h2 vs pyscf audit entrypoint
  - `test_import_h2_grid_convergence_audit_entrypoint()`: import h2 grid convergence audit entrypoint
  - `test_import_h2_regression_baseline()`: import h2 regression baseline
  - `test_import_monitor_grid_entrypoint()`: import monitor grid entrypoint
  - `test_import_monitor_grid_audit_entrypoint()`: import monitor grid audit entrypoint
  - `test_import_monitor_grid_ts_eloc_audit_entrypoint()`: import monitor grid ts eloc audit entrypoint
  - `test_import_monitor_grid_fair_calibration_audit_entrypoint()`: import monitor grid fair calibration audit entrypoint
  - `test_import_monitor_grid_patch_local_audit_entrypoint()`: import monitor grid patch local audit entrypoint
  - `test_import_monitor_grid_patch_hartree_xc_audit_entrypoint()`: import monitor grid patch hartree xc audit entrypoint
  - `test_import_h2_hartree_poisson_comparison_audit_entrypoint()`: import h2 hartree poisson comparison audit entrypoint
  - `test_import_h2_monitor_grid_poisson_operator_audit_entrypoint()`: import h2 monitor grid poisson operator audit entrypoint
  - `test_import_h2_hartree_tail_recheck_audit_entrypoint()`: import h2 hartree tail recheck audit entrypoint
  - `test_import_h2_monitor_grid_fixed_potential_eigensolver_audit_entrypoint()`: import h2 monitor grid fixed potential eigensolver audit entrypoint
  - `test_import_h2_fixed_potential_eigensolver_baseline()`: import h2 fixed potential eigensolver baseline
  - `test_import_h2_monitor_grid_operator_audit_entrypoint()`: import h2 monitor grid operator audit entrypoint
  - `test_import_h2_fixed_potential_operator_baseline()`: import h2 fixed potential operator baseline
  - `test_import_h2_monitor_grid_kinetic_operator_audit_entrypoint()`: import h2 monitor grid kinetic operator audit entrypoint
  - `test_import_h2_kinetic_operator_baseline()`: import h2 kinetic operator baseline
  - `test_import_h2_monitor_grid_kinetic_form_audit_entrypoint()`: import h2 monitor grid kinetic form audit entrypoint
  - `test_import_h2_kinetic_form_baseline()`: import h2 kinetic form baseline
  - `test_import_h2_monitor_grid_geometry_consistency_audit_entrypoint()`: import h2 monitor grid geometry consistency audit entrypoint
  - `test_import_h2_geometry_consistency_baseline()`: import h2 geometry consistency baseline
  - `test_import_h2_monitor_grid_kinetic_green_identity_audit_entrypoint()`: import h2 monitor grid kinetic green identity audit entrypoint
  - `test_import_h2_kinetic_green_identity_baseline()`: import h2 kinetic green identity baseline
  - `test_import_h2_monitor_grid_orbital_shape_audit_entrypoint()`: import h2 monitor grid orbital shape audit entrypoint
  - `test_import_h2_orbital_shape_baseline()`: import h2 orbital shape baseline
  - `test_import_h2_monitor_grid_k2_subspace_audit_entrypoint()`: import h2 monitor grid k2 subspace audit entrypoint
  - `test_import_h2_k2_subspace_baseline()`: import h2 k2 subspace baseline
  - `test_import_h2_monitor_grid_scf_dry_run_audit_entrypoint()`: import h2 monitor grid scf dry run audit entrypoint
  - `test_import_h2_scf_dry_run_baseline()`: import h2 scf dry run baseline
  - `test_import_h2_monitor_grid_singlet_stability_audit_entrypoint()`: import h2 monitor grid singlet stability audit entrypoint
  - `test_import_h2_singlet_stability_baseline()`: import h2 singlet stability baseline
  - `test_import_h2_monitor_grid_diis_scf_audit_entrypoint()`: import h2 monitor grid diis scf audit entrypoint
  - `test_import_h2_diis_scf_baseline()`: import h2 diis scf baseline
  - `test_import_h2_jax_kernel_consistency_audit_entrypoint()`: import h2 jax kernel consistency audit entrypoint
  - `test_import_h2_jax_kernel_consistency_baseline()`: import h2 jax kernel consistency baseline
  - `test_import_h2_jax_eigensolver_hotpath_audit_entrypoint()`: import h2 jax eigensolver hotpath audit entrypoint
  - `test_import_h2_jax_triplet_reintegration_smoke_audit_entrypoint()`: import h2 jax triplet reintegration smoke audit entrypoint
  - `test_import_h2_jax_triplet_end_to_end_micro_profile_audit_entrypoint()`: import h2 jax triplet end to end micro profile audit entrypoint
  - `test_import_h2_jax_eigensolver_hotpath_baseline()`: import h2 jax eigensolver hotpath baseline
  - `test_import_h2_jax_scf_hotpath_audit_entrypoint()`: import h2 jax scf hotpath audit entrypoint
  - `test_import_h2_jax_scf_hotpath_baseline()`: import h2 jax scf hotpath baseline
  - `test_import_h2_jax_triplet_hartree_energy_audit_entrypoint()`: import h2 jax triplet hartree energy audit entrypoint
  - `test_import_h2_jax_singlet_mainline_audit_entrypoint()`: import h2 jax singlet mainline audit entrypoint
  - `test_import_h2_jax_triplet_hartree_energy_baseline()`: import h2 jax triplet hartree energy baseline
  - `test_import_h2_jax_singlet_mainline_baseline()`: import h2 jax singlet mainline baseline

### `tests/test_local_hamiltonian.py`

- File role: Sanity checks for the first local-Hamiltonian slice.
- Test helpers:
  - `_build_symmetric_trial_orbital()`: Helper used by the tests in this file.
- Test cases:
  - `test_ops_and_local_hamiltonian_audit_modules_import()`: ops and local hamiltonian audit modules import
  - `test_kinetic_operator_runs_on_default_h2_trial_orbital()`: kinetic operator runs on default h2 trial orbital
  - `test_apply_local_hamiltonian_matches_input_shape_and_is_finite()`: apply local hamiltonian matches input shape and is finite
  - `test_constant_field_has_zero_interior_kinetic_action()`: constant field has zero interior kinetic action
  - `test_local_hamiltonian_preserves_basic_h2_mirror_symmetry()`: local hamiltonian preserves basic h2 mirror symmetry

### `tests/test_monitor_geometry.py`

- File role: Focused tests for monitor-grid cell-local geometry helpers.
- Test cases:
  - `test_monitor_cell_local_quadrature_recovers_box_volume_better_than_nodal_cell_volumes()`: monitor cell local quadrature recovers box volume better than nodal cell volumes

### `tests/test_monitor_grid.py`

- File role: Smoke tests for the new 3D monitor-driven grid core.
- Test helpers:
  - `_assert_valid_monitor_geometry()`: Helper used by the tests in this file.
- Test cases:
  - `test_build_h2_monitor_geometry()`: build h2 monitor geometry
  - `test_build_n2_monitor_geometry()`: build n2 monitor geometry
  - `test_build_co_monitor_geometry()`: build co monitor geometry
  - `test_build_h2o_monitor_geometry()`: build h2o monitor geometry

### `tests/test_static_ks_hamiltonian.py`

- File role: Sanity checks for the first static KS Hamiltonian slice.
- Test cases:
  - `test_nonlocal_lsda_and_static_ks_modules_import()`: nonlocal lsda and static ks modules import
  - `test_nonlocal_action_runs_on_default_grid_for_supported_element()`: nonlocal action runs on default grid for supported element
  - `test_lsda_terms_are_finite_for_positive_spin_density()`: lsda terms are finite for positive spin density
  - `test_apply_static_ks_hamiltonian_matches_input_shape_and_is_finite()`: apply static ks hamiltonian matches input shape and is finite
  - `test_static_ks_preserves_basic_h2_mirror_symmetry()`: static ks preserves basic h2 mirror symmetry

