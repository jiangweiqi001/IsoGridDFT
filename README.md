# IsoGridDFT

IsoGridDFT is an adaptive real-space Kohn-Sham DFT project for isolated molecules.

## Scope

- Stage-1 target systems are neutral molecules built from H, C, N, and O.
- The first formal closed-loop target is H2 at `R = 1.4 Bohr`.
- The intended stage-1 physical route is a structured adaptive grid, open-boundary electrostatics, GTH pseudopotentials, and LSDA.
- PySCF is used as the reference and audit baseline before the real-space solver is available.

## Current Status

The repository is still in the groundwork stage. A first minimal H2 single-point SCF closed loop now exists, but it is still a development implementation: the convergence strategy, accuracy, and broader benchmark coverage still need more audit work.

The current single-center `sinh` grid should now be treated as a transitional baseline. The next planned phase is a redesign of the main grid representation toward an atom-centered monitor-driven structured grid, guided by the H2 error-localization audits.

The first monitor-driven main-grid core is now being implemented. The legacy single-center `sinh` grid remains available only as a baseline, while the new 3D monitor grid is not yet connected to the downstream physical operators.

The first physical reconnect on the new A-grid is now focused on `T_s + E_loc,ion`, specifically to test whether the dominant H2 singlet grid error drops before the rest of the Hamiltonian is migrated.

The current monitor-grid work has now moved on to a fair resolution-calibration audit against the legacy grid, with the immediate goal of checking whether `T_s + E_loc,ion` improves once the A-grid is no longer allowed to be smaller or coarser near the H nuclei.

The next A-grid step is now the first patch-assisted local-GTH near-core correction on top of the best fair H2 monitor-grid baseline. This still targets only `T_s + E_loc,ion`; nonlocal, Hartree, XC, and SCF remain on their current paths.

The current A-grid patch work has now moved one step further into the static local-chain audit: `T_s + E_loc,ion + E_H + E_xc` can now be compared between the legacy grid, the raw A-grid, and the A-grid plus local-GTH patch correction. Nonlocal, eigensolver, and SCF still remain on their current paths.

The current focus has now shifted to a dedicated A-grid Hartree / open-boundary Poisson comparison audit, because the local-GTH patch already compresses the `E_loc,ion` gap while the remaining static-local mismatch is now dominated by `E_H`.

The next audit slice now drills one level deeper into the A-grid Poisson operator itself: the current focus is to localize why the monitor-grid Hartree path develops negative far-field `v_H` values and a large `E_H` deficit even when the density integral, multipole boundary order, and solver residual all look well behaved.

That boundary-split / RHS consistency defect has now received a first targeted fix on the monitor-grid Poisson path, and the current operator status should be read from the dedicated Poisson audit outputs rather than inferred from the older broken monitor-grid baseline.

With that monitor Poisson split fix in place, the H2 static local-chain audit has now been rerun on `legacy / A-grid / A-grid+patch`, while nonlocal, eigensolver, and SCF still remain on their current non-monitor paths.

That repaired H2 static local-chain result is now frozen as the current A-grid+patch regression baseline, and the present follow-up is a very small Hartree tail recheck to see whether the remaining `E_H` offset behaves more like geometry/discretization tail than a surviving monitor-Poisson system bias.

The next A-grid handoff has now reached the fixed-potential eigensolver on the repaired static local chain. That migration is auditable, but it is not yet numerically stable enough to replace the legacy fixed-potential path; nonlocal, eigensolver production use, and SCF still remain on their current non-monitor routes.

The current focus is now an operator-level audit of the A-grid static-local fixed-potential path. The static local chain itself is established, but the A-grid fixed-potential eigensolver still fails and is being diagnosed before any SCF migration is attempted.

The current diagnostic slice is now even narrower: the immediate question is whether the A-grid kinetic sub-operator itself is admitting pathological fixed-potential modes, while nonlocal, eigensolver production use, and SCF still remain off the A-grid path.

What is present today:

- a minimal `src/isogrid/` package skeleton
- a formal `isogrid.config` layer for benchmark defaults, minimal audit molecule sets, and JAX runtime setup
- a first structured adaptive grid geometry and mapping layer for geometry-driven structured grids
- a first GTH data layer with both local ionic and first-stage nonlocal projector slices for H, C, N, and O with `gth-pade`
- a first open-boundary Poisson and Hartree slice on the structured adaptive grid
- a first static KS Hamiltonian slice that connects kinetic, GTH local ionic, GTH nonlocal ionic, Hartree, and LSDA local terms without SCF
- a first fixed-potential static-KS eigensolver scaffold that extracts the lowest few orbitals under frozen density and frozen potentials
- a first minimal H2 SCF single-point driver for the singlet and triplet candidates
- a first quantitative H2-vs-PySCF error audit for the singlet/triplet single-point energies and their relative gap
- a first H2 singlet grid/box convergence audit that scans geometry-discretization choices and tracks energy-component drift
- a `PySCF` audit baseline for H2 at `R = 1.4 Bohr`
- a `PySCF` basis-sequence audit script for reference-side convergence checks
- placeholder and sanity tests for imports, audit modules, grid geometry, GTH potentials, Hartree, and the static KS slice

## Audit Baseline

The initial audit scripts live in `src/isogrid/audit/`.

They currently cover:

- H2 singlet and triplet comparison at `R = 1.4 Bohr`
- a basis-sequence scan for PySCF-side reference convergence checks
- a local GTH ionic-potential audit slice on the default H2 structured grid
- a local-Hamiltonian trial-orbital audit on the default H2 structured grid
- a static-KS trial-orbital audit on the default H2 structured grid
- a static-KS-with-Hartree audit on the default H2 structured grid
- a fixed-potential static-KS eigensolver audit on the default H2 structured grid
- a minimal H2 SCF single-point audit for the singlet and triplet candidates
- a quantitative H2 vs PySCF audit with a small parameter-sensitivity scan for error localization
- a formal H2 singlet grid/domain convergence audit for separating resolution sensitivity from finite-box sensitivity
- a formal H2 singlet A-grid fair-calibration audit for matching legacy box size and near-core resolution before comparing `T_s + E_loc,ion`
- a formal H2 singlet A-grid local-GTH patch audit for near-core correction on the best fair A-grid baseline
- a first H2 A-grid+patch fixed-potential eigensolver audit on the repaired static local chain
- a dedicated H2 A-grid static-local operator audit for diagnosing the current fixed-potential eigensolver failure
- a dedicated H2 A-grid kinetic-operator audit for diagnosing the current fixed-potential negative-kinetic failure mode
- a lightweight recorded H2 regression baseline for future PySCF error comparisons

These scripts are intended to support the first formal H2 closed loop, not to replace the future real-space solver.

## Grid Geometry Baseline

The initial structured grid layer lives in `src/isogrid/grid/`.

It currently provides:

- structured logical-to-physical coordinate mapping
- geometry-driven center-fine and far-field-coarse stretching
- 3D grid point coordinates and minimal geometric weights

It does not yet implement the final production adaptive strategy, SCF, or geometry optimization.

## Pseudopotential Baseline

The current `src/isogrid/pseudo/` layer provides the first local and nonlocal GTH slices.

It currently provides:

- internal GTH data objects for H, C, N, O with `gth-pade`
- local ionic pseudopotential evaluation on the structured grid
- first-stage real-space nonlocal projector evaluation and `V_nl psi` application

It does not yet implement the final production nonlocal path or a full SCF-ready Hamiltonian application stack.

## Electrostatics Baseline

The current `src/isogrid/poisson/` layer provides the first Hartree path.

It currently provides:

- a finite-domain Poisson solve on the structured adaptive grid
- free-space boundary values approximated by a low-order multipole expansion
- Hartree potential, Hartree action, and Hartree energy helpers

It does not yet implement the final production free-space solver or SCF coupling.

## Exchange-Correlation Baseline

The current `src/isogrid/xc/` layer provides only the first LSDA slice.

It currently provides:

- spin-polarized `lda,vwn` local energy-density and potential evaluation
- explicit alignment with the current PySCF audit baseline

It does not yet implement GGA, meta-GGA, hybrid functionals, or a standalone internal libxc replacement.

## Minimal Setup

```bash
pip install -e .[test]
```

To run the PySCF, pseudopotential, Hartree, and static-KS audit scripts, install PySCF as well:

```bash
pip install -e .[audit,test]
python -m isogrid.audit.pyscf_h2_reference
python -m isogrid.audit.pyscf_h2_basis_convergence
python -m isogrid.audit.gth_local_h2_audit
python -m isogrid.audit.local_hamiltonian_h2_trial_audit
python -m isogrid.audit.static_ks_h2_trial_audit
python -m isogrid.audit.static_ks_h2_hartree_audit
python -m isogrid.audit.fixed_potential_h2_eigensolver_audit
python -m isogrid.audit.h2_scf_single_point_audit
python -m isogrid.audit.h2_vs_pyscf_audit
python -m isogrid.audit.h2_grid_convergence_audit
python -m isogrid.audit.h2_monitor_grid_fair_calibration_audit
python -m isogrid.audit.h2_monitor_grid_patch_local_audit
python -m isogrid.audit.h2_monitor_grid_patch_hartree_xc_audit
python -m isogrid.audit.h2_hartree_poisson_comparison_audit
python -m isogrid.audit.h2_monitor_grid_poisson_operator_audit
python -m isogrid.audit.h2_monitor_grid_operator_audit
python -m isogrid.audit.h2_monitor_grid_kinetic_operator_audit
```

