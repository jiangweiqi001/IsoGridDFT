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
```

