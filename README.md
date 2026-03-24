# IsoGridDFT

IsoGridDFT is an adaptive real-space Kohn-Sham DFT project for isolated molecules.

## Scope

- Stage-1 target systems are neutral molecules built from H, C, N, and O.
- The first formal closed-loop target is H2 at `R = 1.4 Bohr`.
- The intended stage-1 physical route is a structured adaptive grid, open-boundary electrostatics, GTH pseudopotentials, and LSDA.
- PySCF is used as the reference and audit baseline before the real-space solver is available.

## Current Status

The repository is still in the groundwork stage. The real solver is not implemented yet: there is no production open-boundary Poisson solver, no GTH real-space Hamiltonian, no LSDA main path, no full KS Hamiltonian path, and no SCF driver yet.

What is present today:

- a minimal `src/isogrid/` package skeleton
- a formal `isogrid.config` layer for benchmark defaults and JAX runtime setup
- a first structured adaptive grid geometry and mapping layer for geometry-driven structured grids
- a first GTH data layer and local ionic pseudopotential slice for H, C, N, and O with `gth-pade`
- a first local-Hamiltonian slice with a default kinetic operator and local ionic potential action
- a `PySCF` audit baseline for H2 at `R = 1.4 Bohr`
- a `PySCF` basis-sequence audit script for reference-side convergence checks
- placeholder and sanity tests for imports, audit modules, grid geometry, local GTH potentials, and the local Hamiltonian slice

## Audit Baseline

The initial audit scripts live in `src/isogrid/audit/`.

They currently cover:

- H2 singlet and triplet comparison at `R = 1.4 Bohr`
- a basis-sequence scan for PySCF-side reference convergence checks
- a local GTH ionic-potential audit slice on the default H2 structured grid
- a local-Hamiltonian trial-orbital audit on the default H2 structured grid

These scripts are intended to support the first formal H2 closed loop, not to replace the future real-space solver.

## Grid Geometry Baseline

The initial structured grid layer lives in `src/isogrid/grid/`.

It currently provides:

- structured logical-to-physical coordinate mapping
- geometry-driven center-fine and far-field-coarse stretching
- 3D grid point coordinates and minimal geometric weights

It does not yet implement the final production differential operators, Poisson, KS Hamiltonians, or SCF.

## Pseudopotential Baseline

The current `src/isogrid/pseudo/` layer provides only the first local GTH slice.

It currently provides:

- internal GTH data objects for H, C, N, O with `gth-pade`
- local ionic pseudopotential evaluation on the structured grid
- clear placeholders for future nonlocal projector work

It does not yet implement nonlocal projector action, Hartree, full KS Hamiltonians, or SCF.

## Minimal Setup

```bash
pip install -e .[test]
```

To run the PySCF and local-Hamiltonian audit scripts, install PySCF as well:

```bash
pip install -e .[audit,test]
python -m isogrid.audit.pyscf_h2_reference
python -m isogrid.audit.pyscf_h2_basis_convergence
python -m isogrid.audit.gth_local_h2_audit
python -m isogrid.audit.local_hamiltonian_h2_trial_audit
```
