# IsoGridDFT

IsoGridDFT is an adaptive real-space Kohn-Sham DFT project for isolated molecules.

## Scope

- Stage-1 target systems are neutral molecules built from H, C, N, and O.
- The first formal closed-loop target is H2 at `R = 1.4 Bohr`.
- The intended stage-1 physical route is a structured adaptive grid, open-boundary electrostatics, GTH pseudopotentials, and LSDA.
- PySCF is used as the reference and audit baseline before the real-space solver is available.

## Current Status

The repository is still in the groundwork stage. The real solver is not implemented yet: there is no production open-boundary Poisson solver, no GTH real-space Hamiltonian, no LSDA main path, no KS Hamiltonian path, and no SCF driver yet.

What is present today:

- a minimal `src/isogrid/` package skeleton
- a formal `isogrid.config` layer for benchmark defaults and JAX runtime setup
- a `PySCF` audit baseline for H2 at `R = 1.4 Bohr`
- a `PySCF` basis-sequence audit script for reference-side convergence checks
- placeholder tests that only check imports and module presence

## Audit Baseline

The initial audit scripts live in `src/isogrid/audit/`.

They currently cover:

- H2 singlet and triplet comparison at `R = 1.4 Bohr`
- a basis-sequence scan for PySCF-side reference convergence checks

These scripts are intended to support the first formal H2 closed loop, not to replace the future real-space solver.

## Minimal Setup

```bash
pip install -e .[test]
```

To run the PySCF audit scripts, install PySCF as well:

```bash
pip install -e .[audit,test]
python -m isogrid.audit.pyscf_h2_reference
python -m isogrid.audit.pyscf_h2_basis_convergence
```
