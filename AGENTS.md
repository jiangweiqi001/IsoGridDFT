# AGENTS.md

## Project
IsoGridDFT

Adaptive real-space Kohn–Sham DFT for isolated molecules.

## Mission

Build a research-grade prototype for isolated molecular systems with:

- adaptive structured real-space grid
- free-space / open-boundary electrostatics
- GTH pseudopotentials
- spin-polarized LSDA
- iterative eigensolver inside standard SCF
- automatic comparison of candidate spin states when requested
- initial benchmark target: H2 at R = 1.4 Bohr

This project is not a general-purpose electronic-structure package yet.
Do not optimize for broad feature coverage before the first physical closed loop works.

## Phase priorities

### Phase 0: first real closed loop
The first mandatory closed loop must include:

- open boundary
- adaptive structured grid
- GTH
- LSDA
- iterative eigensolver
- SCF
- H2 singlet/triplet comparison
- single-point total energy output

The first acceptance target is:
- H2, R = 1.4 Bohr
- single-point total energy
- comparable to PySCF reference under the same physical model

### Phase 1: benchmark expansion
After the first H2 closed loop:
- H2 bond scan
- N2 bond scan
- H2O bond stretch / bond breaking scans

### Phase 2: D3(BJ)
D3(BJ) must be designed into interfaces from the beginning,
but it does NOT need to be active in the very first H2 closed loop.

## Scientific constraints

### Target systems
First stage only:
- neutral molecules
- elements: H, C, N, O
- includes open-shell radicals
- includes stretched / near-dissociation geometries
- includes weak intermolecular interactions later in stage 1/2

### Accuracy targets
Hard acceptance target for stage 1:
- absolute total energy error vs converged PySCF reference < 1 mHa
- relative energy difference error vs converged PySCF reference < 1 mHa

For the first closed loop, force accuracy is NOT a formal acceptance requirement.
Do not redesign the code around force evaluation yet.

### Reference model
PySCF reference must use:
- the same GTH pseudopotential definition
- LSDA
- converged Gaussian-basis reference as far as practical
- same spin state definition
- same physical model whenever comparison is claimed

Never claim agreement if the reference model is not aligned.

## Numerical design rules

### Grid
Use:
- adaptive structured grid as the main representation
- geometry-driven adaptation in the first version
- leave room for future density-driven adaptation

Do NOT:
- switch the main representation to a uniform grid
- use a true unstructured FEM-style mesh in stage 1
- let adaptive remeshing inside SCF become a blocker for the first closed loop

### Electrostatics
The intended route is:
- free-space / open-boundary Poisson
- native to the adaptive main grid

Do NOT:
- solve Hartree by projecting to an auxiliary uniform grid in stage 1
- silently replace open-boundary physics with large-box Dirichlet as the main method

### Pseudopotential
Use GTH only in stage 1.
Treat local and nonlocal terms as first-class objects.
Near-core integration may use atom-centered auxiliary refinement ideas,
but do not replace the main grid representation.

### Exchange-correlation
Start with spin-polarized LSDA only.
Do not add GGA before LSDA + GTH + open-boundary + SCF are validated.

### Occupations
Smearing may be used only as a convergence aid.
Final reported energies for acceptance should correspond as closely as possible
to the zero-smearing / zero-temperature definition.

## Software architecture rules

### Language / backend split
Use:
- Python as orchestration layer
- JAX as the main numerical backend for heavy array kernels and GPU execution
- NumPy / SciPy as audit tools, CPU fallback, and prototype-side utilities

Do NOT:
- make SciPy the main hot-path backend
- put the principal GPU path behind Python callbacks
- mix JAX and SciPy inside tight iterative hot loops unless explicitly justified

### Ownership boundary
Main production path:
- Python driver
- JAX numerical kernels
- GPU-first, CPU fallback available

SciPy is allowed for:
- audit code
- small-scale fallback solvers
- regression checks
- debugging and validation utilities

### Precision
Use float64 throughout the scientific path unless there is a clearly documented reason not to.
Never silently downgrade precision in benchmark or validation code.

## Engineering rules

### Change style
Prefer small, reviewable patches.
Each patch should do one of:
- add one well-defined module
- connect one interface
- add one benchmark or regression
- fix one clearly identified numerical defect

Do NOT mix:
- refactor
- new physics
- performance optimization
- benchmark rewrites
in one patch unless required.

### Before coding
Before changing code, always state:
1. what physical/numerical invariant must remain true
2. what module boundary is being introduced or changed
3. how the change will be validated

### After coding
After every nontrivial change, always report:
- files changed
- invariant checked
- tests run
- remaining risks / unknowns

### Tests
Prefer the following order:
1. unit test for math/operator behavior
2. tiny physical sanity test
3. regression test against previous accepted result
4. comparison to PySCF reference

### Benchmark discipline
Never claim success from “curve looks good”.
Always quantify:
- absolute total energy error
- relative energy error
- convergence behavior
- spin-state comparison if relevant

## Immediate roadmap

### first mandatory milestone
Implement the minimum path needed for:

- H2
- R = 1.4 Bohr
- singlet and triplet comparison
- SCF convergence
- total energy output
- comparable PySCF reference script

### preferred module order
1. config / runtime
2. grid mapping + metrics
3. local differential operators
4. GTH local + nonlocal operator application
5. LSDA kernel
6. free-space Poisson operator
7. KS Hamiltonian apply
8. iterative eigensolver scaffold
9. SCF driver
10. PySCF audit script
11. H2 benchmark test

## Communication style for Codex

When proposing changes:
- be explicit
- do not hand-wave physics
- separate confirmed facts from assumptions
- call out any place where the current implementation is only a temporary approximation

When blocked:
- do not expand scope blindly
- do not add unrelated abstractions
- instead propose the smallest next step that preserves the roadmap

When uncertain:
- prefer correctness, debuggability, and scientific auditability over cleverness

## Out of scope for now

Do NOT prioritize:
- periodic systems
- geometry optimization
- forces as a formal acceptance target
- hybrids
- meta-GGA
- transition metals
- production-scale datasets
- large-molecule scaling
- full black-box automation

## Naming
Repository name:
- IsoGridDFT

Python package name:
- isogrid