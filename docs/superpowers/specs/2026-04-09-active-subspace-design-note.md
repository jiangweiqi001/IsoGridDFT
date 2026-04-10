# Active-Subspace Design Note

## Scope

First implementation only changes the `local-only H2 singlet` path. It does not
change `triplet`, `nonlocal`, `full-physics`, or default mainline routing.

## Main Idea

The SCF singlet path should stop treating a single tracked occupied orbital as
its only continuity state. Instead it should track:

- a fixed-size active subspace
- the occupied direction selected inside that subspace

The first version keeps `active_subspace_size = 2`, but the interface is
written so that later experiments can raise it above `2` without rewriting the
driver control flow.

## Module Boundary

Add `src/isogrid/scf/active_subspace.py` with:

- `ActiveSubspaceConfig`
- `ActiveSubspaceState`
- `ActiveSubspaceSelectionResult`
- `initialize_active_subspace(...)`
- `update_active_subspace(...)`

The driver remains responsible for eigensolver calls and density rebuilding.
The new module is only responsible for aligning the raw lowest-`m` block to the
stored reference subspace and selecting the continuity-preserving occupied
direction inside it.

## First-Round Acceptance

- `xlarge` should expose “subspace still present but occupied direction flips”
  through the active-subspace diagnostics.
- `xxlarge` should expose “current subspace size is too small” rather than
  hiding it as an ordinary tracked-orbital continuity failure.
- Existing small/large/xlarge local-only short-run behavior should not regress.

## Follow-Up Note

This design note captured the first active-subspace upgrade only. The current
experimental branch has already moved one layer higher:

- `local-only H2 singlet` still uses the active-subspace alignment machinery
- but it now also has an experimental `projector_mixing` route that treats the
  occupied projector / density response as an explicit object
- and a narrower `guarded_projector_mixing` wrapper that auto-enables that
  route only on strong late-step plateau signatures

So this note is still historically useful, but it no longer describes the full
current experimental singlet-response stack by itself.
