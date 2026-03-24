"""Sanity checks for the first GTH local-potential slice."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from isogrid.grid import build_default_h2_grid_geometry
from isogrid.pseudo import build_default_h2_local_ionic_potential
from isogrid.pseudo import load_gth_pseudo_data


def test_pseudo_module_and_audit_imports() -> None:
    pseudo_module = import_module("isogrid.pseudo")
    audit_module = import_module("isogrid.audit.gth_local_h2_audit")

    assert hasattr(pseudo_module, "evaluate_local_ionic_potential")
    assert hasattr(audit_module, "main")


def test_hcno_gth_data_can_be_loaded() -> None:
    for symbol in ("H", "C", "N", "O"):
        pseudo_data = load_gth_pseudo_data(symbol)
        assert pseudo_data.element == symbol
        assert pseudo_data.family == "gth-pade"
        assert pseudo_data.ionic_charge > 0


def test_unsupported_element_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="supports only H, C, N, O"):
        load_gth_pseudo_data("F")


def test_unsupported_pseudo_family_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="supports only the `gth-pade` family"):
        load_gth_pseudo_data("H", family="gth-blyp")


def test_default_h2_local_potential_can_be_constructed() -> None:
    evaluation = build_default_h2_local_ionic_potential()

    assert len(evaluation.atom_contributions) == 2


def test_local_potential_shape_matches_grid_shape() -> None:
    grid_geometry = build_default_h2_grid_geometry()
    evaluation = build_default_h2_local_ionic_potential(grid_geometry=grid_geometry)

    assert evaluation.total_local_potential.shape == grid_geometry.spec.shape
    for contribution in evaluation.atom_contributions:
        assert contribution.local_potential.shape == grid_geometry.spec.shape


def test_local_potential_values_are_finite() -> None:
    evaluation = build_default_h2_local_ionic_potential()

    assert np.all(np.isfinite(evaluation.total_local_potential))


def test_default_h2_local_potential_is_mirror_symmetric() -> None:
    evaluation = build_default_h2_local_ionic_potential()

    assert np.allclose(
        evaluation.total_local_potential,
        evaluation.total_local_potential[:, :, ::-1],
    )


def test_default_h2_local_potential_is_deeper_near_nuclei_than_far_field() -> None:
    evaluation = build_default_h2_local_ionic_potential()
    grid_geometry = evaluation.grid_geometry
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    centerline = evaluation.total_local_potential[center_ix, center_iy, :]

    assert centerline.min() < centerline[0]
    assert centerline.min() < centerline[-1]
