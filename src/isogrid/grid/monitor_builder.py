"""Builders for the 3D atom-centered monitor grid."""

from __future__ import annotations

from dataclasses import replace

from isogrid.config import BenchmarkCase
from isogrid.config import CO_AUDIT_CASE
from isogrid.config import H2O_AUDIT_CASE
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.config import N2_AUDIT_CASE

from .monitor_geometry import generate_monitor_grid_geometry
from .monitor_model import MonitorGridGeometry
from .monitor_model import MonitorGridSpec
from .monitor_model import NearCoreElementParameters

_DEFAULT_LOGICAL_BOUNDS = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
_DEFAULT_MONITOR_SHAPE_DIATOMIC = (29, 29, 29)
_DEFAULT_MONITOR_SHAPE_POLYATOMIC = (31, 31, 31)
H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE = (67, 67, 81)
H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR = (8.0, 8.0, 10.0)
H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE = 4.0
H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE = 0.70
_GTH_MONITOR_SEED_DATA = {
    "H": {
        "ionic_charge": 1,
        "rloc": 0.2,
        "coefficients": (-4.1802368, 0.72507482),
        "projector_radii": (),
    },
    "C": {
        "ionic_charge": 4,
        "rloc": 0.34883045,
        "coefficients": (-8.5137711, 1.22843203),
        "projector_radii": (0.30455321,),
    },
    "N": {
        "ionic_charge": 5,
        "rloc": 0.28917923,
        "coefficients": (-12.23481988, 1.76640728),
        "projector_radii": (0.25660487,),
    },
    "O": {
        "ionic_charge": 6,
        "rloc": 0.24762086,
        "coefficients": (-16.58031797, 2.39570092),
        "projector_radii": (0.22178614,),
    },
}


def _geometry_center(case: BenchmarkCase) -> tuple[float, float, float]:
    atoms = case.geometry.atoms
    count = float(len(atoms))
    return (
        sum(atom.position[0] for atom in atoms) / count,
        sum(atom.position[1] for atom in atoms) / count,
        sum(atom.position[2] for atom in atoms) / count,
    )


def build_default_near_core_element_parameters(
    case: BenchmarkCase,
) -> dict[str, NearCoreElementParameters]:
    """Build first-stage element parameters from GTH-inspired seed metadata."""

    parameters = {}
    for atom in case.geometry.atoms:
        element = atom.element
        if element in parameters:
            continue
        if element not in _GTH_MONITOR_SEED_DATA:
            raise ValueError(
                "The current monitor-grid core supports only GTH-inspired H/C/N/O seeds; "
                f"received `{element}`."
            )
        seed = _GTH_MONITOR_SEED_DATA[element]
        ionic_charge = int(seed["ionic_charge"])
        rloc = float(seed["rloc"])
        projector_radii = tuple(float(value) for value in seed["projector_radii"])
        max_nonlocal_radius = max(projector_radii, default=1.5 * rloc)
        coefficient_scale = sum(abs(value) for value in seed["coefficients"])
        projector_weight = 0.45 * ionic_charge if projector_radii else 0.0
        parameters[element] = NearCoreElementParameters(
            element=element,
            near_core_radius=max(0.35, 1.8 * rloc + 0.15),
            local_radius=max(0.45, 2.5 * rloc + 0.15),
            projector_radius=max(0.55, 2.4 * max_nonlocal_radius),
            patch_radius=max(1.0, 4.0 * max(rloc, max_nonlocal_radius) + 0.25),
            kinetic_weight=1.5 + 0.9 * ionic_charge,
            local_weight=2.0 + 1.1 * ionic_charge + 0.05 * coefficient_scale,
            projector_weight=projector_weight,
            kinetic_exponent=2.0,
            local_exponent=4.0,
            projector_exponent=2.0,
            monitor_cap=4.0 + 0.8 * ionic_charge,
        )
    return parameters


def _default_box_half_extents(
    case: BenchmarkCase,
    element_parameters: dict[str, NearCoreElementParameters],
) -> tuple[float, float, float]:
    center = _geometry_center(case)
    support_radius = max(
        max(
            parameters.patch_radius,
            parameters.local_radius,
            parameters.projector_radius,
        )
        for parameters in element_parameters.values()
    )
    max_axis_distance = [0.0, 0.0, 0.0]
    for atom in case.geometry.atoms:
        for axis in range(3):
            max_axis_distance[axis] = max(
                max_axis_distance[axis],
                abs(atom.position[axis] - center[axis]),
            )
    padding = max(6.0, 4.0 * support_radius)
    return (
        max_axis_distance[0] + padding,
        max_axis_distance[1] + padding,
        max_axis_distance[2] + padding,
    )


def build_monitor_grid_spec_for_case(
    case: BenchmarkCase,
    *,
    shape: tuple[int, int, int] | None = None,
    box_half_extents: tuple[float, float, float] | None = None,
    element_parameters: dict[str, NearCoreElementParameters] | None = None,
    harmonic_outer_iterations: int = 6,
    harmonic_inner_iterations: int = 180,
    harmonic_tolerance: float = 5.0e-5,
    harmonic_relaxation: float = 0.55,
    inner_relaxation: float = 0.82,
    monitor_smoothing: float = 0.18,
) -> MonitorGridSpec:
    """Build the first formal 3D monitor-grid spec for one case."""

    if element_parameters is None:
        element_parameters = build_default_near_core_element_parameters(case)
    if shape is None:
        shape = (
            _DEFAULT_MONITOR_SHAPE_DIATOMIC
            if len(case.geometry.atoms) <= 2
            else _DEFAULT_MONITOR_SHAPE_POLYATOMIC
        )
    if box_half_extents is None:
        box_half_extents = _default_box_half_extents(case, element_parameters)

    center = _geometry_center(case)
    half_extent_x, half_extent_y, half_extent_z = box_half_extents
    box_bounds = (
        (center[0] - half_extent_x, center[0] + half_extent_x),
        (center[1] - half_extent_y, center[1] + half_extent_y),
        (center[2] - half_extent_z, center[2] + half_extent_z),
    )
    return MonitorGridSpec(
        name=f"{case.geometry.name.lower()}_monitor_grid",
        description=(
            "First full 3D atom-centered monitor-function grid generated by a "
            "non-separable weighted harmonic map. The legacy single-center sinh "
            "grid remains only as a baseline."
        ),
        nx=shape[0],
        ny=shape[1],
        nz=shape[2],
        unit=case.geometry.unit,
        box_bounds=box_bounds,
        logical_bounds=_DEFAULT_LOGICAL_BOUNDS,
        element_parameters=element_parameters,
        harmonic_outer_iterations=harmonic_outer_iterations,
        harmonic_inner_iterations=harmonic_inner_iterations,
        harmonic_tolerance=harmonic_tolerance,
        harmonic_relaxation=harmonic_relaxation,
        inner_relaxation=inner_relaxation,
        monitor_smoothing=monitor_smoothing,
    )


def build_monitor_grid_for_case(
    case: BenchmarkCase,
    *,
    shape: tuple[int, int, int] | None = None,
    box_half_extents: tuple[float, float, float] | None = None,
    element_parameters: dict[str, NearCoreElementParameters] | None = None,
) -> MonitorGridGeometry:
    """Build the full 3D monitor grid geometry for one configured case."""

    spec = build_monitor_grid_spec_for_case(
        case,
        shape=shape,
        box_half_extents=box_half_extents,
        element_parameters=element_parameters,
    )
    return generate_monitor_grid_geometry(case=case, spec=spec)


def build_h2_local_patch_development_element_parameters() -> dict[str, NearCoreElementParameters]:
    """Return the current H2 A-grid development-point parameters for local-GTH patch work."""

    base_parameters = build_default_near_core_element_parameters(H2_BENCHMARK_CASE)
    return {
        element: replace(
            parameters,
            near_core_radius=parameters.near_core_radius * H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE,
            local_radius=parameters.local_radius * H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE,
            kinetic_weight=parameters.kinetic_weight * H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE,
            local_weight=parameters.local_weight * H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE,
        )
        for element, parameters in base_parameters.items()
    }


def build_h2_local_patch_development_monitor_grid() -> MonitorGridGeometry:
    """Build the current best-fair H2 A-grid baseline used for local-GTH patch audits."""

    return build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
        box_half_extents=H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def build_default_h2_monitor_grid() -> MonitorGridGeometry:
    return build_monitor_grid_for_case(H2_BENCHMARK_CASE)


def build_default_n2_monitor_grid() -> MonitorGridGeometry:
    return build_monitor_grid_for_case(N2_AUDIT_CASE)


def build_default_co_monitor_grid() -> MonitorGridGeometry:
    return build_monitor_grid_for_case(CO_AUDIT_CASE)


def build_default_h2o_monitor_grid() -> MonitorGridGeometry:
    return build_monitor_grid_for_case(H2O_AUDIT_CASE)
