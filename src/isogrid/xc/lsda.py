"""First-stage spin-polarized LSDA kernel aligned with the PySCF audit baseline.

This module deliberately implements only the `lda,vwn` path used by the current
PySCF reference scripts. To keep the reference-side physics aligned while the
real-space solver is still being assembled, the pointwise LSDA evaluation is
forwarded to the locally installed PySCF/libxc backend at runtime.

The module itself remains importable without PySCF. A clear runtime error is
raised only when LSDA values are actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPPORTED_LSDA_FUNCTIONAL = "lda,vwn"
_NEGATIVE_DENSITY_TOLERANCE = 1.0e-14


@dataclass(frozen=True)
class LSDAEvaluation:
    """Resolved LSDA energy-density and potential data on a 3D grid."""

    functional: str
    backend: str
    rho_up: np.ndarray
    rho_down: np.ndarray
    rho_total: np.ndarray
    eps_xc: np.ndarray
    energy_density: np.ndarray
    v_xc_up: np.ndarray
    v_xc_down: np.ndarray


def _require_pyscf_libxc():
    try:
        from pyscf.dft import libxc
    except ImportError as error:
        raise ImportError(
            "The current LSDA slice delegates `lda,vwn` evaluation to PySCF/libxc. "
            "Install PySCF to evaluate LSDA energy densities and potentials."
        ) from error
    return libxc


def _normalize_functional_name(functional: str) -> str:
    return functional.strip().lower()


def validate_density_field(rho: np.ndarray, name: str = "rho") -> np.ndarray:
    """Validate a 3D density field and clip tiny negative roundoff to zero."""

    values = np.asarray(rho, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{name} must be a 3D field; received ndim={values.ndim}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(values < -_NEGATIVE_DENSITY_TOLERANCE):
        raise ValueError(f"{name} must be non-negative up to roundoff.")
    return np.maximum(values, 0.0)


def evaluate_lsda_terms(
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    functional: str = SUPPORTED_LSDA_FUNCTIONAL,
) -> LSDAEvaluation:
    """Evaluate the pointwise spin-polarized LSDA terms for one 3D density pair."""

    normalized_functional = _normalize_functional_name(functional)
    if normalized_functional != SUPPORTED_LSDA_FUNCTIONAL:
        raise ValueError(
            "The current IsoGridDFT LSDA layer supports only the PySCF-aligned "
            f"`{SUPPORTED_LSDA_FUNCTIONAL}` functional; received `{functional}`."
        )

    rho_up_field = validate_density_field(rho_up, name="rho_up")
    rho_down_field = validate_density_field(rho_down, name="rho_down")
    if rho_up_field.shape != rho_down_field.shape:
        raise ValueError(
            "rho_up and rho_down must have the same 3D shape; received "
            f"{rho_up_field.shape} and {rho_down_field.shape}."
        )

    libxc = _require_pyscf_libxc()
    flat_up = rho_up_field.reshape(-1)
    flat_down = rho_down_field.reshape(-1)
    eps_xc_flat, vxc, _, _ = libxc.eval_xc(
        normalized_functional,
        (flat_up, flat_down),
        spin=1,
        deriv=1,
    )
    v_rho = np.asarray(vxc[0], dtype=np.float64)
    if v_rho.shape == (flat_up.size, 2):
        v_up_flat = v_rho[:, 0]
        v_down_flat = v_rho[:, 1]
    elif v_rho.shape == (2, flat_up.size):
        v_up_flat = v_rho[0]
        v_down_flat = v_rho[1]
    else:
        raise ValueError(
            "Unexpected PySCF/libxc LSDA potential layout; received shape "
            f"{v_rho.shape}."
        )

    rho_total = rho_up_field + rho_down_field
    eps_xc = np.asarray(eps_xc_flat, dtype=np.float64).reshape(rho_total.shape)
    v_xc_up = np.asarray(v_up_flat, dtype=np.float64).reshape(rho_total.shape)
    v_xc_down = np.asarray(v_down_flat, dtype=np.float64).reshape(rho_total.shape)
    energy_density = rho_total * eps_xc
    return LSDAEvaluation(
        functional=normalized_functional,
        backend="pyscf.libxc",
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        rho_total=rho_total,
        eps_xc=eps_xc,
        energy_density=energy_density,
        v_xc_up=v_xc_up,
        v_xc_down=v_xc_down,
    )


def evaluate_lsda_energy_density(
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    functional: str = SUPPORTED_LSDA_FUNCTIONAL,
) -> np.ndarray:
    """Return the per-electron LSDA xc energy density eps_xc(r)."""

    return evaluate_lsda_terms(
        rho_up=rho_up,
        rho_down=rho_down,
        functional=functional,
    ).eps_xc


def evaluate_lsda_potential(
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    functional: str = SUPPORTED_LSDA_FUNCTIONAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the spin-channel LSDA potentials v_xc_up and v_xc_down."""

    evaluation = evaluate_lsda_terms(
        rho_up=rho_up,
        rho_down=rho_down,
        functional=functional,
    )
    return evaluation.v_xc_up, evaluation.v_xc_down
