"""Differentiable map from the VULCAN pressure grid to the ExoJAX ART grid, linear in log pressure.

``jnp.interp`` clamps outside its samples, so the ART grid must lie inside the chemistry grid; both edges are refused at
build. docs/method.md states the limits of this interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

__all__ = ["make_to_art"]

# Relative slack on the chemistry top: an ART top that equals it up to rounding is not a clamp.
_TOP_MATCH_RTOL = 1.0e-9


def make_to_art(p_bar_vulcan: np.ndarray, p_bar_art: np.ndarray) -> Callable[[Array], Array]:
    """Build the interpolator from the chemistry grid to the ART grid.

    Parameters
    ----------
    p_bar_vulcan : numpy.ndarray [nz]
        VULCAN cell-centre pressures, bar, in either monotonic order.
    p_bar_art : numpy.ndarray [nlayer]
        ART layer pressures, bar.

    Returns
    -------
    callable
        ``to_art(profile)``, from an array of shape [nz] to an array of shape [nlayer].

    Raises
    ------
    ValueError
        If the ART grid reaches outside the VULCAN grid at either end.
    """
    p_bar_vulcan = np.asarray(p_bar_vulcan, dtype=np.float64)
    p_bar_art = np.asarray(p_bar_art, dtype=np.float64)

    order = np.argsort(p_bar_vulcan)
    logP_v_sorted = jnp.asarray(np.log10(p_bar_vulcan[order]))
    logP_art = jnp.asarray(np.log10(p_bar_art))
    order_j = jnp.asarray(order)

    p_top_v, p_btm_v = float(np.min(p_bar_vulcan)), float(np.max(p_bar_vulcan))
    n_top_clamp = int(np.sum(np.asarray(p_bar_art) < p_top_v * (1.0 - _TOP_MATCH_RTOL)))
    n_btm_clamp = int(np.sum(np.asarray(p_bar_art) > p_btm_v))
    if n_btm_clamp:
        raise ValueError(
            f"ART grid bottom ({np.max(p_bar_art):.3e} bar) lies below the VULCAN "
            f"chemistry bottom ({p_btm_v:.3e} bar): {n_btm_clamp} layers would clamp "
            "to the deepest chemistry value. Shrink ART_PBTM_BAR or extend the "
            "chemistry grid."
        )
    if n_top_clamp:
        raise ValueError(
            f"ART grid top ({np.min(p_bar_art):.3e} bar) lies above the VULCAN chemistry "
            f"top ({p_top_v:.3e} bar): {n_top_clamp} layers would clamp to the topmost "
            "chemistry value. Extend the chemistry grid (cfg P_t) to the ART top."
        )

    def to_art(profile_nz: Array) -> Array:
        """Interpolate one profile of shape [nz] onto the ART grid."""
        if getattr(profile_nz, "ndim", None) != 1 or profile_nz.shape[0] != order.size:
            raise ValueError(
                f"profile must be 1-D with {order.size} values on the VULCAN "
                f"chemistry grid; got shape {getattr(profile_nz, 'shape', None)}"
            )
        fp = profile_nz[order_j]
        return jnp.interp(logP_art, logP_v_sorted, fp)

    return to_art
