"""Column geometry for the RT: radius and gravity anchors, layer boundaries and the tau = 2/3 level.

Pure jnp at fixed GM; docs/method.md derives the anchor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vulcan_forward import constants

if TYPE_CHECKING:
    from exojax.rt import ArtEmisPure, ArtTransPure
    from jax import Array


def _gravity_profile_invsq(
    art: ArtTransPure | ArtEmisPure, T_art: Array, mmw_art: Array, radius_btm: Array, gravity_btm: Array
) -> Array:
    """Return the mid-layer inverse-square gravity g(r) = g_btm (R_btm/r)^2, cm/s^2, of shape [nlayer, 1].

    The pinned exojax's ``ArtCommon.gravity_profile`` computes the same profile; this copy keeps the engine's gravity
    pinned by its own test. exojax's height integrator is inverse-square too, so opacity columns and heights stay on one
    gravity (against a chord quadrature: 1/r-linear -51 ppm, inverse-square +1.5 ppm in depth).
    """
    normalized_height, normalized_radius_lower = art.atmosphere_height(T_art, mmw_art, radius_btm, gravity_btm)
    rn_mid = normalized_radius_lower + 0.5 * normalized_height
    return jnp.array([gravity_btm / rn_mid**2]).T


def _anchor_to_grid_bottom(
    lnp_art: Array, T_art: Array, mmw_art: Array, r_ref: Array | float, g_ref: Array | float, p_ref_bar: float
) -> tuple[Array, Array]:
    """Return radius and gravity (cgs) at the lower boundary of the bottom layer, from values quoted at p_ref.

    That boundary is where exojax defines ``radius_btm``; a published radius is the transit radius near a millibar, and
    anchoring to the bottom centre would make the depth depend on ``art_nlayer``. ``lnp_art`` must ascend. The
    derivation is in docs/method.md.
    """
    return _radius_at(lnp_art, T_art, mmw_art, r_ref, g_ref, p_ref_bar, jnp.exp(_lnp_grid_bottom_boundary(lnp_art)))


def _boundary_temperature(T_art: Array) -> Array:
    """Return the temperature at the nlayer + 1 layer boundaries from the layer centres.

    ``ibased_linsap`` takes its source at the boundaries. The grid is uniform in log pressure, so a boundary is the
    midpoint of its two centres. The ends extrapolate with the local gradient: the deep end is the interior source term
    that stands in for everything below the grid, and clamping would cool it.
    """
    mid = 0.5 * (T_art[1:] + T_art[:-1])
    return jnp.concatenate([(1.5 * T_art[0] - 0.5 * T_art[1])[None], mid, (1.5 * T_art[-1] - 0.5 * T_art[-2])[None]])


def _lnp_grid_bottom_boundary(lnp_art: Array) -> Array:
    """Return ln(pressure) at the lower boundary of the bottom layer, half a layer below the last centre."""
    return lnp_art[-1] + 0.5 * (lnp_art[-1] - lnp_art[-2])


def _radius_at(
    lnp_art: Array,
    T_art: Array,
    mmw_art: Array,
    r_ref: Array | float,
    g_ref: Array | float,
    p_ref_bar: float,
    p_target_bar: Array | float,
) -> tuple[Array, Array]:
    """Return radius and gravity (cgs) at ``p_target_bar``, given them at ``p_ref_bar``, at fixed GM.

    Emission needs a different target level than transmission, so this is separate from ``_anchor_to_grid_bottom``. The
    integration nodes extend half a layer past both end centres, holding the end layer's T and mmw: ``jnp.interp``
    clamps at its end nodes, so without them the half layer down to the lower boundary would drop out, and a level in
    the top half layer (where ``_photosphere_lnp`` can put the tau = 2/3 photosphere) would get the top-centre radius
    and no derivative.
    """
    C = constants.K_B_CGS * T_art / (mmw_art * constants.M_U_CGS * g_ref * r_ref**2)
    d = lnp_art[1] - lnp_art[0]
    lnp = jnp.concatenate([lnp_art[:1] - 0.5 * d, lnp_art, _lnp_grid_bottom_boundary(lnp_art)[None]])
    C = jnp.concatenate([C[:1], C, C[-1:]])
    seg = 0.5 * (C[1:] + C[:-1]) * jnp.diff(lnp)
    # The integral is zero at the top centre.
    cum = jnp.concatenate([-seg[:1], jnp.zeros(1), jnp.cumsum(seg[1:])])
    i_ref = jnp.interp(jnp.log(p_ref_bar), lnp, cum)
    i_tgt = jnp.interp(jnp.log(p_target_bar), lnp, cum)
    r = 1.0 / (1.0 / r_ref + (i_tgt - i_ref))
    return r, g_ref * (r_ref / r) ** 2


def _photosphere_lnp(dtau_g: Array, lnp_art: Array) -> Array:
    """Return ln(pressure) where the vertical optical depth from the top reaches tau = 2/3, per (g, band).

    Where it never does, the grid's bottom boundary. Piecewise-linear in tau between layer boundaries, so it
    differentiates through ``dtau_g``.
    """
    d = lnp_art[1] - lnp_art[0]
    lnp_b = jnp.concatenate([lnp_art[:1] - 0.5 * d, lnp_art + 0.5 * d])
    tau_b = jnp.concatenate([jnp.zeros((1, *dtau_g.shape[1:])), jnp.cumsum(dtau_g, axis=0)])
    cols = tau_b.reshape(tau_b.shape[0], -1).T  # (ng*nband, nlayer+1)
    lnp = jax.vmap(lambda t: jnp.interp(constants.TAU_PHOTOSPHERE, t, lnp_b))(cols)
    return lnp.reshape(dtau_g.shape[1:])
