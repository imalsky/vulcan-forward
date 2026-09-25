"""Geometry contracts for the transmission RT.

Handing exojax the consumer's ``rp_cm``/``gs_cgs`` directly as
``radius_btm``/``gravity_btm`` places them at the LOWER boundary of the bottom
layer. A catalogue planet radius is instead the transit radius, near the
terminator photosphere at roughly a millibar, so that stacks the whole
column on top of a radius that already was the photospheric one (notes
register #10).

The tests are cheap: the real ``_radius_at`` / ``_anchor_to_grid_bottom``
against an independent closed form and two physical invariants, with no line
lists, no opacity build and no network.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# The geometry helpers live in exojax_rt, whose MODULE imports exojax -- so
# light CI (jax but no exojax) must skip here, and the import-order contract
# applies (see CLAUDE.md): check with find_spec, never importorskip("exojax"),
# and load vulcan_chem before anything exojax.
if importlib.util.find_spec("exojax") is None:              # pragma: no cover
    pytest.skip("exojax not installed (light-CI environment)",
                allow_module_level=True)
pytest.importorskip("jax", reason="the RT geometry helpers are JAX code")
if importlib.util.find_spec("vulcan_jax") is not None:      # pragma: no cover
    from vulcan_forward import vulcan_chem  # noqa: F401
else:                                                        # pragma: no cover
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                    # noqa: E402

from vulcan_forward import constants                       # noqa: E402
from vulcan_forward.exojax_rt import (                       # noqa: E402
    _anchor_to_grid_bottom, _radius_at)

R_JUP_CM = 7.1492e9
P_TOP, P_BTM, NLAYER = 1.0e-8, 7.0, 100


def _grid(n=NLAYER):
    """An ascending log-pressure grid, as exojax orders its layers.

    These are layer CENTRES (exojax's ``pressure_layer_logspace``); the level
    ``radius_btm`` is defined at is half a log-layer deeper. ``_p_boundary``
    below is that level.
    """
    return jnp.asarray(np.log(np.logspace(np.log10(P_TOP), np.log10(P_BTM), n)))


def _p_boundary(n=NLAYER):
    """Pressure at the LOWER BOUNDARY of the bottom layer = p[-1] * k**-0.5,
    which is where exojax defines radius_btm (atm/atmprof.py).
    """
    lnp = np.asarray(_grid(n))
    return float(np.exp(lnp[-1] + 0.5 * (lnp[-1] - lnp[-2])))


def _isothermal(T=1097.0, mmw=2.502, n=NLAYER):
    return jnp.full(n, T), jnp.full(n, mmw)


@pytest.mark.parametrize("n", (20, 60, 100, 400))
def test_matches_the_isothermal_closed_form_at_every_level(n):
    """u = 1/r is linear in lnP, so an isothermal column has an exact answer:
    1/r(p) = 1/r_ref + C ln(p/p_ref), C = k T / (mu m_u g_ref r_ref^2), with
    GM fixed so g = g_ref (r_ref/r)^2.

    Checked at the top boundary, inside the top half layer, an interior level,
    a FIXED 1 bar level (so r(p) is the same at every layer count) and the
    bottom boundary: the integration nodes must reach half a layer beyond
    both end centres (``jnp.interp`` clamps outside them, which returned the
    top-centre radius and no derivative in the top half layer).
    ``_anchor_to_grid_bottom`` must land on the bottom layer's LOWER
    BOUNDARY, not its centre, at any resolution: anchoring to the centre made
    the transit depth depend on art_nlayer. Deeper is then smaller and
    heavier by construction, and the correction scales with H/Rp.
    """
    T, mmw, r_ref, g_ref, p_ref = 1097.0, 2.502, 1.279 * R_JUP_CM, 422.0, 1.0e-3
    lnp = _grid(n)
    Ti, mi = _isothermal(T, mmw, n)
    d = float(lnp[1] - lnp[0])
    c = constants.K_B_CGS * T / (mmw * constants.M_U_CGS * g_ref * r_ref ** 2)

    def exact(lnp_t):
        return 1.0 / (1.0 / r_ref + c * (lnp_t - np.log(p_ref)))

    for lnp_t in (float(lnp[0]) - 0.5 * d,      # top boundary
                  float(lnp[0]) - 0.25 * d,     # inside the top half layer
                  float(lnp[3]) + 0.3 * d,      # an interior level
                  0.0,                          # 1 bar
                  np.log(_p_boundary(n))):      # bottom boundary
        r, g = _radius_at(lnp, Ti, mi, r_ref, g_ref, p_ref, np.exp(lnp_t))
        assert float(r) == pytest.approx(exact(lnp_t), rel=1e-12), lnp_t
        assert float(g) == pytest.approx(g_ref * (r_ref / exact(lnp_t)) ** 2,
                                         rel=1e-12)
    r_btm, g_btm = _anchor_to_grid_bottom(lnp, Ti, mi, r_ref, g_ref, p_ref)
    assert float(r_btm) == pytest.approx(exact(np.log(_p_boundary(n))), rel=1e-12)
    assert float(g_btm) == pytest.approx(g_ref * (r_ref / float(r_btm)) ** 2,
                                         rel=1e-12)


def test_a_hotter_deep_atmosphere_pushes_the_bottom_radius_down():
    """The correction uses the REAL T(p), not a constant: a puffier deep
    column means more altitude between the bottom and the mbar level, hence
    a smaller bottom radius. If T were ever dropped from the integrand this
    test fails."""
    lnp = _grid()
    _, mi = _isothermal()
    r_ref, g_ref = 1.279 * R_JUP_CM, 422.0
    cold = jnp.full(NLAYER, 1097.0)
    hot = jnp.asarray(np.linspace(870.0, 2246.0, NLAYER))   # ascending with P
    r_cold, _ = _anchor_to_grid_bottom(lnp, cold, mi, r_ref, g_ref, 1e-3)
    r_hot, _ = _anchor_to_grid_bottom(lnp, hot, mi, r_ref, g_ref, 1e-3)
    assert float(r_hot) < float(r_cold)


def test_is_differentiable():
    """The retrieval differentiates the depth end to end, and the planner takes
    jvps through it for Fisher rows. A cumulative integral keeps this clean, but
    an ODE solve or a np.interp slipped in here would silently break it."""
    import jax
    lnp = _grid()
    Ti, mi = _isothermal()

    def f(g):
        r, _ = _anchor_to_grid_bottom(lnp, Ti, mi, 1.279 * R_JUP_CM, g, 1.0e-3)
        return r

    d = float(jax.grad(f)(422.0))
    assert np.isfinite(d)
    assert d > 0.0        # weaker gravity -> more altitude -> smaller r_btm
