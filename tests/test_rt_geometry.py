"""Geometry contracts for the transmission RT.

These exist because of a real, shipped defect. Until v0.4.0 the engine handed
exojax the consumer's ``rp_cm``/``gs_cgs`` directly as ``radius_btm``/
``gravity_btm``, which exojax defines at the LOWER boundary of the bottom layer
(7 bar). A catalogue planet radius is instead the transit radius, near the
terminator photosphere at roughly a millibar. The engine therefore stacked the
whole 7 bar -> mbar column on top of a radius that already was the photospheric
one: WASP-39 b's modelled transit depth came out at 26,754 ppm against a
published 21,381, with 1.42x too much spectral contrast.

Nothing caught it. The two suites that could have (this one and the planner's)
never called into the RT geometry at all, so the bug was invisible to CI while
every published number it produced was wrong.

The tests below are deliberately cheap: they exercise the real
``_anchor_to_grid_bottom`` against an independent closed-form solution and a set
of physical invariants, with no line lists, no opacity build and no network.
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


def test_matches_the_closed_form_for_an_isothermal_column():
    """u = 1/r is linear in lnP, so an isothermal column has an exact answer.

    du/dlnP = kT/(mu m_H g_ref r_ref^2) = const, hence
    u_btm = u_ref + const * ln(P_btm/P_ref). If the cumulative integral in
    _anchor_to_grid_bottom ever drifts (wrong sign, wrong direction, an
    off-by-one in the trapezoid) this catches it exactly.
    """
    T, mmw, r_ref, g_ref, p_ref = 1097.0, 2.502, 1.279 * R_JUP_CM, 422.0, 1.0e-3
    lnp = _grid()
    Ti, mi = _isothermal(T, mmw)

    r_btm, g_btm = _anchor_to_grid_bottom(lnp, Ti, mi, r_ref, g_ref, p_ref)

    c = constants.K_B_CGS * T / (mmw * constants.M_U_CGS * g_ref * r_ref ** 2)
    u_exact = 1.0 / r_ref + c * np.log(_p_boundary() / p_ref)
    assert float(r_btm) == pytest.approx(1.0 / u_exact, rel=1e-9)
    # GM is held fixed, so gravity must follow the inverse square exactly.
    assert float(g_btm) == pytest.approx(
        g_ref * (r_ref / float(r_btm)) ** 2, rel=1e-12)


def test_deeper_is_smaller_and_heavier():
    """The direction of the correction. Getting this backwards would inflate
    the radius instead of shrinking it, which is the original bug with a sign
    flip and would still look plausible on a single planet."""
    lnp = _grid()
    Ti, mi = _isothermal()
    r_ref, g_ref = 1.279 * R_JUP_CM, 422.0
    r_btm, g_btm = _anchor_to_grid_bottom(lnp, Ti, mi, r_ref, g_ref, 1.0e-3)
    assert float(r_btm) < r_ref
    assert float(g_btm) > g_ref




def test_the_anchor_level_is_the_grid_bottom_BOUNDARY_at_any_resolution():
    """``radius_btm`` means the radius at the bottom layer's LOWER BOUNDARY.

    v0.4.0 anchored to the deepest layer CENTRE instead, half a log-layer
    shallower, and the size of that half layer shrinks with resolution. A
    requested 1 mbar reference therefore landed at 1.11 mbar on the planner's
    100-layer grid and 1.71 mbar on vulcan-retrieval's 20-layer SMOKE preset,
    which made the absolute transit depth depend on art_nlayer (+0.20% and
    +1.02%). Here _anchor_to_grid_bottom must agree with _radius_at aimed
    explicitly at the boundary, at every layer count.
    """
    r_ref, g_ref, p_ref = 1.279 * R_JUP_CM, 422.0, 1.0e-3
    for n in (20, 60, 100, 400):
        lnp = _grid(n)
        Ti, mi = _isothermal(n=n)
        got = float(_anchor_to_grid_bottom(lnp, Ti, mi, r_ref, g_ref, p_ref)[0])
        want = float(_radius_at(lnp, Ti, mi, r_ref, g_ref, p_ref,
                                _p_boundary(n))[0])
        assert got == pytest.approx(want, rel=1e-12), n


def test_radius_at_a_fixed_pressure_is_resolution_independent():
    """The physical content: r(p) at a FIXED pressure must not depend on how
    many layers the grid happens to have. Isothermal, so the integrand carries
    no discretisation error of its own and any spread is the anchoring."""
    r_ref, g_ref, p_ref, p_tgt = 1.279 * R_JUP_CM, 422.0, 1.0e-3, 1.0
    got = [float(_radius_at(_grid(n), *_isothermal(n=n), r_ref, g_ref,
                            p_ref, p_tgt)[0])
           for n in (20, 60, 100, 400)]
    assert max(got) / min(got) - 1.0 < 1e-9, got


def test_correction_scales_with_scale_height():
    """The error the bug caused scales as H/Rp, which is why it was worst on the
    low-gravity planets (WASP-107 b, chi2/N 490 against its published spectrum)
    and mildest on the high-gravity ones. Halving gravity must roughly double
    the altitude correction."""
    lnp = _grid()
    Ti, mi = _isothermal()
    r_ref = 1.279 * R_JUP_CM
    dz_hi = r_ref - float(_anchor_to_grid_bottom(lnp, Ti, mi, r_ref, 844.0, 1e-3)[0])
    dz_lo = r_ref - float(_anchor_to_grid_bottom(lnp, Ti, mi, r_ref, 422.0, 1e-3)[0])
    assert dz_lo / dz_hi == pytest.approx(2.0, rel=0.05)


def test_wasp39b_lands_on_the_published_transit_radius():
    """The number that matters, on the planet with published data.

    WASP-39 b: Rp = 1.279 R_Jup, g = 422 cm/s^2, isothermal at the ~1097 K
    photosphere. Converting the catalogue radius down to 7 bar must give
    ~1.18 R_Jup. The pre-fix code used 1.279 there, and that 8% radius error is
    the 25% median transit-depth error the JWST ERS comparison showed.
    """
    lnp = _grid()
    Ti, mi = _isothermal()
    r_btm, _ = _anchor_to_grid_bottom(
        lnp, Ti, mi, 1.279 * R_JUP_CM, 422.0, 1.0e-3)
    assert float(r_btm) / R_JUP_CM == pytest.approx(1.180, abs=0.01)


def test_a_hotter_deep_atmosphere_pushes_the_bottom_radius_down():
    """The correction uses the REAL T(p), not a constant. WASP-39 b's shipped
    table runs 2246 K at 7.6 bar, far hotter than its 1097 K photosphere, and a
    puffier deep column means more altitude between 7 bar and the mbar level,
    hence a smaller bottom radius. If T were ever dropped from the integrand
    this test fails."""
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
