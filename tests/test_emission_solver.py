"""Contracts for the emission solver's bottom boundary.

Until v0.5.0 the engine ran exojax's ``ibased`` solver, which drops the
bottom-boundary term entirely ("with no surface", its own docstring). Every
photon entering the model column from below was lost, so any wavelength that
saw through the grid bottom was underestimated -- silently, because the number
returned is a perfectly ordinary-looking flux. The consumer's answer was to
REFUSE any column with min bottom optical depth below 3, which is honest but
meant emission did not run at all on the shipped 7 bar grid.

v0.5.0 switches to ``ibased_linsap``, which carries the interior source and is
also the more accurate scheme at production layer counts. These tests pin the
part that can regress silently: that the term is present, that it vanishes when
the column is opaque, and that the boundary temperatures fed to it are right.

Cheap by construction -- no line lists, no opacity build, no chemistry.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# IMPORT ORDER IS LOAD-BEARING (see CLAUDE.md): vulcan_chem fixes jax x64 and
# the VULCAN_JAX_* import-frozen env vars, and refuses to load if exojax got
# there first. So check for the RT stack WITHOUT importing it -- an
# importorskip("exojax") here would poison the ordering contract for every
# later test module in the session.
if importlib.util.find_spec("exojax") is None:              # pragma: no cover
    pytest.skip("RT stack (exojax) not installed; run where exojax is present.",
                allow_module_level=True)
if importlib.util.find_spec("vulcan_jax") is not None:      # pragma: no cover
    from vulcan_forward import vulcan_chem  # noqa: F401
else:                                                        # pragma: no cover
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from exojax.rt import ArtEmisPure                           # noqa: E402
from exojax.rt.planck import piBarr                         # noqa: E402

from vulcan_forward.exojax_rt import _boundary_temperature   # noqa: E402

NU = jnp.asarray([2000.0, 5000.0])          # 5 um and 2 um
NLAYER = 30


def _art():
    return ArtEmisPure(nu_grid=NU, pressure_top=1e-8, pressure_btm=100.0,
                       nlayer=NLAYER, rtsolver="ibased_linsap", nstream=8)


def _column(tau_total, t_lo=900.0, t_hi=2600.0):
    dtau = jnp.full((NLAYER, NU.size), tau_total / NLAYER)
    T = jnp.asarray(np.linspace(t_lo, t_hi, NLAYER))
    return dtau, T


def test_boundary_temperatures_are_midpoints_with_extrapolated_ends():
    """The solver's source lives at the nlayer+1 layer boundaries; chemistry
    hands us the nlayer centres. On a log-uniform grid a boundary is the
    midpoint of its two neighbours, and the two ends extrapolate."""
    T = jnp.asarray([100.0, 200.0, 400.0, 800.0])
    got = np.asarray(_boundary_temperature(T))
    assert got.shape == (5,)
    assert got[1:-1] == pytest.approx([150.0, 300.0, 600.0])
    # ends carry the local gradient, they are NOT clamped to the end centre
    assert got[0] == pytest.approx(50.0)
    assert got[-1] == pytest.approx(1000.0)
    assert got[-1] > float(T[-1]), "the deep end must extrapolate, not clamp"


def test_a_transparent_column_returns_the_interior_blackbody():
    """The whole point of the switch. With ``ibased`` this returned ~0: the
    flux from below the grid was simply dropped, so a see-through window read
    as an empty sky."""
    art = _art()
    dtau = jnp.full((NLAYER, NU.size), 1e-12)
    T_btm = 2600.0
    T = jnp.full(NLAYER, T_btm)
    got = np.asarray(art.run(dtau, _boundary_temperature(T)))
    want = np.asarray(piBarr(jnp.asarray([T_btm]), NU))[0]
    assert got == pytest.approx(want, rel=1e-5)


def test_an_opaque_column_forgets_the_interior_entirely():
    """The bottom term is exp(-tau_total/mu), so a thick column must be
    insensitive to what is below it. If this ever fails, the interior
    temperature is leaking into a spectrum it has no business reaching."""
    art = _art()
    dtau, T = _column(200.0)
    hot = np.asarray(art.run(dtau, _boundary_temperature(T)))
    T_cooler = T.at[-1].set(float(T[-1]) - 800.0)
    cool = np.asarray(art.run(dtau, _boundary_temperature(T_cooler)))
    # the deepest layer still radiates, so allow a little; the BOUNDARY term
    # itself must be gone
    assert np.max(np.abs(hot - cool) / hot) < 0.02


def test_the_interior_term_dominates_a_marginally_thick_column():
    """tau_bottom = 3 is exactly the band the consumer's gate admits, and it is
    where dropping the term did the damage. Moving only the interior
    temperature must move the flux there."""
    art = _art()
    dtau, T = _column(3.0)
    base = np.asarray(art.run(dtau, _boundary_temperature(T)))
    hotter = np.asarray(art.run(
        dtau, _boundary_temperature(T.at[-1].set(float(T[-1]) + 1500.0))))
    rel = (hotter - base) / base
    assert np.all(rel > 0.05), rel


def test_is_differentiable_through_the_boundary_source():
    """The planner takes jvps through the emission depth for Fisher rows and
    the retrieval differentiates it end to end. A finite gradient with respect
    to the interior temperature is what proves the new term is in the graph."""
    art = _art()
    dtau, T = _column(3.0)

    def f(t_int):
        Tb = _boundary_temperature(T.at[-1].set(t_int))
        return jnp.sum(art.run(dtau, Tb))

    d = float(jax.grad(f)(float(T[-1])))
    assert np.isfinite(d) and d > 0.0


# --------------------------------------------------------------------------
# Correlated-k emission (v0.7.0)
#
# Before 0.7.0 this path did not exist: build_emis_model REFUSED a CKD parent,
# because exojax's own ArtEmisPure.run_ckd hard-codes the "ibased" solver and
# would have silently undone the interior-source fix above. So emission stayed
# on sampled line-by-line at R = 1,477 and kept the full grid bias that
# transmission had just been fixed for.
#
# _run_emis_ckd_linsap is upstream's flatten-solve-reweight with "ibased_linsap"
# in its place. The tests below pin the two things that can regress silently:
# the flatten/tile index convention (a transpose here mixes bands into
# g-ordinates and still returns plausible numbers), and the presence of the
# interior term.
# --------------------------------------------------------------------------

from vulcan_forward.exojax_rt import _run_emis_ckd_linsap    # noqa: E402
from vulcan_forward import ckd                               # noqa: E402


def _gquad(ng=16):
    g, w = ckd.gauss_legendre(ng)
    return jnp.asarray(g), jnp.asarray(w)


def test_ckd_emission_matches_line_by_line_when_every_g_is_identical():
    """If k does not vary across the band, the k-distribution is degenerate and
    the CKD flux MUST equal the plain line-by-line flux on the same bands.

    This is the index-convention test. dtau is reshaped (nlayer, ng*nband) and
    the source is tiled along its last axis; if either side flattened
    band-major instead of g-major, bands would be paired with the wrong source
    and this identity would break. Measured max relative difference: 3.3e-16.
    """
    art = _art()
    ng = 16
    gg, gw = _gquad(ng)
    rng = np.random.default_rng(0)
    base = np.abs(rng.lognormal(-3.0, 1.5, size=(NLAYER, NU.size)))
    T = jnp.asarray(np.linspace(900.0, 2600.0, NLAYER))
    Tb = _boundary_temperature(T)

    dtau_g = jnp.asarray(np.repeat(base[:, None, :], ng, axis=1))
    got = np.asarray(_run_emis_ckd_linsap(art, dtau_g, Tb, NU, gw))
    want = np.asarray(art.run(jnp.asarray(base), Tb))
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_ckd_emission_keeps_the_interior_source_upstream_run_ckd_drops():
    """The reason we do not call ArtEmisPure.run_ckd. On a thin column its
    "ibased" solver loses the interior blackbody entirely; measured, it returns
    22-99% less flux than the linsap version over the band. If this test ever
    fails because the two agree, someone has routed emission back through
    upstream's CKD entry point.
    """
    art = _art()
    art_ibased = ArtEmisPure(nu_grid=NU, pressure_top=1e-8, pressure_btm=100.0,
                             nlayer=NLAYER, rtsolver="ibased", nstream=8)
    ng = 16
    gg, gw = _gquad(ng)
    T = jnp.asarray(np.linspace(900.0, 2600.0, NLAYER))
    thin = jnp.full((NLAYER, ng, NU.size), 1e-4 / NLAYER)

    ours = np.asarray(_run_emis_ckd_linsap(art, thin, _boundary_temperature(T),
                                           NU, gw))
    upstream = np.asarray(art_ibased.run_ckd(thin, T, gw, NU))
    assert np.all(ours > upstream)
    assert (1.0 - upstream / ours).min() > 0.05


def test_ckd_emission_is_not_the_same_as_using_the_band_mean_opacity():
    """Sanity that the k-distribution is doing work. A real k(g) spread at
    FIXED band-mean optical depth must give a different flux from running that
    mean directly -- that gap (measured 24-38% here) is precisely the grey
    within-band averaging error correlated-k exists to remove. If these agreed,
    the g-axis would be decorative.
    """
    art = _art()
    ng = 16
    gg, gw = _gquad(ng)
    rng = np.random.default_rng(1)
    w = np.asarray(gw)
    base = np.abs(rng.lognormal(-3.0, 1.5, size=(NLAYER, NU.size)))
    T = jnp.asarray(np.linspace(900.0, 2600.0, NLAYER))
    Tb = _boundary_temperature(T)

    kg = np.sort(rng.lognormal(0.0, 2.0, size=(NLAYER, ng, NU.size)), axis=1)
    kg *= base[:, None, :] / np.einsum("g,lgb->lb", w, kg)[:, None, :]
    spread = np.asarray(_run_emis_ckd_linsap(art, jnp.asarray(kg), Tb, NU, gw))
    grey = np.asarray(art.run(jnp.asarray(base), Tb))
    assert np.median(np.abs(spread / grey - 1.0)) > 0.05


def test_ckd_emission_tangent_sign_follows_the_lapse_rate():
    """Differentiability plus a physics check in one. More opacity moves the
    photosphere UP; on a normal profile that is into cooler gas so the flux
    dims, and on an inversion it is into hotter gas so the flux brightens.
    Getting both signs right is what shows the solver tracks where the
    photosphere actually sits, rather than merely returning a finite number.
    """
    art = _art()
    ng = 8
    gg, gw = _gquad(ng)
    rng = np.random.default_rng(2)
    base = np.abs(rng.lognormal(-3.0, 1.5, size=(NLAYER, NU.size)))
    dtau_g = jnp.asarray(np.repeat(base[:, None, :], ng, axis=1))

    # index 0 is the TOP of exojax's column
    for t_top, t_btm in ((900.0, 2600.0), (2600.0, 900.0)):
        Tb = _boundary_temperature(jnp.asarray(np.linspace(t_top, t_btm, NLAYER)))

        def f(scale, _Tb=Tb):
            return jnp.sum(_run_emis_ckd_linsap(art, dtau_g * scale, _Tb, NU, gw))

        _, tan = jax.jvp(f, (1.0,), (1.0,))
        assert np.isfinite(float(tan)) and float(tan) != 0.0
        assert (float(tan) < 0.0) == (t_top < t_btm), (t_top, t_btm, float(tan))
