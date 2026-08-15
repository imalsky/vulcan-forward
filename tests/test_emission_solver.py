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
