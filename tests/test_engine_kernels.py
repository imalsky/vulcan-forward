"""The two engine kernels that fail SILENTLY when wrong.

`interp_map.make_to_art` and `exojax_rt._blend_h2he_broadening` had no test in
this repo or in either consumer (found by the 2026-08-14 bloat audit, whose
only under-testing finding this was). Both are silent failure modes: a wrong
interpolation returns a plausible profile, and a wrong broadening blend returns
plausible line widths. Neither raises, so nothing downstream notices.

`make_to_art` needs jax (it returns a jnp interpolator); `_blend_h2he_broadening`
is pure numpy and runs against a stub `mdb`, so it never touches exojax.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# interp_map.make_to_art
# ---------------------------------------------------------------------------
jnp = pytest.importorskip("jax.numpy", reason="make_to_art returns a jnp interpolator")

# x64 must be on before the first jnp array is built, exactly as
# `vulcan_chem` does it in production. Without this the interpolator runs in
# float32 and agrees only to ~1e-7 -- worth knowing: `make_to_art` inherits
# whatever precision the process was configured with and never asserts it.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _to_art(p_vulcan, p_art):
    from vulcan_forward.interp_map import make_to_art
    return make_to_art(np.asarray(p_vulcan, float), np.asarray(p_art, float))


def test_to_art_is_exact_on_the_source_grid():
    """Interpolating a profile onto its OWN pressures must return it."""
    p = np.logspace(-6, 1, 40)               # bar, ascending
    prof = jnp.asarray(np.linspace(-3.0, -8.0, 40))
    out = np.asarray(_to_art(p, p)(prof))
    assert np.allclose(out, np.asarray(prof), rtol=0, atol=1e-12)


def test_to_art_interpolates_linearly_in_log_pressure():
    """A profile linear in log10(P) must stay linear after mapping.

    This is the property the whole module exists for: VMRs are interpolated in
    log-pressure, not in pressure. A pressure-linear implementation would pass
    the identity test above and fail here.
    """
    p_v = np.logspace(-6, 1, 30)
    prof = jnp.asarray(2.0 * np.log10(p_v) + 5.0)
    p_art = np.logspace(-5.5, 0.5, 17)
    out = np.asarray(_to_art(p_v, p_art)(prof))
    assert np.allclose(out, 2.0 * np.log10(p_art) + 5.0, rtol=1e-12, atol=1e-10)


def test_to_art_accepts_a_descending_source_grid():
    """`p_bar_vulcan` is documented as "any monotonic order"; prove it.

    VULCAN hands pressures bottom-to-top or top-to-bottom depending on the
    caller, and the argsort inside make_to_art is what makes both work. A
    regression here would silently REVERSE every profile.
    """
    p_up = np.logspace(-6, 1, 30)
    prof_up = 2.0 * np.log10(p_up) + 5.0
    p_art = np.logspace(-5.0, 0.0, 11)
    out_up = np.asarray(_to_art(p_up, p_art)(jnp.asarray(prof_up)))
    # same physical profile, handed over in the opposite order
    out_down = np.asarray(
        _to_art(p_up[::-1], p_art)(jnp.asarray(prof_up[::-1])))
    assert np.allclose(out_up, out_down, rtol=0, atol=1e-12)


def test_to_art_refuses_an_art_grid_deeper_than_the_chemistry():
    """Deep clamping would fabricate deep-atmosphere chemistry: it must raise."""
    p_v = np.logspace(-6, 0, 20)             # chemistry bottom = 1 bar
    p_art = np.logspace(-5, 1, 15)           # ART bottom = 10 bar, too deep
    with pytest.raises(ValueError, match="below the VULCAN"):
        _to_art(p_v, p_art)


def test_to_art_allows_a_shallower_art_top_with_a_notice(capsys):
    """The TOP clamp is deliberate and only logged, unlike the bottom one."""
    p_v = np.logspace(-4, 0, 20)
    p_art = np.logspace(-6, -0.5, 15)        # extends ABOVE the chemistry top
    fn = _to_art(p_v, p_art)
    assert "above the" in capsys.readouterr().out
    out = np.asarray(fn(jnp.asarray(np.linspace(-3, -9, 20))))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# exojax_rt._blend_h2he_broadening
# ---------------------------------------------------------------------------
class _StubMdb:
    """Minimal stand-in for an exojax mdb: only the columns the blend reads."""

    def __init__(self, n=6, **cols):
        self.gamma_air = np.full(n, 0.05)
        self.n_air = np.full(n, 0.60)
        for k, v in cols.items():
            setattr(self, k, np.asarray(v, float))


def _blend(mdb, key="H2O"):
    from vulcan_forward.exojax_rt import _blend_h2he_broadening
    _blend_h2he_broadening(mdb, key)
    return mdb


def test_blend_is_the_number_weighted_mix_when_both_partners_cover():
    from vulcan_forward import constants
    f_h2, f_he = constants.H2HE_BROADENING_MIX
    n = 6
    g_h2, g_he = 0.070, 0.030
    n_h2, n_he = 0.70, 0.40
    mdb = _StubMdb(n, gamma_h2=np.full(n, g_h2), n_h2=np.full(n, n_h2),
                   gamma_he=np.full(n, g_he), n_he=np.full(n, n_he))
    _blend(mdb)
    want_g = f_h2 * g_h2 + f_he * g_he
    # the exponent is the GAMMA-weighted mean, not the number-weighted one
    want_n = (f_h2 * g_h2 * n_h2 + f_he * g_he * n_he) / want_g
    assert np.allclose(mdb.gamma_air, want_g, rtol=1e-12)
    assert np.allclose(mdb.n_air, want_n, rtol=1e-12)
    # air must be genuinely replaced, not blended with
    assert not np.isclose(mdb.gamma_air[0], 0.05)


def test_uncovered_lines_fall_back_to_air_for_that_partner_only():
    """A line with an invalid H2 width keeps gamma_air for H2 and its real He."""
    from vulcan_forward import constants
    f_h2, f_he = constants.H2HE_BROADENING_MIX
    n = 4
    g_h2 = np.array([0.070, 0.070, np.nan, -1.0])   # last two invalid
    g_he = np.full(n, 0.030)
    mdb = _StubMdb(n, gamma_h2=g_h2, n_h2=np.full(n, 0.70),
                   gamma_he=g_he, n_he=np.full(n, 0.40))
    _blend(mdb)
    covered = f_h2 * 0.070 + f_he * 0.030
    fell_back = f_h2 * 0.05 + f_he * 0.030      # gamma_air for the H2 part
    assert np.allclose(mdb.gamma_air[:2], covered, rtol=1e-12)
    assert np.allclose(mdb.gamma_air[2:], fell_back, rtol=1e-12)
    assert np.all(np.isfinite(mdb.gamma_air)), "a NaN width must never survive"


def test_a_missing_partner_column_still_blends_with_the_other():
    """He-only coverage: the H2 share uses gamma_air, and nothing raises."""
    from vulcan_forward import constants
    f_h2, f_he = constants.H2HE_BROADENING_MIX
    n = 5
    mdb = _StubMdb(n, gamma_he=np.full(n, 0.030), n_he=np.full(n, 0.40))
    _blend(mdb)
    assert np.allclose(mdb.gamma_air, f_h2 * 0.05 + f_he * 0.030, rtol=1e-12)


def test_no_h2_or_he_columns_at_all_raises():
    """Silently returning terrestrial-air widths is the failure to prevent."""
    mdb = _StubMdb(4)
    with pytest.raises(RuntimeError, match="no H2/He broadening"):
        _blend(mdb, key="CO2")


def test_blend_reports_per_molecule_coverage(capsys):
    """Coverage must be printed: a 3%-covered molecule is nearly air."""
    n = 10
    g_h2 = np.where(np.arange(n) < 3, 0.070, np.nan)   # 30% covered
    mdb = _StubMdb(n, gamma_h2=g_h2, n_h2=np.full(n, 0.70),
                   gamma_he=np.full(n, 0.030), n_he=np.full(n, 0.40))
    _blend(mdb, key="CH4")
    out = capsys.readouterr().out
    assert "CH4" in out and "30.0%" in out and "He 100.0%" in out
