"""The engine kernel that fails SILENTLY when wrong.

`interp_map.make_to_art` returns a plausible profile when its interpolation is
wrong, and nothing downstream notices. It needs jax (it returns a jnp
interpolator) and runs in this repo's light CI.
"""

from __future__ import annotations

import numpy as np
import pytest


# interp_map.make_to_art
jnp = pytest.importorskip("jax.numpy", reason="make_to_art returns a jnp interpolator")

# x64 must be on before the first jnp array is built, exactly as
# `vulcan_chem` does it in production. Without this the interpolator runs in
# float32 and agrees only to ~1e-7 -- worth knowing: `make_to_art` inherits
# whatever precision the process was configured with and never asserts it.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

ROUND_TOL = 1e-12   # float64 rounding bar


def _to_art(p_vulcan, p_art):
    from vulcan_forward.interp_map import make_to_art
    return make_to_art(np.asarray(p_vulcan, float), np.asarray(p_art, float))


def test_to_art_is_exact_on_the_source_grid():
    """Interpolating a profile onto its OWN pressures must return it."""
    p = np.logspace(-6, 1, 40)               # bar, ascending
    prof = jnp.asarray(np.linspace(-3.0, -8.0, 40))
    out = np.asarray(_to_art(p, p)(prof))
    assert np.allclose(out, np.asarray(prof), rtol=0, atol=ROUND_TOL)


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
    assert np.allclose(out, 2.0 * np.log10(p_art) + 5.0, rtol=ROUND_TOL, atol=1e-10)


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
    assert np.allclose(out_up, out_down, rtol=0, atol=ROUND_TOL)


@pytest.mark.parametrize("p_v,p_art", [
    # clamping fabricates chemistry the grid never solved, at either end
    (np.logspace(-6, 0, 20), np.logspace(-5, 1, 15)),     # ART bottom too deep
    (np.logspace(-4, 0, 20), np.logspace(-6, -0.5, 15)),  # ART top too high
    ([1e-5, 1e-4, 1e-4, 1e-2], np.logspace(-5, -2, 4)),   # duplicate pressure
    ([1e-5, np.nan, 1e-3, 1e-2], np.logspace(-5, -2, 4)), # non-finite
    ([1e-5, 0.0, 1e-3, 1e-2], np.logspace(-5, -2, 4)),    # non-positive
])
def test_to_art_refuses_clamped_or_invalid_grids(p_v, p_art):
    """A coincident top or bottom is allowed: the identity test above maps a
    grid onto itself."""
    with pytest.raises(ValueError):
        _to_art(p_v, p_art)
