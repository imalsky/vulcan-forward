"""Regression: the transmission opacity columns use INVERSE-SQUARE gravity.

`exojax_rt._gravity_profile_invsq` is the one physics choice in this package
that a refactor can silently undo. There are two wrong neighbours it could
collapse onto, and neither raises:

* constant `g_btm` -- what the code used before the 2026-07 audit, and what
  `tests/parity_picaso/scripts/run_native_rt_parity.py` still claimed in its
  docstring long after it stopped being true;
* `art.gravity_profile`, ExoJAX's own helper (<=2.2.3), which is LINEAR in
  1/r while its height integrator `normalized_layer_height` is inverse-square.
  Using it leaves heights and opacity columns on different gravities.

Measured on an isothermal gray W39b-like column at nlayer=60 (audit 2026-07-28):
-101.8 ppm constant-g, -50.8 ppm `gravity_profile`, +1.5 ppm this profile,
against an independent chord quadrature. So the difference between these three
is a real transmission bias, not a rounding choice.

The test drives the helper with a tiny FAKE ART whose `atmosphere_height`
returns known arrays, so it pins the algebra without building an opacity grid,
reading a line list, or running RT. Deliberately narrow: this is not an RT
validation framework.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# `exojax_rt` imports exojax at module scope, so this file needs the RT stack.
# Check WITHOUT importing: `vulcan_forward.vulcan_chem` refuses to load if exojax
# reached sys.modules first, so an `importorskip` here would poison the ordering
# contract for every later test in the session (the failure mode fixed in
# vulcan-retrieval's test_condensation_inference_gate.py).
if importlib.util.find_spec("exojax") is None:                  # pragma: no cover
    pytest.skip(
        "RT stack (exojax) not installed; this is the one test in the package "
        "that needs it. Run locally where exojax is present.",
        allow_module_level=True,
    )

# Honor the load-bearing import order: vulcan_chem owns the first jax import
# (it sets x64 and the VULCAN_JAX_* import-frozen env vars). Import it first
# when the chemistry side is installed; otherwise enable x64 directly, because
# the exactness assertions below are meaningless in float32.
if importlib.util.find_spec("vulcan_jax") is not None:          # pragma: no cover
    from vulcan_forward import vulcan_chem  # noqa: F401
else:                                                            # pragma: no cover
    import jax

    jax.config.update("jax_enable_x64", True)

from vulcan_forward.exojax_rt import _gravity_profile_invsq      # noqa: E402


class _FakeArt:
    """Minimal stand-in exposing only `atmosphere_height`.

    Returns fixed normalized heights/lower radii so the expected profile is a
    closed-form array rather than the output of an ExoJAX integration.
    """

    def __init__(self, normalized_height, normalized_radius_lower):
        self._h = np.asarray(normalized_height, dtype=np.float64)
        self._r = np.asarray(normalized_radius_lower, dtype=np.float64)
        self.calls = []

    def atmosphere_height(self, T_art, mmw_art, radius_btm, gravity_btm):
        self.calls.append((radius_btm, gravity_btm))
        return self._h, self._r


# A 5-layer column with genuinely non-zero altitude: r/R_btm runs 1.00 -> 1.20,
# so g(r) falls by ~30% top-to-bottom and the three candidate profiles are
# numerically far apart.
_NLAYER = 5
_HEIGHT = np.array([0.04, 0.04, 0.04, 0.04, 0.04])
_R_LOWER = np.array([1.00, 1.04, 1.08, 1.12, 1.16])
_G_BTM = 2140.0        # cgs, W39b-like
_R_BTM = 6.5e9         # cm


def _profile():
    art = _FakeArt(_HEIGHT, _R_LOWER)
    g = _gravity_profile_invsq(art, T_art=None, mmw_art=None,
                               radius_btm=_R_BTM, gravity_btm=_G_BTM)
    return art, np.asarray(g)


def test_gravity_profile_is_exactly_inverse_square():
    """g(r) == g_btm / r_mid**2 on the mid-layer normalized radius."""
    _, g = _profile()
    r_mid = _R_LOWER + 0.5 * _HEIGHT
    expected = (_G_BTM / r_mid**2).reshape(_NLAYER, 1)
    # Same operations in the same precision -> exact, not merely close.
    np.testing.assert_array_equal(g, expected)




def test_gravity_profile_uses_the_supplied_geometry():
    """radius_btm / gravity_btm reach `atmosphere_height` unmodified.

    A helper that quietly substituted a default planet would still pass the
    algebra tests above, so pin the pass-through too.
    """
    art, _ = _profile()
    assert art.calls == [(_R_BTM, _G_BTM)]
