"""The transmission opacity columns use inverse-square gravity.

There are two wrong neighbours a refactor could collapse onto, and neither raises: constant ``g_btm``, and a profile
linear in 1/r, which leaves heights and opacity columns on different gravities. Against an independent chord quadrature
the three differ by tens of ppm in transit depth (constant g -102 ppm, 1/r-linear -51 ppm, inverse-square +1.5 ppm).

The test drives the helper with a fake ART whose ``atmosphere_height`` returns known arrays, so it pins the algebra with
no opacity grid.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# Import order: vulcan_chem before exojax; find_spec, never importorskip("exojax").
if importlib.util.find_spec("exojax") is None:
    pytest.skip("exojax not installed (light-CI environment)", allow_module_level=True)
if importlib.util.find_spec("vulcan3") is not None:
    from vulcan_forward import vulcan_chem  # noqa: F401  # vulcan_chem first: the import order is the contract
else:
    import jax

    jax.config.update("jax_enable_x64", True)

from vulcan_forward._rt_geometry import _gravity_profile_invsq


class _FakeArt:
    """Stand-in for an ART with only ``atmosphere_height``, returning fixed normalized heights and lower radii."""

    def __init__(self, normalized_height, normalized_radius_lower):
        self._h = np.asarray(normalized_height, dtype=np.float64)
        self._r = np.asarray(normalized_radius_lower, dtype=np.float64)
        self.calls = []

    def atmosphere_height(self, _T_art, _mmw_art, radius_btm, gravity_btm):
        self.calls.append((radius_btm, gravity_btm))
        return self._h, self._r


# A 5-layer column with real altitude: r/R_btm runs 1.00 to 1.20, so g(r) falls by ~30% top to bottom and the three
# candidate profiles are far apart.
_HEIGHT = np.array([0.04, 0.04, 0.04, 0.04, 0.04])
_R_LOWER = np.array([1.00, 1.04, 1.08, 1.12, 1.16])
_G_BTM = 2140.0  # cgs, W39b-like
_R_BTM = 6.5e9  # cm


def test_gravity_profile_is_exactly_inverse_square():
    """g(r) is g_btm / r_mid**2 on the mid-layer normalized radius, and radius_btm and gravity_btm reach
    ``atmosphere_height`` unmodified (a helper that substituted a default planet would still pass the algebra).
    """
    art = _FakeArt(_HEIGHT, _R_LOWER)
    g = np.asarray(_gravity_profile_invsq(art, T_art=None, mmw_art=None, radius_btm=_R_BTM, gravity_btm=_G_BTM))
    r_mid = _R_LOWER + 0.5 * _HEIGHT
    expected = (_G_BTM / r_mid**2).reshape(-1, 1)
    # The same operations in the same precision: exact, not merely close.
    np.testing.assert_array_equal(g, expected)
    assert art.calls == [(_R_BTM, _G_BTM)]
