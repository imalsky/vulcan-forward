"""The elemental repair and the warm guess, on a small hand-made network with no solve.

The column is the cool one the carbon repair exists for: CH4 holds the carbon and CO only 1e-3 of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vulcan_forward import constants, vulcan_chem  # noqa: F401  # vulcan_chem sets jax x64
from vulcan_forward._elemental import _build_elemental

SPECIES = {
    "H2": {"H": 2},
    "He": {"He": 1},
    "H2O": {"H": 2, "O": 1},
    "CO": {"C": 1, "O": 1},
    "CH4": {"C": 1, "H": 4},
    "N2": {"N": 2},
    "H2S": {"H": 2, "S": 1},
}
SIDX = {sp: i for i, sp in enumerate(SPECIES)}
# Number density per layer; the two layers differ only by a factor.
COOL = np.array([0.85, 0.14, 5.0e-3, 4.0e-6, 4.0e-3, 1.0e-3, 1.0e-4])
Y0 = np.stack([COOL, 2.0 * COOL])


def _compo() -> np.ndarray:
    out = np.zeros((len(SPECIES), max(constants.ATOM_COLS.values()) + 1))
    for sp, atoms in SPECIES.items():
        for atom, n in atoms.items():
            out[SIDX[sp], constants.ATOM_COLS[atom]] = n
    return out


def _elem(co_fixed_o: bool = True):
    return _build_elemental(_compo(), SIDX, Y0, len(SPECIES), co_fixed_o)


def test_a_carbon_excess_is_repaired_through_ch4_as_well_as_co():
    """1% too much carbon is 10 times the CO column: a CO-only repair would need a negative factor."""
    excess = Y0.copy()
    excess[:, SIDX["CH4"]] *= 1.01
    y, min_adj = _elem().project(excess, excess.sum(axis=1), 0.0, 0.0, 0.0)
    y = np.asarray(y)
    assert np.all(y > 0.0)
    assert 0.98 < float(min_adj) < 1.0
    # the column's C/H is back at the baseline's, exactly up to the fixed-iteration residual
    compo = _compo()
    c_col, h_col = constants.ATOM_COLS["C"], constants.ATOM_COLS["H"]
    ratio = (y @ compo[:, c_col]).sum() / (y @ compo[:, h_col]).sum()
    want = (Y0 @ compo[:, c_col]).sum() / (Y0 @ compo[:, h_col]).sum()
    assert ratio == pytest.approx(want, rel=1e-13)


@pytest.mark.parametrize("co_fixed_o", [True, False])
def test_the_warm_guess_applies_only_the_composition_increments(co_fixed_o):
    """A guess from a column at (lnZ_ref, c_o_ref, s_o_ref) depends on the differences alone."""
    guess = _elem(co_fixed_o).guess
    with_refs = np.asarray(guess(1.0, 0.3, 0.2, Y0, 0.4, 0.1, 0.05))
    from_zero = np.asarray(guess(0.6, 0.2, 0.15, Y0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(with_refs, from_zero, rtol=1e-13, atol=0.0)
    # N2 is a metal and nothing else: its factor is e^(lnZ - lnZ_ref) alone
    ratio = with_refs[:, SIDX["N2"]] / Y0[:, SIDX["N2"]]
    np.testing.assert_allclose(ratio, np.exp(0.6), rtol=1e-14, atol=0.0)
