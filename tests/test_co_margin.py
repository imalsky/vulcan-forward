"""The fixed-O C/O seed map's positivity margin (numpy-only).

One invariant: the margin is the worst layer's ln(1 + O_only/O_in_C), exactly
0 once some layer holds no O-only carrier, NaN (refused by ``not m > x``)
once a layer holds no oxygen at all. The engine evaluates it on the build
column (``co_bz_bound``) and on the converged column the AD tangent starts
from (``co_bz_margin``); both route through this function.
"""
import numpy as np

from vulcan_forward.vulcan_chem import bz_margin

# species: H2O (O-only), CO (C and O), CH4 (C, no O), H2 (neither)
N_O = np.array([1.0, 1.0, 0.0, 0.0])
C_MASK = np.array([0.0, 1.0, 1.0, 0.0])
OO_MASK = np.array([1.0, 0.0, 0.0, 0.0])


def test_margin_is_the_worst_layer_and_vanishes_without_o_only_carriers():
    # layer 0: O_only/O_C = 3/1; layer 1: 1/4 -> the min sets the margin
    y = np.array([[3.0, 1.0, 5.0, 100.0],
                  [1.0, 4.0, 5.0, 100.0]])
    assert bz_margin(y, N_O, C_MASK, OO_MASK) == np.log(1.0 + 0.25)
    # a layer with every O atom in CO: no compensation possible, margin 0
    y[1, 0] = 0.0
    assert bz_margin(y, N_O, C_MASK, OO_MASK) == 0.0
    # a layer with no oxygen at all is NaN, and the gate form refuses it
    y[1, 1] = 0.0
    m = bz_margin(y, N_O, C_MASK, OO_MASK)
    assert np.isnan(m) and not m > 0.1
