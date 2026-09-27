"""The fixed-O C/O seed map's positivity margin, and the ChemParams layout.

One invariant: the margin is the worst layer's ln(1 + O_only/O_in_C), exactly
0 once some layer holds no O-only carrier, NaN (refused by ``not m > x``)
once a layer holds no oxygen at all. The engine evaluates it on the build
column (``co_bz_bound``) and on the converged column the AD tangent starts
from (``co_bz_margin``); both route through this function.

The ChemParams test sits here because this module imports vulcan_chem before
anything imports exojax, as the import-order guard requires.
"""
import numpy as np

from vulcan_forward.vulcan_chem import ChemParams, bz_margin, params_from_vector

import jax  # after vulcan_chem, which owns the first jax import

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


def test_chem_params_named_api_matches_the_vector_form():
    """The engine's primitive is named ChemParams; the positional vector stays
    supported as the adapter a sampler or a jvp needs."""
    # with a tp_eval hook the tail is that hook's parameter block
    theta = [0.5, -0.25, 1.5, 1200.0, -1.0]
    p = params_from_vector(theta, n_tp_params=2, has_tp_eval=True)
    assert (float(p.lnZ), float(p.c_o), float(p.lnKzz)) == (0.5, -0.25, 1.5)
    assert np.allclose(np.asarray(p.tp), [1200.0, -1.0])
    assert np.allclose(np.asarray(p.to_vector()), theta)

    # without one, the tail is a single uniform temperature offset in K
    q = params_from_vector([0.0, 0.0, 0.0, 25.0], n_tp_params=0, has_tp_eval=False)
    assert np.asarray(q.tp).shape == (1,)
    assert float(q.tp[0]) == 25.0

    # a NamedTuple, so it is a JAX pytree and round-trips through to_vector
    assert len(jax.tree_util.tree_leaves(ChemParams(1.0, 2.0, 3.0, (4.0,)))) == 4
    assert np.allclose(
        np.asarray(ChemParams(1.0, 2.0, 3.0, (4.0,)).to_vector()),
        [1.0, 2.0, 3.0, 4.0])
