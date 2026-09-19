"""``converged_y_batch``: the solver's batched runner as a drop-in for the
production per-lane ``jax.vmap(converged_y)``.

Two properties, and they pull in opposite directions:

* a lane must be its OWN solve -- lanes freeze at their own exits, so lane k of
  an N-lane batch is BIT-IDENTICAL to the same theta run as a batch of one.
  Without this the batch would be a different physical model at every width;
* against the route it replaces (a vmap of the per-lane solve) the agreement is
  at the CONVERGENCE scale, not bitwise, and that is by design: the batched
  runner's while loop sits above the lane vmap, so photolysis and the geometry
  refresh key to the loop's iteration tick instead of a lane's accept count.
  The certificate (``conv_normal``) must still match lane for lane.

The measured differences are printed (``pytest -s``): they are the number that
says whether the batched primal is still the same answer.

Cheap profile: nz=20, photochemistry off, no build-time warm-up solve -- the
solve itself is what is under test, not the column.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# vulcan_chem drives VULCAN-JAX's runner, which light CI does not install; the
# import-order contract applies (see CLAUDE.md): check with find_spec, never
# importorskip("exojax"), and load vulcan_chem before anything else jax.
if importlib.util.find_spec("vulcan_jax") is None:           # pragma: no cover
    pytest.skip("vulcan_jax not installed (light-CI environment)",
                allow_module_level=True)
pytest.importorskip("jax", reason="the batched runner is JAX code")

from vulcan_forward import vulcan_chem                       # noqa: E402

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

PROFILE = {"use_photo": False, "yconv_cri": 1.0e-2, "nz": 20,
           "abundance_mode": "elemental", "skip_warmup": True}
# [lnZ, c_o, lnKzz, T-offset]: the baseline column and a perturbed one.
THETAS = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.3, 0.1, 0.5, 40.0]], dtype=np.float64)
MIX_FLOOR = 1.0e-10   # cells below this carry no observable and no certificate
REL_MAX = 5.0e-2


@pytest.fixture(scope="module")
def runs():
    """The three routes, solved once: batched, per-lane vmap, batch-of-one."""
    try:
        chem = vulcan_chem.build_chem_model(PROFILE)
    except (FileNotFoundError, OSError) as e:                # pragma: no cover
        pytest.skip(f"chem model data unavailable: {e}")
    th = jnp.asarray(THETAS)
    y_b, cd_b = chem.converged_y_batch(th, return_conv_diag=True)
    y_v, cd_v = jax.vmap(
        lambda t: chem.converged_y(t, return_conv_diag=True))(th)
    y_one = [np.asarray(chem.converged_y_batch(th[k:k + 1]))[0]
             for k in range(THETAS.shape[0])]
    return np.asarray(y_b), cd_b, np.asarray(y_v), cd_v, y_one


@pytest.mark.parametrize("k", range(THETAS.shape[0]))
def test_batched_lane_is_its_own_solve_and_agrees_with_the_vmap(runs, k):
    y_b, cd_b, y_v, cd_v, y_one = runs
    acc_b = int(np.asarray(cd_b.accept_count)[k])
    acc_v = int(np.asarray(cd_v.accept_count)[k])

    mix = y_b[k] / y_b[k].sum(axis=1, keepdims=True)
    obs = mix > MIX_FLOOR
    rel = np.abs(y_b[k] - y_v[k]) / np.maximum(np.abs(y_v[k]), 1e-300)
    print(f"[lane {k}] batch-vs-vmap over {int(obs.sum())} cells > {MIX_FLOOR:g} "
          f"VMR: max rel {rel[obs].max():.3e}, median {np.median(rel[obs]):.3e}; "
          f"accept_count batched={acc_b} vmap={acc_v}", flush=True)

    assert np.array_equal(y_b[k], y_one[k]), (
        "lane is not independent of the batch: lane k of the N-lane batch "
        "differs from the same theta as a batch of one")
    assert bool(np.asarray(cd_b.conv_normal)[k]) == bool(
        np.asarray(cd_v.conv_normal)[k])
    assert rel[obs].max() < REL_MAX
