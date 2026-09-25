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

The warm-continuation arguments are covered too: the mutation cap
(``warm_cap``) and the PER-LANE reference composition, both of which the
retrieval's warm mutation kernel needs from a batched call.

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
           "skip_warmup": True}
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
    return chem, np.asarray(y_b), cd_b, np.asarray(y_v), cd_v, y_one


def test_batched_lanes_are_their_own_solves_and_agree_with_the_vmap(runs):
    _chem, y_b, cd_b, y_v, cd_v, y_one = runs
    for k in range(THETAS.shape[0]):
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


WARM_CMAX = 5     # under count_min=120: a warm continuation cannot certify
COLD_CMAX = 50    # well above WARM_CMAX, so the cap that binds is identifiable


@pytest.fixture(scope="module")
def chem_capped():
    """A model whose warm cap (5) is far below its cold cap (50), the shape the
    retrieval's mutation path runs (1500 against 5000)."""
    try:
        return vulcan_chem.build_chem_model(
            {**PROFILE, "warm_count_max": WARM_CMAX, "count_max": COLD_CMAX})
    except (FileNotFoundError, OSError) as e:                # pragma: no cover
        pytest.skip(f"chem model data unavailable: {e}")


def test_warm_cap_binds_per_lane_batched_and_queued(chem_capped):
    """``warm_cap=True`` cuts every lane of a BATCH and of the QUEUE at
    warm_count_max, exactly as it cuts the solo ``converged_y``.

    The cap rides the runner carry (``count_max_dyn``), not a second compiled
    runner, so the batched calls reproduce the mutation-path semantics without
    one: termination is ``accept_count > cap``, hence cap + 1 accepted steps on
    every lane. The uncapped arm of the same batch marches on to the cold cap,
    which is what identifies the warm cap as the thing that stopped it."""
    chem = chem_capped
    th = jnp.asarray(THETAS)
    yw = jnp.broadcast_to(jnp.asarray(chem.y0), (THETAS.shape[0],) + chem.y0.shape)
    _y, cd_cap = chem.converged_y_batch(th, warm_y=yw, warm_cap=True,
                                        return_conv_diag=True)
    _y, cd_cold = chem.converged_y_batch(th, warm_y=yw, warm_cap=False,
                                         return_conv_diag=True)
    _y, cd_queue = chem.converged_y_queue(th, 1, chunk=1, warm_y=yw, warm_cap=True)
    solo = [chem.converged_y(th[k], warm_y=yw[k], warm_cap=True,
                             return_conv_diag=True)[1]
            for k in range(THETAS.shape[0])]
    acc_cap = np.asarray(cd_cap.accept_count)
    acc_queue = np.asarray(cd_queue.accept_count)
    acc_solo = np.array([int(np.asarray(cd.accept_count)) for cd in solo])
    print(f"[warm cap {WARM_CMAX} / cold cap {COLD_CMAX}] accept_count batched "
          f"{acc_cap.tolist()} queued {acc_queue.tolist()} solo "
          f"{acc_solo.tolist()} uncapped-batch "
          f"{np.asarray(cd_cold.accept_count).tolist()}", flush=True)
    assert np.array_equal(acc_cap, acc_solo)
    assert np.all(acc_cap == WARM_CMAX + 1)
    assert np.all(acc_queue == WARM_CMAX + 1)
    assert np.all(np.asarray(cd_cold.accept_count) == COLD_CMAX + 1)
    # neither arm can certify this far below count_min: the cap, not
    # convergence, ended both runs
    assert not bool(np.any(np.asarray(cd_cap.conv_normal)))
    assert not bool(np.any(np.asarray(cd_cold.conv_normal)))


def test_per_lane_references_match_the_scalar_calls(runs):
    """``lnZ_ref`` / ``c_o_ref`` as (N,) arrays give every lane the reference
    composition ITS carried column was converged at -- what the warm mutation
    needs, since each particle carries its own.

    Reference: the same warm continuation as a batch of ONE with the scalar
    reference. The bound is the convergence scale (a traced reference is not the
    folded constant of the scalar call) with the certificate lane for lane; the
    measured difference is 0 -- subtracting a reference that rides the carry is
    the same arithmetic as subtracting the constant."""
    chem, y_conv, _cd_b, _y_v, _cd_v, _y_one = runs
    th0 = jnp.asarray(THETAS)
    th1 = jnp.asarray(THETAS + np.array([0.05, 0.02, 0.1, 5.0])[None, :])
    yw = jnp.asarray(y_conv)
    # distinct per-lane references: each column's own (lnZ, c_o)
    y_a, cd_a = chem.converged_y_batch(th1, warm_y=yw, lnZ_ref=th0[:, 0],
                                       c_o_ref=th0[:, 1], return_conv_diag=True)
    y_a = np.asarray(y_a)
    assert len({float(x) for x in np.asarray(THETAS)[:, 0]}) > 1  # not vacuous
    for k in range(THETAS.shape[0]):
        y_s, cd_s = chem.converged_y_batch(
            th1[k:k + 1], warm_y=yw[k:k + 1], lnZ_ref=float(THETAS[k, 0]),
            c_o_ref=float(THETAS[k, 1]), return_conv_diag=True)
        y_s = np.asarray(y_s)[0]
        mix = y_s / y_s.sum(axis=1, keepdims=True)
        obs = mix > MIX_FLOOR
        rel = np.abs(y_a[k] - y_s) / np.maximum(np.abs(y_s), 1e-300)
        print(f"[refs lane {k}] array-vs-scalar reference over {int(obs.sum())} "
              f"cells: max rel {rel[obs].max():.3e} median "
              f"{np.median(rel[obs]):.3e}; accept_count "
              f"{int(np.asarray(cd_a.accept_count)[k])} vs "
              f"{int(np.asarray(cd_s.accept_count)[0])}", flush=True)
        assert bool(np.asarray(cd_a.conv_normal)[k]) == bool(
            np.asarray(cd_s.conv_normal)[0])
        assert rel[obs].max() < REL_MAX


def test_inputs_that_would_be_ignored_are_refused(chem_capped):
    """A profile key the engine does not read (a typo, or a retired knob such
    as `fastchem_met_scale`) would leave its default in place silently, so it
    is refused. A warm column with the wrong trailing shape would BROADCAST --
    (N, 1, ni) seeds every layer from one layer -- so it is refused too."""
    chem = chem_capped
    with pytest.raises(ValueError, match="'warm_count_mx'"):
        vulcan_chem.build_chem_model({**PROFILE, "warm_count_mx": 10})
    with pytest.raises(ValueError, match="warm_y"):
        chem.converged_y_batch(jnp.asarray(THETAS),
                               warm_y=jnp.ones((THETAS.shape[0], 1, chem.ni)))
