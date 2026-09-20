"""``converged_y_queue``: the batched cold solve on a fixed number of lanes fed
from a queue of thetas (vulcan-jax ``OuterLoop.run_queue``).

Three properties:

* with at least as many lanes as thetas nothing is ever refilled, so every lane
  runs the ticks the plain batch gives it: same accept_count, same certificate,
  and the y's agree at the convergence scale. NOT bitwise -- ``run_queue``
  builds the seed inside its jitted loop while ``converged_y_batch`` builds it
  outside, and the two compilations of ``_prep`` differ by a ulp in y_ini
  (measured 1.1e-15), which the trajectory amplifies;
* with fewer lanes a theta enters the loop at the tick its lane was freed at,
  which moves the photolysis / geometry cadence exactly as the batch already
  moves it against the solo solve: agreement is again at the CONVERGENCE
  scale, and the certificate must still match theta for theta;
* the warm-started stage 2 of a two-stage cold solve -- the shape
  vulcan-retrieval's cold path runs -- keeps every theta certified and leaves
  the species the spectrum reads alone.

The measured differences are printed (``pytest -s``).

Cheap profile: nz=20, photochemistry off, no build-time warm-up solve.
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

import jax.numpy as jnp                                      # noqa: E402

PROFILE = {"use_photo": False, "yconv_cri": 1.0e-2, "nz": 20,
           "abundance_mode": "elemental", "skip_warmup": True}
# [lnZ, c_o, lnKzz, T-offset]: the baseline column and three perturbed ones,
# heterogeneous enough that they do not all converge at the same tick.
THETAS = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.3, 0.1, 0.5, 40.0],
                   [-0.3, -0.1, 0.0, -40.0],
                   [0.1, 0.0, 0.2, 80.0]], dtype=np.float64)
MIX_FLOOR = 1.0e-10   # cells below this carry no observable and no certificate
REL_MAX = 5.0e-2
# Two-stage check: the species a W39b spectrum reads, split into the ones that
# are PINNED (measured <= 2.6e-4, bound at 4x that) and the ones only PRINTED,
# which carry the slow chemistry the two arms disagree on.
PINNED_SPECIES = ("H2O", "CO", "H2", "He")
PRINTED_SPECIES = ("CO2", "CH4", "SO2")
SPECIES_FLOOR = 1.0e-6   # layers below this carry no opacity
SPECIES_MAX = 1.0e-3


@pytest.fixture(scope="module")
def chem():
    try:
        return vulcan_chem.build_chem_model(PROFILE)
    except (FileNotFoundError, OSError) as e:                # pragma: no cover
        pytest.skip(f"chem model data unavailable: {e}")


@pytest.fixture(scope="module")
def ref(chem):
    """The plain batch, solved once: the reference both tests compare to."""
    y_b, cd_b = chem.converged_y_batch(jnp.asarray(THETAS), return_conv_diag=True)
    return np.asarray(y_b), cd_b


def _report(tag, y_q, cd_q, y_b, cd_b):
    """Print and check the queue-vs-batch difference job by job."""
    y_q = np.asarray(y_q)
    # Certified in BOTH arms: equal certificates alone would pass with both
    # arms uncertified.
    assert bool(np.all(np.asarray(cd_q.conv_normal)))
    assert bool(np.all(np.asarray(cd_b.conv_normal)))
    assert np.array_equal(np.asarray(cd_q.conv_normal),
                          np.asarray(cd_b.conv_normal))
    assert np.array_equal(np.asarray(cd_q.conv_branch),
                          np.asarray(cd_b.conv_branch))
    for k in range(THETAS.shape[0]):
        mix_b = y_b[k] / y_b[k].sum(axis=1, keepdims=True)
        obs = mix_b > MIX_FLOOR
        rel = np.abs(y_q[k] - y_b[k]) / np.maximum(np.abs(y_b[k]), 1e-300)
        print(f"[{tag} job {k}] queue-vs-batch over {int(obs.sum())} cells > "
              f"{MIX_FLOOR:g} VMR: max rel {rel[obs].max():.3e}, median "
              f"{np.median(rel[obs]):.3e}; accept_count queue="
              f"{int(np.asarray(cd_q.accept_count)[k])} batch="
              f"{int(np.asarray(cd_b.accept_count)[k])}", flush=True)
        assert rel[obs].max() < REL_MAX


def test_queue_with_enough_lanes_runs_the_batch_ticks(chem, ref):
    y_b, cd_b = ref
    th = jnp.asarray(THETAS)
    y_q, cd_q = chem.converged_y_queue(th, n_lanes=THETAS.shape[0])
    # The same call signature reuses the same compiled program: the init_fn /
    # out_fn pair is memoized, so run_queue's cache does not grow per call.
    # The second call takes the thetas REVERSED, which no frozen first-call
    # result could pass: with n_lanes >= N nothing refills, every lane starts
    # at tick 0 and is independent of its neighbours, so the answer must be
    # the first call's, permuted, job for job.
    n_programs = len(chem._integ._vrunner_queue)
    y_q2, cd_q2 = chem.converged_y_queue(th[::-1], n_lanes=THETAS.shape[0])
    y_q2, y_q = np.asarray(y_q2), np.asarray(y_q)
    print(f"[lanes=4 reversed] accept_count {np.asarray(cd_q2.accept_count).tolist()}"
          f" vs first call reversed {np.asarray(cd_q.accept_count)[::-1].tolist()};"
          f" max|y2 - y1[::-1]| {np.abs(y_q2 - y_q[::-1]).max():.3e}", flush=True)
    assert len(chem._integ._vrunner_queue) == n_programs
    assert np.array_equal(y_q2, y_q[::-1])
    assert np.array_equal(np.asarray(cd_q2.accept_count),
                          np.asarray(cd_q.accept_count)[::-1])
    assert np.array_equal(np.asarray(cd_q2.conv_normal),
                          np.asarray(cd_q.conv_normal)[::-1])
    # No refill happens, so every job sees the ticks the plain batch gives it:
    # the step counts match exactly, the y's only to the seed's ulp.
    assert np.array_equal(np.asarray(cd_q.accept_count),
                          np.asarray(cd_b.accept_count))
    _report("lanes=4", y_q, cd_q, y_b, cd_b)


def test_queue_with_refill_agrees_at_the_convergence_scale(chem, ref):
    y_b, cd_b = ref
    y_q, cd_q = chem.converged_y_queue(jnp.asarray(THETAS), n_lanes=2, chunk=1)
    _report("lanes=2 chunk=1", y_q, cd_q, y_b, cd_b)


def test_two_stage_warm_queue_keeps_the_observed_species(chem):
    """The shape vulcan-retrieval's cold path runs: stage 1 at baseline
    composition (batched once, shared), stage 2 warm-started from its columns
    and queued on 2 lanes.

    What is pinned is the certificate and the species the spectrum reads. What
    is NOT pinned, deliberately: from a converged warm start the loose branch
    (longdy < yconv_min, 0.1 in the production configs too) can fire early, so
    the two arms certify at different relaxation depths -- 408 accepted steps
    batched against 124 queued, measured on job 3 -- and may certify on
    different branches. The difference that buys concentrates in the slow
    sulfur chemistry (worst cells S / SH just above 1e-6 VMR, up to 5.9e-2)
    and in CO2 / CH4 / SO2 at the percent level, all of it printed here. A
    worst-cell bound would have to sit above the cold cases' own, which would
    pin nothing; the production gate for `cold_lanes` is the per-draw
    log-likelihood comparison of the GPU bench, not this test."""
    th = jnp.asarray(THETAS)
    y1 = chem.converged_y_batch(th.at[:, 0].set(0.0).at[:, 1].set(0.0))
    y_b, cd_b = chem.converged_y_batch(th, warm_y=y1, lnZ_ref=0.0, c_o_ref=0.0,
                                       return_conv_diag=True)
    y_q, cd_q = chem.converged_y_queue(th, n_lanes=2, chunk=1, warm_y=y1,
                                       lnZ_ref=0.0, c_o_ref=0.0)
    y_b, y_q = np.asarray(y_b), np.asarray(y_q)
    print(f"[two-stage lanes=2 chunk=1] accept_count batch "
          f"{np.asarray(cd_b.accept_count).tolist()} queue "
          f"{np.asarray(cd_q.accept_count).tolist()}", flush=True)
    assert bool(np.all(np.asarray(cd_b.conv_normal)))
    assert bool(np.all(np.asarray(cd_q.conv_normal)))
    assert np.array_equal(np.asarray(cd_q.conv_normal),
                          np.asarray(cd_b.conv_normal))
    mix = [y_b[k] / y_b[k].sum(axis=1, keepdims=True)
           for k in range(THETAS.shape[0])]
    for name in PINNED_SPECIES + PRINTED_SPECIES:
        i = chem.sidx[name]
        pinned = name in PINNED_SPECIES
        worst = []
        for k in range(THETAS.shape[0]):
            obs = mix[k][:, i] > SPECIES_FLOOR
            if not obs.any():   # this job has no opacity-carrying layer of it
                worst.append(np.nan)
                continue
            rel = (np.abs(y_q[k, :, i] - y_b[k, :, i])
                   / np.maximum(np.abs(y_b[k, :, i]), 1e-300))
            worst.append(float(rel[obs].max()))
        print(f"[two-stage {name:>3} {'pinned ' if pinned else 'printed'}] max rel "
              f"over layers > {SPECIES_FLOOR:g} VMR per job: "
              + "  ".join("    n/a" if not np.isfinite(w) else f"{w:.2e}"
                            for w in worst), flush=True)
        if pinned:
            assert np.all(np.isfinite(worst)), name
            assert max(worst) < SPECIES_MAX, name
