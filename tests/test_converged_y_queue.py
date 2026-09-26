"""``converged_y_queue``: the batched cold solve on a fixed number of lanes fed
from a queue of thetas (vulcan-jax ``OuterLoop.run_queue``).

Three properties:

* with at least as many lanes as thetas nothing is ever refilled, so every lane
  runs the ticks the plain batch gives it: from one identical starting column,
  the same accept_count and certificate. Cold, the y's agree at the
  convergence scale only -- ``run_queue`` builds the seed inside its jitted
  loop while ``converged_y_batch`` builds it outside, and the two compilations
  of ``_prep`` can differ by a ulp in y_ini, which the trajectory amplifies;
* with fewer lanes a theta enters the loop at the tick its lane was freed at,
  which moves the photolysis / geometry cadence exactly as the batch already
  moves it against the solo solve: agreement is again at the CONVERGENCE
  scale, and the certificate must still match theta for theta;
* the warm-started stage 2 of a two-stage cold solve -- the shape
  vulcan-retrieval's cold path runs -- keeps every theta certified and leaves
  the species the spectrum reads alone.

The two acceptance tests at the bottom pin a consumer's batched GRADIENT
against the per-theta solve: the cold two-stage map, and the WARM-capped
mutation map with a per-lane reference composition.

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

from vulcan_forward import vulcan_chem                       # noqa: E402

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

# dt_max: the production step cap (vulcan-retrieval's case sets 1e11 s).
PROFILE = {"use_photo": False, "yconv_cri": 1.0e-2, "nz": 20,
           "skip_warmup": True, "dt_max": 1.0e11}
# [lnZ, c_o, lnKzz, T-offset]: the baseline column and three perturbed ones,
# heterogeneous enough that they do not all converge at the same tick.
THETAS = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.3, 0.1, 0.5, 40.0],
                   [-0.3, -0.1, 0.0, -40.0],
                   [0.1, 0.0, 0.2, 80.0]], dtype=np.float64)
MIX_FLOOR = 1.0e-10   # cells below this carry no observable and no certificate
REL_MAX = 5.0e-2
# The queued tangent against the batched one: a refilled lane enters at a
# later tick, so its geometry-refresh cadence differs from the batch's even at
# the same accept count. Theta 2 (a loose-branch exit with its bottom-layer
# sulfur still relaxing) has no determined queue tangent -- a 1e-12 nudge to
# lnZ moves it by ~7e-2 and central FD at h 1e-4 and 1e-5 disagree by 0.23
# there (notes §0) -- so it keeps a wiring bound only: finite, and not off by
# an order of magnitude.
QUEUE_DY_MAX = 1.0e-2
QUEUE_DY_MAX_UNSETTLED = 1.0
UNSETTLED_THETAS = (2,)
# Two-stage check: the species a W39b spectrum reads, split into the ones that
# are PINNED and the ones only PRINTED, which carry the slow chemistry the two
# arms disagree on.
PINNED_SPECIES = ("H2O", "CO", "H2", "He")
PRINTED_SPECIES = ("CO2", "CH4", "SO2")
SPECIES_FLOOR = 1.0e-6   # layers below this carry no opacity
SPECIES_MAX = 1.0e-3


@pytest.fixture(scope="module")
def chem():
    return vulcan_chem.build_chem_model(PROFILE)


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
    # `conv_branch` is NOT compared: the queue's seed is built inside its
    # jitted loop, and when the controlling cell is an ultratrace species (S4
    # at 1e-20 VMR on theta 0) the same accept count certifies tight in one
    # compilation and loose in another (longdy 0.002 against 0.07).
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


def test_queue_agrees_with_the_batch_with_and_without_refill(chem, ref):
    """Four lanes for four thetas: nothing refills, so every job runs the
    plain batch's ticks. Started from one identical column (the model's y0,
    as a continuation) the two routes take the same accept count and
    certificate; cold, each builds its own seed (the queue inside its jitted
    loop), which can differ at the last bit, so the cold arm and the refilled
    two-lane arm (a theta enters at the tick its lane was freed at) agree at
    the convergence scale, with the certificate theta for theta."""
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
    _report("lanes=4", y_q, cd_q, y_b, cd_b)

    # The batch-tick check: both routes from one column, so only the queue's
    # tick scheduling is under test.
    y0 = jnp.broadcast_to(jnp.asarray(chem.y0), (THETAS.shape[0],) + chem.y0.shape)
    same = {"warm_y": y0, "lnZ_ref": 0.0, "c_o_ref": 0.0}
    y_qs, cd_qs = chem.converged_y_queue(th, n_lanes=THETAS.shape[0], **same)
    y_bs, cd_bs = chem.converged_y_batch(th, return_conv_diag=True, **same)
    assert np.array_equal(np.asarray(cd_qs.accept_count),
                          np.asarray(cd_bs.accept_count))
    _report("lanes=4 from y0", y_qs, cd_qs, np.asarray(y_bs), cd_bs)

    y_q, cd_q = chem.converged_y_queue(th, n_lanes=2, chunk=1)
    _report("lanes=2 chunk=1", y_q, cd_q, y_b, cd_b)


def test_two_stage_warm_queue_keeps_the_observed_species(chem):
    """The shape vulcan-retrieval's cold path runs: stage 1 at baseline
    composition (batched once, shared), stage 2 warm-started from its columns
    and queued on 2 lanes.

    What is pinned is the certificate and the species the spectrum reads. What
    is NOT pinned, deliberately: from a converged warm start the loose branch
    (longdy < yconv_min, 0.1 in the production configs too) can fire early, so
    the two arms certify at different relaxation depths and may certify on
    different branches. The difference that buys concentrates in the slow
    sulfur chemistry and in CO2 / CH4 / SO2 at the percent level, all of it
    printed here (readings: notes §2 "The speed branch"). A
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


def _dy_rel(dy, dy_ref, mix):
    """max|dy - dy_ref| / max|dy_ref| over the cells the spectrum can see.

    Two AD routes are compared against the DIRECTION's own scale, never
    component by component: a cell whose tangent is accidentally tiny turns
    floating-point accumulation into a meaningless ratio.
    """
    obs = mix > MIX_FLOOR
    return float(np.abs(dy - dy_ref)[obs].max() / np.abs(dy_ref[obs]).max())


def test_queue_is_differentiable(chem, ref):
    """A plain ``jax.jvp`` runs through the queue, and through the batch.

    This is what lets a consumer take the WHOLE cold gradient on the batched
    runner instead of a per-theta vmap of jvps: JAX carries the tangent
    leaves through the refill scatters, the chunk selection and the lane
    `lax.cond` as it does through the lockstep batch. The stopping rule is
    unchanged -- the loop predicate reads the primal only -- so the tangent is
    the plain jvp's, and the three routes (queue, plain batch, solo
    ``converged_y``) agree at the convergence scale their primals agree at.

    Forward-mode callers that need a CERTIFIED tangent still go through
    ``converged_y_jvp``; this test pins the plain-jvp route the retrieval's
    cold gradient uses.
    """
    y_b_ref, cd_b_ref = ref
    th = jnp.asarray(THETAS)
    # one direction (lnZ), the same for every theta in the stack
    v = jnp.zeros_like(th).at[:, 0].set(1.0)

    (y_q, cd_q), (dy_q, _dcd) = jax.jvp(
        lambda t: chem.converged_y_queue(t, 2, chunk=1), (th,), (v,))
    (y_b, cd_b), (dy_b, _dcd_b) = jax.jvp(
        lambda t: chem.converged_y_batch(t, return_conv_diag=True), (th,), (v,))
    dy_q, dy_b, y_b = np.asarray(dy_q), np.asarray(dy_b), np.asarray(y_b)
    assert np.all(np.isfinite(dy_q)) and np.all(np.isfinite(dy_b))
    # the primal of the jvp'd batch is the primal-only batch, bit for bit
    assert np.array_equal(y_b, y_b_ref)
    assert np.array_equal(np.asarray(cd_b.conv_normal),
                          np.asarray(cd_b_ref.conv_normal))
    _report("jvp lanes=2 chunk=1", y_q, cd_q, y_b, cd_b)
    for k in range(THETAS.shape[0]):
        mix = y_b[k] / y_b[k].sum(axis=1, keepdims=True)
        rel = _dy_rel(dy_q[k], dy_b[k], mix)
        print(f"[jvp lanes=2 chunk=1 job {k}] queue-vs-batch tangent "
              f"max|ddy|/max|dy| {rel:.3e}", flush=True)
        assert rel < (QUEUE_DY_MAX_UNSETTLED if k in UNSETTLED_THETAS else QUEUE_DY_MAX)

    # The contract a batched-gradient consumer relies on: one jvp through the
    # batch is the per-theta jvp through the solo runner, at the convergence
    # scale and with the same certificate (a lane is its own solve, so the
    # four-lane batch's first two lanes stand for a batch of two).
    for k in range(2):
        (y_s, cd_s), (dy_s, _ds) = jax.jvp(
            lambda t: chem.converged_y(t, return_conv_diag=True),
            (th[k],), (v[k],))
        y_s, dy_s = np.asarray(y_s), np.asarray(dy_s)
        mix = y_s / y_s.sum(axis=1, keepdims=True)
        rel = _dy_rel(dy_b[k], dy_s, mix)
        print(f"[jvp batch-vs-solo job {k}] tangent max|ddy|/max|dy| {rel:.3e}; "
              f"accept_count batch {int(np.asarray(cd_b.accept_count)[k])} "
              f"solo {int(np.asarray(cd_s.accept_count))}", flush=True)
        assert bool(np.asarray(cd_s.conv_normal))
        assert bool(np.asarray(cd_b.conv_normal)[k])
        assert rel < REL_MAX


# --- the three-route acceptance test for a batched cold GRADIENT consumer ---
# The complete two-stage cold map (stage 1 at baseline composition, stage 2 warm
# from its own column) with PHOTOCHEMISTRY ON, differentiated along every
# chemistry direction, on three routes:
#   solo   per theta, vmap over directions of jvp(converged_y)
#   batch  vmap over directions of jvp(converged_y_batch)
#   queue  vmap over directions of jvp(converged_y_queue) on 2 lanes, chunk 1
# plus the queue replayed with the jobs REVERSED (its lanes then free at other
# ticks). Photo on is the point: the photolysis and geometry-refresh cadences are
# what the batched runner moves from the lane's own accept count to the loop tick.
# warm_count_max is the retrieval's production mutation cap against a cold cap
# of 30000: it makes the warm route below run the CAPPED carry, and it leaves
# the cold route untouched (warm_cap=False keeps the cold cap).
PHOTO_PROFILE = {"use_photo": True, "yconv_cri": 1.0e-2, "nz": 20,
                 "skip_warmup": True,
                 "warm_count_max": 1500}
# An MCMC-sized proposal step away from the carried column, per direction.
WARM_DELTA = np.array([0.05, 0.02, 0.1, 5.0], dtype=np.float64)
# PREDECLARED from the measurement (worst over the four jobs and all four
# comparisons): column worst cell 1.70e-1, column median 1.99e-4, tangent
# stack-relative 2.87e-5. The bounds sit a modest factor above each.
COL_MAX = 4.0e-1
COL_MED = 1.0e-3
DY_STACK_MAX = 2.0e-4


@pytest.fixture(scope="module")
def chem_photo():
    return vulcan_chem.build_chem_model(PHOTO_PROFILE)


def _stack_thetas(outs):
    """The per-theta solo results stacked along a leading theta axis. A Python
    loop, not a vmap over thetas: each theta is its own solve (the vmap would
    run them in lockstep), and one compiled program serves all four."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *outs)


def _two_stage_routes(chem, th):
    """(solo, batch, queue, queue-reversed) results of the two-stage cold map,
    each ``(y (N, nz, ni), ConvDiag over N, dy (N, D, nz, ni))``."""
    n_dir = th.shape[1]
    eye = jnp.eye(n_dir, dtype=jnp.float64)

    def pad(dy1):
        # directions 0,1 (lnZ, c_o) carry no stage-1 tangent: stage 1 zeroes them
        return jnp.zeros((n_dir,) + dy1.shape[1:], dy1.dtype).at[2:].set(dy1)

    def solo(TH):
        def s1(t):
            return chem.converged_y(t.at[0].set(0.0).at[1].set(0.0))

        def s2(t, y1):
            return chem.converged_y(t, warm_y=y1, lnZ_ref=0.0, c_o_ref=0.0,
                                    return_conv_diag=True)

        def one(t):
            y1_l, dy1 = jax.vmap(lambda v: jax.jvp(s1, (t,), (v,)))(eye[2:])
            (y_l, cd_l), (dy_l, _d) = jax.vmap(
                lambda v, dy: jax.jvp(s2, (t, y1_l[0]), (v, dy)))(eye, pad(dy1))
            return y_l[0], jax.tree_util.tree_map(lambda x: x[0], cd_l), dy_l
        return _stack_thetas([one(TH[k]) for k in range(TH.shape[0])])

    def stacked(solve1, solve2, TH):
        bc = lambda v: jnp.broadcast_to(v, TH.shape)     # noqa: E731
        Y1_l, dY1 = jax.vmap(lambda v: jax.jvp(solve1, (TH,), (bc(v),)))(eye[2:])
        (Y_l, CD_l), (dY_l, _d) = jax.vmap(
            lambda v, dY: jax.jvp(solve2, (TH, Y1_l[0]), (bc(v), dY))
        )(eye, pad(dY1))
        return (Y_l[0], jax.tree_util.tree_map(lambda x: x[0], CD_l),
                jnp.swapaxes(dY_l, 0, 1))

    def batch(TH):
        return stacked(
            lambda C: chem.converged_y_batch(C.at[:, 0].set(0.0).at[:, 1].set(0.0)),
            lambda C, Y1: chem.converged_y_batch(C, warm_y=Y1, lnZ_ref=0.0,
                                                 c_o_ref=0.0, return_conv_diag=True),
            TH)

    def queue(TH):
        return stacked(
            lambda C: chem.converged_y_queue(
                C.at[:, 0].set(0.0).at[:, 1].set(0.0), 2, chunk=1)[0],
            lambda C, Y1: chem.converged_y_queue(C, 2, chunk=1, warm_y=Y1,
                                                 lnZ_ref=0.0, c_o_ref=0.0),
            TH)

    rev = queue(th[::-1])
    rev = (rev[0][::-1], jax.tree_util.tree_map(lambda x: x[::-1], rev[1]),
           rev[2][::-1])
    return solo(th), batch(th), queue(th), rev


def _cmp(tag, a, b, n_dir):
    """Print and bound one route pair. ``b`` is the reference.

    The column is judged over the cells the spectrum can see; the tangent by the
    DIRECTION STACK's own scale (max|ddy| over all directions / max|dy| over all
    directions). Per-direction ratios are PRINTED with each direction's share of
    that scale, not asserted: the lnKzz tangent is ~1e-6 of the lnZ tangent here,
    and a norm-relative error on a direction that carries no signal is
    floating-point accumulation, not disagreement.
    """
    ya, cda, dya = np.asarray(a[0]), a[1], np.asarray(a[2])
    yb, cdb, dyb = np.asarray(b[0]), b[1], np.asarray(b[2])
    assert np.array_equal(np.asarray(cda.conv_normal), np.asarray(cdb.conv_normal))
    for k in range(yb.shape[0]):
        mix = yb[k] / yb[k].sum(axis=1, keepdims=True)
        obs = mix > MIX_FLOOR
        rel = np.abs(ya[k] - yb[k]) / np.maximum(np.abs(yb[k]), 1e-300)
        scale = float(np.abs(dyb[k][:, obs]).max())
        stack = float(np.abs(dya[k] - dyb[k])[:, obs].max() / scale)
        per = [float(np.abs(dya[k, i] - dyb[k, i])[obs].max()
                     / max(float(np.abs(dyb[k, i][obs]).max()), 1e-300))
               for i in range(n_dir)]
        frac = [float(np.abs(dyb[k, i][obs]).max() / scale) for i in range(n_dir)]
        print(f"[{tag} job {k}] y max rel {rel[obs].max():.2e} median "
              f"{np.median(rel[obs]):.2e} over {int(obs.sum())} cells; dy "
              f"stack-rel {stack:.2e}; per direction "
              + " ".join(f"{p:.2e}(scale {f:.0e})" for p, f in zip(per, frac)),
              flush=True)
        assert rel[obs].max() < COL_MAX
        assert float(np.median(rel[obs])) < COL_MED
        assert stack < DY_STACK_MAX


def test_two_stage_cold_gradient_three_routes(chem_photo):
    """The batched and queued cold GRADIENT agree with the per-theta one.

    The acceptance test for a consumer's cold two-stage gradient on the
    batched runner (and the queue), against a per-theta solve: the
    tangents stay finite, every theta keeps its certificate on every route, and
    the columns and the direction stack agree at the convergence scale the
    primals already agree at. Accept counts are PRINTED, not pinned: a batched
    lane's photo / refresh cadence rides the loop tick, and a queued lane enters
    at the tick its lane was freed at, so the routes certify at different
    relaxation depths -- which is the same freedom the batch already has against
    the solo solve.
    """
    th = jnp.asarray(THETAS)
    n_dir = int(th.shape[1])
    solo, batch, queue, rev = _two_stage_routes(chem_photo, th)
    for tag, r in (("solo", solo), ("batch", batch), ("queue", queue),
                   ("queue-rev", rev)):
        y, cd, dy = np.asarray(r[0]), r[1], np.asarray(r[2])
        print(f"[{tag}] conv_normal {np.asarray(cd.conv_normal).astype(int).tolist()} "
              f"branch {np.asarray(cd.conv_branch).astype(int).tolist()} "
              f"accept {np.asarray(cd.accept_count).astype(int).tolist()}", flush=True)
        # every theta certifies on every route, and its tangent is finite
        assert bool(np.all(np.asarray(cd.conv_normal))), tag
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(dy)), tag
        assert float(np.abs(dy).max()) > 0.0, tag
    # Not vacuous: the four thetas are heterogeneous, so their columns differ by
    # far more than any route-to-route difference below.
    y0 = np.asarray(solo[0])
    mix = y0[0] / y0[0].sum(axis=1, keepdims=True)
    spread = float((np.abs(y0[1] - y0[0])
                    / np.maximum(np.abs(y0[0]), 1e-300))[mix > MIX_FLOOR].max())
    print(f"[spread] job1-vs-job0 column max rel {spread:.2e}", flush=True)
    assert spread > COL_MAX
    _cmp("batch-vs-solo", batch, solo, n_dir)
    _cmp("queue-vs-batch", queue, batch, n_dir)
    _cmp("queue-vs-solo", queue, solo, n_dir)
    _cmp("queuerev-vs-queue", rev, queue, n_dir)


def _warm_routes(chem, th, yw, r0, r1):
    """(solo, batch, queue, queue-reversed) results of the WARM-capped
    continuation map -- each theta from its OWN carried column ``yw`` at its
    OWN reference composition (r0, r1) -- as
    ``(y (N, nz, ni), ConvDiag over N, dy (N, D, nz, ni))``.

    The carried column and the references are CONSTANTS of the map (the
    consumer differentiates with respect to theta only), so they ride in as
    values and only theta carries a tangent.
    """
    n_dir = th.shape[1]
    eye = jnp.eye(n_dir, dtype=jnp.float64)

    def solo(TH):
        def one(t, y_w, a, b):
            def f(tt):
                return chem.converged_y(tt, warm_y=y_w, lnZ_ref=a, c_o_ref=b,
                                        warm_cap=True, return_conv_diag=True)
            (y_l, cd_l), (dy_l, _d) = jax.vmap(
                lambda v: jax.jvp(f, (t,), (v,)))(eye)
            return y_l[0], jax.tree_util.tree_map(lambda x: x[0], cd_l), dy_l
        return _stack_thetas([one(TH[k], yw[k], r0[k], r1[k])
                              for k in range(TH.shape[0])])

    def stacked(solve, TH):
        bc = lambda v: jnp.broadcast_to(v, TH.shape)     # noqa: E731
        (Y_l, CD_l), (dY_l, _d) = jax.vmap(
            lambda v: jax.jvp(solve, (TH,), (bc(v),)))(eye)
        return (Y_l[0], jax.tree_util.tree_map(lambda x: x[0], CD_l),
                jnp.swapaxes(dY_l, 0, 1))

    def batch(TH, y_w=yw, a=r0, b=r1):
        return stacked(lambda C: chem.converged_y_batch(
            C, warm_y=y_w, lnZ_ref=a, c_o_ref=b, warm_cap=True,
            return_conv_diag=True), TH)

    def queue(TH, y_w=yw, a=r0, b=r1):
        return stacked(lambda C: chem.converged_y_queue(
            C, 2, chunk=1, warm_y=y_w, lnZ_ref=a, c_o_ref=b, warm_cap=True), TH)

    rev = queue(TH=th[::-1], y_w=yw[::-1], a=r0[::-1], b=r1[::-1])
    rev = (rev[0][::-1], jax.tree_util.tree_map(lambda x: x[::-1], rev[1]),
           rev[2][::-1])
    return solo(th), batch(th), queue(th), rev


def test_warm_capped_gradient_three_routes(chem_photo):
    """The batched and queued WARM mutation gradient agree with the per-theta one.

    The acceptance test for vulcan-retrieval's warm mutation kernel: the cap
    rides the runner carry per lane (that it binds is pinned in
    test_converged_y_batch), each lane gets its own carried column AND
    its own reference composition, the tangents stay finite, every theta keeps
    its certificate on every route, and the columns and the direction stack
    agree at the convergence scale the primals agree at. Accept counts are
    PRINTED, not pinned -- a batched lane's photo / refresh cadence rides the
    loop tick and a queued lane enters at the tick its lane was freed at, so
    from a converged warm start the loose branch can fire at different
    relaxation depths.
    """
    chem = chem_photo
    th0 = jnp.asarray(THETAS)
    th1 = jnp.asarray(THETAS + WARM_DELTA[None, :])
    n_dir = int(th1.shape[1])
    # the carried cloud: each theta's own converged column, with the reference
    # composition it was converged at (distinct per lane)
    yw, cd0 = chem.converged_y_batch(th0, return_conv_diag=True)
    assert bool(np.all(np.asarray(cd0.conv_normal))), "carried columns not certified"
    r0, r1 = th0[:, 0], th0[:, 1]
    assert len(set(np.asarray(THETAS)[:, 0].tolist())) > 1   # references differ
    solo, batch, queue, rev = _warm_routes(chem, th1, yw, r0, r1)
    for tag, r in (("solo", solo), ("batch", batch), ("queue", queue),
                   ("queue-rev", rev)):
        y, cd, dy = np.asarray(r[0]), r[1], np.asarray(r[2])
        print(f"[warm {tag}] conv_normal "
              f"{np.asarray(cd.conv_normal).astype(int).tolist()} branch "
              f"{np.asarray(cd.conv_branch).astype(int).tolist()} accept "
              f"{np.asarray(cd.accept_count).astype(int).tolist()}", flush=True)
        assert bool(np.all(np.asarray(cd.conv_normal))), tag
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(dy)), tag
        assert float(np.abs(dy).max()) > 0.0, tag
    _cmp("warm batch-vs-solo", batch, solo, n_dir)
    _cmp("warm queue-vs-batch", queue, batch, n_dir)
    _cmp("warm queue-vs-solo", queue, solo, n_dir)
    _cmp("warm queuerev-vs-queue", rev, queue, n_dir)
