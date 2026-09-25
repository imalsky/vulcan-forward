"""``converged_y_jvp``: D directions in ONE stacked tangent, not D calls.

The stacked ``(D, n_theta)`` tangent exists so the solver integrates the primal
once for all D (the tangent certificate is in the loop predicate, so a vmap of
the call would integrate it D times). The property that makes it a substitute:
direction d of the stack is the SAME derivative the single-direction call
returns.

Endpoint control: ``count_min = count_max = K`` pins every run at exactly K+1
accepted steps regardless of the certificate, so the stacked and the single
runs stop at the same primal state -- otherwise "every direction settled"
(stacked) and "this direction settled" (single) exit at different steps and the
two derivatives are of different functions.

Measured on this profile: the primal is bit-identical between the routes and
the tangents agree to max rel 6.2e-9 over cells above 1e-10 VMR (median 0);
the bar is 10x that. ``tangent_longdy`` (the tangent's lookback change,
normalised by primal quantities in ``outer_loop._tangent_conv``) carries the
same empirical bar.

A second model leaves the endpoint free, so there the tangent certificate
decides when the run stops.

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
pytest.importorskip("jax", reason="the tangent-certified runner is JAX code")

from vulcan_forward import vulcan_chem                       # noqa: E402

import jax.numpy as jnp                                      # noqa: E402

K = 40
PROFILE = {"use_photo": False, "yconv_cri": 1.0e-2, "nz": 20,
           "skip_warmup": True,
           "count_min": K, "count_max": K, "warm_count_max": K}
THETA = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
# [lnZ, c_o, lnKzz, T-offset]: unit directions in lnZ and in the T offset.
DIRS = np.eye(4, dtype=np.float64)[[0, 3]]
MIX_FLOOR = 1.0e-10   # cells below this carry no observable and no certificate
REL_MAX = 6.2e-8      # 10x the measured 6.2e-9
TL_REL_MAX = REL_MAX  # empirical: 2.9e-10 and 4.3e-9 measured, see the docstring


@pytest.fixture(scope="module")
def runs():
    """One stacked call and one call per direction, same theta, same endpoint."""
    try:
        chem = vulcan_chem.build_chem_model(PROFILE)
    except (FileNotFoundError, OSError) as e:                # pragma: no cover
        pytest.skip(f"chem model data unavailable: {e}")
    th = jnp.asarray(THETA)
    y_s, dy_s, cd_s = chem.converged_y_jvp(th, jnp.asarray(DIRS))
    singles = [chem.converged_y_jvp(th, jnp.asarray(DIRS[d]))
               for d in range(DIRS.shape[0])]
    return np.asarray(y_s), np.asarray(dy_s), cd_s, singles


def test_stacked_directions_reproduce_the_single_direction_tangents(runs):
    y_s, dy_s, cd_s, singles = runs
    assert dy_s.shape == (DIRS.shape[0],) + y_s.shape

    mix = y_s / y_s.sum(axis=1, keepdims=True)
    obs = mix > MIX_FLOOR
    for d, (y_1, dy_1, cd_1) in enumerate(singles):
        assert np.array_equal(y_s, np.asarray(y_1)), (
            f"direction {d}: the stacked call integrated a different primal")
        assert int(np.asarray(cd_1.accept_count)) == int(
            np.asarray(cd_s.accept_count)) == K + 1
        b = np.asarray(dy_1)
        rel = np.abs(dy_s[d] - b) / np.maximum(np.abs(b), 1e-300)
        print(f"[dir {d}] stacked-vs-single over {int(obs.sum())} cells > "
              f"{MIX_FLOOR:g} VMR: max rel {rel[obs].max():.3e}, median "
              f"{np.median(rel[obs]):.3e}", flush=True)
        assert rel[obs].max() < REL_MAX

    tl_s = float(cd_s.tangent_longdy)
    tl_max = max(float(cd_1.tangent_longdy) for _y, _dy, cd_1 in singles)
    print(f"[tangent_longdy] stacked {tl_s:.12g} vs max of the singles "
          f"{tl_max:.12g}", flush=True)
    assert abs(tl_s - tl_max) / tl_max < TL_REL_MAX


# The endpoint is free here: count_min is low enough that the certificate, not
# the floor, ends a warm continuation, and count_max far above any exit.
SETTLE_PROFILE = {"use_photo": False, "yconv_cri": 1.0e-2, "nz": 20,
                  "skip_warmup": True,
                  "count_min": 10, "count_max": 3000}
THETA_WARM = np.array([0.3, 0.1, 0.5, 40.0], dtype=np.float64)


@pytest.fixture(scope="module")
def chem_eq():
    try:
        return vulcan_chem.build_chem_model(SETTLE_PROFILE)
    except (FileNotFoundError, OSError) as e:                # pragma: no cover
        pytest.skip(f"chem model data unavailable: {e}")


def test_eq_seed_certifies_and_the_tangent_certificate_ends_the_run(chem_eq):
    """The equilibrium seed lands on the theta targets (the elemental
    projection closes it to rounding) and its cold solve certifies. From that
    converged column, a warm continuation to another theta: the primal alone
    certifies at some step; with a tangent along lnZ the solver runs on until
    the tangent settles too, and stops there -- after the primal's own exit
    and before the cap, certified (``conv_normal`` carries ``tangent_ok``)."""
    chem = chem_eq
    th0 = jnp.asarray(THETA)
    audit = chem.audit_init(th0)
    print(f"[eq seed] ratio_max_rel_err {audit['ratio_max_rel_err']:.3e}", flush=True)
    assert audit["ratio_max_rel_err"] < 1e-12
    y0, cd0 = chem.converged_y(th0, return_conv_diag=True)
    assert bool(cd0.conv_normal)

    th1 = jnp.asarray(THETA_WARM)
    warm = dict(warm_y=y0, lnZ_ref=float(THETA[0]), c_o_ref=float(THETA[1]))
    _y, cd_p = chem.converged_y(th1, return_conv_diag=True, **warm)
    _y, dy, cd_j = chem.converged_y_jvp(th1, jnp.asarray(DIRS[0]), **warm)
    acc_p, acc_j = int(cd_p.accept_count), int(cd_j.accept_count)
    print(f"[settle] accept_count primal {acc_p} jvp {acc_j} (cap "
          f"{chem.count_max}); tangent_longdy {float(cd_j.tangent_longdy):.3e}",
          flush=True)
    assert bool(cd_p.conv_normal) and bool(cd_j.conv_normal)
    assert np.all(np.isfinite(np.asarray(dy)))
    assert acc_p < acc_j <= chem.count_max
