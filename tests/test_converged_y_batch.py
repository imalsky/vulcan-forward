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

The measured differences are printed (``pytest -s``).

The warm-continuation arguments are covered too: the PER-LANE reference
composition the retrieval's warm starts need from a batched call.

Cheap profile: nz=20, photochemistry off -- the solve itself is what is under
test, not the column.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# Import order: skip via find_spec, never importorskip("exojax"); vulcan_chem before jax.
if importlib.util.find_spec("vulcan3") is None:           # pragma: no cover
    pytest.skip("vulcan3 not installed (light-CI environment)",
                allow_module_level=True)

from vulcan_forward import vulcan_chem                       # noqa: E402

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

PROFILE = {"use_photo": False, "yconv_cri": 1.0e-2, "nz": 20}
# [lnZ, c_o, lnKzz, T-offset]: the baseline column and a perturbed one.
THETAS = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.3, 0.1, 0.5, 40.0]], dtype=np.float64)
MIX_FLOOR = 1.0e-10   # cells below this carry no observable and no certificate
# Batch against vmap / scalar routes: two converged states at this loose
# yconv_cri differ at the convergence scale, measured up to 6e-2 and 8e-2 across
# Python 3.10-3.12 and numpy 1.26-2.x; a lane-mixing bug is O(1).
REL_MAX = 1.0e-1


@pytest.fixture(scope="module")
def runs():
    """The three routes, solved once: batched, per-lane vmap, batch-of-one."""
    chem = vulcan_chem.build_chem_model(PROFILE)
    th = jnp.asarray(THETAS)
    y_b, cd_b = chem.converged_y_batch(th, return_conv_diag=True)
    y_v, cd_v = jax.vmap(
        lambda t: chem.converged_y(t, return_conv_diag=True))(th)
    y_one = [np.asarray(chem.converged_y_batch(th[k:k + 1]))[0]
             for k in range(THETAS.shape[0])]
    return chem, np.asarray(y_b), cd_b, np.asarray(y_v), cd_v, y_one


def test_initial_column_carries_the_theta_elemental_ratios(runs):
    """lnZ scales every metal's ratio to H by e^lnZ and c_o scales carbon's by
    e^c_o on top (O stays at e^lnZ, He stays put), read off the built column
    against the baseline column, independently of the projection's targets."""
    chem = runs[0]
    lnZ, c_o = 0.3, 0.1
    audit = chem.audit_init(jnp.asarray([lnZ, c_o, 0.0, 0.0]))
    got, base = audit["ratios_to_H"], audit["baseline_ratios_to_H"]
    want = {"He": 1.0, "C": np.exp(lnZ + c_o)}
    for e in got:
        np.testing.assert_allclose(got[e] / base[e], want.get(e, np.exp(lnZ)),
                                   rtol=1e-10, err_msg=e)


@pytest.fixture(scope="module")
def chem_x():
    """The model with both optional entries, [lnZ, c_o, lnKzz, s_o, kzz_slope,
    T-offset], and Kzz as a power law about the reference level. No solve: the
    tests below read the built initial column and its Kzz."""
    return vulcan_chem.build_chem_model({**PROFILE, "kzz_mode": "powerlaw"},
                                        chem_extras=vulcan_chem.CHEM_EXTRAS)


def test_sulfur_knob_moves_only_sulfur(chem_x):
    """s_o scales sulfur's ratio to H by e^s_o on top of e^lnZ and nothing
    else, from the cold seed and from a warm guess at another sulfur
    reference, with every repair factor positive."""
    lnZ, c_o, s_o = 0.3, 0.1, -0.8
    th = jnp.asarray([lnZ, c_o, 0.0, s_o, 0.0, 0.0])
    audit = chem_x.audit_init(th)
    got, base = audit["ratios_to_H"], audit["baseline_ratios_to_H"]
    want = {"He": 1.0, "C": np.exp(lnZ + c_o), "S": np.exp(lnZ + s_o)}
    for e in got:
        np.testing.assert_allclose(got[e] / base[e], want.get(e, np.exp(lnZ)),
                                   rtol=1e-10, err_msg=e)
    assert abs(audit["dln_SO_achieved"] - s_o) < 1e-10
    assert audit["min_repair_factor"] > 0 and audit["ratio_max_rel_err"] < 1e-9
    warm = chem_x.audit_init(th, warm_y=jnp.asarray(chem_x.y0), lnZ_ref=0.0,
                             c_o_ref=0.0, s_o_ref=0.4)
    assert warm["min_repair_factor"] > 0 and warm["ratio_max_rel_err"] < 1e-9
    assert abs(warm["dln_SO_achieved"] - s_o) < 1e-10


def test_kzz_power_law_is_anchored_on_the_profile_at_the_reference_level(runs, chem_x):
    """kzz_mode="powerlaw": slope 0 is a constant Kzz equal to the config
    profile's value at the reference level (log-log interpolated), and a slope
    tilts it as (P / P_ref)^-slope times e^lnKzz; the default mode is the
    profile times e^lnKzz."""
    from vulcan3.atm_setup import compute_pico
    from vulcan_forward import constants
    chem = runs[0]
    p_int = (np.asarray(compute_pico(jnp.asarray(chem.p_bar * constants.BAR_CGS)))[1:-1]
             / constants.BAR_CGS)
    kzz_prof = np.asarray(chem.prep_pv(jnp.asarray([0.0, 0.0, 0.0, 0.0])).Kzz)
    np.testing.assert_allclose(
        np.asarray(chem.prep_pv(jnp.asarray([0.0, 0.0, np.log(3.0), 0.0])).Kzz),
        3.0 * kzz_prof, rtol=1e-12)
    flat = np.asarray(chem_x.prep_pv(jnp.zeros(6)).Kzz)
    assert np.ptp(flat) == 0.0
    o = np.argsort(p_int)
    ref = np.exp(np.interp(np.log(constants.KZZ_P_REF_BAR), np.log(p_int[o]),
                           np.log(kzz_prof[o])))
    np.testing.assert_allclose(flat[0], ref, rtol=1e-12)
    slope = 0.5
    tilted = np.asarray(chem_x.prep_pv(
        jnp.asarray([0.0, 0.0, np.log(2.0), 0.0, slope, 0.0])).Kzz)
    np.testing.assert_allclose(
        tilted, 2.0 * ref * (p_int / constants.KZZ_P_REF_BAR) ** (-slope), rtol=1e-12)


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


def test_per_lane_references_match_the_scalar_calls(runs):
    """``lnZ_ref`` / ``c_o_ref`` as (N,) arrays give every lane the reference
    composition ITS carried column was converged at -- what the retrieval's
    warm starts need, since each row starts from its own anchor.

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


def test_inputs_that_would_be_ignored_are_refused(runs):
    """A profile key the engine does not read (a typo, or a retired knob such
    as `warm_count_max`) would leave its default in place silently, so it
    is refused. A warm column with the wrong trailing shape would BROADCAST --
    (N, 1, ni) seeds every layer from one layer -- so it is refused too."""
    chem = runs[0]
    with pytest.raises(ValueError, match="'warm_count_max'"):
        vulcan_chem.build_chem_model({**PROFILE, "warm_count_max": 10})
    with pytest.raises(ValueError, match="warm_y"):
        chem.converged_y_batch(jnp.asarray(THETAS),
                               warm_y=jnp.ones((THETAS.shape[0], 1, chem.ni)))
