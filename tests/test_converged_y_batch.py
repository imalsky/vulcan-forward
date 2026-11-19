"""``converged_y_batch``: the batched runner as a drop-in for a per-lane ``jax.vmap(converged_y)``.

A lane must be its own solve: lanes freeze at their own exits, so lane k of an N-lane batch is bit-identical to the same
theta beside other lanes. The reference keeps the width, because a batch of one compiles to a different XLA program and
can split by one ulp (notes "Batched routes"). Against a vmap of the per-lane solve the agreement is at the convergence
scale by design: the while loop sits above the lane vmap, so photolysis and the geometry refresh key to the loop's tick.
The certificate must match lane for lane. The per-lane reference composition of a warm batch is covered too.

The measured differences are printed (``pytest -s``). Cheap profile: nz=20, photochemistry off.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# Import order: skip via find_spec, never importorskip("exojax"); vulcan_chem before jax.
if importlib.util.find_spec("vulcan3") is None:
    pytest.skip("vulcan3 not installed (light-CI environment)", allow_module_level=True)

from vulcan_forward import vulcan_chem

import jax
import jax.numpy as jnp

# The chemistry every test here solves: the 2025 sulfur network on the WASP-39 b config.
CHEM_ID = {
    "network": "thermo/SNCHO_photo_network_2025.txt",
    "atom_list": ["H", "O", "C", "N", "S"],
    "vulcan_cfg_name": "W39b",
}
PROFILE = {**CHEM_ID, "use_photo": False, "yconv_cri": 1.0e-2, "nz": 20}
# [lnZ, c_o, lnKzz, T-offset]: the baseline column and a perturbed one.
THETAS = np.array([[0.0, 0.0, 0.0, 0.0], [0.3, 0.1, 0.5, 40.0]], dtype=np.float64)
MIX_FLOOR = 1.0e-10  # cells below this carry no observable and no certificate
# Batch against the vmap and scalar routes: two converged states at this loose yconv_cri differ at the convergence
# scale, measured up to 6e-2 and 8e-2 across Python 3.10-3.12 and numpy 1.26-2.x; a lane-mixing bug is O(1).
REL_MAX = 1.0e-1


@pytest.fixture(scope="module")
def runs():
    """The three routes, solved once: batched, per-lane vmap, and each theta beside a copy of itself."""
    chem = vulcan_chem.build_chem_model(PROFILE)
    th = jnp.asarray(THETAS)
    y_b, cd_b = chem.converged_y_batch(th, return_conv_diag=True)
    y_v, cd_v = jax.vmap(lambda t: chem.converged_y(t, return_conv_diag=True))(th)
    y_pair = [np.asarray(chem.converged_y_batch(th[jnp.array([k, k])]))[0] for k in range(THETAS.shape[0])]
    return chem, np.asarray(y_b), cd_b, np.asarray(y_v), cd_v, y_pair


def test_initial_column_carries_the_theta_elemental_ratios(runs):
    """LnZ scales every metal's ratio to H by e^lnZ and c_o scales carbon's by e^c_o on top (O stays at e^lnZ, He stays
    put), read off the built column against the baseline column, independently of the projection's targets.
    """
    chem = runs[0]
    lnZ, c_o = 0.3, 0.1
    audit = chem.audit_init(jnp.asarray([lnZ, c_o, 0.0, 0.0]))
    got, base = audit["ratios_to_H"], audit["baseline_ratios_to_H"]
    want = {"He": 1.0, "C": np.exp(lnZ + c_o)}
    for e in got:
        np.testing.assert_allclose(got[e] / base[e], want.get(e, np.exp(lnZ)), rtol=1e-10, err_msg=e)


@pytest.fixture(scope="module")
def chem_x():
    """The model with the optional entry, [lnZ, c_o, lnKzz, s_o, T-offset].

    No solve: the test below reads the built initial column.
    """
    return vulcan_chem.build_chem_model(PROFILE, chem_extras=vulcan_chem.CHEM_EXTRAS)


def test_sulfur_knob_moves_only_sulfur(chem_x):
    """s_o scales sulfur's ratio to H by e^s_o on top of e^lnZ and nothing else.

    It holds from the cold seed and from a warm guess at another sulfur reference, with every repair factor positive.
    """
    lnZ, c_o, s_o = 0.3, 0.1, -0.8
    th = jnp.asarray([lnZ, c_o, 0.0, s_o, 0.0])
    audit = chem_x.audit_init(th)
    got, base = audit["ratios_to_H"], audit["baseline_ratios_to_H"]
    want = {"He": 1.0, "C": np.exp(lnZ + c_o), "S": np.exp(lnZ + s_o)}
    for e in got:
        np.testing.assert_allclose(got[e] / base[e], want.get(e, np.exp(lnZ)), rtol=1e-10, err_msg=e)
    assert abs(audit["dln_SO_achieved"] - s_o) < 1e-10
    assert audit["ratio_max_rel_err"] < 1e-9
    warm = chem_x.audit_init(th, warm_y=jnp.asarray(chem_x.y0), lnZ_ref=0.0, c_o_ref=0.0, s_o_ref=0.4)
    assert warm["ratio_max_rel_err"] < 1e-9
    assert abs(warm["dln_SO_achieved"] - s_o) < 1e-10


def test_lnkzz_scales_the_config_profile(runs):
    """e^lnKzz multiplies the config's Kzz profile."""
    chem = runs[0]
    kzz_prof = np.asarray(chem.prep_pv(jnp.asarray([0.0, 0.0, 0.0, 0.0])).Kzz)
    np.testing.assert_allclose(
        np.asarray(chem.prep_pv(jnp.asarray([0.0, 0.0, np.log(3.0), 0.0])).Kzz), 3.0 * kzz_prof, rtol=1e-12
    )


def test_batched_lanes_are_their_own_solves_and_agree_with_the_vmap(runs):
    _chem, y_b, cd_b, y_v, cd_v, y_pair = runs
    for k in range(THETAS.shape[0]):
        acc_b = int(np.asarray(cd_b.accept_count)[k])
        acc_v = int(np.asarray(cd_v.accept_count)[k])
        mix = y_b[k] / y_b[k].sum(axis=1, keepdims=True)
        obs = mix > MIX_FLOOR
        rel = np.abs(y_b[k] - y_v[k]) / np.maximum(np.abs(y_v[k]), 1e-300)
        print(
            f"[lane {k}] batch-vs-vmap over {int(obs.sum())} cells > {MIX_FLOOR:g} "
            f"VMR: max rel {rel[obs].max():.3e}, median {np.median(rel[obs]):.3e}; "
            f"accept_count batched={acc_b} vmap={acc_v}",
            flush=True,
        )
        assert np.array_equal(y_b[k], y_pair[k]), (
            "lane is not independent of the batch: lane k of the N-lane batch "
            "differs from the same theta beside a copy of itself"
        )
        assert bool(np.asarray(cd_b.conv_normal)[k]) == bool(np.asarray(cd_v.conv_normal)[k])
        assert rel[obs].max() < REL_MAX


def test_per_lane_references_match_the_scalar_calls(runs):
    """``lnZ_ref`` and ``c_o_ref`` as (N,) arrays give every lane the composition its own column was converged at.

    A warm-starting consumer needs this, since each row starts from its own anchor. Reference: the same warm
    continuation as a batch of one with the scalar reference. The bound is the convergence scale (a traced reference is
    not the folded constant of the scalar call), with the certificate lane for lane; the measured difference is 0, since
    subtracting a reference that rides the carry is the same arithmetic as subtracting the constant.
    """
    chem, y_conv, _cd_b, _y_v, _cd_v, _y_pair = runs
    th0 = jnp.asarray(THETAS)
    th1 = jnp.asarray(THETAS + np.array([0.05, 0.02, 0.1, 5.0])[None, :])
    yw = jnp.asarray(y_conv)
    # distinct per-lane references: each column's own (lnZ, c_o)
    y_a, cd_a = chem.converged_y_batch(th1, warm_y=yw, lnZ_ref=th0[:, 0], c_o_ref=th0[:, 1], return_conv_diag=True)
    y_a = np.asarray(y_a)
    assert len({float(x) for x in np.asarray(THETAS)[:, 0]}) > 1  # not vacuous
    for k in range(THETAS.shape[0]):
        y_s, cd_s = chem.converged_y_batch(
            th1[k : k + 1],
            warm_y=yw[k : k + 1],
            lnZ_ref=float(THETAS[k, 0]),
            c_o_ref=float(THETAS[k, 1]),
            return_conv_diag=True,
        )
        y_s = np.asarray(y_s)[0]
        mix = y_s / y_s.sum(axis=1, keepdims=True)
        obs = mix > MIX_FLOOR
        rel = np.abs(y_a[k] - y_s) / np.maximum(np.abs(y_s), 1e-300)
        print(
            f"[refs lane {k}] array-vs-scalar reference over {int(obs.sum())} "
            f"cells: max rel {rel[obs].max():.3e} median "
            f"{np.median(rel[obs]):.3e}; accept_count "
            f"{int(np.asarray(cd_a.accept_count)[k])} vs "
            f"{int(np.asarray(cd_s.accept_count)[0])}",
            flush=True,
        )
        assert bool(np.asarray(cd_a.conv_normal)[k]) == bool(np.asarray(cd_s.conv_normal)[0])
        assert rel[obs].max() < REL_MAX


def test_inputs_that_would_be_ignored_are_refused(runs):
    """A profile key the engine does not read, such as ``warm_count_max``, would change nothing, so it is refused.

    A warm column with the wrong trailing shape would broadcast, (N, 1, ni) seeding every layer from one layer, so it is
    refused too.
    """
    chem = runs[0]
    with pytest.raises(ValueError, match="'warm_count_max'"):
        vulcan_chem.build_chem_model({**PROFILE, "warm_count_max": 10})
    with pytest.raises(ValueError, match="warm_y"):
        chem.converged_y_batch(jnp.asarray(THETAS), warm_y=jnp.ones((THETAS.shape[0], 1, chem.ni)))


@pytest.mark.usefixtures("runs")  # vulcan3 is imported on this module's network before the refusals are tried
@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"network": None}, r"profile is missing \['network'\], which have no default"),
        ({"atom_list": None, "vulcan_cfg_name": None}, r"profile is missing \['atom_list', 'vulcan_cfg_name'\]"),
        (
            {"network": "thermo/NCHO_photo_network.txt", "atom_list": ["H", "O", "C", "N"]},
            "vulcan3 is already imported",
        ),
        ({"cfg_overrides": {"nz": 30}}, "at the top level of the profile and again in cfg_overrides"),
        ({"cfg_overrides": {"runtime": 1.0e20}}, "'runtime' without 'dt_max'"),
        ({"cfg_overrides": {"network": "thermo/NCHO_photo_network.txt"}}, "freezes at its first import"),
        ({"rp_cm": 9.1e9, "gs_cgs": 4220.0}, "the chemistry and the RT describe different planets"),
        ({"use_photo": True, "rstar_cm": 7.0e10}, "the chemistry and the RT describe different stars"),
    ],
)
def test_the_chemistry_is_named_by_the_profile_and_stated_once(change, match):
    """The network, its atoms and the config have no default; a setting stated twice, or one the build would ignore, or
    a planet or star that is not the RT's, is refused before any solve.
    """
    with pytest.raises((ValueError, RuntimeError), match=match):
        vulcan_chem.build_chem_model({**PROFILE, **change})
    # WASP-39 b's own radius, gravity and star pass the one-planet and one-star checks
    vulcan_chem._configured_cfg(
        {**PROFILE, "use_photo": True, "rp_cm": 9143826800.0, "gs_cgs": 422.0, "rstar_cm": 64839240000.0}
    )


def test_a_0_or_1_flag_in_cfg_overrides_loads_as_vulcan3_loads_it():
    """A 0 / 1 flag is a bool, as vulcan3's load_config reads it, so use_solar=1 takes the preset; 2 is refused."""
    cfg = vulcan_chem._configured_cfg({**PROFILE, "cfg_overrides": {"use_solar": 1, "O_H": 0.003}})
    assert cfg.use_solar is True
    with pytest.raises(ValueError, match=r"use_solar=2 must be true or false"):
        vulcan_chem._configured_cfg({**PROFILE, "cfg_overrides": {"use_solar": 2}})


def test_the_cold_seed_reads_the_run_config():
    """The seed is traced after setup has restored vulcan3's process default, so it must read the run's own config."""
    from types import SimpleNamespace

    from vulcan3 import composition
    from vulcan3.ini_abun import ratio_indices, seed_elements

    from vulcan_forward._chem_model import _ChemSetup

    T = jnp.full(4, 1500.0)
    p_bar = jnp.logspace(1.0, -2.0, 4)
    o_to_h = {}
    for frac in (0.0, 0.5):
        cfg = vulcan_chem._configured_cfg({**PROFILE, "cfg_overrides": {"sequestered_frac": {"O": frac}}})
        setup = SimpleNamespace(cfg=cfg, p_bar=p_bar, ratio_idx=ratio_indices([], cfg))
        y = np.asarray(_ChemSetup._eq_seed(setup, T, jnp.zeros(0), jnp.ones(4)))
        elements = list(seed_elements(cfg))
        atoms = y @ np.asarray(composition.compo_array)[:, [composition.atom_list.index(e) for e in elements]]
        o_to_h[frac] = atoms[:, elements.index("O")] / atoms[:, elements.index("H")]
    np.testing.assert_allclose(o_to_h[0.5] / o_to_h[0.0], 0.5, rtol=1e-6)


@pytest.mark.usefixtures("runs")  # vulcan3 is imported on this module's network
def test_a_star_the_case_does_not_state_is_refused(tmp_path):
    """A config file that omits the star would take default.yaml's star; the case must state it, in its file or in
    cfg_overrides. Photochemistry off reads no star, so nothing is required then.
    """
    from importlib import resources

    star = {"r_star": 0.932, "orbit_radius": 0.04828, "sflux_file": "atm/stellar_flux/sflux-W39b_Tsai2023.txt"}
    w39b = resources.files("vulcan3").joinpath("configs", "W39b.yaml").read_text().splitlines()
    unstated = tmp_path / "no_star.yaml"
    unstated.write_text("\n".join(line for line in w39b if line.split(":")[0] not in star) + "\n")
    case = {**PROFILE, "vulcan_cfg_name": str(unstated), "use_photo": True}
    with pytest.raises(ValueError, match=r"do not state \['r_star', 'orbit_radius', 'sflux_file'\]"):
        vulcan_chem._configured_cfg(case)
    with pytest.raises(ValueError, match=r"do not state \['orbit_radius'\]"):
        vulcan_chem._configured_cfg({**case, "cfg_overrides": {"r_star": 0.932, "sflux_file": star["sflux_file"]}})
    assert vulcan_chem._configured_cfg({**case, "cfg_overrides": star}).r_star == star["r_star"]
    vulcan_chem._configured_cfg({**case, "use_photo": False})


def test_the_start_check_covers_the_repair_factor_and_a_differentiated_call():
    """A repair factor at or below zero is refused like a negative density, NaN under jit, and an eager jvp gets the
    same error, not a tracer conversion failure.
    """
    from vulcan_forward._chem_model import _physical_start

    good = jnp.ones((3, 2))
    assert np.array_equal(np.asarray(_physical_start(good, jnp.asarray(0.5))), np.ones((3, 2)))
    for y0, factor in (
        (good, 0.0),
        (good, -21.0),
        (good.at[1, 0].set(-1.0e-100), 1.0),
        (good.at[0, 1].set(jnp.nan), 1.0),
    ):
        with pytest.raises(ValueError, match="start column is not physical"):
            _physical_start(y0, jnp.asarray(factor))
        assert np.all(np.isnan(np.asarray(jax.jit(_physical_start)(y0, jnp.asarray(factor)))))
    with pytest.raises(ValueError, match="start column is not physical"):
        jax.jvp(lambda f: _physical_start(good * f, f - 2.0), (jnp.asarray(1.0),), (jnp.asarray(1.0),))


def test_a_start_column_that_is_not_physical_is_refused(runs):
    """A warm start with a negative density: an error when eager (from the audit too), an uncertified NaN column
    under jit.
    """
    chem, y_b = runs[0], runs[1]
    th = jnp.asarray(THETAS[1])
    bad = jnp.asarray(y_b[0]).at[3, chem.sidx["H2O"]].set(-1.0e5)
    with pytest.raises(ValueError, match="start column is not physical"):
        chem.converged_y(th, warm_y=bad)
    with pytest.raises(ValueError, match="start column is not physical"):
        chem.audit_init(th, warm_y=bad)
    y, cd = jax.jit(lambda t: chem.converged_y(t, warm_y=bad, return_conv_diag=True))(th)
    assert not bool(cd.conv_normal)
    assert np.all(np.isnan(np.asarray(y)))


def test_the_start_check_without_a_solve_is_the_solves_own(runs):
    """``warm_start_ok`` says, with no solve, which warm starts the solve refuses: a negative density, and a C/O above
    what the column's oxygen can compensate.
    """
    chem, y_b = runs[0], runs[1]
    th = jnp.asarray(THETAS[1])
    good = jnp.asarray(y_b[0])
    ok = jax.jit(jax.vmap(chem.warm_start_ok, in_axes=(None, 0)))
    bad = good.at[3, chem.sidx["H2O"]].set(-1.0e5)
    assert np.asarray(ok(th, jnp.stack([good, bad]))).tolist() == [True, False]
    # Past the column's positivity margin the fixed-O map turns an O-only carrier negative.
    high_co = th.at[1].set(float(chem.co_bz_margin(np.asarray(good))) + 0.5)
    assert not bool(chem.warm_start_ok(high_co, good))
    with pytest.raises(ValueError, match="start column is not physical"):
        chem.audit_init(high_co, warm_y=good)
    assert bool(chem.warm_start_ok(th.at[1].set(-0.5), good))
    chem.audit_init(th.at[1].set(-0.5), warm_y=good)


def test_an_uncertified_column_is_never_returned_as_converged():
    """A column stopped on its step cap: an error when eager, NaN under jit, and returned as it is only on request."""
    chem = vulcan_chem.build_chem_model({**PROFILE, "count_max": 5})
    th = jnp.asarray(THETAS[1])
    y, cd = chem.converged_y(th, return_conv_diag=True)
    assert not bool(cd.conv_normal)
    with pytest.raises(RuntimeError, match="did not certify"):
        chem.converged_y(th)
    with pytest.raises(RuntimeError, match="did not certify"):
        chem.converged_y_batch(jnp.asarray(THETAS))
    assert np.all(np.isnan(np.asarray(jax.jit(chem.converged_y)(th))))
    assert np.array_equal(np.asarray(chem.converged_y(th, allow_uncertified=True)), np.asarray(y))


def test_the_certified_return_gives_its_own_error_under_an_eager_jvp():
    """The certificate is concrete while its diagnostics carry tangents: the refusal is still the engine's message."""
    from vulcan_forward._chem_layout import ConvDiag
    from vulcan_forward._chem_model import _certified

    def solve(x):
        fields = dict.fromkeys(ConvDiag._fields, jnp.asarray(0))
        cd = ConvDiag(**{**fields, "conv_normal": jnp.asarray(False), "longdy": 2.0 * x})
        return _certified(jnp.ones((3, 2)) * x, cd)

    with pytest.raises(RuntimeError, match="did not certify: accepted steps 0, termination_reason 0"):
        jax.jvp(solve, (jnp.asarray(1.0),), (jnp.asarray(1.0),))


@pytest.mark.parametrize(("span", "refused"), [((0.5, 1.0e6), False), ((115.0, 283.0), True), ((0.5, 400.0), True)])
def test_a_stellar_spectrum_must_cover_the_photolysis_thresholds(span, refused):
    """vulcan3 cuts the photolysis grid to the stellar file without a word, so a short file is refused at the build."""
    from types import SimpleNamespace

    var = SimpleNamespace(def_bin_min=span[0], def_bin_max=span[1], photo_sp={"H2O", "CH4", "HCO"})
    cfg = SimpleNamespace(sflux_file="atm/stellar_flux/a_star.txt")
    if refused:
        with pytest.raises(ValueError, match="photolysis thresholds"):
            vulcan_chem._check_stellar_span(cfg, var)
    else:
        vulcan_chem._check_stellar_span(cfg, var)


def test_a_network_without_its_nasa9_folder_is_refused(tmp_path):
    """The Gibbs data comes from the network's own tree: no folder beside it is an error, never another tree's data."""
    net = tmp_path / "thermo" / "network.txt"
    net.parent.mkdir()
    net.write_text("")
    with pytest.raises(FileNotFoundError, match="no NASA9 folder beside the network"):
        vulcan_chem._thermo_dir(net)
    (net.parent / "NASA9").mkdir()
    assert vulcan_chem._thermo_dir(net) == net.parent
