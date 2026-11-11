"""The RT builders and observables on miniature k-tables and CIA files, so they run with no data tree.

Each check is an identity of the physics or of the routes, not a value: the pRT comparisons on real tables are in
tests/test_e2e_rt_reference.py.
"""

from __future__ import annotations

import contextlib
import importlib.util

import numpy as np
import pytest

# Import order: vulcan_chem before exojax; find_spec, never importorskip("exojax").
if importlib.util.find_spec("exojax") is None:
    pytest.skip("exojax not installed (light-CI environment)", allow_module_level=True)
if importlib.util.find_spec("h5py") is None:
    pytest.skip("h5py not installed", allow_module_level=True)
if importlib.util.find_spec("vulcan3") is not None:
    from vulcan_forward import vulcan_chem  # noqa: F401  # vulcan_chem first: the import order is the contract
else:
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from exojax.rt.planck import piBarr
from test_exomolop import _write

from vulcan_forward import exojax_rt, exomolop, paths

NU0 = 2000.0
NB = 20
NLAYER = 30
ROUND_TOL = 1e-12
PROFILE = {
    "molecules": ["H2O", "CO2"],
    "omit_core_molecules": ["CO"],  # the 4.9-5.0 um band also holds CO; two miniature tables are enough here
    "nu_min": NU0,
    "nu_max": NU0 * np.exp((NB + 0.5) / 1000.0),
    "art_nlayer": NLAYER,
    "art_ptop_bar": 1.0e-5,
    "art_pbtm_bar": 10.0,
    "p_ref_bar": 1.0,
    "rp_cm": 7.1e9,
    "gs_cgs": 1.0e3,
    "rstar_cm": 7.0e10,
}
ZEROS = jnp.zeros(NLAYER)
HALF = jnp.full(NLAYER, 0.5)
T_ISO = jnp.full(NLAYER, 1000.0)
MMW = jnp.full(NLAYER, 2.33)


def _write_cia(path, nu_lo=500.0, nu_hi=5000.0):
    """A miniature HITRAN CIA file: three temperatures, a coefficient that grows with T."""
    nu = np.arange(nu_lo, nu_hi + 1.0, 1.0)
    label = path.name.split("_")[0]
    blocks = []
    for t in (300.0, 1000.0, 3000.0):
        head = f"{label:>20} {nu[0]:9.3f} {nu[-1]:9.3f} {nu.size:6d} {t:6.1f} 1.0E-45 -.999"
        blocks.append(head + "\n" + "\n".join(f"{n:10.4f} {1.0e-45 * t / 1000.0:10.3E}" for n in nu))
    path.write_text("\n".join(blocks) + "\n")


@contextlib.contextmanager
def miniature_root(data, molecules, nb, nu0):
    """Point the engine at a data root holding miniature tables on ``nb`` bands from ``nu0``, and both CIA files."""
    (data / "exomolop").mkdir()
    (data / "opacity_cache").mkdir()
    mp = pytest.MonkeyPatch()
    mp.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    saved = paths._root_override
    paths.set_data_root(data)
    for m in molecules:
        _write(exomolop.table_path(m), nb=nb, nu0=nu0)
    _write_cia(paths.cia_h2h2_file())
    _write_cia(paths.cia_h2he_file())
    try:
        yield data
    finally:
        paths._root_override = saved
        mp.undo()


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    with miniature_root(tmp_path_factory.mktemp("mini_data"), PROFILE["molecules"], NB, NU0) as data:
        yield data


@pytest.fixture(scope="module")
def models(root):  # noqa: ARG001  # the data root must be in place
    rt = exojax_rt.build_rt_model(PROFILE)
    return rt, exojax_rt.build_emis_model(rt, PROFILE)


def _vmr(h2o=1.0e-3, co2=1.0e-4):
    return {"H2O": jnp.full(NLAYER, h2o), "CO2": jnp.full(NLAYER, co2)}


def test_build_reads_the_miniature_grid(models):
    rt, em = models
    assert rt.nu_grid.size == NB
    assert rt.nu_edges.size == NB + 1  # each band value is a mean between two of these
    assert np.all((rt.nu_edges[:-1] < rt.nu_grid) & (rt.nu_grid < rt.nu_edges[1:]))
    assert np.array_equal(em.nu_edges, rt.nu_edges)
    assert rt.molecules == PROFILE["molecules"]
    assert rt.p_art_bar.size == NLAYER
    np.testing.assert_allclose(rt.wl_um, 1.0e4 / rt.nu_grid)
    assert em.p_art_bar.size == NLAYER
    assert float(rt.art_ptop_bar) == PROFILE["art_ptop_bar"]


def test_each_opacity_source_deepens_the_transit(models):
    rt, _ = models
    bare = np.asarray(rt.transmission_depth(_vmr(0.0, 0.0), ZEROS, T_ISO, MMW, vmr_he=ZEROS))
    assert np.all(np.isfinite(bare))
    for vmr, h2, he, name in (
        (_vmr(1.0e-3, 0.0), ZEROS, ZEROS, "H2O"),
        (_vmr(0.0, 1.0e-3), ZEROS, ZEROS, "CO2"),
        (_vmr(0.0, 0.0), HALF, ZEROS, "H2-H2 CIA"),
        (_vmr(0.0, 0.0), HALF, HALF, "H2-He CIA"),
    ):
        got = np.asarray(rt.transmission_depth(vmr, h2, T_ISO, MMW, vmr_he=he))
        assert np.all(got > bare), name
    # H2-He adds to H2-H2 rather than replacing it
    h2_only = np.asarray(rt.transmission_depth(_vmr(0.0, 0.0), HALF, T_ISO, MMW, vmr_he=ZEROS))
    both = np.asarray(rt.transmission_depth(_vmr(0.0, 0.0), HALF, T_ISO, MMW, vmr_he=HALF))
    assert np.all(both > h2_only)


def test_a_missing_he_profile_is_refused_not_dropped(models):
    rt, _ = models
    with pytest.raises(ValueError, match="vmr_he"):
        rt.transmission_depth(_vmr(), HALF, T_ISO, MMW)


def test_leave_one_out_rows_are_the_zeroed_solves_bit_for_bit(models):
    rt, _ = models
    vmr = _vmr()
    full, wo = rt.transmission_depth_r(vmr, HALF, T_ISO, MMW, 0.0, vmr_he=HALF, wo_mols=PROFILE["molecules"])
    assert np.array_equal(
        np.asarray(full), np.asarray(rt.transmission_depth_r(vmr, HALF, T_ISO, MMW, 0.0, vmr_he=HALF))
    )
    for i, m in enumerate(PROFILE["molecules"]):
        zeroed = {**vmr, m: jnp.zeros(NLAYER)}
        want = np.asarray(rt.transmission_depth_r(zeroed, HALF, T_ISO, MMW, 0.0, vmr_he=HALF))
        assert np.array_equal(np.asarray(wo[i]), want), m


def test_lnr0_and_lnm_are_the_model_rebuilt_at_that_radius_and_mass(models):
    """lnR0 is a radius change at fixed mass, and lnM a mass change: the same as quoting the planet there."""
    rt, _ = models
    h, m = 1.0e-2, 0.1
    moved = {
        **PROFILE,
        "rp_cm": PROFILE["rp_cm"] * np.exp(h),
        "gs_cgs": PROFILE["gs_cgs"] * np.exp(m - 2.0 * h),
    }
    want = np.asarray(exojax_rt.build_rt_model(moved).transmission_depth(_vmr(), HALF, T_ISO, MMW, vmr_he=HALF))
    got = np.asarray(rt.transmission_depth_r(_vmr(), HALF, T_ISO, MMW, h, vmr_he=HALF, lnM=m))
    np.testing.assert_allclose(got, want, rtol=ROUND_TOL, atol=0.0)


def test_a_deck_and_a_cloud_only_deepen_and_an_opaque_deck_sets_the_depth(models):
    rt, _ = models
    clear = np.asarray(rt.transmission_depth_r(_vmr(), HALF, T_ISO, MMW, 0.0, vmr_he=HALF))
    cloud = np.asarray(
        rt.transmission_depth_r(_vmr(), HALF, T_ISO, MMW, 0.0, vmr_he=HALF, cloud=jnp.asarray([-1.0, 0.0]))
    )
    high = np.asarray(rt.transmission_depth_r(_vmr(), HALF, T_ISO, MMW, 0.0, vmr_he=HALF, deck=-4.0))
    low = np.asarray(rt.transmission_depth_r(_vmr(), HALF, T_ISO, MMW, 0.0, vmr_he=HALF, deck=-1.0))
    assert np.all(cloud > clear)
    assert np.all(low >= clear)
    assert np.all(high > low)
    # a deck this high hides the molecular bands: the spectrum goes flat
    assert np.ptp(high) < np.ptp(clear)


def test_an_isothermal_column_radiates_pi_planck_whatever_its_opacity(models):
    """The source equals the sink everywhere, so k cancels; a unit, weight or boundary slip breaks this at once."""
    _, em = models
    want = np.asarray(piBarr(jnp.asarray([1000.0]), jnp.asarray(em.nu_grid)))[0]
    for vmr in (_vmr(1.0e-2, 1.0e-2), _vmr(1.0e-5, 0.0)):
        f = np.asarray(em.emission_flux(vmr, HALF, T_ISO, MMW, vmr_he=HALF))
        np.testing.assert_allclose(f, want, rtol=1.0e-11, atol=0.0)


def test_emission_routes_agree_and_opacity_sets_the_bottom_optical_depth(models):
    _, em = models
    vmr = _vmr()
    T = jnp.linspace(800.0, 1800.0, NLAYER)
    mols = PROFILE["molecules"]
    f, tau, f_wo, tau_wo = em.emission_flux_tau(vmr, HALF, T, MMW, vmr_he=HALF, wo_mols=mols)
    assert np.array_equal(np.asarray(f), np.asarray(em.emission_flux(vmr, HALF, T, MMW, vmr_he=HALF)))
    assert np.array_equal(np.asarray(tau), np.asarray(em.tau_bottom(vmr, HALF, T, MMW, HALF)))
    for i, m in enumerate(mols):
        zeroed = {**vmr, m: jnp.zeros(NLAYER)}
        assert np.array_equal(np.asarray(f_wo[i]), np.asarray(em.emission_flux(zeroed, HALF, T, MMW, vmr_he=HALF))), m
        assert np.all(np.asarray(tau_wo[i]) < np.asarray(tau)), m
    thick = em.tau_bottom(_vmr(1.0e-1, 1.0e-1), HALF, T, MMW, HALF)
    assert np.all(np.asarray(thick) > np.asarray(tau))


def test_the_eclipse_flux_at_fixed_mass_is_the_model_rebuilt_there(models):
    rt, em = models
    h = 1.0e-2
    moved = {**PROFILE, "rp_cm": PROFILE["rp_cm"] * np.exp(h), "gs_cgs": PROFILE["gs_cgs"] * np.exp(-2.0 * h)}
    em_h = exojax_rt.build_emis_model(exojax_rt.build_rt_model(moved), moved)
    T = jnp.linspace(800.0, 1800.0, NLAYER)
    got = (*em.eclipse_flux_tau(_vmr(), HALF, T, MMW, vmr_he=HALF, lnR0=h), em.emission_radius(T, MMW, lnR0=h))
    want = (*em_h.eclipse_flux_tau(_vmr(), HALF, T, MMW, vmr_he=HALF), em_h.emission_radius(T, MMW))
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=ROUND_TOL, atol=0.0)
    # an eclipse is the flux times the photosphere's area, so a thicker column raises it on an isothermal profile
    assert rt.nu_grid.size == np.asarray(got[0]).size
    thin, _ = em.eclipse_flux_tau(_vmr(1.0e-5, 0.0), HALF, T_ISO, MMW, vmr_he=HALF)
    thick, _ = em.eclipse_flux_tau(_vmr(1.0e-2, 1.0e-2), HALF, T_ISO, MMW, vmr_he=HALF)
    assert np.all(np.asarray(thick) > np.asarray(thin))


@pytest.mark.usefixtures("root")
@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"opacity_mode": "lbl"}, "opacity_mode"),
        ({"art_ptop_bar": 1.0, "art_pbtm_bar": 0.1}, "need 0 < top < bottom"),
        ({"art_pbtm_bar": 3.0e2}, "k-table pressure ceiling"),
        ({"p_ref_bar": 100.0}, "p_ref_bar"),
        ({"not_a_key": 1}, "not_a_key"),
        # Either band reads the same miniature tables, so only the band check can refuse it.
        ({"nu_max": 1.2e4}, "supported band"),
        ({"nu_min": 600.0}, "supported band"),
        # the band needs CO; naming a molecule that is in the RT, or one that is not core, is a mistake
        ({"omit_core_molecules": []}, r"needs core absorbers.*CO \(absorbs at 4.3-5.2 um\)"),
        ({"omit_core_molecules": ["CO", "H2O"]}, "omit_core_molecules names"),
        ({"omit_core_molecules": ["CO", "OCS"]}, "omit_core_molecules names"),
        ({"gs_cgs": 0.0}, "finite and positive"),
        ({"gs_cgs": -1.0e3}, "finite and positive"),
        ({"rp_cm": float("nan")}, "finite and positive"),
        ({"rstar_cm": -7.0e10}, "finite and positive"),
    ],
)
def test_build_refuses_a_bad_profile(over, match):
    with pytest.raises(ValueError, match=match):
        exojax_rt.build_rt_model({**PROFILE, **over})


def test_rayleigh_is_on_unless_the_profile_turns_it_off(models, root):  # noqa: ARG001  # the data root must be in place
    """H2 and He alone: the default build carries Rayleigh, and it deepens every band."""
    rt, _ = models

    def depth(model):
        return np.asarray(model.transmission_depth(_vmr(0.0, 0.0), HALF, T_ISO, MMW, vmr_he=HALF))

    on = depth(exojax_rt.build_rt_model({**PROFILE, "use_rayleigh": True}))
    off = depth(exojax_rt.build_rt_model({**PROFILE, "use_rayleigh": False}))
    assert np.array_equal(depth(rt), on)
    assert np.all(on > off)


@pytest.mark.parametrize(
    ("what", "value", "match"),
    [
        ("T", 299.0, "opacity window"),
        ("T", 3001.0, "opacity window"),
        ("T", np.nan, "opacity window"),
        ("mmw", 0.5, "mean molecular weight"),
        ("mmw", -2.33, "mean molecular weight"),
        ("h2o", -1.0e-3, "H2O mixing ratio"),
        ("h2o", 1.5, "H2O mixing ratio"),
        ("h2", np.nan, "H2 mixing ratio"),
        ("he", 1.1, "He mixing ratio"),
    ],
)
def test_a_column_the_model_does_not_cover_is_refused(models, what, value, match):
    """One bad layer: an error naming the quantity when eager, a NaN spectrum under jit, and no clamp either way.

    Outside 300-3000 K the k-table and CIA interpolations would clamp; a mean molecular weight below atomic hydrogen
    or a mixing ratio outside [0, 1] is not a gas.
    """
    import jax

    rt, em = models
    good = {"T": T_ISO, "mmw": MMW, "h2o": _vmr()["H2O"], "h2": HALF, "he": HALF}

    def depth(c):
        return rt.transmission_depth({**_vmr(), "H2O": c["h2o"]}, c["h2"], c["T"], c["mmw"], vmr_he=c["he"])

    def flux(c):
        return em.emission_flux({**_vmr(), "H2O": c["h2o"]}, c["h2"], c["T"], c["mmw"], vmr_he=c["he"])

    bad = {**good, what: good[what].at[3].set(value)}
    for observable in (depth, flux):
        with pytest.raises(ValueError, match=match):
            observable(bad)
        jitted = jax.jit(observable)
        assert np.all(np.isnan(np.asarray(jitted(bad))))
        assert np.all(np.isfinite(np.asarray(jitted(good))))


def test_build_refuses_a_band_past_a_cia_table_and_a_missing_he_table(root):
    he = paths.cia_h2he_file()
    saved = he.read_text()
    try:
        _write_cia(he, nu_lo=2500.0, nu_hi=5000.0)
        with pytest.raises(ValueError, match="H2-He CIA table covers"):
            exojax_rt.build_rt_model(PROFILE)
        he.unlink()
        with pytest.raises(FileNotFoundError, match="H2-He CIA table missing"):
            exojax_rt.build_rt_model(PROFILE)
    finally:
        he.write_text(saved)
    assert root.is_dir()


@pytest.mark.parametrize("absent", ["cia_h2h2_file", "cia_h2he_file"])
def test_an_absent_cia_table_is_refused_with_its_url(tmp_path, absent):
    """An absent table would be downloaded by exojax, behind a downloader that swallows failures."""
    with miniature_root(tmp_path, PROFILE["molecules"], NB, NU0):
        getattr(paths, absent)().unlink()
        with pytest.raises(FileNotFoundError, match=r"hitran\.org/data/CIA/main/H2-H[2e]_2011\.cia"):
            exojax_rt.build_rt_model(PROFILE)
