"""End-to-end RT reference tests against committed petitRADTRANS spectra.

Three canonical cases, run THROUGH THE PUBLIC API (build_rt_model /
build_emis_model) and compared per band against pRT 3.4.0 reading the SAME
ExoMolOP k-table files. The reference arrays are committed fixtures in
tests/data/; each fixture's ``meta`` (JSON inside the npz) records the pRT
version, opacity files, case constants, the full generating pRT script, the
achieved agreement, and the tolerances asserted here (3x achieved -- loose
enough not to flake, tight enough that any physics change trips them).

petitRADTRANS is NEVER a test dependency: the pRT side regenerates from the
scripts in the Desktop validation bundle (rt_verification/), whose text each
fixture carries verbatim.

Gating (repo rule: never ``pytest.importorskip("exojax")`` at module scope):
skips cleanly without exojax/h5py/the data root/the H2O k-table, so light CI
is green and a local run with data executes everything. The 8-species case
additionally wants VULCAN_FORWARD_RUN_E2E=1 (it loads eight k-tables).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("exojax") is None:
    pytest.skip("exojax not installed (light-CI environment)",
                allow_module_level=True)
if importlib.util.find_spec("h5py") is None:
    pytest.skip("h5py not installed", allow_module_level=True)

# IMPORT ORDER IS LOAD-BEARING (see CLAUDE.md): vulcan_chem fixes jax x64 and
# the VULCAN_JAX_* env vars and refuses to load if exojax got there first --
# so it must come before exojax_rt here, or collecting THIS module would
# poison every later test module that imports vulcan_chem.
if importlib.util.find_spec("vulcan_jax") is not None:
    from vulcan_forward import vulcan_chem  # noqa: F401
else:
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

from vulcan_forward import paths  # noqa: E402

try:
    paths.data_root()
except RuntimeError:
    pytest.skip("VULCAN_FORWARD_DATA not configured", allow_module_level=True)

from vulcan_forward import exomolop  # noqa: E402

if not exomolop.table_path("H2O").exists():
    pytest.skip("ExoMolOP H2O k-table absent; fetch with "
                "python -m vulcan_forward.fetch_exomolop --molecules H2O",
                allow_module_level=True)

import jax.numpy as jnp  # noqa: E402

from vulcan_forward import exojax_rt  # noqa: E402

DATA = Path(__file__).parent / "data"
RSTAR_W39 = 0.932 * 6.957e10


def _load(name):
    z = np.load(DATA / name)
    meta = json.loads(bytes(np.asarray(z["meta"])))
    return z, meta


def _assert_stats(r, meta, label):
    m = float(np.mean(r))
    rms = float(np.std(r / m - 1.0) * 100.0)
    mx = float(np.max(np.abs(r / m - 1.0)) * 100.0)
    tol = meta["tol"]
    msg = (f"{label}: mean_ratio {m:.6f} rms {rms:.4f}% max {mx:.4f}% "
           f"(tol: mean 1+/-{tol['mean_ratio_abs']:.2e}, "
           f"rms {tol['rms_pct']:.4f}%, max {tol['max_dev_pct']:.4f}%). "
           "A tolerance breach means the RT physics moved relative to the "
           "2026-08-15 petitRADTRANS verification; if the change is intended, "
           "regenerate the fixture (meta carries the recipe) and say so in "
           "the commit.")
    print("\n  " + msg)
    assert abs(m - meta["stats"]["mean_ratio"]) <= tol["mean_ratio_abs"], msg
    assert rms <= tol["rms_pct"], msg
    assert mx <= tol["max_dev_pct"], msg


def test_transmission_isothermal_h2o_matches_prt():
    z, meta = _load("prt_ref_isothermal_h2o_trans.npz")
    nlay = 100
    trt = exojax_rt.build_rt_model(dict(
        molecules=["H2O"], nu_min=2000.0, nu_max=10000.0,
        opacity_mode="exomolop", art_nlayer=nlay,
        art_ptop_bar=1.0e-6, art_pbtm_bar=1.0e2, p_ref_bar=10.0,
        rp_cm=7.1492e9, gs_cgs=1.0e3, rstar_cm=RSTAR_W39))
    assert trt.opacity_mode == "exomolop"
    zeros = jnp.zeros(nlay)
    depth = np.asarray(trt.transmission_depth(
        {"H2O": jnp.full(nlay, 1.0e-3)}, zeros, jnp.full(nlay, 1000.0),
        jnp.full(nlay, 2.33), vmr_he=zeros))
    o = np.argsort(trt.wl_um)
    wl, radius = trt.wl_um[o], np.sqrt(depth[o]) * RSTAR_W39
    m = (wl > 1.01) & (wl < 4.95)
    r = radius[m] / np.interp(wl[m], z["wl_um"], z["prt_radius_cm"])
    _assert_stats(r, meta, "isothermal H2O radius vs pRT")


@pytest.mark.skipif(os.environ.get("VULCAN_FORWARD_RUN_E2E") != "1",
                    reason="8-species case: set VULCAN_FORWARD_RUN_E2E=1")
def test_transmission_w39b_8species_matches_prt():
    z, meta = _load("prt_ref_w39b_8species_trans.npz")
    mols = ["H2O", "CH4", "CO", "CO2", "H2S", "SH", "SO", "SO2"]
    missing = [m for m in mols if not exomolop.table_path(m).exists()]
    if missing:
        pytest.skip("ExoMolOP tables absent for "
                    f"{missing}; python -m vulcan_forward.fetch_exomolop "
                    f"--molecules {','.join(missing)}")
    cols = ["P_dyn", "T", "z", "mu", "H2", "H2O", "CH4", "CO", "CO2", "H2S",
            "S", "S2", "SH", "SO", "SO2"]
    raw = np.genfromtxt(DATA / "wasp39b_10Xsolar_evening_vulcan.txt",
                        skip_header=2)
    c = {k: raw[:, i] for i, k in enumerate(cols)}
    ps = c["P_dyn"] / 1e6
    order = np.argsort(ps)
    ps = ps[order]

    def regrid(ys, pd, log=True):
        if log:
            return 10.0 ** np.interp(np.log10(pd), np.log10(ps),
                                     np.log10(np.maximum(ys[order], 1e-300)))
        return np.interp(np.log10(pd), np.log10(ps), ys[order])

    nlay = 80
    trt = exojax_rt.build_rt_model(dict(
        molecules=mols, nu_min=1.0e4 / 5.0, nu_max=1.0e4 / 3.0,
        opacity_mode="exomolop", art_nlayer=nlay,
        art_ptop_bar=1.0e-8, art_pbtm_bar=50.0, p_ref_bar=0.8995,
        rp_cm=9.1438268e9, gs_cgs=426.0, rstar_cm=RSTAR_W39))
    p = trt.p_art_bar
    zeros = jnp.zeros(nlay)
    depth = np.asarray(trt.transmission_depth(
        {m: jnp.asarray(regrid(c[m], p)) for m in mols}, zeros,
        jnp.asarray(regrid(c["T"], p, log=False)),
        jnp.asarray(regrid(c["mu"], p, log=False)), vmr_he=zeros))
    o = np.argsort(trt.wl_um)
    wl, dep = trt.wl_um[o], depth[o] * 1e6
    m = (wl > 3.0 * 1.005) & (wl < 5.0 * 0.995)
    r = dep[m] / np.interp(wl[m], z["wl_um"], z["prt_depth_ppm"])
    _assert_stats(r, meta, "Tsai W39b 8-species depth vs pRT")


def test_emission_isothermal_atmosphere_radiates_pi_planck():
    """Absolute physics check, no reference code involved: an isothermal
    pure-absorption column must emit exactly pi*B(T) at every wavenumber,
    INDEPENDENT of its opacity (the source equals the sink everywhere, so k
    cancels). Measured through the full public path with real H2O k-tables:
    machine precision, max |ratio-1| = 6.7e-16 at VMR 1e-3 and 1e-5 alike.
    A g-weight normalization error, a flux unit slip, or a wrong boundary
    temperature each break this by orders of magnitude, which is why the
    tolerance is 1e-12 and not something forgiving."""
    nlay = 80
    prof = dict(molecules=["H2O"], nu_min=1.0e4 / 5.0, nu_max=1.0e4 / 3.0,
                opacity_mode="exomolop", art_nlayer=nlay,
                art_ptop_bar=1.0e-6, art_pbtm_bar=1.0e2,
                rp_cm=100 * 7.1492e9, gs_cgs=1.0e3, rstar_cm=RSTAR_W39)
    trt = exojax_rt.build_rt_model(prof)
    emod = exojax_rt.build_emis_model(trt, prof)
    from exojax.rt.planck import piBarr
    T = 1500.0
    zeros = jnp.zeros(nlay)
    want = np.asarray(piBarr(jnp.asarray([T]), jnp.asarray(emod.nu_grid)))[0]
    flux = {vmr: np.asarray(emod.emission_flux(
                {"H2O": jnp.full(nlay, vmr)}, zeros, jnp.full(nlay, T),
                jnp.full(nlay, 2.33), vmr_he=zeros))
            for vmr in (1.0e-3, 1.0e-5)}
    for vmr, f in flux.items():
        dev = np.max(np.abs(f / want - 1.0))
        assert dev < 1e-12, (
            f"isothermal column at VMR {vmr:.0e} deviates from pi*B(T) by "
            f"{dev:.3e}; the emission path has a normalization, unit, or "
            "boundary-source error")
    assert np.max(np.abs(flux[1.0e-3] / flux[1.0e-5] - 1.0)) < 1e-12, (
        "isothermal flux depends on opacity; k must cancel exactly")


def test_wo_batch_is_bit_identical_to_separate_solves():
    """The leave-one-out batch (transmission_depth_r / emission_flux_tau with
    wo_mols) must reproduce, BITWISE, what separate single solves return: the
    full spectrum unchanged, and every wo row equal to a from-scratch solve
    with that molecule's VMR zeroed. That is the whole contract of the fold
    prefix reuse -- fewer folds, zero numerical change. Grouped: transmission
    full + rows, emission flux/tau full + rows, and the unknown-molecule
    refusal."""
    mols = ["H2O"] + (["CO2"] if exomolop.table_path("CO2").exists() else [])
    nlay = 40
    prof = dict(molecules=mols, nu_min=1.0e4 / 5.0, nu_max=1.0e4 / 3.0,
                opacity_mode="exomolop", art_nlayer=nlay,
                art_ptop_bar=1.0e-6, art_pbtm_bar=1.0e2,
                rp_cm=7.1492e9, gs_cgs=1.0e3, rstar_cm=RSTAR_W39)
    trt = exojax_rt.build_rt_model(prof)
    emod = exojax_rt.build_emis_model(trt, prof)
    zeros = jnp.zeros(nlay)
    T = jnp.linspace(900.0, 2200.0, nlay)
    mmw = jnp.full(nlay, 2.33)
    vmr = {m: jnp.full(nlay, v) for m, v in zip(mols, (1.0e-3, 3.0e-4))}

    def wo_vmr(m):
        return {**vmr, m: jnp.zeros_like(vmr[m])}

    d_full, d_wo = trt.transmission_depth_r(vmr, zeros, T, mmw, 0.0,
                                            vmr_he=zeros, wo_mols=mols)
    assert np.array_equal(
        np.asarray(d_full),
        np.asarray(trt.transmission_depth_r(vmr, zeros, T, mmw, 0.0,
                                            vmr_he=zeros)))
    for i, m in enumerate(mols):
        assert np.array_equal(
            np.asarray(d_wo[i]),
            np.asarray(trt.transmission_depth_r(wo_vmr(m), zeros, T, mmw, 0.0,
                                                vmr_he=zeros))), m

    f, tau, f_wo, tau_wo = emod.emission_flux_tau(vmr, zeros, T, mmw,
                                                  vmr_he=zeros, wo_mols=mols)
    assert np.array_equal(np.asarray(f), np.asarray(
        emod.emission_flux(vmr, zeros, T, mmw, vmr_he=zeros)))
    assert np.array_equal(np.asarray(tau), np.asarray(
        emod.tau_bottom(vmr, zeros, T, mmw, zeros)))
    for i, m in enumerate(mols):
        assert np.array_equal(np.asarray(f_wo[i]), np.asarray(
            emod.emission_flux(wo_vmr(m), zeros, T, mmw, vmr_he=zeros))), m
        assert np.array_equal(np.asarray(tau_wo[i]), np.asarray(
            emod.tau_bottom(wo_vmr(m), zeros, T, mmw, zeros))), m

    with pytest.raises(ValueError, match="wo_mols"):
        trt.transmission_depth_r(vmr, zeros, T, mmw, 0.0, vmr_he=zeros,
                                 wo_mols=["NOT_A_MOLECULE"])


def test_emission_h2o_matches_prt():
    z, meta = _load("prt_ref_emission_h2o.npz")
    nlay = 80
    prof = dict(molecules=["H2O"], nu_min=1.0e4 / 5.0, nu_max=1.0e4 / 3.0,
                opacity_mode="exomolop", art_nlayer=nlay,
                art_ptop_bar=1.0e-6, art_pbtm_bar=1.0e2,
                # pRT ran constant g = 1e3; a huge rp makes the hydrostatic
                # g-variation over the column < 1e-4 relative (meta records it)
                rp_cm=100 * 7.1492e9, gs_cgs=1.0e3, rstar_cm=RSTAR_W39)
    trt = exojax_rt.build_rt_model(prof)
    emod = exojax_rt.build_emis_model(trt, prof)
    p = np.asarray(emod.p_art_bar)
    x = (np.log10(p) + 6.0) / 8.0            # 900 K at 1e-6 bar -> 2200 at 1e2
    zeros = jnp.zeros(nlay)
    flux = np.asarray(emod.emission_flux(
        {"H2O": jnp.full(nlay, 1.0e-3)}, zeros,
        jnp.asarray(900.0 + 1300.0 * x), jnp.full(nlay, 2.33), vmr_he=zeros))
    wl = 1.0e4 / np.asarray(emod.nu_grid)
    o = np.argsort(wl)
    wl, flux = wl[o], flux[o]
    m = (wl > 3.0 * 1.01) & (wl < 5.0 * 0.99)
    # fixture stores pRT flux already converted to this engine's per-cm^-1
    # convention (the conversion and the unit trap are documented in meta)
    r = flux[m] / np.interp(wl[m], z["wl_um"], z["prt_flux_per_cm1"])
    _assert_stats(r, meta, "H2O emission flux vs pRT")
