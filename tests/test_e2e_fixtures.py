"""Always-on sanity for the committed pRT reference fixtures (numpy-only).

Runs in the exojax-less light CI, so the fixtures cannot rot silently:
arrays stay parseable and finite, provenance stays complete, and the stats
recorded inside each fixture stay EQUAL to the values hardcoded here -- the
one permitted duplication. It pins that the fixtures really are the
2026-08-15 petitRADTRANS verification (independent record:
Desktop vulcan_validation/models/rt_verification.json); silently
regenerating them to looser numbers fails this test.

The tolerances the e2e tests assert live in the same meta, held at 3x the
achieved stats (checked here as a [2, 4] ratio so neither side can drift).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"

# One authoritative measurement (build_prt_fixtures, 2026-08-15), duplicated
# here on purpose: fixture meta and this table must agree exactly.
PINNED_STATS = {
    "prt_ref_isothermal_h2o_trans.npz": dict(
        n_wl=1610, arrays=("wl_um", "prt_radius_cm"),
        mean_ratio=1.0000341406917026, rms_pct=0.0012825771808827554,
        max_dev_pct=0.004962960960619434),
    "prt_ref_w39b_8species_trans.npz": dict(
        n_wl=511, arrays=("wl_um", "prt_depth_ppm"),
        mean_ratio=0.9982884571858963, rms_pct=0.09576590862575744,
        max_dev_pct=0.3406240168612462),
    "prt_ref_emission_h2o.npz": dict(
        n_wl=511, arrays=("wl_um", "prt_flux_per_cm1", "prt_flux_lambda"),
        mean_ratio=1.0004052868780968, rms_pct=0.020925888022426987,
        max_dev_pct=0.05851848060187681),
}

META_KEYS = ("case", "prt_version", "generated", "opacity", "config",
             "estimator", "prt_script", "prt_script_text", "exo_method",
             "bundle_record", "stats", "tol")


@pytest.mark.parametrize("name", sorted(PINNED_STATS))
def test_fixture_arrays_and_provenance(name):
    pin = PINNED_STATS[name]
    z = np.load(DATA / name)
    meta = json.loads(bytes(np.asarray(z["meta"])))
    for key in META_KEYS:
        assert key in meta, f"{name}: meta missing {key!r}"
    assert meta["prt_version"] == "3.4.0"
    for arr in pin["arrays"]:
        a = np.asarray(z[arr])
        assert a.shape == (pin["n_wl"],), (name, arr, a.shape)
        assert np.all(np.isfinite(a)), (name, arr)
    wl = np.asarray(z["wl_um"])
    assert np.all(np.diff(wl) > 0), f"{name}: wl_um not strictly ascending"


@pytest.mark.parametrize("name", sorted(PINNED_STATS))
def test_fixture_stats_are_the_2026_08_15_verification(name):
    pin = PINNED_STATS[name]
    meta = json.loads(bytes(np.asarray(np.load(DATA / name)["meta"])))
    st = meta["stats"]
    for key in ("mean_ratio", "rms_pct", "max_dev_pct"):
        assert st[key] == pytest.approx(pin[key], rel=1e-9), (name, key)


@pytest.mark.parametrize("name", sorted(PINNED_STATS))
def test_tolerances_are_3x_achieved(name):
    meta = json.loads(bytes(np.asarray(np.load(DATA / name)["meta"])))
    st, tol = meta["stats"], meta["tol"]
    for key in ("rms_pct", "max_dev_pct", "max_abs_dev_from_unity_pct"):
        ratio = tol[key] / st[key]
        assert 2.0 <= ratio <= 4.0, (name, key, ratio)


def test_emission_fixture_unit_conversion_is_recorded_and_consistent():
    """The pRT flux is stored BOTH ways; the per-cm^-1 array must equal the
    per-cm-wavelength array divided by nu_tilde^2 (the documented unit trap:
    dividing by c instead is wrong by ~1e18)."""
    z = np.load(DATA / "prt_ref_emission_h2o.npz")
    nutilde = 1.0e4 / np.asarray(z["wl_um"])
    want = np.asarray(z["prt_flux_lambda"]) / nutilde ** 2
    assert np.allclose(np.asarray(z["prt_flux_per_cm1"]), want, rtol=1e-12)
    meta = json.loads(bytes(np.asarray(z["meta"])))
    assert "nu_tilde^2" in meta["units"]


def test_chemistry_table_parses():
    raw = np.genfromtxt(DATA / "wasp39b_10Xsolar_evening_vulcan.txt",
                        skip_header=2)
    assert raw.shape[1] == 15, raw.shape
    assert raw.shape[0] > 100, raw.shape
    assert np.all(np.isfinite(raw))
