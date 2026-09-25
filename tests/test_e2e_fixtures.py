"""Always-on sanity for the committed pRT reference fixtures (numpy-only).

The e2e tests read their tolerances from each fixture's meta, so a fixture
silently regenerated to looser numbers would still pass them. This file runs
in the exojax-less light CI and pins the stats recorded inside each fixture
EQUAL to the values hardcoded here (the one permitted duplication), the
tolerances at 3x those stats (a [2, 4] ratio so neither side can drift), and
the arrays finite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"

# One authoritative measurement (build_prt_fixtures), duplicated here on
# purpose: fixture meta and this table must agree exactly.
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


@pytest.mark.parametrize("name", sorted(PINNED_STATS))
def test_fixture_is_the_pinned_verification(name):
    pin = PINNED_STATS[name]
    z = np.load(DATA / name)
    meta = json.loads(bytes(np.asarray(z["meta"])))
    assert meta["prt_version"] == "3.4.0"
    for arr in pin["arrays"]:
        a = np.asarray(z[arr])
        assert a.shape == (pin["n_wl"],), (name, arr, a.shape)
        assert np.all(np.isfinite(a)), (name, arr)
    assert np.all(np.diff(np.asarray(z["wl_um"])) > 0), name
    st, tol = meta["stats"], meta["tol"]
    for key in ("mean_ratio", "rms_pct", "max_dev_pct"):
        assert st[key] == pytest.approx(pin[key], rel=1e-9), (name, key)
    for key in ("rms_pct", "max_dev_pct", "max_abs_dev_from_unity_pct"):
        assert 2.0 <= tol[key] / st[key] <= 4.0, (name, key)


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
