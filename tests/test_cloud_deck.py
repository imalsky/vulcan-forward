"""The power-law cloud deck, against closed-form physics.

The deck is exojax's ``powerlaw_clouds`` plus this repo's conversion to optical depth in ``_ckd_continuum``, dtau =
10^c0 (nu / CLOUD_NUC0)^alphac dP[bar] 1e6 / g. For an isothermal, constant-g, uniformly mixed atmosphere with R0 >> H
the chain has an exact answer (Heng & Kitzmann 2017, MNRAS 470, 2972):

    R_eff = R0 + H * [gamma_E + ln(tau0) + E1(tau0)]
    tau0  = (kappa * P0 / g) * sqrt(2 pi R0 / H)
    dR_eff/dlog10(kappa0) = H * ln(10) * (1 - exp(-tau0))

The case here is thin-shell (R0/H near 1e4), so those forms are exact to well under the tolerances; at production
geometry the closed form is not a valid reference. Gas is zeroed while mmw_art is kept, which kills the lines, the CIA
and Rayleigh but keeps the scale height, so a failure can only come from the cloud path.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("exojax") is None:
    pytest.skip("exojax not installed (light-CI environment)", allow_module_level=True)
if importlib.util.find_spec("h5py") is None:
    pytest.skip("h5py not installed", allow_module_level=True)

if importlib.util.find_spec("vulcan3") is not None:
    from vulcan_forward import vulcan_chem  # noqa: F401  # vulcan_chem first: the import order is the contract
else:
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.special import exp1

from vulcan_forward import constants
from vulcan_forward.exojax_rt import build_emis_model, build_rt_model
from test_rt_miniature import miniature_root

KB, M_U, GAMMA_E = 1.380649e-16, 1.66053906660e-24, 0.5772156649015329
# thin shell: R0/H = 1.05e4, so the column spans 0.2% of R0
CASE = {
    "T": 800.0,
    "mmw": 2.33,
    "g": 3.0e4,
    "rp_cm": 1.0e10,
    "rstar_cm": 6.957e10,
    "p_ref_bar": 1.0,
    "ptop": 1e-9,
    "pbtm": 100.0,
    "nlayer": 200,
}
H_SCALE = KB * CASE["T"] / (CASE["mmw"] * M_U * CASE["g"])
# R1000 bands from below 4.2 um to above 2.9 um.
MINI_NU0, MINI_BANDS = 2370.0, 400


@pytest.fixture(scope="module")
def rt_ins(tmp_path_factory):
    """(rt model, gas-free isothermal inputs) over a band straddling 3.5 um.

    With no gas, a miniature table's band grid is all that is read.
    """
    with miniature_root(tmp_path_factory.mktemp("mini_cloud"), ["H2O"], MINI_BANDS, MINI_NU0):
        rt = build_rt_model(
            {
                "molecules": ["H2O"],
                # a gas-free column: only the band grid is read
                "omit_core_molecules": ["CO2", "CO", "CH4", "NH3", "HCN", "H2S", "SO2"],
                "nu_min": 1e4 / 4.2,
                "nu_max": 1e4 / 2.9,
                "art_nlayer": CASE["nlayer"],
                "art_ptop_bar": CASE["ptop"],
                "art_pbtm_bar": CASE["pbtm"],
                "rp_cm": CASE["rp_cm"],
                "gs_cgs": CASE["g"],
                "rstar_cm": CASE["rstar_cm"],
                "p_ref_bar": CASE["p_ref_bar"],
            }
        )
    z = jnp.zeros(len(rt.p_art_bar))
    ins = {
        "vmr": {"H2O": z},
        "vmr_h2": z,
        "vmr_he": z,
        "T_art": jnp.full(z.size, CASE["T"]),
        "mmw_art": jnp.full(z.size, CASE["mmw"]),
    }
    return rt, ins


@pytest.fixture(scope="module")
def rt_cloud_only(rt_ins):
    """(depth_fn, nu_grid): the power-law cloud alone."""
    rt, ins = rt_ins
    return (lambda c: rt.transmission_depth(cloud=c, **ins), np.asarray(rt.nu_grid))


def test_grey_deck_puts_the_transit_radius_at_the_deck(rt_ins):
    """An opaque grey deck in a gas-free isothermal column puts the transit radius at the deck, band-independently.

    The radius is the deck's hydrostatic radius R0 + H ln(P0 / P_deck), plus the switch width times the ~13 e-folds the
    deck is thick along the chord (0.26 H at the shipped width), within half a scale height, and equal to the last bit
    across the band. A deck three decades below the grid bottom leaves the deck-free spectrum as it is.
    """
    rt, ins = rt_ins
    base = np.asarray(rt.transmission_depth(**ins))
    far = np.asarray(rt.transmission_depth(deck=np.log10(CASE["pbtm"]) + 3.0, **ins))
    np.testing.assert_allclose(far, base, rtol=1e-12, atol=0.0)
    for lgp in (-3.0, -1.0):
        depth = np.asarray(rt.transmission_depth(deck=lgp, **ins))
        assert np.ptp(depth) == 0.0
        reff = np.sqrt(depth[0]) * CASE["rstar_cm"]
        r_deck = CASE["rp_cm"] + H_SCALE * np.log(CASE["p_ref_bar"] / 10.0**lgp)
        above = (reff - r_deck) / H_SCALE
        assert -0.1 < above < 0.5, above
    # both cloud terms together: the deck below a thin power-law haze adds to it
    both = np.asarray(rt.transmission_depth(cloud=jnp.array([-4.0, 0.0]), deck=-1.0, **ins))
    assert np.all(both >= np.asarray(rt.transmission_depth(deck=-1.0, **ins)))


def test_grey_deck_emits_at_the_deck_temperature(rt_ins):
    """An opaque grey deck under a gas-free column radiates as a black body at the level it sits at.

    The flux is pi B(T) at the deck's emitting level, which the switch width puts within 0.2 dex above the deck
    pressure, and it does not see the temperature below the deck. A deck three decades below the grid bottom leaves the
    deck-free flux as it is.
    """
    rt, ins = rt_ins
    em = build_emis_model(rt, {"rp_cm": CASE["rp_cm"], "gs_cgs": CASE["g"]})
    lgp = np.log10(np.asarray(rt.p_art_bar))

    def temp(x):
        """Temperature, K, rising with log10 P (bar): 600 K at the top, 2250 K at 100 bar."""
        return 600.0 + 150.0 * (x + 9.0)

    ins = {**ins, "T_art": jnp.asarray(temp(lgp))}
    base = np.asarray(em.emission_flux(**ins))
    far = np.asarray(em.emission_flux(deck=np.log10(CASE["pbtm"]) + 3.0, **ins))
    np.testing.assert_allclose(far, base, rtol=1e-12, atol=0.0)

    nu = np.asarray(rt.nu_grid)
    h, c = 6.62607015e-27, 2.99792458e10

    def planck_flux(t):
        return np.pi * 2.0 * h * c**2 * nu**3 / np.expm1(h * c * nu / (KB * t))

    for deck in (-3.0, -1.0):
        flux = np.asarray(em.emission_flux(deck=deck, **ins))
        assert np.all(flux > planck_flux(temp(deck - 0.2)))
        assert np.all(flux < planck_flux(temp(deck)))
        hidden = jnp.where(jnp.asarray(lgp) > deck + 0.3, 1000.0, ins["T_art"])
        below = np.asarray(em.emission_flux(deck=deck, **{**ins, "T_art": hidden}))
        np.testing.assert_allclose(below, flux, rtol=1e-10, atol=0.0)
        tau = np.asarray(
            em.tau_bottom(ins["vmr"], ins["vmr_h2"], ins["T_art"], ins["mmw_art"], ins["vmr_he"], deck=deck)
        )
        assert np.all(tau > 1.0e3)


def _tau0(kappa):
    return kappa * CASE["p_ref_bar"] * 1e6 / CASE["g"] * np.sqrt(2.0 * np.pi * CASE["rp_cm"] / H_SCALE)


def _reff_analytic(kappa):
    t0 = _tau0(kappa)
    return CASE["rp_cm"] + H_SCALE * (GAMMA_E + np.log(t0) + np.asarray(exp1(jnp.asarray(t0))))


def _kappa(nu, c0, alpha):
    return 10.0**c0 * (nu / constants.CLOUD_NUC0) ** alpha


@pytest.mark.parametrize(("c0", "alpha"), [(-2.0, 0.0), (0.0, 0.0), (-2.0, 2.0), (-2.0, 4.0)])
def test_depth_matches_closed_form(rt_cloud_only, c0, alpha):
    """The transit radius must match Heng & Kitzmann to a few 1e-2 H.

    The engine's inverse-square gravity, which the constant-g closed form does not carry, dominates the residual:
    measured worst case over these four decks 1.04e-2 H, gated at 3x that. A grey deck (alpha = 0) must also be
    band-independent to the last bit: a wrong power, a wrong reference wavenumber, or a continuum that fails to
    broadcast identically over the g-ordinates would leak a wavelength dependence.
    """
    depth_of, nu = rt_cloud_only
    depth = np.asarray(depth_of(jnp.array([c0, alpha])))
    if alpha == 0.0:
        assert np.ptp(depth) == 0.0
    reff = np.sqrt(depth) * CASE["rstar_cm"]
    ana = _reff_analytic(_kappa(nu, c0, alpha))
    assert np.max(np.abs(reff - ana)) / H_SCALE < 3.1e-2


def test_alpha_derivative_vanishes_at_the_reference_wavenumber(rt_cloud_only):
    """d(depth)/d(alpha) is proportional to ln(nu/nu0), so it crosses zero at CLOUD_NUC0 and nowhere else.

    A reference-wavenumber or sign error cannot survive this, and it needs no tolerance on the derivative's magnitude.
    """
    depth_of, nu = rt_cloud_only
    g = np.asarray(jax.jvp(depth_of, (jnp.array([-1.0, 2.0]),), (jnp.array([0.0, 1.0]),))[1])
    sign_changes = np.flatnonzero(np.diff(np.sign(g)))
    assert sign_changes.size == 1
    i = int(sign_changes[0])
    x0, x1 = np.log(nu[i]), np.log(nu[i + 1])
    nu_cross = np.exp(x0 - g[i] * (x1 - x0) / (g[i + 1] - g[i]))
    assert abs(nu_cross / constants.CLOUD_NUC0 - 1.0) < 1e-6


def test_kappa_gradient_matches_the_analytic_derivative(rt_cloud_only):
    """AD against an exact derivative, not a finite difference.

    dR_eff/dlog10(kappa0) = H ln(10) (1 - exp(-tau0)), independent of wavelength, alpha, R0, P0 and kappa0 itself.
    Measured 1.61e-3 relative; the gate is 3x that.
    """
    depth_of, nu = rt_cloud_only
    c0, alpha = -1.0, 2.0
    g = np.asarray(jax.jvp(depth_of, (jnp.array([c0, alpha]),), (jnp.array([1.0, 0.0]),))[1])
    kappa = _kappa(nu, c0, alpha)
    reff = _reff_analytic(kappa)
    expect = 2.0 * reff / CASE["rstar_cm"] ** 2 * H_SCALE * np.log(10.0) * (-np.expm1(-_tau0(kappa)))
    assert np.max(np.abs(g / expect - 1.0)) < 5e-3
