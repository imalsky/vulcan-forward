"""The power-law cloud deck, against closed-form physics.

The deck is ``exojax.atm.simple_clouds.powerlaw_clouds`` -- a one-line power
law -- plus this repo's conversion to optical depth in ``_ckd_continuum``:

    dtau = 10^c0 * (nu / CLOUD_NUC0)^alphac * dP[bar] * 1e6 / g

so everything that can be wrong lives in the six lines around exojax: the
per-gram-of-atmosphere convention, the bar->cgs factor and the 1/g, the sign
of alphac, the reference wavenumber, and how a smooth continuum stacks into
the correlated-k dtau and then the chord integral.

For an isothermal, constant-g, uniformly mixed atmosphere with R0 >> H that
chain has an EXACT answer (Heng & Kitzmann 2017, MNRAS 470, 2972):

    R_eff = R0 + H * [gamma_E + ln(tau0) + E1(tau0)]
    tau0  = (kappa * P0 / g) * sqrt(2 pi R0 / H)
    dR_eff/dlog10(kappa0) = H * ln(10) * (1 - exp(-tau0))

The case here is deliberately thin-shell (R0/H ~ 1e4) so those forms are
exact to well under the tolerances asserted; at production geometry the
atmosphere spans ~24% of R and the closed form is NOT a valid reference.
Gas is zeroed while mmw_art is kept, which kills the lines, the CIA
(~vmr_h2^2) and Rayleigh (~mmr_h2) but keeps the scale height, so a failure
can only come from the cloud path.

Gating follows test_e2e_rt_reference.py: never importorskip exojax at module
scope, and vulcan_chem must precede any exojax import.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("exojax") is None:
    pytest.skip("exojax not installed (light-CI environment)",
                allow_module_level=True)
if importlib.util.find_spec("h5py") is None:
    pytest.skip("h5py not installed", allow_module_level=True)

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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.scipy.special import exp1  # noqa: E402

from vulcan_forward import constants  # noqa: E402
from vulcan_forward.exojax_rt import build_rt_model  # noqa: E402

KB, M_U, GAMMA_E = 1.380649e-16, 1.66053906660e-24, 0.5772156649015329
# thin shell on purpose: R0/H = 1.05e4, so the column spans 0.2% of R0
CASE = dict(T=800.0, mmw=2.33, g=3.0e4, rp_cm=1.0e10, rstar_cm=6.957e10,
            p_ref_bar=1.0, ptop=1e-9, pbtm=100.0, nlayer=200)
H_SCALE = KB * CASE["T"] / (CASE["mmw"] * M_U * CASE["g"])


@pytest.fixture(scope="module")
def rt_cloud_only():
    """(depth_fn, nu_grid): cloud-only model over a band straddling 3.5 um."""
    rt = build_rt_model({
        "molecules": ["H2O"], "nu_min": 1e4 / 4.2, "nu_max": 1e4 / 2.9,
        "art_nlayer": CASE["nlayer"], "art_ptop_bar": CASE["ptop"],
        "art_pbtm_bar": CASE["pbtm"], "rp_cm": CASE["rp_cm"],
        "gs_cgs": CASE["g"], "rstar_cm": CASE["rstar_cm"],
        "p_ref_bar": CASE["p_ref_bar"],
    })
    z = jnp.zeros(len(rt.p_art_bar))
    ins = dict(vmr={"H2O": z}, vmr_h2=z, vmr_he=z,
               T_art=jnp.full(z.size, CASE["T"]),
               mmw_art=jnp.full(z.size, CASE["mmw"]))
    return (lambda c: rt.transmission_depth(cloud=c, **ins),
            np.asarray(rt.nu_grid))


def _tau0(kappa):
    return (kappa * CASE["p_ref_bar"] * 1e6 / CASE["g"]
            * np.sqrt(2.0 * np.pi * CASE["rp_cm"] / H_SCALE))


def _reff_analytic(kappa):
    t0 = _tau0(kappa)
    return CASE["rp_cm"] + H_SCALE * (GAMMA_E + np.log(t0)
                                      + np.asarray(exp1(jnp.asarray(t0))))


def _kappa(nu, c0, alpha):
    return 10.0 ** c0 * (nu / constants.CLOUD_NUC0) ** alpha


@pytest.mark.parametrize("c0,alpha", [(-2.0, 0.0), (0.0, 0.0),
                                      (-2.0, 2.0), (-2.0, 4.0)])
def test_depth_matches_closed_form(rt_cloud_only, c0, alpha):
    """The transit radius must match Heng & Kitzmann to a few 1e-2 H.

    The residual is dominated by the engine's inverse-square gravity, which
    the constant-g closed form does not carry. Measured worst case over these
    four decks: 1.04e-2 H; the gate is 3x that. A grey deck (alpha = 0) must
    also be band-independent to the last bit: a wrong power, a wrong
    reference wavenumber, or a continuum that fails to broadcast identically
    over the g-ordinates would leak a wavelength dependence.
    """
    depth_of, nu = rt_cloud_only
    depth = np.asarray(depth_of(jnp.array([c0, alpha])))
    if alpha == 0.0:
        assert np.ptp(depth) == 0.0
    reff = np.sqrt(depth) * CASE["rstar_cm"]
    ana = _reff_analytic(_kappa(nu, c0, alpha))
    assert np.max(np.abs(reff - ana)) / H_SCALE < 3.1e-2


def test_alpha_derivative_vanishes_at_the_reference_wavenumber(rt_cloud_only):
    """d(depth)/d(alpha) is proportional to ln(nu/nu0), so it must cross zero
    EXACTLY at CLOUD_NUC0 and nowhere else.

    A reference-wavenumber or sign error cannot survive this, and it needs no
    tolerance on the derivative's magnitude.
    """
    depth_of, nu = rt_cloud_only
    g = np.asarray(jax.jvp(depth_of, (jnp.array([-1.0, 2.0]),),
                           (jnp.array([0.0, 1.0]),))[1])
    sign_changes = np.flatnonzero(np.diff(np.sign(g)))
    assert sign_changes.size == 1
    i = int(sign_changes[0])
    x0, x1 = np.log(nu[i]), np.log(nu[i + 1])
    nu_cross = np.exp(x0 - g[i] * (x1 - x0) / (g[i + 1] - g[i]))
    assert abs(nu_cross / constants.CLOUD_NUC0 - 1.0) < 1e-6


def test_kappa_gradient_matches_the_analytic_derivative(rt_cloud_only):
    """AD against an EXACT derivative, not a finite difference.

    dR_eff/dlog10(kappa0) = H ln(10) (1 - exp(-tau0)), independent of
    wavelength, alpha, R0, P0 and kappa0 itself. Measured 1.61e-3 relative;
    the gate is 3x that.
    """
    depth_of, nu = rt_cloud_only
    c0, alpha = -1.0, 2.0
    g = np.asarray(jax.jvp(depth_of, (jnp.array([c0, alpha]),),
                           (jnp.array([1.0, 0.0]),))[1])
    kappa = _kappa(nu, c0, alpha)
    reff = _reff_analytic(kappa)
    expect = (2.0 * reff / CASE["rstar_cm"] ** 2
              * H_SCALE * np.log(10.0) * (-np.expm1(-_tau0(kappa))))
    assert np.max(np.abs(g / expect - 1.0)) < 5e-3
