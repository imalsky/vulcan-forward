"""ExoJax side of the engine: differentiable ``ArtTransPure`` / ``ArtEmisPure`` models.

``build_rt_model(profile)`` builds (once) the correlated-k band grid from the
ExoMolOP k-tables of the molecules named in ``profile``, the H2-H2 and H2-He
collision-induced-absorption tables, and an ``ArtTransPure`` radiative-transfer
object. It returns a model whose ``transmission_depth(vmr, vmr_h2, T_art,
mmw_art, vmr_he)`` maps per-layer VMR + temperature + mean-molecular-weight
profiles (already interpolated onto the ART grid) to the transit depth
``(R_p(lambda)/R_star)^2``. ``build_emis_model`` shares the opacities for the
emergent-flux observable.

Everything inside the run functions is pure JAX, so forward-mode tangents from
the chemistry pass straight through to the spectrum. Opacities are float64 (x64
is globally enabled), matching the chemistry side -- no dtype break.

Correlated-k over the published ExoMolOP tables is the ONLY opacity path. The
sampled line-by-line mode and the Mie condensate deck were removed: sampling
below exojax's critical R = 700,000 is measurably biased and the Mie deck was
the only thing that still needed it. A profile that asks for either is
refused, never silently mapped onto this path.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # the k-tables are float64; safe if already set
import jax.numpy as jnp

from vulcan_forward import constants, paths

from exojax.opacity import OpaCIA
from exojax.rt import ArtTransPure, ArtEmisPure
from exojax.atm.atmconvert import vmr_to_mmr
from exojax.atm.simple_clouds import powerlaw_clouds
from exojax.opacity.rayleigh import xsvector_rayleigh_gas
from exojax.atm.polarizability import polarizability as _POLARIZABILITY
from exojax.database.contdb import CdbCIA
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_linsap

_H2_MOLMASS, _HE_MOLMASS = 2.016, 4.0026   # g/mol, for the Rayleigh mmr conversion

# bar -> dyn/cm^2, for per-MASS (cm^2/g of atmosphere) opacities: dtau = kappa*dP_cgs/g.
# (exojax's layer_optical_depth folds bar_cgs/m_u into its opacity_factor because its
# xs are per-molecule cross sections; a per-gram kappa must NOT pick up the 1/m_u.)
_BAR_CGS = 1.0e6

def _check_profile(profile: dict) -> None:
    """Unknown keys raise (``constants.check_profile_keys``), and so does any
    opacity mode but correlated-k: the line-by-line mode and the Mie deck were
    removed, and a profile asking for them must not be mapped onto this path."""
    constants.check_profile_keys(profile)
    mode = str(profile.get("opacity_mode", "exomolop"))
    if mode != "exomolop":
        raise ValueError(
            f"opacity_mode={mode!r} is not available: the sampled line-by-line "
            "mode ('lbl') and the Mie condensate deck were removed. "
            "Correlated-k over the published ExoMolOP tables ('exomolop', the "
            "default -- drop the key) is the only opacity path.")


def _gravity_profile_invsq(art, T_art, mmw_art, radius_btm, gravity_btm):
    """Mid-layer inverse-square gravity profile, g(r) = g_btm * (R_btm/r)^2.

    ExoJax's own ``ArtCommon.gravity_profile`` (<=2.2.3) returns
    ``g_btm / rn`` -- LINEAR in 1/r -- while its height integrator uses the
    physical inverse-square ``g_btm / rn**2``. Using ``gravity_profile`` for
    the opacity columns leaves heights and columns on DIFFERENT gravities and
    removes only about half of the constant-g bias (measured vs an independent
    chord quadrature: -101.8 ppm constant-g, -50.8 ppm gravity_profile,
    +1.5 ppm this profile). Same (nlayer, 1) shape contract as
    ``gravity_profile`` so it broadcasts through the dtau kernels.
    """
    normalized_height, normalized_radius_lower = art.atmosphere_height(
        T_art, mmw_art, radius_btm, gravity_btm)
    rn_mid = normalized_radius_lower + 0.5 * normalized_height
    return jnp.array([gravity_btm / rn_mid**2]).T


def _anchor_to_grid_bottom(lnp_art, T_art, mmw_art, r_ref, g_ref, p_ref_bar):
    """Radius and gravity at the RT grid bottom, from values quoted at p_ref.

    ExoJax defines ``radius_btm``/``gravity_btm`` at the LOWER boundary of the
    bottom layer. A published planet radius is instead the transit radius, i.e.
    the radius at roughly the photosphere (~mbar). Handing exojax the literature
    pair stacks the whole p_ref -> p_btm column ON TOP OF a radius that already
    is the photospheric one, which inflated WASP-39 b's transit depth by
    5472 ppm (26,853 vs a measured 21,381) and its spectral contrast by 1.42x.

    Integrating hydrostatic equilibrium with GM held fixed, so that
    ``g(r) = g_ref (r_ref/r)^2``::

        dr/dlnP = -kT/(mu m_H g(r)) = -C(P) r^2,   C = kT/(mu m_H g_ref r_ref^2)

    the substitution u = 1/r linearises it exactly::

        du/dlnP = +C(P)      ->      u_btm = u_ref + int_{lnP_ref}^{lnP_btm} C dlnP

    so no ODE solve is needed and the whole thing stays differentiable. Deeper
    means smaller radius and stronger gravity, as it must.

    The target level is the bottom layer's LOWER BOUNDARY, not its
    representative (centre) pressure. exojax builds ``art.pressure`` with
    ``pressure_layer_logspace``, whose entries are layer CENTRES, and defines
    ``radius_btm`` at ``pressure_lower_logspace(...) = p[-1] * k**-0.5``
    (atmprof.py) -- half a log-layer deeper. Anchoring to ``p[-1]`` instead put
    the reference radius at 1.11 mbar for a requested 1 mbar on the planner's
    100-layer grid and at 1.71 mbar on a 20-layer one, i.e. it made the
    absolute transit depth depend on ``art_nlayer`` (+0.20% to +1.02%).

    lnp_art must be ascending (exojax orders its grid top-to-bottom).
    Returns (r_btm, g_btm) in cgs.
    """
    return _radius_at(lnp_art, T_art, mmw_art, r_ref, g_ref, p_ref_bar,
                      jnp.exp(_lnp_grid_bottom_boundary(lnp_art)))


def _boundary_temperature(T_art):
    """Temperature at the layer BOUNDARIES (nlayer+1) from the centres.

    exojax's ``ibased_linsap`` solver takes its source at the boundaries; the
    chemistry hands us representative (centre) values. The ART grid is uniform
    in log pressure, so a boundary is the midpoint of the two centres around
    it, and the two ends extrapolate with the local gradient.

    Extrapolated, not clamped, at the DEEP end on purpose: that last entry is
    the interior source term the solver multiplies by exp(-tau_total/mu), the
    one level standing in for everything below the grid. Clamping it to the
    deepest centre would systematically cool it. With an optically thick bottom
    the term is negligible either way, which is the point -- it should be
    negligible because the column is opaque, not because it was set too low.
    """
    mid = 0.5 * (T_art[1:] + T_art[:-1])
    return jnp.concatenate([
        (1.5 * T_art[0] - 0.5 * T_art[1])[None],
        mid,
        (1.5 * T_art[-1] - 0.5 * T_art[-2])[None]])


def _lnp_grid_bottom_boundary(lnp_art):
    """ln(pressure) at the LOWER BOUNDARY of exojax's bottom layer.

    The grid is uniform in log pressure, so the boundary sits half a layer
    below the last centre. Equivalent to exojax's ``p[-1] * k**-0.5``.
    """
    return lnp_art[-1] + 0.5 * (lnp_art[-1] - lnp_art[-2])


def _radius_at(lnp_art, T_art, mmw_art, r_ref, g_ref, p_ref_bar, p_target_bar):
    """Radius and gravity at ``p_target_bar``, given them at ``p_ref_bar``.

    The workhorse behind _anchor_to_grid_bottom; see that docstring for the
    derivation. Separate because emission needs a DIFFERENT target level than
    transmission: the two observables probe different depths, so a single anchor
    cannot serve both (Fortney et al. 2019, ApJL 880, L16).

    The integration nodes extend HALF A LAYER above the top grid centre and
    below the deepest one, holding the end layer's T and mmw over each half
    layer (which is what a representative layer value means). Without the
    extra nodes ``jnp.interp`` clamps at the end centres: below, that silently
    dropped the half layer down to the grid's lower boundary -- the level
    exojax actually defines ``radius_btm`` at; above, it pinned any level in
    the top half layer (where ``_photosphere_lnp`` can put the tau = 2/3
    photosphere) to the top-centre radius with a zero derivative.
    """
    C = constants.K_B_CGS * T_art / (
        mmw_art * constants.M_U_CGS * g_ref * r_ref ** 2)
    d = lnp_art[1] - lnp_art[0]
    lnp = jnp.concatenate([lnp_art[:1] - 0.5 * d, lnp_art,
                           _lnp_grid_bottom_boundary(lnp_art)[None]])
    C = jnp.concatenate([C[:1], C, C[-1:]])
    seg = 0.5 * (C[1:] + C[:-1]) * jnp.diff(lnp)
    # The integral is zero at the top CENTRE, exactly as before the top node
    # existed, so every level at or below it evaluates bitwise as it did.
    cum = jnp.concatenate([-seg[:1], jnp.zeros(1), jnp.cumsum(seg[1:])])
    i_ref = jnp.interp(jnp.log(p_ref_bar), lnp, cum)
    i_tgt = jnp.interp(jnp.log(p_target_bar), lnp, cum)
    r = 1.0 / (1.0 / r_ref + (i_tgt - i_ref))
    return r, g_ref * (r_ref / r) ** 2


def _accumulate_dtau_ckd(art, pack, mols, molmass, opacia, opacia_he,
                         vmr, vmr_h2, vmr_he, T_art, mmw_art, g_btm,
                         cloud=None, rayleigh_xs=None):
    """Per-layer, per-g-ordinate, per-band optical depth ``(nlayer, ng, nband)``.

    The line opacity of each molecule comes from its k-table, interpolated to
    the layer (T, P) and combined across molecules by random-overlap
    resort-rebin (``ckd.fold``, one scan over the molecules). The
    continua -- CIA, Rayleigh, the power-law cloud -- are smooth across a band,
    so their band value adds identically to every g-ordinate; that is exact,
    not an approximation, to the accuracy of a smooth function over one R=1000
    band.

    Shared by the transmission and emission models. Rayleigh is
    transmission-only BY DESIGN: it is scattering, not absorption, and the
    pure-absorption emission solver must not count it as thermal extinction
    (it is also negligible at the >1 um thermal bands).
    """
    from vulcan_forward import ckd as _ckd

    P = jnp.asarray(art.pressure)
    dts = jnp.stack([_ckd_dt_one(art, pack, key, molmass, vmr[key], T_art,
                                 mmw_art, g_btm, P) for key in mols])
    cont = _ckd_continuum(art, pack, opacia, opacia_he, vmr_h2, vmr_he,
                          T_art, mmw_art, g_btm, cloud, rayleigh_xs)
    return _ckd.fold(dts, pack.gg, pack.gw) + cont[:, None, :]


def _ckd_dt_one(art, pack, key, molmass, vmr_key, T_art, mmw_art, g_btm, P):
    """One molecule's (nlayer, ng, nband) optical-depth tensor."""
    from vulcan_forward import ckd as _ckd

    lk = _ckd._interp_logk(pack.logk[key], pack.t_grid, pack.p_grid,
                           T_art, P)                        # (nlayer, ng, nb)
    mmr = vmr_to_mmr(vmr_key, molmass[key], mmw_art)
    # layer_optical_depth_ckd multiplies a (nlayer, ng, nband) tensor by
    # dParr[:, None, None] / gravity, so gravity needs the third axis too;
    # the continuum terms stay on the 2-D (nlayer, nband) form.
    return art.opacity_profile_xs_ckd(jnp.exp(lk), mmr, molmass[key],
                                      g_btm[..., None])


def _ckd_continuum(art, pack, opacia, opacia_he, vmr_h2, vmr_he,
                   T_art, mmw_art, g_btm, cloud, rayleigh_xs):
    """Band continuum (nlayer, nband): CIA + Rayleigh + power-law cloud.

    ``cloud`` is the optional (2,) array [log10 kappac0 (cm^2/g at
    constants.CLOUD_NUC0), alphac] for ``exojax.atm.simple_clouds.powerlaw_clouds``
    (alphac=0 -> gray cloud; per-gram-of-atmosphere opacity, uniformly mixed:
    dtau = kappa(nu)*dP_cgs/g). ``rayleigh_xs`` = (xs_h2, xs_he), each (nband,)
    from exojax ``xsvector_rayleigh_gas`` -- zero-free-parameter known physics
    that matters short of ~1.5 um; omitting it would bias a retrieved haze slope.
    """
    def _cia(opa_, va, vb):
        # opacity_profile_cia divides a (nlayer, nband) matrix by mmw, so mmw
        # must broadcast as (nlayer, 1) here.
        return art.opacity_profile_cia(opa_.logacia_matrix(T_art), T_art,
                                       va, vb, mmw_art[:, None], g_btm)

    cont = _cia(opacia, vmr_h2, vmr_h2)
    if opacia_he is not None and vmr_he is not None:
        cont = cont + _cia(opacia_he, vmr_h2, vmr_he)
    if rayleigh_xs is not None:
        xs_h2, xs_he = rayleigh_xs
        nl, nb = art.pressure.shape[0], xs_h2.shape[0]
        cont = cont + art.opacity_profile_xs(
            jnp.broadcast_to(xs_h2[None, :], (nl, nb)),
            vmr_to_mmr(vmr_h2, _H2_MOLMASS, mmw_art), _H2_MOLMASS, g_btm)
        if vmr_he is not None:
            cont = cont + art.opacity_profile_xs(
                jnp.broadcast_to(xs_he[None, :], (nl, nb)),
                vmr_to_mmr(vmr_he, _HE_MOLMASS, mmw_art), _HE_MOLMASS, g_btm)
    if cloud is not None:
        # ExoJax's shipped retrieval cloud (pRT convention, per gram of atmosphere).
        kappa_c = powerlaw_clouds(pack.nu_bands_j, kappac0=10.0 ** cloud[0],
                                  nuc0=constants.CLOUD_NUC0, alphac=cloud[1])
        dP = jnp.asarray(art.dParr)
        cont = cont + kappa_c[None, :] * (dP[:, None] * _BAR_CGS / g_btm)
    return cont


def _ckd_dtau_batch(art, pack, mols, molmass, opacia, opacia_he,
                    vmr, vmr_h2, vmr_he, T_art, mmw_art, g_btm,
                    wo_mols, finish, cloud=None, rayleigh_xs=None):
    """Full + leave-one-out observables in one pass over the k-fold.

    ``finish(dtau_g)`` maps a (nlayer, ng, nband) optical depth to its
    observable. Returns ``(finish(full), [finish(wo) ...])`` with the wo list
    aligned to ``wo_mols``. The dropped molecule is still folded, as the zero
    tensor its zeroed VMR produces through the SAME op pipeline, so every
    output is bit-identical to a from-scratch solve with that VMR zeroed;
    only the shared fold prefix is reused (see ckd._fold_wo). The
    per-molecule tensors are held for the
    fold tails (~n x 35 MB at planner defaults); wo totals are finished and
    freed one at a time.
    """
    from vulcan_forward import ckd as _ckd

    bad = [m for m in wo_mols if m not in mols]
    if bad or len(set(wo_mols)) != len(wo_mols):
        raise ValueError(
            f"wo_mols {list(wo_mols)!r} must be unique members of the RT "
            f"molecule set {list(mols)!r}")
    P = jnp.asarray(art.pressure)
    dts = [_ckd_dt_one(art, pack, k, molmass, vmr[k], T_art, mmw_art, g_btm, P)
           for k in mols]
    cont3 = _ckd_continuum(art, pack, opacia, opacia_he, vmr_h2, vmr_he,
                           T_art, mmw_art, g_btm, cloud,
                           rayleigh_xs)[:, None, :]

    def zero_of(i):
        return _ckd_dt_one(art, pack, mols[i], molmass,
                           jnp.zeros_like(vmr[mols[i]]), T_art, mmw_art,
                           g_btm, P)

    wo_idx = [mols.index(m) for m in wo_mols]
    full_tot, wo = _ckd._fold_wo(dts, zero_of, pack.gg, pack.gw, wo_idx,
                                 finish=lambda t: finish(t + cont3))
    by_idx = dict(wo)
    return finish(full_tot + cont3), [by_idx[i] for i in wo_idx]


# Vertical optical depth of the emitting photosphere (Fortney, Lupu, Morley,
# Freedman & Hood 2019, ApJL 880, L16): the eclipse-depth prefactor is the
# radius where tau = 2/3 at EACH wavelength, not one radius for the band.
TAU_PHOTOSPHERE = 2.0 / 3.0


def _photosphere_lnp(dtau_g, lnp_art):
    """ln(pressure) where the vertical optical depth from the top reaches
    TAU_PHOTOSPHERE, per (g, band) of a ``(nlayer, ng, nband)`` optical depth;
    the grid's bottom boundary where it never does (a see-through column is the
    thin-bottom gate's business, not this function's). Piecewise-linear in tau
    between layer boundaries, so it differentiates through dtau."""
    d = lnp_art[1] - lnp_art[0]
    lnp_b = jnp.concatenate([lnp_art[:1] - 0.5 * d, lnp_art + 0.5 * d])
    tau_b = jnp.concatenate([jnp.zeros((1,) + dtau_g.shape[1:]),
                             jnp.cumsum(dtau_g, axis=0)])
    cols = tau_b.reshape(tau_b.shape[0], -1).T              # (ng*nband, nlayer+1)
    lnp = jax.vmap(lambda t: jnp.interp(TAU_PHOTOSPHERE, t, lnp_b))(cols)
    return lnp.reshape(dtau_g.shape[1:])


def _run_emis_ckd_linsap(art, dtau_g, T_boundary, nu_bands, gw, weight_g=None):
    """Emergent flux (nband,) from a ``(nlayer, ng, nband)`` CKD optical depth,
    solved with the LINEAR-SOURCE scheme. ``weight_g`` (ng, nband) multiplies
    each g-ordinate's flux before the g-average: the photospheric-radius
    factor of ``eclipse_flux_tau``, which differs between the k-ordinates of
    one band exactly as the flux does.

    exojax's ``ArtEmisPure.run_ckd`` hard-codes ``rtrun_emis_pureabs_ibased``,
    which has no bottom-boundary term, so every photon entering the grid from
    below is lost (measured flux deficits: notes.md §1.2, register #8).
    This is upstream's own flatten-solve-reweight structure with
    ``ibased_linsap`` in its place, so CKD emission keeps the interior source
    term the solver carries.

    The flatten is g-major / band-minor, matching ``jnp.tile``'s last-axis
    tiling of the source, exactly as upstream's version does it: element
    ``(g, b)`` of the (ng, nband) block lands at flat index ``g * nband + b``
    on both sides.

    Weighting the FLUX by the g-weights (rather than some intermediate) is what
    makes this correct: the k-distribution is a reordering of wavenumber within
    the band, the solver is linear in nothing but acts wavenumber-by-
    wavenumber, and the band-integrated flux is the g-average of the flux
    computed at each k. Same identity the transmission path uses.
    """
    nlayer, ng, nband = dtau_g.shape
    # piBarr -> (nlayer + 1, nband); tile the last axis to (nlayer + 1, ng*nband)
    src = jnp.tile(piBarr(T_boundary, nu_bands), ng)
    flux = rtrun_emis_pureabs_ibased_linsap(
        dtau_g.reshape((nlayer, ng * nband)), src, art.mus, art.weights)
    flux = flux.reshape((ng, nband))
    if weight_g is not None:
        flux = flux * weight_g
    return jnp.einsum("g,gb->b", gw, flux)


_GEOMETRY_KEYS = {
    "rp_cm": "planet radius in cm at p_ref_bar",
    "gs_cgs": "gravity in cm/s^2 at p_ref_bar",
    "rstar_cm": "stellar radius in cm",
}


def _require_geometry(profile: dict, *keys: str) -> None:
    """Refuse a profile missing planet geometry.

    These keys set the transit-depth normalization and the hydrostatic
    scale, so there is no safe default (standing fail-loud rule). A missing
    one must never fall back to another planet's constants.
    """
    missing = [k for k in keys if k not in profile]
    if missing:
        raise ValueError(
            f"profile is missing required planet geometry {missing}: "
            + "; ".join(f"{k} ({_GEOMETRY_KEYS[k]})" for k in missing) + ".")


def _require_he(vmr_he):
    """He is ~14% by number and its CIA is real continuum physics; a None
    would silently drop the term, so both observables refuse it."""
    if vmr_he is None:
        raise ValueError(
            "vmr_he is required: pass the He VMR profile (chem.sidx['He']) so the "
            "H2-He CIA term is included. There is no supported He-less mode.")


def _check_cia_span(cdb, nu_grid, label):
    """Refuse a CIA table that does not cover the requested wavenumber grid.

    `CdbCIA` reads only the rows inside the requested range and never checks
    coverage, so bands past the file's edge silently reuse the edge
    coefficient -- the same clamp the k-table pressure ceiling is refused for.
    """
    lo, hi = float(np.min(cdb.nucia)), float(np.max(cdb.nucia))
    g_lo, g_hi = float(np.min(nu_grid)), float(np.max(nu_grid))
    if g_lo < lo or g_hi > hi:
        raise ValueError(
            f"{label} CIA table covers [{lo:g}, {hi:g}] cm^-1 but the requested "
            f"grid spans [{g_lo:g}, {g_hi:g}] cm^-1. Bands outside the table "
            "would silently reuse its edge coefficient. Narrow nu_min/nu_max "
            "or install a table that covers the band.")


def build_rt_model(profile: dict) -> SimpleNamespace:
    """Build the transmission-spectrum model for the molecules named in ``profile``.

    Returns a SimpleNamespace with:
        transmission_depth(vmr, vmr_h2, T_art, mmw_art, vmr_he) -> (n_nu,) transit depth
        nu_grid : (n_nu,) band-centre wavenumber grid (cm^-1)
        wl_um   : (n_nu,) wavelength grid (micron), descending->ascending sorted handled by caller
        p_art_bar : (nlayer,) ART pressure grid (bar)
        molecules : list[str]
    """
    t0 = time.time()
    _check_profile(profile)
    mols = list(profile["molecules"])
    # Published ExoMol/HITEMP opacities with H2/He broadening already applied.
    # See vulcan_forward.exomolop for the three measured defects this closes
    # over building tables from HITRAN; the band grid and the split quadrature
    # come from the files.
    from vulcan_forward import exomolop as _exo
    ckd_pack = _exo.load_tables(
        mols, float(profile["nu_min"]), float(profile["nu_max"]),
        molecule_table=profile.get("molecule_table"))
    nu_grid = jnp.asarray(ckd_pack.nu_bands)
    ckd_pack.nu_bands_j = nu_grid
    # Profile-overridable RT knobs, validated loudly here: an out-of-range
    # value must never build a wrong model.
    ptop = float(profile.get("art_ptop_bar", constants.ART_PTOP_BAR))
    pbtm = float(profile.get("art_pbtm_bar", constants.ART_PBTM_BAR))
    if not (0.0 < ptop < pbtm):
        raise ValueError(
            f"art_ptop_bar={ptop:g} / art_pbtm_bar={pbtm:g}: need "
            "0 < top < bottom (bar)")
    # _interp_logk clamps log P at BOTH ends. The low-P clamp is defensible
    # (k is Doppler-dominated and pressure-independent there); the deep one is
    # not -- pressure broadening grows with P, so a layer below the table
    # ceiling reuses an under-broadened, under-opaque entry. Refuse it.
    p_ceiling = float(np.asarray(ckd_pack.p_grid)[-1])
    if pbtm > p_ceiling:
        raise ValueError(
            f"art_pbtm_bar={pbtm:g} bar is below the k-table pressure ceiling "
            f"({p_ceiling:g} bar). Layers deeper than the ceiling reuse its "
            "entry, which is under-broadened and therefore under-opaque. Raise "
            "the column bottom or install tables that reach deeper.")
    integration = str(profile.get("rt_integration", "simpson"))
    if integration not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={integration!r}: exojax ArtTransPure supports "
            "'simpson' (default) or 'trapezoid'")
    # The molecule table is INJECTABLE: a consumer adding a molecule passes its
    # own table rather than editing a constant inside this package.
    mol_table = profile.get("molecule_table") or constants.MOLECULES
    _unknown = [k for k in mols if k not in mol_table]
    if _unknown:
        raise KeyError(
            f"no molecule spec for {_unknown} in the molecule table "
            f"(have: {sorted(mol_table)}). Pass profile['molecule_table'] "
            "with one entry per molecule: vulcan, molmass.")
    molmass = {key: float(mol_table[key]["molmass"]) for key in mols}

    art = ArtTransPure(
        pressure_top=ptop,
        pressure_btm=pbtm,
        nlayer=int(profile["art_nlayer"]),
        integration=integration)
    p_art_bar = np.asarray(art.pressure)
    # ascending (exojax orders top-to-bottom); _anchor_to_grid_bottom relies on it
    if not np.all(np.diff(p_art_bar) > 0):
        raise RuntimeError(
            "the ART pressure grid is not ascending; _anchor_to_grid_bottom's "
            "cumulative integral and jnp.interp both assume it is")
    lnp_art = jnp.asarray(np.log(p_art_bar))
    print(f"[rt] ArtTransPure {profile['art_nlayer']} layers, "
          f"P=[{p_art_bar.min():.1e},{p_art_bar.max():.1e}] bar, "
          f"chord integration {integration}",
          flush=True)

    def _cia(path, label):
        cdb = CdbCIA(str(path), nurange=nu_grid)
        _check_cia_span(cdb, nu_grid, label)
        return OpaCIA(cdb, nu_grid=nu_grid)

    cia_h2h2 = paths.cia_h2h2_file()
    if not cia_h2h2.exists():
        # exojax auto-fetches it (~24 MB from hitran.org), but its downloader
        # swallows failures -- say up front what is about to happen so an
        # offline failure is attributable (fail-loud rule).
        print(f"[rt] H2-H2 CIA absent at {cia_h2h2}; exojax will "
              "download ~24 MB from https://hitran.org/data/CIA/main/"
              "H2-H2_2011.cia now (network required)", flush=True)
    opacia = _cia(cia_h2h2, "H2-H2")
    # H2-He CIA is required physics (He is ~14% by number): without it the
    # spectrum would lose a real continuum term with no error.
    cia_h2he = paths.cia_h2he_file()
    if not cia_h2he.exists():
        raise FileNotFoundError(
            f"H2-He CIA table missing ({cia_h2he}). Download "
            "https://hitran.org/data/CIA/main/H2-He_2011.cia (~147 MB; note the "
            "/main/ path -- the bare /data/CIA/ URL 404s) to that exact path. "
            "Refusing to build the RT without it (silently skipping the He "
            "continuum would bias the spectrum).")
    opacia_he = _cia(cia_h2he, "H2-He")
    print(f"[rt] CIA + RT built; total {time.time()-t0:.1f}s", flush=True)

    # H2/He Rayleigh cross sections (nu-only, precomputed once; opt-in via profile)
    if profile.get("use_rayleigh", False):
        rayleigh_xs = (
            xsvector_rayleigh_gas(nu_grid, _POLARIZABILITY["H2"]),
            xsvector_rayleigh_gas(nu_grid, _POLARIZABILITY["He"]),
        )
        print("[rt] H2/He Rayleigh scattering enabled", flush=True)
    else:
        rayleigh_xs = None

    _require_geometry(profile, "rp_cm", "gs_cgs", "rstar_cm")
    Rp_ref = float(profile["rp_cm"])
    g_ref = float(profile["gs_cgs"])
    rstar_cm = float(profile["rstar_cm"])
    # Pressure at which rp_cm / gs_cgs are quoted. A published planet radius is
    # the TRANSIT radius, i.e. the radius at roughly the photosphere, NOT the
    # radius at the bottom of a model grid. See _anchor_to_grid_bottom.
    p_ref_bar = float(profile.get("p_ref_bar", constants.P_REF_BAR))
    if not (ptop <= p_ref_bar <= pbtm):
        raise ValueError(
            f"p_ref_bar={p_ref_bar:g} lies outside the RT grid "
            f"[{ptop:g}, {pbtm:g}] bar. It is the pressure at which rp_cm and "
            "gs_cgs are defined, so it must be a level the grid actually covers.")

    def transmission_depth(vmr, vmr_h2, T_art, mmw_art, vmr_he=None,
                           cloud=None):
        """Transit depth (R_p(lambda)/R_star)^2 from ART-grid profiles.

        vmr : dict molecule -> (nlayer,) VMR; vmr_h2 : (nlayer,) H2 VMR (for CIA);
        vmr_he : (nlayer,) He VMR (H2-He CIA partner; REQUIRED -- the None default
        exists only so an omission raises the explanatory ValueError, not TypeError).
        Optional: cloud=[log10 kappac0, alphac] (ExoJax powerlaw_clouds).
        """
        _require_he(vmr_he)
        # rp_cm/gs_cgs are quoted at p_ref_bar (the transit radius, ~mbar), not at
        # the grid bottom where exojax wants them -- convert first.
        Rp_btm, g_btm = _anchor_to_grid_bottom(
            lnp_art, T_art, mmw_art, Rp_ref, g_ref, p_ref_bar)
        # g(r) self-consistency: use the SAME inverse-square gravity ExoJax
        # uses for the chord heights in the pressure->column-mass conversion.
        # Constant g_btm makes upper-layer tau too small; art.gravity_profile
        # is 1/r-linear and is NOT this profile (see _gravity_profile_invsq).
        # Emission is plane-parallel and correctly keeps constant g_btm.
        g_prof = _gravity_profile_invsq(art, T_art, mmw_art, Rp_btm, g_btm)  # (nlayer,1)
        dtau_g = _accumulate_dtau_ckd(
            art, ckd_pack, mols, molmass, opacia, opacia_he,
            vmr, vmr_h2, vmr_he, T_art, mmw_art, g_prof,
            cloud=cloud, rayleigh_xs=rayleigh_xs)
        Rp2 = art.run_ckd(dtau_g, T_art, mmw_art, Rp_btm, g_btm,
                          ckd_pack.gw)
        return Rp2 * (Rp_btm / rstar_cm) ** 2

    def transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, lnR0, vmr_he=None,
                             cloud=None, wo_mols=None):
        """transmission_depth with a reference-radius scaling: the radius at the bottom
        pressure P_btm is Rp_btm * e^lnR0 (gravity held fixed -- the standard xR_p
        normalization nuisance, cf. Batalha & Line 2017). lnR0 = 0 reproduces
        transmission_depth exactly; the lnR0 jvp is the exact geometric+hydrostatic
        response, RT-only (chemistry profiles enter frozen). Because gravity is held
        fixed, lnR0 must be read as a pressure-radius normalization, NOT a physical
        planet-radius change at fixed mass.

        wo_mols: None (default) returns the depth (n_nu,) exactly as before.
        A list of molecule names returns ``(depth, depth_wo)`` with one row per
        entry, each the depth with that molecule's VMR zeroed -- bit-identical
        to a separate call on the zeroed profile, but reusing the shared
        correlated-k fold prefix (~2x fewer overlap folds for a full set)."""
        _require_he(vmr_he)
        Rp_btm, g_btm = _anchor_to_grid_bottom(
            lnp_art, T_art, mmw_art, Rp_ref, g_ref, p_ref_bar)
        Rp_r = Rp_btm * jnp.exp(lnR0)
        # g(r) at the lnR0-scaled reference radius (gravity g_btm held fixed, per
        # the xR_p normalization; the height grid and thus g(r) shift with Rp_r) --
        # inverse-square, matching the g(r) art.run uses for the heights (see
        # transmission_depth; art.gravity_profile is 1/r-linear and is NOT used).
        g_prof = _gravity_profile_invsq(art, T_art, mmw_art, Rp_r, g_btm)  # (nlayer,1)

        def _depth_of(Rp2):
            # one expression and op order for both paths, so the single and wo
            # results agree bitwise
            return (Rp2 * (Rp_btm / rstar_cm) ** 2                  # (radius/R_star)^2
                    * jnp.exp(2.0 * lnR0))

        def _finish(dtau_g):
            return _depth_of(art.run_ckd(dtau_g, T_art, mmw_art, Rp_r,
                                         g_btm, ckd_pack.gw))

        if wo_mols is None:
            dtau_g = _accumulate_dtau_ckd(
                art, ckd_pack, mols, molmass, opacia, opacia_he,
                vmr, vmr_h2, vmr_he, T_art, mmw_art, g_prof,
                cloud=cloud, rayleigh_xs=rayleigh_xs)
            return _finish(dtau_g)
        depth, rows = _ckd_dtau_batch(
            art, ckd_pack, mols, molmass, opacia, opacia_he,
            vmr, vmr_h2, vmr_he, T_art, mmw_art, g_prof,
            list(wo_mols), _finish, cloud=cloud, rayleigh_xs=rayleigh_xs)
        return depth, (jnp.stack(rows) if rows
                       else jnp.zeros((0,) + depth.shape))

    return SimpleNamespace(
        transmission_depth=transmission_depth,
        transmission_depth_r=transmission_depth_r,
        nu_grid=np.asarray(nu_grid),
        wl_um=1e4 / np.asarray(nu_grid),
        p_art_bar=p_art_bar,
        molecules=mols,
        # echo of the profile-overridable RT knobs, so downstream consumers
        # (jwst-transit-authority) can VERIFY the engine honored them -- an older
        # engine that ignores an unknown profile key must fail loudly there,
        # never silently compute a different model than the cache key claims
        art_ptop_bar=ptop,
        art_pbtm_bar=pbtm,
        # the pressure rp_cm/gs_cgs were taken to apply at; consumers verify this
        # echo, because a tool that silently reverted to bottom-of-grid anchoring
        # would inflate every transit depth (see _anchor_to_grid_bottom)
        p_ref_bar=p_ref_bar,
        rt_integration=integration,
        # which opacity path ran; there is only one, and consumers still
        # verify the echo so an engine/tool version mismatch is loud
        opacity_mode="exomolop",
        # internals reused by build_emis_model (so opacities aren't rebuilt)
        _nu_grid=nu_grid, _molmass=molmass, _opacia=opacia,
        _opacia_he=opacia_he, _ckd_pack=ckd_pack,
    )


def build_emis_model(trt, profile: dict) -> SimpleNamespace:
    """Build an ArtEmisPure thermal-emission model that SHARES trt's opacities/grid.

    Returns a model whose ``emission_flux(vmr, vmr_h2, T_art, mmw_art, vmr_he)`` maps
    the same ART-grid profiles to the planet's EMERGENT flux spectrum
    (erg s^-1 cm^-2 / cm^-1). This is the top-of-atmosphere planetary flux, NOT an
    eclipse depth / planet-star contrast -- do not compare it to an observed
    secondary-eclipse spectrum without dividing by the stellar flux and applying
    (Rp/Rstar)^2. Opacity terms match transmission (lines + H2-H2 + H2-He CIA,
    optional cloud); Rayleigh scattering is deliberately excluded here (see
    _accumulate_dtau_ckd -- a pure-absorption solver must not count scattering
    as thermal absorption, and it is negligible in the thermal bands).
    """
    constants.check_profile_keys(profile)
    # CKD emission runs through _run_emis_ckd_linsap, NOT ArtEmisPure.run_ckd:
    # upstream's version hard-codes the "ibased" solver, which has no interior
    # source term.
    ckd_pack = trt._ckd_pack
    nu_grid = trt._nu_grid
    molmass, opacia, mols = trt._molmass, trt._opacia, trt.molecules
    opacia_he = trt._opacia_he
    _require_geometry(profile, "gs_cgs")
    # Emission is plane-parallel, so ArtEmisPure needs ONE gravity for the whole
    # column: it converts pressure to column mass as dP/g. That gravity must be
    # the same physical quantity the transmission path anchors, otherwise the two
    # observables silently disagree about the planet. gs_cgs is quoted at
    # p_ref_bar (~1 mbar), while the column is dominated by the emission
    # photosphere near 0.1 bar, so the gravity is re-anchored there.
    #
    # The RADIUS must be re-anchored with it. The eclipse depth prefactor is
    # (R_p/R_star)^2, and R_p there is the radius of the emitting surface. Using
    # the 1 mbar transit radius with the 0.1 bar gravity makes the pair imply a
    # planet mass 8.9% off its own GM (measured on WASP-39 b: r(0.1 bar) =
    # 1.2257 RJ against the 1.2790 RJ transit radius, so the eclipse depth ran
    # 8.9% high). ``emission_radius`` below returns the consistent value; the
    # consumer must use it for the prefactor rather than the catalogue radius.
    #
    # That single radius is the ANCHOR only. The prefactor a consumer must use
    # is the wavelength-dependent photospheric radius at vertical tau = 2/3
    # (Fortney, Lupu, Morley, Freedman & Hood 2019, ApJL 880, L16; what
    # POSEIDON and PLATON II compute): ``eclipse_flux_tau`` folds
    # (R_phot/R_em)^2 into the flux per k-ordinate, so depth = that flux / F_s
    # x (R_em/R_star)^2 is exact. Measured on the planner's WASP-39 b dayside
    # (g ~ 470 cm/s2) the single-radius depth was low by a median 7% and by 13%
    # in the CO2 core against 4% in the neighbouring continuum -- a feature
    # error, not an offset; on HD 189733 b (g ~ 2200) under 2.5% everywhere.
    # ``emission_flux`` stays the plain emergent flux (the pRT comparison and
    # the pi*B(T) check are flux tests).
    _require_geometry(profile, "rp_cm")
    g_ref_em = float(profile["gs_cgs"])
    r_ref_em = float(profile["rp_cm"])
    p_ref_em = float(profile.get("p_ref_emission_bar",
                                 constants.P_REF_EMISSION_BAR))
    if not (trt.art_ptop_bar <= p_ref_em <= trt.art_pbtm_bar):
        # jnp.interp CLAMPS at the grid ends, so without this an out-of-grid
        # emission anchor silently returns the grid-edge gravity and radius.
        # Same refusal the transmission p_ref_bar gets; standing loud-errors
        # rule, and art_pbtm_bar is profile-overridable so this is reachable.
        raise ValueError(
            f"p_ref_emission_bar={p_ref_em:g} lies outside the RT grid "
            f"[{trt.art_ptop_bar:g}, {trt.art_pbtm_bar:g}] bar. It is the "
            "level the emission column's radius and gravity are anchored at, "
            "so the grid must actually cover it.")
    # The transit anchor comes from trt, where it was validated against the
    # grid -- both observables must share it. A conflicting per-profile value
    # would silently anchor emission on a different planet, so it raises.
    p_ref_used = float(trt.p_ref_bar)
    if "p_ref_bar" in profile and float(profile["p_ref_bar"]) != p_ref_used:
        raise ValueError(
            f"p_ref_bar={float(profile['p_ref_bar']):g} disagrees with the "
            f"transmission model's validated {p_ref_used:g}; the two "
            "observables share one column anchor (build both from one "
            "profile).")

    # pressure bounds follow the transmission model's (possibly profile-
    # overridden) grid -- the two share opacities and must share the column
    #
    # rtsolver "ibased_linsap", not "ibased". Two measured reasons.
    # (1) INTERIOR SOURCE. "ibased" drops the bottom-boundary term entirely
    # (rtransfer.py: its docstring says "with no surface"), so every photon
    # entering the grid from below is lost. At the tau_bottom = 3 the consumer
    # gate admits, that term is 12-52% of the flux. linsap carries it: its last
    # source row is multiplied by exp(-tau_total/mu) over the same streams, so
    # a transparent column returns the interior blackbody and an opaque one
    # returns 1.9e-86 of it. (2) ACCURACY. linsap is the linear-source
    # (Olson & Kunasz) scheme rather than isothermal-layer; at nlayer = 60 and
    # tau_bottom = 10 it sits within 0.13-0.40% of its own converged answer
    # where "ibased" sits 2.9-5.3% away. It costs one extra source row.
    # nlayer follows trt's grid too: a differing value would put the emission
    # column on different layer centres than trt.p_art_bar with no error.
    art_nlayer = int(np.asarray(trt.p_art_bar).size)
    if "art_nlayer" in profile and int(profile["art_nlayer"]) != art_nlayer:
        raise ValueError(
            f"art_nlayer={int(profile['art_nlayer'])} disagrees with the "
            f"transmission grid's {art_nlayer} layers; the two observables "
            "share opacities and must share the column.")
    art = ArtEmisPure(nu_grid=nu_grid, pressure_top=trt.art_ptop_bar,
                      pressure_btm=trt.art_pbtm_bar, nlayer=art_nlayer,
                      rtsolver="ibased_linsap", nstream=8)
    lnp_em = jnp.asarray(np.log(np.asarray(art.pressure)))
    print(f"[rt] ArtEmisPure {art_nlayer} layers (shares opacities)", flush=True)

    def _dtau(vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud):
        return _accumulate_dtau_ckd(
            art, ckd_pack, mols, molmass, opacia, opacia_he,
            vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud=cloud,
            rayleigh_xs=None)

    def emission_flux(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None):
        """Emergent thermal flux (n_nu,) from ART-grid VMR/T/mmw profiles.

        vmr_he is REQUIRED (H2-He CIA -- same continuum physics as transmission;
        the None default only upgrades the omission error message)."""
        _require_he(vmr_he)
        _, g_em = _emission_anchor(T_art, mmw_art)
        # linsap wants the source at the layer BOUNDARIES (nlayer + 1), not at
        # the representative centres. Passing the centres raises a broadcasting
        # error rather than modelling the wrong column, so a half-done switch
        # cannot ship quietly.
        dtau_g = _dtau(vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud)
        return _run_emis_ckd_linsap(art, dtau_g, _boundary_temperature(T_art),
                                    nu_grid, ckd_pack.gw)

    def _emission_anchor(T_art, mmw_art):
        """(radius, gravity) at p_ref_emission_bar -- ONE consistent pair."""
        return _radius_at(lnp_em, T_art, mmw_art, r_ref_em, g_ref_em,
                          p_ref_used, p_ref_em)

    # lnR0 CONVENTION for consumers. The eclipse prefactor multiplies this
    # radius by exp(2*lnR0), matching transmission. That is an approximation
    # here: this geometry shifts radii ADDITIVELY at fixed g_btm, which makes
    # the consistent exponent 2*Rp_btm/r_em (1.888 on WASP-39 b; finite
    # differences give 1.8814), and it also drops the g_em response to lnR0.
    # Measured, both are below the eclipse-depth error budget, so the exponent
    # is left at 2. If it is ever tightened, tighten the gravity response with
    # it -- correcting one alone moves the answer the wrong way.
    def emission_radius(T_art, mmw_art):
        """Planet radius (cm) at p_ref_emission_bar, for the eclipse-depth
        (R_p/R_star)^2 prefactor.

        The catalogue radius is the TRANSIT radius at p_ref_bar (~1 mbar); the
        emitting surface sits ~2 decades deeper, so the two differ by several
        percent and pairing the transit radius with the emission gravity
        misstates GM. Differentiable, and traced through T/mmw exactly like the
        gravity the same column uses.
        """
        return _emission_anchor(T_art, mmw_art)[0]

    def tau_bottom(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud=None):
        """Total vertical optical depth at the BOTTOM of the RT column, per band
        (n_nu,), reduced over the g-ordinates by the MINIMUM.

        The linsap solver DOES carry an interior source term, so a transparent
        column returns the deep boundary blackbody rather than nothing. That
        makes this a statement about assumption sensitivity, not about lost
        flux: where tau_bottom is small the emergent flux is set by an
        extrapolated boundary temperature standing in for everything below the
        grid, so a caller should check min(tau_bottom) and flag windows that
        see through it. Not on the hot AD path (diagnostic).

        The minimum over g is the conservative reduction: a caller takes min
        over wavenumber, and within a band the smallest g is the least opaque
        wavenumber, so the gate stays conservative in the same sense.
        """
        _require_he(vmr_he)
        _, g_em = _emission_anchor(T_art, mmw_art)
        dtau_g = _dtau(vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud)
        return jnp.min(jnp.sum(dtau_g, axis=0), axis=0)

    def _flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud, wo_mols,
                  photosphere):
        _require_he(vmr_he)
        r_em, g_em = _emission_anchor(T_art, mmw_art)

        def _finish(dtau_g):
            w = None
            if photosphere:
                r_phot, _ = _radius_at(lnp_em, T_art, mmw_art, r_ref_em, g_ref_em,
                                       p_ref_used,
                                       jnp.exp(_photosphere_lnp(dtau_g, lnp_em)))
                w = (r_phot / r_em) ** 2
            return (_run_emis_ckd_linsap(art, dtau_g,
                                         _boundary_temperature(T_art),
                                         nu_grid, ckd_pack.gw, weight_g=w),
                    jnp.min(jnp.sum(dtau_g, axis=0), axis=0))

        if wo_mols is None:
            return _finish(_dtau(vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud))
        (flux, tau), rows = _ckd_dtau_batch(
            art, ckd_pack, mols, molmass, opacia, opacia_he,
            vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em,
            list(wo_mols), _finish, cloud=cloud, rayleigh_xs=None)
        nb = tau.shape[0]
        flux_wo = (jnp.stack([f for f, _ in rows]) if rows
                   else jnp.zeros((0, nb)))
        tau_wo = (jnp.stack([t for _, t in rows]) if rows
                  else jnp.zeros((0, nb)))
        return flux, tau, flux_wo, tau_wo

    def emission_flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None,
                          wo_mols=None):
        """Emergent flux AND bottom optical depth from ONE optical-depth build.

        Returns ``(flux, tau_bottom)`` -- bitwise what the separate
        ``emission_flux`` / ``tau_bottom`` calls return, without building the
        same optical depth twice. With ``wo_mols`` (a list of molecule names)
        returns ``(flux, tau_bottom, flux_wo, tau_wo)``, the wo rows aligned to
        ``wo_mols``: each is the observable with that molecule's VMR zeroed,
        bit-identical to a from-scratch call on the zeroed profile but reusing
        the shared correlated-k fold prefix (see ckd._fold_wo).
        """
        return _flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud, wo_mols,
                         photosphere=False)

    def eclipse_flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None,
                         wo_mols=None):
        """``emission_flux_tau`` with the tau = 2/3 photospheric radius folded
        in: the flux is ``sum_g w_g F_g (R_phot,g / R_em)^2`` per band, so
        eclipse depth = flux / F_star x (R_em / R_star)^2 with ``R_em`` from
        ``emission_radius``. This is the quantity an eclipse depth is built
        from; ``emission_flux_tau`` is the plain emergent flux."""
        return _flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud, wo_mols,
                         photosphere=True)

    return SimpleNamespace(
        emission_flux=emission_flux,
        emission_flux_tau=emission_flux_tau,
        eclipse_flux_tau=eclipse_flux_tau,
        emission_radius=emission_radius,
        tau_bottom=tau_bottom,
        nu_grid=np.asarray(nu_grid),
        wl_um=trt.wl_um,
        p_art_bar=np.asarray(art.pressure),
        art_pbtm_bar=float(trt.art_pbtm_bar),
        art_ptop_bar=float(trt.art_ptop_bar),
        art_nlayer=art_nlayer,
        # echoed so a consumer can verify the engine honored the key
        p_ref_emission_bar=p_ref_em,
        p_ref_bar=p_ref_used,
        molecules=mols,
    )
