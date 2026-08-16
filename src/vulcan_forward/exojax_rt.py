"""ExoJax side of the demo: a differentiable ``ArtTransPure`` transmission model.

``build_rt_model(profile)`` builds (once) the wavenumber grid, one premodit opacity
per molecule, the H2-H2 collision-induced-absorption table, and an ``ArtTransPure``
radiative-transfer object. It returns a model whose ``transmission_depth(vmr, vmr_h2,
T_art, mmw_art)`` maps per-layer VMR + temperature + mean-molecular-weight profiles
(already interpolated onto the ART grid) to the transit depth ``(R_p(lambda)/R_star)^2``.

Everything inside ``transmission_depth`` is pure JAX, so forward-mode tangents from the
chemistry pass straight through to the spectrum. Opacities are built in float64 (x64 is
globally enabled), matching the chemistry side -- no dtype break.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # OpaPremodit refuses 32-bit; safe if already set
import jax.numpy as jnp

from vulcan_forward import constants, paths

from exojax.utils.grids import wavenumber_grid
from exojax.database.api import MdbExomol, MdbHitran
from exojax.opacity import OpaPremodit, OpaCIA
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


def _blend_h2he_broadening(mdb, key: str) -> None:
    """Overwrite ``mdb.gamma_air``/``n_air`` with an H2/He-weighted Lorentz width.

    HITRAN's default gamma_air/n_air describe TERRESTRIAL air, which is the wrong
    perturber for an H2/He-dominated envelope. With ``nonair_broadening=True`` exojax
    exposes the HITRAN planetary-broadener columns (gamma_h2/n_h2, gamma_he/n_he)
    where the database provides them; this blends them with the fixed number-fraction
    mix ``constants.H2HE_BROADENING_MIX`` and writes the result into the gamma_air/n_air
    slots that OpaPremodit (and MDBSnapshot) consume, so the rest of the opacity
    pipeline is untouched. Lines lacking a valid H2/He entry fall back to gamma_air
    FOR THAT PARTNER, and the per-molecule coverage is printed loudly; a molecule
    with NO coverage at all raises (use broadening="air" for it, knowingly).
    The temperature exponent is the gamma-weighted mean (exact at T_ref, first-order
    in the mix elsewhere).
    """
    g_air = np.asarray(mdb.gamma_air, dtype=np.float64)
    n_air = np.asarray(mdb.n_air, dtype=np.float64)
    f_h2, f_he = constants.H2HE_BROADENING_MIX
    parts = []
    for partner, frac in (("h2", f_h2), ("he", f_he)):
        g = getattr(mdb, f"gamma_{partner}", None)
        n = getattr(mdb, f"n_{partner}", None)
        if g is None or n is None:
            parts.append((frac, None, None, 0.0))
            continue
        g = np.asarray(g, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        ok = np.isfinite(g) & (g > 0.0)
        n_ok = np.where(np.isfinite(n), n, n_air)
        parts.append((frac, np.where(ok, g, g_air), np.where(ok, n_ok, n_air),
                      float(ok.mean()) if g.size else 0.0))
    if all(p[1] is None for p in parts):
        raise RuntimeError(
            f"broadening='h2he' requested but HITRAN supplies no H2/He broadening "
            f"columns for {key} in this band (or the cache predates "
            "nonair_broadening=True -- h2he mode uses a separate 'h2he/<db>' cache "
            "dir precisely to force a fresh extended download). Either drop the "
            f"molecule, or run it with broadening='air' knowingly.")
    gamma_mix = np.zeros_like(g_air)
    gn_mix = np.zeros_like(g_air)
    for frac, g, n, _cov in parts:
        g_eff = g_air if g is None else g
        n_eff = n_air if n is None else n
        gamma_mix = gamma_mix + frac * g_eff
        gn_mix = gn_mix + frac * g_eff * n_eff
    n_mix = gn_mix / np.where(gamma_mix > 0.0, gamma_mix, 1.0)
    cov = {p: parts[i][3] for i, p in enumerate(("H2", "He"))}
    print(f"[rt]   {key}: H2/He broadening blend f=({f_h2:.2f},{f_he:.2f}); line "
          f"coverage H2 {cov['H2']:.1%}, He {cov['He']:.1%} (uncovered lines keep "
          f"gamma_air for that partner); median gamma_mix/gamma_air = "
          f"{np.median(gamma_mix / np.where(g_air > 0, g_air, np.nan)):.3f}", flush=True)
    mdb.gamma_air = gamma_mix
    mdb.n_air = n_mix


def _build_opa(key: str, spec: dict, nu_grid, broadening: str = "air",
               dit_grid_resolution: float = 1.0):
    """Build one premodit opacity for ``key`` (CO cached; others downloaded).

    ``broadening``: "air" (HITRAN default, terrestrial perturber -- documented
    approximation) or "h2he" (HITRAN planetary H2/He widths where available, blended
    per ``constants.H2HE_BROADENING_MIX``; separate ``h2he/<db>`` download cache).
    ExoMol sources already carry their own default broadening and ignore the knob.
    ``dit_grid_resolution``: the PreMODIT broadening-parameter grid spacing;
    1.0 is this pipeline's long-standing value, smaller resolves the
    pressure-broadening grid finer (exojax's own default is 0.2) at a slower
    opacity build. Profile-overridable via ``profile["dit_grid_resolution"]``.
    """
    src = spec["source"]
    if src in ("exomol", "exomol_cached"):
        path = paths.resolve_db(spec["db"], src)
        mdb = MdbExomol(path, nurange=nu_grid)
    elif src == "hitran":
        # isotope=1 (main isotopologue): isotope=0 pulls minor isotopologues whose
        # TIPS partition functions are missing from hapi (e.g. SO2 (9,3) -> KeyError).
        # HITRAN line intensities include the terrestrial isotopic abundance factor,
        # so applying the main-isotopologue opacity to the TOTAL molecular VMR is the
        # standard (slightly conservative) approximation documented in constants.py.
        if broadening == "h2he":
            # separate h2he/<db> cache dir: forces a fresh nonair_broadening
            # download AND keeps the path stem a valid HITRAN molecule token
            # (a "<db>_h2he" suffix breaks exojax>=2.x molecule parsing)
            mdb = MdbHitran(str(paths.linelist_dir() / "h2he" / spec["db"]),
                            nurange=nu_grid, isotope=1, nonair_broadening=True)
            _blend_h2he_broadening(mdb, key)
        elif broadening == "air":
            mdb = MdbHitran(paths.resolve_db(spec["db"], src), nurange=nu_grid,
                            isotope=1)
        else:
            raise ValueError(f"unknown broadening mode {broadening!r} "
                             "(expected 'air' or 'h2he')")
    else:
        raise ValueError(f"unknown opacity source {src!r} for {key}")
    # exojax >= 2.x deprecated the dit_grid_resolution= constructor argument:
    # pass the SAME value through broadening_resolution. The public profile
    # key stays "dit_grid_resolution" -- identical semantics.
    _broad_res = {"mode": "manual", "value": float(dit_grid_resolution)}
    try:
        opa = OpaPremodit.from_snapshot(
            mdb.to_snapshot(), nu_grid,
            auto_trange=(constants.T_OPA_MIN_K, constants.T_OPA_MAX_K),
            broadening_resolution=_broad_res)
    except AttributeError:
        opa = OpaPremodit(
            mdb, nu_grid,
            auto_trange=(constants.T_OPA_MIN_K, constants.T_OPA_MAX_K),
            broadening_resolution=_broad_res)
    n_lines = int(np.asarray(getattr(mdb, "nu_lines", np.zeros(0)).shape[0])) if hasattr(mdb, "nu_lines") else -1
    return opa, n_lines


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

    The integration nodes extend HALF A LAYER below the deepest grid centre,
    holding the bottom layer's T and mmw over that half layer (which is what a
    representative layer value means). Without the extra node ``jnp.interp``
    clamps at ``lnp_art[-1]`` and silently drops the half layer between the
    deepest centre and the grid's lower boundary -- the level exojax actually
    defines ``radius_btm`` at.
    """
    C = constants.K_B_CGS * T_art / (
        mmw_art * constants.M_U_CGS * g_ref * r_ref ** 2)
    lnp = jnp.concatenate(
        [lnp_art, _lnp_grid_bottom_boundary(lnp_art)[None]])
    C = jnp.concatenate([C, C[-1:]])
    cum = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(0.5 * (C[1:] + C[:-1]) * jnp.diff(lnp))])
    i_ref = jnp.interp(jnp.log(p_ref_bar), lnp, cum)
    i_tgt = jnp.interp(jnp.log(p_target_bar), lnp, cum)
    r = 1.0 / (1.0 / r_ref + (i_tgt - i_ref))
    return r, g_ref * (r_ref / r) ** 2


def _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_btm,
                     vmr, vmr_h2, T_art, mmw_art,
                     opacia_he=None, vmr_he=None, cloud=None, rayleigh_xs=None,
                     mie_pack=None, mie=None):
    """Per-layer optical-depth matrix ``dtau`` (nlayer, n_nu) on the ART pressure grid.

    The sum of each molecule's line opacity plus the H2-H2 CIA continuum -- and,
    optionally, H2-He CIA (``opacia_he`` + ``vmr_he``), the ExoJax power-law
    retrieval cloud (``cloud``), and H2/He Rayleigh scattering (``rayleigh_xs``).
    Shared by the transmission and emission models: line + CIA + cloud terms are
    identical; Rayleigh is transmission-only BY DESIGN (it is scattering, not
    absorption -- adding it to the pure-absorption ibased emission solver would
    fake thermal extinction; it is also negligible at the >1 um thermal bands).

    Parameters
    ----------
    art : ArtTransPure | ArtEmisPure   provides the pressure grid + opacity_profile_* ops
    vmr : dict molecule -> (nlayer,) volume mixing ratio
    vmr_h2 : (nlayer,) H2 volume mixing ratio (both H2-H2 CIA collision partners)
    opacia_he, vmr_he : optional H2-He CIA table + He VMR profile (term skipped if
        either is None -- keeps the parent demo callers byte-compatible)
    cloud : optional (2,) array [log10 kappac0 (cm^2/g at constants.CLOUD_NUC0), alphac]
        for ``exojax.atm.simple_clouds.powerlaw_clouds`` (alphac=0 -> gray cloud;
        per-gram-of-atmosphere opacity, uniformly mixed: dtau = kappa(nu)*dP_cgs/g)

    Pure JAX throughout, so forward-mode tangents from the chemistry pass straight through.

    Each molecule's line-opacity term (and each CIA term) is wrapped in
    ``jax.checkpoint``: reverse-mode differentiation otherwise stores every
    molecule's PreMODIT intermediates until the backward pass (~30-50 GB per
    spectrum on the gpu-preset grid; OOM'd a GH200 at a 6-wide particle
    chunk). With checkpoint the backward recomputes one molecule at a time.
    Exact same values; forward eval and forward-mode jvp are unaffected.
    """
    dtau = jnp.zeros((art.pressure.shape[0], nu_grid.shape[0]))
    for key in mols:
        def _line_term(T_art_, vmr_key_, mmw_art_, _key=key):
            xs = opas[_key].xsmatrix(T_art_, art.pressure)         # (nlayer, n_nu)
            mmr = vmr_to_mmr(vmr_key_, molmass[_key], mmw_art_)
            return art.opacity_profile_xs(xs, mmr, molmass[_key], g_btm)
        dtau = dtau + jax.checkpoint(_line_term)(T_art, vmr[key], mmw_art)
    # opacity_profile_cia divides a (nlayer, n_nu) matrix by mmw, so mmw must broadcast
    # as (nlayer, 1) here -- note art.run separately wants the 1-D (nlayer,) form.

    def _cia_term(opacia_, T_art_, vmr_a, vmr_b, mmw_art_):
        logacia_ = opacia_.logacia_matrix(T_art_)
        return art.opacity_profile_cia(
            logacia_, T_art_, vmr_a, vmr_b, mmw_art_[:, None], g_btm)

    dtau = dtau + jax.checkpoint(
        lambda t, va, vb, m: _cia_term(opacia, t, va, vb, m))(
        T_art, vmr_h2, vmr_h2, mmw_art)
    if opacia_he is not None and vmr_he is not None:
        dtau = dtau + jax.checkpoint(
            lambda t, va, vb, m: _cia_term(opacia_he, t, va, vb, m))(
            T_art, vmr_h2, vmr_he, mmw_art)
    if cloud is not None:
        # ExoJax's shipped retrieval cloud (pRT convention, per gram of atmosphere).
        kappa_c = powerlaw_clouds(nu_grid, kappac0=10.0 ** cloud[0],
                                  nuc0=constants.CLOUD_NUC0, alphac=cloud[1])  # (n_nu,)
        dP = jnp.asarray(art.dParr)                                # (nlayer,) bar
        dtau = dtau + kappa_c[None, :] * (dP[:, None] * _BAR_CGS / g_btm)
    if mie is not None:
        # Mie cloud deck (exojax PdbCloud/OpaMie): column-uniform condensate
        # MMR, one lognormal size distribution; mie = [log10 rg (cm), sigmag,
        # log10 MMR]. sigma_extinction is the correct chord attenuation.
        # Differentiable: the miegrid interp is piecewise-linear, edge-clamped
        # (callers keep parameters strictly inside the grid).
        if mie_pack is None:
            raise ValueError(
                "mie parameters passed but the RT was built without "
                "mie_condensate: rebuild with profile['mie_condensate'] set "
                "(never silently ignore a requested cloud).")
        opa_mie_, rho_c = mie_pack
        rg = 10.0 ** mie[0]
        sigmag = mie[1]
        mmr = 10.0 ** mie[2]
        sig_ext, _sig_sca, _g_asym = opa_mie_.mieparams_vector(rg, sigmag)
        mmr_prof = jnp.full((art.pressure.shape[0],), 1.0) * mmr
        dtau = dtau + art.opacity_profile_cloud_lognormal(
            sig_ext, rho_c, mmr_prof, rg, sigmag, g_btm)
    if rayleigh_xs is not None:
        # H2 (+He) Rayleigh scattering -- zero-free-parameter known physics that
        # matters short of ~1.5 um; omitting it would bias the retrieved haze slope.
        # rayleigh_xs = (xs_h2, xs_he), each (n_nu,) from exojax xsvector_rayleigh_gas.
        xs_h2, xs_he = rayleigh_xs
        nlayer = art.pressure.shape[0]
        mmr_h2 = vmr_to_mmr(vmr_h2, _H2_MOLMASS, mmw_art)
        dtau = dtau + art.opacity_profile_xs(
            jnp.broadcast_to(xs_h2[None, :], (nlayer, xs_h2.shape[0])),
            mmr_h2, _H2_MOLMASS, g_btm)
        if vmr_he is not None:
            mmr_he = vmr_to_mmr(vmr_he, _HE_MOLMASS, mmw_art)
            dtau = dtau + art.opacity_profile_xs(
                jnp.broadcast_to(xs_he[None, :], (nlayer, xs_he.shape[0])),
                mmr_he, _HE_MOLMASS, g_btm)
    return dtau


def _accumulate_dtau_ckd(art, pack, mols, molmass, opacia, opacia_he,
                         vmr, vmr_h2, vmr_he, T_art, mmw_art, g_btm,
                         cloud=None, rayleigh_xs=None):
    """Per-layer, per-g-ordinate, per-band optical depth ``(nlayer, ng, nband)``.

    The line opacity of each molecule comes from its k-table, interpolated to
    the layer (T, P) and combined across molecules by random-overlap
    resort-rebin (see ``vulcan_forward.ckd``). The continua -- CIA, Rayleigh,
    the power-law cloud -- are smooth across a band, so their band value adds
    identically to every g-ordinate; that is exact, not an approximation, to
    the accuracy of a smooth function over one R=500 band.

    Mie is deliberately unsupported here: its extinction has structure across a
    band and folding it into a k-distribution built from line opacity alone
    would be wrong. The caller refuses instead of silently approximating.
    """
    from vulcan_forward import ckd as _ckd

    P = jnp.asarray(art.pressure)
    tot = None
    for key in mols:
        dt = _ckd_dt_one(art, pack, key, molmass, vmr[key], T_art, mmw_art,
                         g_btm, P)
        tot = dt if tot is None else _ckd.overlap(tot, dt, pack.gg, pack.gw)
    cont = _ckd_continuum(art, pack, opacia, opacia_he, vmr_h2, vmr_he,
                          T_art, mmw_art, g_btm, cloud, rayleigh_xs)
    return tot + cont[:, None, :]


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
    """Band continuum (nlayer, nband): CIA + Rayleigh + power-law cloud."""
    def _cia(opa_, va, vb):
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
    aligned to ``wo_mols``. Each wo fold keeps today's semantics exactly — the
    dropped molecule is still folded, as the zero tensor its zeroed VMR
    produces through the SAME op pipeline — so every output is bit-identical
    to a from-scratch solve with that VMR zeroed; only the shared fold prefix
    is reused (see ckd._fold_wo). The per-molecule tensors are held for the
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


def _run_emis_ckd_linsap(art, dtau_g, T_boundary, nu_bands, gw):
    """Emergent flux (nband,) from a ``(nlayer, ng, nband)`` CKD optical depth,
    solved with the LINEAR-SOURCE scheme.

    exojax's ``ArtEmisPure.run_ckd`` hard-codes ``rtrun_emis_pureabs_ibased``,
    which has no bottom-boundary term, so every photon entering the grid from
    below is lost (measured flux deficits: README, "Emission uses it too").
    This is upstream's own flatten-solve-reweight structure with
    ``ibased_linsap`` in its place, so CKD emission keeps the interior source
    the line-by-line path has carried since 0.5.0.

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
    return jnp.einsum("g,gb->b", gw, flux.reshape((ng, nband)))


def _require_geometry(profile: dict, *keys: str) -> None:
    """Refuse a profile missing planet geometry.

    Before extraction these keys fell back to WASP-39 b constants, so a
    caller who forgot ``rp_cm`` silently modeled a different planet. They
    set the transit-depth normalization and the hydrostatic scale, so there
    is no safe default (standing fail-loud rule).
    """
    missing = [k for k in keys if k not in profile]
    if missing:
        raise ValueError(
            f"profile is missing required planet geometry {missing}: rp_cm "
            "(planet radius in cm at art_pbtm_bar), gs_cgs (surface gravity "
            "in cm/s^2), rstar_cm (stellar radius in cm).")


def build_rt_model(profile: dict) -> SimpleNamespace:
    """Build the transmission-spectrum model for the molecules named in ``profile``.

    Returns a SimpleNamespace with:
        transmission_depth(vmr, vmr_h2, T_art, mmw_art) -> (n_nu,) transit depth
        nu_grid : (n_nu,) wavenumber grid (cm^-1)
        wl_um   : (n_nu,) wavelength grid (micron), descending->ascending sorted handled by caller
        p_art_bar : (nlayer,) ART pressure grid (bar)
        molecules : list[str]
    """
    t0 = time.time()
    mode = str(profile.get("opacity_mode", "lbl"))
    if mode not in ("lbl", "exomolop"):
        raise ValueError(
            f"opacity_mode={mode!r}: choose 'exomolop' (correlated-k from the "
            "published ExoMolOP high-temperature tables -- the accurate "
            "default) or 'lbl' (direct line-by-line sampling on the output "
            "grid, which is only correct at R >= 700,000; kept for the Mie "
            "deck -- see vulcan_forward.ckd). The interim 'ckd' mode "
            "(correlated-k built here from HITRAN) was removed in 0.8.0: its "
            "line data was measurably wrong for a hot hydrogen atmosphere.")
    # Asked once here rather than by string comparison at each use: a missed
    # comparison would silently take the line-by-line branch with a band
    # grid, which is not a mode this engine supports.
    is_ckd = mode == "exomolop"
    mols = list(profile["molecules"])
    ckd_pack = None
    if is_ckd:
        # Published ExoMol/HITEMP opacities with H2/He broadening already
        # applied. See vulcan_forward.exomolop for the three measured defects
        # this closes over building tables from HITRAN; the band grid and the
        # split quadrature come from the files.
        from vulcan_forward import exomolop as _exo
        ckd_pack = _exo.load_tables(
            mols, float(profile["nu_min"]), float(profile["nu_max"]),
            molecule_table=profile.get("molecule_table"))
        nu_grid = jnp.asarray(ckd_pack.nu_bands)
        ckd_pack.nu_bands_j = nu_grid
        # The band R actually in force, DERIVED from the file's edges rather
        # than hard-coded: a future non-R1000 table variant must not mis-echo.
        resolution = float(
            1.0 / np.median(np.diff(np.log(ckd_pack.band_edges))))
    else:
        nu_grid, wav, resolution = wavenumber_grid(
            profile["nu_min"], profile["nu_max"], profile["nu_pts"],
            unit="cm-1", xsmode="premodit")
        if resolution < 7.0e5:
            # exojax's own critical resolution (utils.grids.warn_resolution).
            # It emits a UserWarning that is easily lost in the build log; say
            # it loudly, because below it the sampled spectrum is biased
            # (measurements: README, "Opacity: correlated-k").
            print(f"[rt] WARNING: line-by-line grid R~{resolution:.0f} is below "
                  "exojax's critical resolution 700000. The sampled cross "
                  "section is biased (too opaque, and more so in strong-line "
                  "regions, which inflates spectral contrast). Use "
                  "opacity_mode='exomolop'.", flush=True)
        print(f"[rt] nu_grid {nu_grid.shape[0]} pts, R~{resolution:.0f}, "
              f"lambda[{1e4/profile['nu_max']:.2f},{1e4/profile['nu_min']:.2f}] um",
              flush=True)
    broadening = str(profile.get("broadening", constants.BROADENING))
    if mode == "exomolop":
        # The knob does not apply: ExoMolOP's tables were integrated with H2/He
        # broadening already baked in. Printing "air" here would be a false
        # statement about the model that just got built.
        broadening = "h2he (from the ExoMolOP tables)"
        print("[rt] pressure broadening: H2/He, as published in the ExoMolOP "
              "tables; the 'broadening' profile key does not apply in this "
              "mode", flush=True)
    else:
        print(f"[rt] pressure broadening: {broadening}"
              + (" (terrestrial-air widths -- documented approximation, and "
                 "WRONG for a hydrogen atmosphere; measurements in the README, "
                 "'Opacity data: ExoMolOP'. Use opacity_mode='exomolop'.)"
                 if broadening == "air" else ""), flush=True)
    # Profile-overridable RT knobs (defaults = the historical hard-coded
    # values, so every existing consumer is byte-unchanged). Validated
    # loudly here -- an out-of-range value must never build a wrong model.
    dit_res = float(profile.get("dit_grid_resolution", 1.0))
    if not 0.05 <= dit_res <= 2.0:
        raise ValueError(
            f"dit_grid_resolution={dit_res:g} outside [0.05, 2.0] (PreMODIT "
            "broadening-grid spacing; 1.0 = this pipeline's default, 0.2 = "
            "exojax's own default)")
    ptop = float(profile.get("art_ptop_bar", constants.ART_PTOP_BAR))
    pbtm = float(profile.get("art_pbtm_bar", constants.ART_PBTM_BAR))
    if not (0.0 < ptop < pbtm):
        raise ValueError(
            f"art_ptop_bar={ptop:g} / art_pbtm_bar={pbtm:g}: need "
            "0 < top < bottom (bar)")
    integration = str(profile.get("rt_integration", "simpson"))
    if integration not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={integration!r}: exojax ArtTransPure supports "
            "'simpson' (default) or 'trapezoid'")
    # The opacity table is INJECTABLE: a consumer adding a molecule passes its
    # own table rather than editing a constant inside this package (which is
    # what downstream tools had to do before extraction).
    mol_table = profile.get("molecule_table") or constants.MOLECULES
    _unknown = [k for k in mols if k not in mol_table]
    if _unknown:
        raise KeyError(
            f"no opacity spec for {_unknown} in the molecule table "
            f"(have: {sorted(mol_table)}). Pass profile['molecule_table'] "
            "with one entry per molecule: vulcan, molmass, source, db.")
    opas, molmass = {}, {}
    for key in mols:
        spec = mol_table[key]
        molmass[key] = float(spec["molmass"])
        if is_ckd:
            # the line opacity already lives in the k-table; building a
            # premodit object here would cost minutes and be unused
            continue
        tb = time.time()
        opa, n_lines = _build_opa(key, spec, nu_grid, broadening=broadening,
                                  dit_grid_resolution=dit_res)
        opas[key] = opa
        print(f"[rt]   {key}: {n_lines} lines, opa built in {time.time()-tb:.1f}s", flush=True)

    art = ArtTransPure(
        pressure_top=ptop,
        pressure_btm=pbtm,
        nlayer=int(profile["art_nlayer"]),
        integration=integration)
    art.change_temperature_range(constants.T_OPA_MIN_K, constants.T_OPA_MAX_K)
    p_art_bar = np.asarray(art.pressure)
    # ascending (exojax orders top-to-bottom); _anchor_to_grid_bottom relies on it
    if not np.all(np.diff(p_art_bar) > 0):
        raise RuntimeError(
            "the ART pressure grid is not ascending; _anchor_to_grid_bottom's "
            "cumulative integral and jnp.interp both assume it is")
    lnp_art = jnp.asarray(np.log(p_art_bar))
    print(f"[rt] ArtTransPure {profile['art_nlayer']} layers, "
          f"P=[{p_art_bar.min():.1e},{p_art_bar.max():.1e}] bar, "
          f"chord integration {integration}, dit_grid_resolution {dit_res:g}",
          flush=True)

    cia_h2h2 = paths.cia_h2h2_file()
    if not cia_h2h2.exists():
        # exojax auto-fetches it (~24 MB from hitran.org), but its downloader
        # swallows failures -- say up front what is about to happen so an
        # offline failure is attributable (fail-loud rule).
        print(f"[rt] H2-H2 CIA absent at {cia_h2h2}; exojax will "
              "download ~24 MB from https://hitran.org/data/CIA/main/"
              "H2-H2_2011.cia now (network required)", flush=True)
    cdb = CdbCIA(str(cia_h2h2), nurange=nu_grid)
    opacia = OpaCIA(cdb, nu_grid=nu_grid)
    # H2-He CIA is required physics (He is ~14% by number). Missing it
    # would silently drop a real continuum term -> a wrong spectrum with no error.
    # Fail loud instead; the file ships in data/opacity_cache/ and the PBS preflight
    # also checks for it.
    cia_h2he = paths.cia_h2he_file()
    if not cia_h2he.exists():
        raise FileNotFoundError(
            f"H2-He CIA table missing ({cia_h2he}). It ships in the bundle's "
            "data/opacity_cache/; if absent download "
            "https://hitran.org/data/CIA/main/H2-He_2011.cia (~147 MB; note the "
            "/main/ path -- the bare /data/CIA/ URL 404s) to that exact path. "
            "Refusing to build the RT without it (silently skipping the He "
            "continuum would bias the spectrum).")
    opacia_he = OpaCIA(CdbCIA(str(cia_h2he), nurange=nu_grid), nu_grid=nu_grid)
    print("[rt] H2-He CIA loaded", flush=True)
    print(f"[rt] CIA + RT built; total {time.time()-t0:.1f}s", flush=True)

    # H2/He Rayleigh cross sections (nu-only, precomputed once; opt-in via profile
    # so the parent demo's published outputs are untouched)
    if profile.get("use_rayleigh", False):
        rayleigh_xs = (
            xsvector_rayleigh_gas(nu_grid, _POLARIZABILITY["H2"]),
            xsvector_rayleigh_gas(nu_grid, _POLARIZABILITY["He"]),
        )
        print("[rt] H2/He Rayleigh scattering enabled", flush=True)
    else:
        rayleigh_xs = None

    # Mie cloud deck: opt-in via profile["mie_condensate"] + a pinned ABSOLUTE
    # mie_data_dir (exojax's own default is CWD-relative, banned here). The
    # forward model only LOADS a pre-generated miegrid; a missing grid raises
    # with the generation command. The virga archive auto-downloads on first
    # use -- announced up front so an offline failure is attributable.
    mie_pack = None
    mie_cond = profile.get("mie_condensate") or None
    if mie_cond:
        from pathlib import Path as _Path
        from exojax.database.pardb import PdbCloud
        from exojax.opacity import OpaMie
        mie_dir = profile.get("mie_data_dir")
        if not mie_dir:
            raise ValueError(
                "mie_condensate set but mie_data_dir missing: pass the "
                "absolute Mie cache directory (a CWD-relative default would "
                "scatter caches per launch dir).")
        if not (_Path(mie_dir) / "virga.zip").exists():
            print(f"[rt] virga refractive-index archive absent under "
                  f"{mie_dir}; exojax will download ~4 MB from Zenodo now "
                  "(network required)", flush=True)
        pdb = PdbCloud(str(mie_cond), nurange=nu_grid, path=str(mie_dir))
        if not pdb.miegrid_path.exists():
            raise FileNotFoundError(
                f"Mie grid for {mie_cond} not found at {pdb.miegrid_path}. "
                "Generate it once (vulcan-jwst-tool: python "
                f"tools/generate_miegrid.py {mie_cond}, ~1 h/condensate); "
                "the forward model only loads grids, never a silent "
                "fallback.")
        pdb.load_miegrid()
        opa_mie = OpaMie(pdb, nu_grid)
        mie_pack = (opa_mie, float(pdb.condensate_substance_density))
        print(f"[rt] Mie cloud deck: {mie_cond} (rho = "
              f"{pdb.condensate_substance_density:g} g/cm3, miegrid loaded)",
              flush=True)

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

    def _require_he(vmr_he):
        # He is ~14% by number and its CIA is real continuum physics;
        # an accidental None here used to SILENTLY drop the term (the sensitivity
        # demo shipped that way). Fail loud instead -- standing repo rule.
        if vmr_he is None:
            raise ValueError(
                "vmr_he is required: pass the He VMR profile (chem.sidx['He']) so the "
                "H2-He CIA term is included. There is no supported He-less mode.")

    def transmission_depth(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None,
                           mie=None):
        """Transit depth (R_p(lambda)/R_star)^2 from ART-grid profiles.

        vmr : dict molecule -> (nlayer,) VMR; vmr_h2 : (nlayer,) H2 VMR (for CIA);
        vmr_he : (nlayer,) He VMR (H2-He CIA partner; REQUIRED -- the None default
        exists only so an omission raises the explanatory ValueError, not TypeError).
        Optional: cloud=[log10 kappac0, alphac] (ExoJax powerlaw_clouds);
        mie=[log10 rg (cm), sigmag, log10 MMR] (needs mie_condensate at build).
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
        if is_ckd:
            if mie is not None:
                raise ValueError(
                    "Mie cloud is not supported on the correlated-k path: its "
                    "extinction has structure across a band, so folding it "
                    "into a k-distribution built from line opacity would be "
                    "wrong. Use opacity_mode='lbl' for a Mie deck, or the "
                    "power-law cloud, which is smooth across a band.")
            dtau_g = _accumulate_dtau_ckd(
                art, ckd_pack, mols, molmass, opacia, opacia_he,
                vmr, vmr_h2, vmr_he, T_art, mmw_art, g_prof,
                cloud=cloud, rayleigh_xs=rayleigh_xs)
            Rp2 = art.run_ckd(dtau_g, T_art, mmw_art, Rp_btm, g_btm,
                              ckd_pack.gw)
            return Rp2 * (Rp_btm / rstar_cm) ** 2
        dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_prof,
                                vmr, vmr_h2, T_art, mmw_art,
                                opacia_he=opacia_he, vmr_he=vmr_he, cloud=cloud,
                                rayleigh_xs=rayleigh_xs,
                                mie_pack=mie_pack, mie=mie)
        Rp2 = art.run(dtau, T_art, mmw_art, Rp_btm, g_btm)          # (radius/Rp_btm)^2
        return Rp2 * (Rp_btm / rstar_cm) ** 2                       # (radius/R_star)^2

    def transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, lnR0, vmr_he=None,
                             cloud=None, mie=None, wo_mols=None):
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
            # identical expression and op order to the pre-batch return, so the
            # single and wo paths agree bitwise
            return (Rp2 * (Rp_btm / rstar_cm) ** 2                  # (radius/R_star)^2
                    * jnp.exp(2.0 * lnR0))

        if is_ckd:
            if mie is not None:
                raise ValueError(
                    "Mie cloud is not supported on the correlated-k path: its "
                    "extinction has structure across a band, so folding it "
                    "into a k-distribution built from line opacity would be "
                    "wrong. Use opacity_mode='lbl' for a Mie deck.")

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

        def _one(vmr_):
            dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia,
                                    g_prof, vmr_, vmr_h2, T_art, mmw_art,
                                    opacia_he=opacia_he, vmr_he=vmr_he,
                                    cloud=cloud, rayleigh_xs=rayleigh_xs,
                                    mie_pack=mie_pack, mie=mie)
            return _depth_of(art.run(dtau, T_art, mmw_art, Rp_r, g_btm))

        depth = _one(vmr)
        if wo_mols is None:
            return depth
        bad = [m for m in wo_mols if m not in mols]
        if bad or len(set(wo_mols)) != len(wo_mols):
            raise ValueError(
                f"wo_mols {list(wo_mols)!r} must be unique members of the RT "
                f"molecule set {list(mols)!r}")
        rows = [_one({**vmr, m: jnp.zeros_like(vmr[m])}) for m in wo_mols]
        return depth, (jnp.stack(rows) if rows
                       else jnp.zeros((0,) + depth.shape))

    return SimpleNamespace(
        transmission_depth=transmission_depth,
        transmission_depth_r=transmission_depth_r,
        nu_grid=np.asarray(nu_grid),
        wl_um=1e4 / np.asarray(nu_grid),
        p_art_bar=p_art_bar,
        molecules=mols,
        broadening=broadening,
        # echo of the profile-overridable RT knobs, so downstream consumers
        # (vulcan-jwst-tool) can VERIFY the engine honored them -- an older
        # engine that ignores an unknown profile key must fail loudly there,
        # never silently compute a different model than the cache key claims
        art_ptop_bar=ptop,
        art_pbtm_bar=pbtm,
        # the pressure rp_cm/gs_cgs were taken to apply at; consumers verify this
        # echo, because a tool that silently reverted to bottom-of-grid anchoring
        # would inflate every transit depth (see _anchor_to_grid_bottom)
        p_ref_bar=p_ref_bar,
        rt_integration=integration,
        dit_grid_resolution=dit_res,
        # which opacity path actually ran. A consumer that asked for 'ckd' and
        # silently got sampled line-by-line would be modelling a spectrum with
        # ~1500 ppm of grid bias, so the echo is checked, not assumed.
        opacity_mode=mode,
        ckd_ng=(int(ckd_pack.ng) if ckd_pack is not None else 0),
        # ExoMolOP's band grid is fixed by the published tables (R = 1000), so
        # echo the resolution actually in force rather than a profile key the
        # engine would have ignored in that mode.
        ckd_r_band=(float(resolution) if ckd_pack is not None else 0.0),
        has_cia_h2he=opacia_he is not None,
        # Mie deck echo: "" when no Mie cloud was built -- consumers verify it
        # against what they requested (the echo makes a silent ignore loud)
        mie_condensate=str(mie_cond or ""),
        # internals reused by build_emis_model (so opacities aren't rebuilt)
        _nu_grid=nu_grid, _opas=opas, _molmass=molmass, _opacia=opacia,
        _opacia_he=opacia_he, _mie_pack=mie_pack, _ckd_pack=ckd_pack,
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
    _accumulate_dtau -- a pure-absorption solver must not count scattering as
    thermal absorption, and it is negligible in the thermal bands).
    """
    # CKD emission runs through _run_emis_ckd_linsap, NOT ArtEmisPure.run_ckd:
    # upstream's version hard-codes the "ibased" solver, which has no interior
    # source term. Using it here would have silently undone the 0.5.0 linsap
    # fix, so 0.7.0 reimplements the same flatten-solve-reweight with linsap
    # instead. Before that this branch refused outright, which left emission
    # sampling line-by-line at R = 1,477 and carrying the full grid bias.
    ckd_mode = getattr(trt, "opacity_mode", "lbl") == "exomolop"
    ckd_pack = getattr(trt, "_ckd_pack", None)
    if ckd_mode and ckd_pack is None:
        raise ValueError(
            "transmission model reports a correlated-k opacity_mode but "
            "carries no k-table pack; it was not built by this engine's "
            "build_rt_model.")
    nu_grid = trt._nu_grid
    opas, molmass, opacia, mols = trt._opas, trt._molmass, trt._opacia, trt.molecules
    opacia_he = trt._opacia_he
    mie_pack = getattr(trt, "_mie_pack", None)   # shared Mie deck (v16)
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
    # SCOPE, stated because it is a real and quantified limitation: this is
    # still a SINGLE radius. Fortney, Lupu, Morley, Freedman & Hood 2019
    # (ApJL 880, L16) show the correct prefactor is the wavelength-dependent
    # photospheric radius at vertical tau = 2/3, which POSEIDON
    # (use_photosphere_radius, default True) and PLATON II both compute. A
    # single radius biases planet-to-star flux ratios by ~5% typically and
    # 10-25% for low-gravity hot Jupiters, and it MUTES or ENHANCES features
    # rather than offsetting them. Implementing that is the right next step; it
    # is deliberately not done here.
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
    profile = dict(profile)
    profile["p_ref_bar_used"] = float(
        profile.get("p_ref_bar", constants.P_REF_BAR))

    # pressure bounds follow the transmission model's (possibly profile-
    # overridden) grid -- the two share opacities and must share the column
    #
    # rtsolver "ibased_linsap", not "ibased" (v0.5.0). Two measured reasons.
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
    art = ArtEmisPure(nu_grid=nu_grid, pressure_top=trt.art_ptop_bar,
                      pressure_btm=trt.art_pbtm_bar, nlayer=int(profile["art_nlayer"]),
                      rtsolver="ibased_linsap", nstream=8)
    art.change_temperature_range(constants.T_OPA_MIN_K, constants.T_OPA_MAX_K)
    lnp_em = jnp.asarray(np.log(np.asarray(art.pressure)))
    print(f"[rt] ArtEmisPure {profile['art_nlayer']} layers (shares opacities)", flush=True)

    def emission_flux(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None,
                      mie=None):
        """Emergent thermal flux (n_nu,) from ART-grid VMR/T/mmw profiles.

        vmr_he is REQUIRED (H2-He CIA -- same continuum physics as transmission;
        the None default only upgrades the omission error message)."""
        if vmr_he is None:
            raise ValueError(
                "vmr_he is required: pass the He VMR profile so the H2-He CIA term "
                "is included in the emission opacity (parity with transmission).")
        if mie is not None:
            # ArtEmisPure is pure-absorption: Mie sigma_extinction would count
            # scattering as thermal absorption (a conservative cloud would
            # radiate like a blackbody instead of zero). Refuse; the absorbing
            # power-law cloud stays allowed.
            raise ValueError(
                "Mie cloud is not supported in EMISSION: ArtEmisPure is a "
                "pure-absorption solver, so Mie scattering would be counted as "
                "thermal absorption and violate the conservative-scattering "
                "zero-emission limit. Use transmission, or the (absorbing) "
                "power-law cloud, until a scattering-aware emission solver "
                "lands.")
        _, g_em = _emission_anchor(T_art, mmw_art)
        # linsap wants the source at the layer BOUNDARIES (nlayer + 1), not at
        # the representative centres. Passing the centres raises a broadcasting
        # error rather than modelling the wrong column, so a half-done switch
        # cannot ship quietly.
        if ckd_mode:
            dtau_g = _accumulate_dtau_ckd(
                art, ckd_pack, mols, molmass, opacia, opacia_he,
                vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud=cloud,
                rayleigh_xs=None)
            return _run_emis_ckd_linsap(art, dtau_g,
                                        _boundary_temperature(T_art),
                                        nu_grid, ckd_pack.gw)
        dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_em,
                                vmr, vmr_h2, T_art, mmw_art,
                                opacia_he=opacia_he, vmr_he=vmr_he, cloud=cloud,
                                mie_pack=mie_pack, mie=None)
        return art.run(dtau, _boundary_temperature(T_art))

    def _emission_anchor(T_art, mmw_art):
        """(radius, gravity) at p_ref_emission_bar -- ONE consistent pair."""
        return _radius_at(lnp_em, T_art, mmw_art, r_ref_em, g_ref_em,
                          float(profile["p_ref_bar_used"]), p_ref_em)

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

    def tau_bottom(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud=None, mie=None):
        """Total vertical optical depth at the BOTTOM of the RT column,
        per wavenumber (n_nu,).

        The linsap solver DOES carry an interior source term, so a transparent
        column returns the deep boundary blackbody rather than nothing. That
        makes this a statement about assumption sensitivity, not about lost
        flux: where tau_bottom is small the emergent flux is set by an
        extrapolated boundary temperature standing in for everything below the
        grid, so a caller should check min(tau_bottom) and flag windows that
        see through it. Not on the hot AD path (diagnostic).

        In CKD mode the return is per BAND, reduced over the g-ordinates by the
        MINIMUM. That is the analogue of the line-by-line form: a caller takes
        min over wavenumber, and within a band the smallest g is the least
        opaque wavenumber, so the gate stays conservative in the same sense.
        """
        _, g_em = _emission_anchor(T_art, mmw_art)
        if ckd_mode:
            dtau_g = _accumulate_dtau_ckd(
                art, ckd_pack, mols, molmass, opacia, opacia_he,
                vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud=cloud,
                rayleigh_xs=None)
            return jnp.min(jnp.sum(dtau_g, axis=0), axis=0)
        dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_em,
                                vmr, vmr_h2, T_art, mmw_art,
                                opacia_he=opacia_he, vmr_he=vmr_he, cloud=cloud,
                                mie_pack=mie_pack, mie=mie)
        return jnp.sum(dtau, axis=0)

    def emission_flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None,
                          mie=None, wo_mols=None):
        """Emergent flux AND bottom optical depth from ONE optical-depth build.

        Returns ``(flux, tau_bottom)`` -- bitwise what the separate
        ``emission_flux`` / ``tau_bottom`` calls return, without building the
        same optical depth twice. With ``wo_mols`` (a list of molecule names)
        returns ``(flux, tau_bottom, flux_wo, tau_wo)``, the wo rows aligned to
        ``wo_mols``: each is the observable with that molecule's VMR zeroed,
        bit-identical to a from-scratch call on the zeroed profile but reusing
        the shared correlated-k fold prefix (see ckd._fold_wo).
        """
        if vmr_he is None:
            raise ValueError(
                "vmr_he is required: pass the He VMR profile so the H2-He CIA term "
                "is included in the emission opacity (parity with transmission).")
        if mie is not None:
            raise ValueError(
                "Mie cloud is not supported in EMISSION: ArtEmisPure is a "
                "pure-absorption solver, so Mie scattering would be counted as "
                "thermal absorption and violate the conservative-scattering "
                "zero-emission limit. Use transmission, or the (absorbing) "
                "power-law cloud, until a scattering-aware emission solver "
                "lands.")
        _, g_em = _emission_anchor(T_art, mmw_art)
        if ckd_mode:
            def _finish(dtau_g):
                return (_run_emis_ckd_linsap(art, dtau_g,
                                             _boundary_temperature(T_art),
                                             nu_grid, ckd_pack.gw),
                        jnp.min(jnp.sum(dtau_g, axis=0), axis=0))

            if wo_mols is None:
                dtau_g = _accumulate_dtau_ckd(
                    art, ckd_pack, mols, molmass, opacia, opacia_he,
                    vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud=cloud,
                    rayleigh_xs=None)
                return _finish(dtau_g)
            (flux, tau), rows = _ckd_dtau_batch(
                art, ckd_pack, mols, molmass, opacia, opacia_he,
                vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em,
                list(wo_mols), _finish, cloud=cloud, rayleigh_xs=None)
        else:
            def _one(vmr_):
                dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass,
                                        opacia, g_em, vmr_, vmr_h2, T_art,
                                        mmw_art, opacia_he=opacia_he,
                                        vmr_he=vmr_he, cloud=cloud,
                                        mie_pack=mie_pack, mie=None)
                return (art.run(dtau, _boundary_temperature(T_art)),
                        jnp.sum(dtau, axis=0))

            flux, tau = _one(vmr)
            if wo_mols is None:
                return flux, tau
            bad = [m for m in wo_mols if m not in mols]
            if bad or len(set(wo_mols)) != len(wo_mols):
                raise ValueError(
                    f"wo_mols {list(wo_mols)!r} must be unique members of the "
                    f"RT molecule set {list(mols)!r}")
            rows = [_one({**vmr, m: jnp.zeros_like(vmr[m])}) for m in wo_mols]
        nb = tau.shape[0]
        flux_wo = (jnp.stack([f for f, _ in rows]) if rows
                   else jnp.zeros((0, nb)))
        tau_wo = (jnp.stack([t for _, t in rows]) if rows
                  else jnp.zeros((0, nb)))
        return flux, tau, flux_wo, tau_wo

    return SimpleNamespace(
        emission_flux=emission_flux,
        emission_flux_tau=emission_flux_tau,
        emission_radius=emission_radius,
        tau_bottom=tau_bottom,
        nu_grid=np.asarray(nu_grid),
        wl_um=trt.wl_um,
        p_art_bar=np.asarray(art.pressure),
        art_pbtm_bar=float(trt.art_pbtm_bar),
        # echoed so a consumer can verify the engine honored the key
        p_ref_emission_bar=p_ref_em,
        # which opacity path actually ran, same contract as transmission: a
        # consumer that asked for 'ckd' and silently got sampled line-by-line
        # would be modelling a spectrum with the full grid bias in it
        opacity_mode=str(getattr(trt, "opacity_mode", "lbl")),
        molecules=mols,
    )
