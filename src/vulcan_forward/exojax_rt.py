# size-limit-ok: one closure per observable, with its numpy docs
"""Radiative transfer: correlated-k opacities, geometry anchors, transmission depth and emergent flux.

``build_rt_model`` builds the band grid from the ExoMolOP tables named in the profile, the two CIA tables and an
``ArtTransPure``; ``build_emis_model`` shares those opacities for the emergent flux. The run functions are pure JAX, in
float64, so forward-mode tangents from the chemistry pass through to the spectrum. docs/method.md derives the anchors
and the source term.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # the k-tables are float64
import jax.numpy as jnp
from exojax.atm.polarizability import polarizability as _polarizability
from exojax.database.contdb import CdbCIA
from exojax.opacity import OpaCIA
from exojax.opacity.rayleigh import xsvector_rayleigh_gas
from exojax.rt import ArtEmisPure, ArtTransPure
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_linsap

from vulcan_forward import constants, paths
from vulcan_forward._rt_geometry import (
    _anchor_to_grid_bottom,
    _boundary_temperature,
    _gravity_profile_invsq,
    _photosphere_lnp,
    _radius_at,
)
from vulcan_forward._rt_opacity import _accumulate_dtau_ckd, _ckd_dtau_batch

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

    Profile = dict[str, Any]

__all__ = ["EMISSION_NSTREAM", "build_emis_model", "build_rt_model"]

logger = logging.getLogger(__name__)

# exojax ArtEmisPure's default stream count.
EMISSION_NSTREAM = 8


def _check_profile(profile: Profile) -> None:
    """Refuse unknown keys and any ``opacity_mode`` other than "exomolop"."""
    constants.check_profile_keys(profile)
    mode = str(profile.get("opacity_mode", "exomolop"))
    if mode != "exomolop":
        raise ValueError(
            f"opacity_mode={mode!r} is not available: correlated-k over the "
            "ExoMolOP tables ('exomolop', the default) is the only opacity "
            "path; drop the key."
        )


def _run_emis_ckd_linsap(
    art: ArtEmisPure,
    dtau_g: Array,
    T_boundary: Array,
    nu_bands: Array,
    gw: Array,
    weight_g: Array | None = None,
) -> Array:
    """Return the emergent flux [nband] from a CKD optical depth, with the linear-source scheme.

    exojax's ``ArtEmisPure.run_ckd`` hard-codes ``rtrun_emis_pureabs_ibased``, which has no bottom-boundary term; this
    is upstream's own flatten-solve-reweight structure with ``ibased_linsap`` in its place. The flatten is g-major,
    band-minor, matching ``jnp.tile``'s last-axis tiling of the source, and the g-weights multiply the flux.
    ``weight_g`` [ng, nband] multiplies each g-ordinate's flux first: the photospheric-radius factor of
    ``eclipse_flux_tau``.
    """
    nlayer, ng, nband = dtau_g.shape
    # piBarr -> (nlayer + 1, nband); tile the last axis to (nlayer + 1, ng*nband)
    src = jnp.tile(piBarr(T_boundary, nu_bands), ng)
    flux = rtrun_emis_pureabs_ibased_linsap(dtau_g.reshape((nlayer, ng * nband)), src, art.mus, art.weights)
    flux = flux.reshape((ng, nband))
    if weight_g is not None:
        flux = flux * weight_g
    return jnp.einsum("g,gb->b", gw, flux)


_GEOMETRY_KEYS = {
    "rp_cm": "planet radius in cm at p_ref_bar",
    "gs_cgs": "gravity in cm/s^2 at p_ref_bar",
    "rstar_cm": "stellar radius in cm",
}


def _require_geometry(profile: Profile, *keys: str) -> None:
    """Refuse a profile missing planet geometry: it sets the depth normalization and has no default."""
    missing = [k for k in keys if k not in profile]
    if missing:
        raise ValueError(
            f"profile is missing required planet geometry {missing}: "
            + "; ".join(f"{k} ({_GEOMETRY_KEYS[k]})" for k in missing)
            + "."
        )


def _require_he(vmr_he: Array | None) -> Array:
    """Return the He profile, refusing a missing one, which would silently drop the H2-He CIA term."""
    if vmr_he is None:
        raise ValueError(
            "vmr_he is required: pass the He VMR profile (chem.sidx['He']) so the "
            "H2-He CIA term is included. There is no supported He-less mode."
        )
    return vmr_he


def _check_cia_span(cdb: CdbCIA, nu_grid: Array, label: str) -> None:
    """Refuse a CIA table that does not cover the requested wavenumber grid.

    ``CdbCIA`` never checks coverage, so bands past the file's edge would reuse the edge coefficient.
    """
    lo, hi = float(np.min(cdb.nucia)), float(np.max(cdb.nucia))
    g_lo, g_hi = float(np.min(nu_grid)), float(np.max(nu_grid))
    if g_lo < lo or g_hi > hi:
        raise ValueError(
            f"{label} CIA table covers [{lo:g}, {hi:g}] cm^-1 but the requested "
            f"grid spans [{g_lo:g}, {g_hi:g}] cm^-1. Bands outside the table "
            "would silently reuse its edge coefficient. Narrow nu_min/nu_max "
            "or install a table that covers the band."
        )


# size-limit-ok: the refusals in order, then the closures over one grid and its opacities
def build_rt_model(profile: Profile) -> SimpleNamespace:
    """Build the transmission model for the molecules named in ``profile``.

    Parameters
    ----------
    profile : dict
        The engine's settings; keys in ``constants.PROFILE_KEYS``. Requires ``molecules``, ``nu_min``, ``nu_max``
        (cm^-1), ``art_nlayer``, ``rp_cm``, ``gs_cgs`` and ``rstar_cm``.

    Returns
    -------
    types.SimpleNamespace
        ``transmission_depth``, ``transmission_depth_r``, ``nu_grid`` (cm^-1), ``wl_um``, ``p_art_bar``, ``molecules``,
        and echoes of the settings it resolved: ``art_ptop_bar``, ``art_pbtm_bar``, ``p_ref_bar``, ``rt_integration``,
        ``opacity_mode``.

    Raises
    ------
    ValueError
        On an unknown key, a bad grid or anchor, an uncovered band, or missing geometry.
    FileNotFoundError
        If the H2-He CIA table is missing.
    """
    t0 = time.time()
    _check_profile(profile)
    mols = list(profile["molecules"])
    from vulcan_forward import exomolop as _exo

    ckd_pack = _exo.load_tables(
        mols, float(profile["nu_min"]), float(profile["nu_max"]), molecule_table=profile.get("molecule_table")
    )
    nu_grid = jnp.asarray(ckd_pack.nu_bands)
    ckd_pack.nu_bands_j = nu_grid
    ptop = float(profile.get("art_ptop_bar", constants.ART_PTOP_BAR))
    pbtm = float(profile.get("art_pbtm_bar", constants.ART_PBTM_BAR))
    if not (0.0 < ptop < pbtm):
        raise ValueError(f"art_ptop_bar={ptop:g} / art_pbtm_bar={pbtm:g}: need 0 < top < bottom (bar)")
    # _interp_logk clamps log P at both ends. The low-P clamp is defensible (k is Doppler-dominated and
    # pressure-independent there); the deep one is not: a layer below the ceiling reuses an under-broadened entry.
    p_ceiling = float(np.asarray(ckd_pack.p_grid)[-1])
    if pbtm > p_ceiling:
        raise ValueError(
            f"art_pbtm_bar={pbtm:g} bar is below the k-table pressure ceiling "
            f"({p_ceiling:g} bar). Layers deeper than the ceiling reuse its "
            "entry, which is under-broadened and therefore under-opaque. Raise "
            "the column bottom or install tables that reach deeper."
        )
    integration = str(profile.get("rt_integration", "simpson"))
    if integration not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={integration!r}: exojax ArtTransPure supports 'simpson' (default) or 'trapezoid'"
        )
    mol_table = profile.get("molecule_table") or constants.MOLECULES
    molmass = {key: float(mol_table[key]["molmass"]) for key in mols}

    art = ArtTransPure(pressure_top=ptop, pressure_btm=pbtm, nlayer=int(profile["art_nlayer"]), integration=integration)
    p_art_bar = np.asarray(art.pressure)  # ascending: exojax orders it top to bottom
    lnp_art = jnp.asarray(np.log(p_art_bar))
    logger.info(
        "[rt] ArtTransPure %s layers, P=[%.1e,%.1e] bar, chord integration %s",
        profile["art_nlayer"],
        p_art_bar.min(),
        p_art_bar.max(),
        integration,
    )

    def _cia(path: Path, label: str) -> OpaCIA:
        """Load one CIA table, refusing one that does not cover the band."""
        cdb = CdbCIA(str(path), nurange=nu_grid)
        _check_cia_span(cdb, nu_grid, label)
        return OpaCIA(cdb, nu_grid=nu_grid)

    cia_h2h2 = paths.cia_h2h2_file()
    if not cia_h2h2.exists():
        # exojax fetches it but its downloader swallows failures, so say first what is about to happen.
        logger.warning(
            "[rt] H2-H2 CIA absent at %s; exojax will download 24 MB from "
            "https://hitran.org/data/CIA/main/H2-H2_2011.cia (network required)",
            cia_h2h2,
        )
    opacia = _cia(cia_h2h2, "H2-H2")
    cia_h2he = paths.cia_h2he_file()
    if not cia_h2he.exists():
        raise FileNotFoundError(
            f"H2-He CIA table missing ({cia_h2he}). Download "
            "https://hitran.org/data/CIA/main/H2-He_2011.cia (~147 MB; the "
            "/main/ segment is required) to that path."
        )
    opacia_he = _cia(cia_h2he, "H2-He")
    logger.info("[rt] CIA + RT built; total %.1fs", time.time() - t0)

    if profile.get("use_rayleigh", False):
        rayleigh_xs = (
            xsvector_rayleigh_gas(nu_grid, _polarizability["H2"]),
            xsvector_rayleigh_gas(nu_grid, _polarizability["He"]),
        )
        logger.info("[rt] H2/He Rayleigh scattering enabled")
    else:
        rayleigh_xs = None

    _require_geometry(profile, "rp_cm", "gs_cgs", "rstar_cm")
    Rp_ref = float(profile["rp_cm"])
    g_ref = float(profile["gs_cgs"])
    rstar_cm = float(profile["rstar_cm"])
    p_ref_bar = float(profile.get("p_ref_bar", constants.P_REF_BAR))
    if not (ptop <= p_ref_bar <= pbtm):
        raise ValueError(
            f"p_ref_bar={p_ref_bar:g} lies outside the RT grid "
            f"[{ptop:g}, {pbtm:g}] bar. It is the pressure at which rp_cm and "
            "gs_cgs are defined, so it must be a level the grid actually covers."
        )

    def transmission_depth(
        vmr: dict[str, Array],
        vmr_h2: Array,
        T_art: Array,
        mmw_art: Array,
        vmr_he: Array | None = None,
        cloud: Array | None = None,
        deck: Array | float | None = None,
    ) -> Array:
        """Return the transit depth (R_p(lambda)/R_star)^2 from ART-grid profiles.

        Parameters
        ----------
        vmr : dict of str to jax.Array [nlayer]
            Volume mixing ratio per molecule.
        vmr_h2 : jax.Array [nlayer]
            H2 volume mixing ratio, the CIA partner.
        T_art : jax.Array [nlayer]
            Temperature, K.
        mmw_art : jax.Array [nlayer]
            Mean molecular weight, amu.
        vmr_he : jax.Array [nlayer]
            He volume mixing ratio, the H2-He CIA partner. Required: the default None raises the explanatory error.
        cloud : jax.Array [2], optional
            ``[log10 kappac0, alphac]`` of the exojax power-law cloud.
        deck : float, optional
            log10 pressure (bar) of an opaque grey deck.

        Returns
        -------
        jax.Array [nband]
            The depth.
        """
        vmr_he = _require_he(vmr_he)
        Rp_btm, g_btm = _anchor_to_grid_bottom(lnp_art, T_art, mmw_art, Rp_ref, g_ref, p_ref_bar)
        g_prof = _gravity_profile_invsq(art, T_art, mmw_art, Rp_btm, g_btm)  # (nlayer,1)
        dtau_g = _accumulate_dtau_ckd(
            art,
            ckd_pack,
            mols,
            molmass,
            opacia,
            opacia_he,
            vmr,
            vmr_h2,
            vmr_he,
            T_art,
            mmw_art,
            g_prof,
            cloud=cloud,
            rayleigh_xs=rayleigh_xs,
            deck=deck,
        )
        Rp2 = art.run_ckd(dtau_g, T_art, mmw_art, Rp_btm, g_btm, ckd_pack.gw)
        return Rp2 * (Rp_btm / rstar_cm) ** 2

    # size-limit-ok: most of the length is the numpy docs
    def transmission_depth_r(
        vmr: dict[str, Array],
        vmr_h2: Array,
        T_art: Array,
        mmw_art: Array,
        lnR0: Array | float,
        vmr_he: Array | None = None,
        cloud: Array | None = None,
        wo_mols: list[str] | None = None,
        lnM: Array | float = 0.0,
        deck: Array | float | None = None,
    ) -> Array | tuple[Array, Array]:
        """Return ``transmission_depth`` with the reference radius scaled at fixed planet mass.

        The radius at p_ref_bar is Rp_ref e^lnR0 and the gravity there g_ref e^(lnM - 2 lnR0), so lnR0 keeps GM
        unchanged and lnM scales it. lnR0 = 0 reproduces ``transmission_depth`` (bitwise eager; under jit the folded
        gravity can move by one ulp). Both are RT-only: the chemistry's hydrostatic column is not re-solved. Holding g
        instead would make the planet e^(2 lnR0) times heavier and keep every scale height at its lnR0 = 0 value, which
        the retrieval would pay for in T and mu.

        With ``wo_mols``, each row is bit-identical on the CPU to a separate call on the zeroed profile, reusing the
        shared fold prefix: (n-1)(n/2+2) overlap folds instead of (n+1)(n-1).

        Parameters
        ----------
        vmr : dict of str to jax.Array [nlayer]
            Volume mixing ratio per molecule.
        vmr_h2 : jax.Array [nlayer]
            H2 volume mixing ratio, the CIA partner.
        T_art : jax.Array [nlayer]
            Temperature, K.
        mmw_art : jax.Array [nlayer]
            Mean molecular weight, amu.
        lnR0 : float
            ln of the radius multiple at p_ref_bar.
        vmr_he : jax.Array [nlayer]
            He volume mixing ratio, the H2-He CIA partner. Required: the default None raises the explanatory error.
        cloud : jax.Array [2], optional
            ``[log10 kappac0, alphac]`` of the exojax power-law cloud.
        wo_mols : list of str, optional
            Molecules to drop, one row each.
        lnM : float, optional
            ln of the planet-mass multiple.
        deck : float, optional
            log10 pressure (bar) of an opaque grey deck.

        Returns
        -------
        jax.Array [nband], or tuple of jax.Array
            The depth; with ``wo_mols``, ``(depth, depth_wo)`` with ``depth_wo`` of shape [len(wo_mols), nband].
        """
        vmr_he = _require_he(vmr_he)
        Rp_btm, g_btm = _anchor_to_grid_bottom(
            lnp_art, T_art, mmw_art, Rp_ref * jnp.exp(lnR0), g_ref * jnp.exp(lnM - 2.0 * lnR0), p_ref_bar
        )
        g_prof = _gravity_profile_invsq(art, T_art, mmw_art, Rp_btm, g_btm)  # (nlayer,1)

        def _depth_of(Rp2: Array) -> Array:
            """Scale one spectrum to a depth; one expression for both paths, so they agree bitwise."""
            return Rp2 * (Rp_btm / rstar_cm) ** 2  # (radius/R_star)^2

        def _finish(dtau_g: Array) -> Array:
            """Map an optical depth to its transit depth."""
            return _depth_of(art.run_ckd(dtau_g, T_art, mmw_art, Rp_btm, g_btm, ckd_pack.gw))

        if wo_mols is None:
            dtau_g = _accumulate_dtau_ckd(
                art,
                ckd_pack,
                mols,
                molmass,
                opacia,
                opacia_he,
                vmr,
                vmr_h2,
                vmr_he,
                T_art,
                mmw_art,
                g_prof,
                cloud=cloud,
                rayleigh_xs=rayleigh_xs,
                deck=deck,
            )
            return _finish(dtau_g)
        depth, rows = _ckd_dtau_batch(
            art,
            ckd_pack,
            mols,
            molmass,
            opacia,
            opacia_he,
            vmr,
            vmr_h2,
            vmr_he,
            T_art,
            mmw_art,
            g_prof,
            list(wo_mols),
            _finish,
            cloud=cloud,
            rayleigh_xs=rayleigh_xs,
            deck=deck,
        )
        return depth, (jnp.stack(rows) if rows else jnp.zeros((0, *depth.shape)))

    return SimpleNamespace(
        transmission_depth=transmission_depth,
        transmission_depth_r=transmission_depth_r,
        nu_grid=np.asarray(nu_grid),
        wl_um=constants.UM_PER_CM / np.asarray(nu_grid),
        p_art_bar=p_art_bar,
        molecules=mols,
        # Echoes of what the profile resolved to, so a consumer can check them.
        art_ptop_bar=ptop,
        art_pbtm_bar=pbtm,
        p_ref_bar=p_ref_bar,
        rt_integration=integration,
        opacity_mode="exomolop",
        # Reused by build_emis_model.
        _nu_grid=nu_grid,
        _molmass=molmass,
        _opacia=opacia,
        _opacia_he=opacia_he,
        _ckd_pack=ckd_pack,
    )


# size-limit-ok: the observables close over one column and its anchors
def build_emis_model(trt: SimpleNamespace, profile: Profile) -> SimpleNamespace:  # noqa: C901  # counts its closures
    """Build the thermal-emission model, sharing the opacities and grid of a transmission model.

    The flux is the planet's top-of-atmosphere emergent flux, erg s^-1 cm^-2 / cm^-1, not an eclipse depth: dividing by
    the stellar flux and applying (R/R_star)^2 is the consumer's. Opacity matches transmission without Rayleigh.

    Parameters
    ----------
    trt : types.SimpleNamespace
        The ``build_rt_model`` result.
    profile : dict
        The same settings dict; requires ``gs_cgs`` and ``rp_cm``.

    Returns
    -------
    types.SimpleNamespace
        ``emission_flux``, ``emission_flux_tau``, ``eclipse_flux_tau``, ``emission_radius``, ``tau_bottom``,
        ``nu_grid``, ``wl_um``, ``p_art_bar``, ``art_pbtm_bar``, ``art_ptop_bar``, ``art_nlayer``,
        ``p_ref_emission_bar``, ``p_ref_bar``, ``molecules``.

    Raises
    ------
    ValueError
        If the emission anchor lies outside the grid, or ``p_ref_bar`` or ``art_nlayer`` disagree with ``trt``.
    """
    constants.check_profile_keys(profile)
    ckd_pack = trt._ckd_pack
    nu_grid = trt._nu_grid
    molmass, opacia, mols = trt._molmass, trt._opacia, trt.molecules
    opacia_he = trt._opacia_he
    _require_geometry(profile, "gs_cgs", "rp_cm")
    # Radius and gravity are re-anchored together at p_ref_emission_bar: the transit radius with this gravity is 8.9%
    # off in GM.
    g_ref_em = float(profile["gs_cgs"])
    r_ref_em = float(profile["rp_cm"])
    p_ref_em = constants.P_REF_EMISSION_BAR
    if not (trt.art_ptop_bar <= p_ref_em <= trt.art_pbtm_bar):
        raise ValueError(
            f"the emission anchor P_REF_EMISSION_BAR={p_ref_em:g} lies outside "
            f"the RT grid [{trt.art_ptop_bar:g}, {trt.art_pbtm_bar:g}] bar. It "
            "is the level the emission column's radius and gravity are "
            "anchored at, so the grid must cover it: deepen art_pbtm_bar."
        )
    # Both observables share the transit anchor, validated against the grid in trt.
    p_ref_used = float(trt.p_ref_bar)
    if "p_ref_bar" in profile and float(profile["p_ref_bar"]) != p_ref_used:
        raise ValueError(
            f"p_ref_bar={float(profile['p_ref_bar']):g} disagrees with the "
            f"transmission model's validated {p_ref_used:g}; the two "
            "observables share one column anchor (build both from one "
            "profile)."
        )

    # ibased_linsap keeps the interior source term that "ibased" (and ArtEmisPure.run_ckd) drops.
    art_nlayer = int(np.asarray(trt.p_art_bar).size)
    if "art_nlayer" in profile and int(profile["art_nlayer"]) != art_nlayer:
        raise ValueError(
            f"art_nlayer={int(profile['art_nlayer'])} disagrees with the "
            f"transmission grid's {art_nlayer} layers; the two observables "
            "share opacities and must share the column."
        )
    art = ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=trt.art_ptop_bar,
        pressure_btm=trt.art_pbtm_bar,
        nlayer=art_nlayer,
        rtsolver="ibased_linsap",
        nstream=EMISSION_NSTREAM,
    )
    lnp_em = jnp.asarray(np.log(np.asarray(art.pressure)))
    logger.info("[rt] ArtEmisPure %d layers (shares opacities)", art_nlayer)

    def _dtau(
        vmr: dict[str, Array],
        vmr_h2: Array,
        vmr_he: Array,
        T_art: Array,
        mmw_art: Array,
        g_em: Array,
        cloud: Array | None,
    ) -> Array:
        """Return the emission optical depth, without Rayleigh."""
        return _accumulate_dtau_ckd(
            art,
            ckd_pack,
            mols,
            molmass,
            opacia,
            opacia_he,
            vmr,
            vmr_h2,
            vmr_he,
            T_art,
            mmw_art,
            g_em,
            cloud=cloud,
            rayleigh_xs=None,
        )

    def _anchor(lnR0: Array | float) -> tuple[Array, Array]:
        """Return (radius, gravity) at p_ref_bar for lnR0, at fixed planet mass."""
        return r_ref_em * jnp.exp(lnR0), g_ref_em * jnp.exp(-2.0 * lnR0)

    def _g_layers(T_art: Array, mmw_art: Array, lnR0: Array | float) -> Array:
        """Return inverse-square g(r) at the layer centres, [nlayer, 1], the g(r) of the anchor and the photosphere."""
        return _radius_at(lnp_em, T_art, mmw_art, *_anchor(lnR0), p_ref_used, jnp.exp(lnp_em))[1][:, None]

    def emission_flux(
        vmr: dict[str, Array],
        vmr_h2: Array,
        T_art: Array,
        mmw_art: Array,
        vmr_he: Array | None = None,
        cloud: Array | None = None,
        lnR0: Array | float = 0.0,
    ) -> Array:
        """Return the emergent thermal flux from ART-grid profiles.

        Parameters
        ----------
        vmr : dict of str to jax.Array [nlayer]
            Volume mixing ratio per molecule.
        vmr_h2 : jax.Array [nlayer]
            H2 volume mixing ratio, the CIA partner.
        T_art : jax.Array [nlayer]
            Temperature, K.
        mmw_art : jax.Array [nlayer]
            Mean molecular weight, amu.
        vmr_he : jax.Array [nlayer]
            He volume mixing ratio, the H2-He CIA partner. Required: the default None raises the explanatory error.
        cloud : jax.Array [2], optional
            ``[log10 kappac0, alphac]`` of the exojax power-law cloud.
        lnR0 : float, optional
            ln of the radius multiple at p_ref_bar, at fixed planet mass.

        Returns
        -------
        jax.Array [nband]
            The flux, erg s^-1 cm^-2 / cm^-1.
        """
        vmr_he = _require_he(vmr_he)
        dtau_g = _dtau(vmr, vmr_h2, vmr_he, T_art, mmw_art, _g_layers(T_art, mmw_art, lnR0), cloud)
        return _run_emis_ckd_linsap(art, dtau_g, _boundary_temperature(T_art), nu_grid, ckd_pack.gw)

    def _emission_anchor(T_art: Array, mmw_art: Array, lnR0: Array | float) -> tuple[Array, Array]:
        """Return (radius, gravity) at p_ref_emission_bar, one consistent pair."""
        return _radius_at(lnp_em, T_art, mmw_art, *_anchor(lnR0), p_ref_used, p_ref_em)

    def emission_radius(T_art: Array, mmw_art: Array, lnR0: Array | float = 0.0) -> Array:
        """Return the planet radius at p_ref_emission_bar, for the eclipse-depth prefactor.

        The catalogue radius is the transit radius near 1 mbar; the emitting surface sits about 2 decades deeper, so the
        two differ by several percent, and pairing the transit radius with the emission gravity misstates GM.

        Parameters
        ----------
        T_art : jax.Array [nlayer]
            Temperature, K.
        mmw_art : jax.Array [nlayer]
            Mean molecular weight, amu.
        lnR0 : float, optional
            ln of the radius multiple at p_ref_bar, at fixed planet mass.

        Returns
        -------
        jax.Array
            The radius, cm.
        """
        return _emission_anchor(T_art, mmw_art, lnR0)[0]

    def tau_bottom(
        vmr: dict[str, Array],
        vmr_h2: Array,
        T_art: Array,
        mmw_art: Array,
        vmr_he: Array,
        cloud: Array | None = None,
        lnR0: Array | float = 0.0,
    ) -> Array:
        """Return the minimum over g-ordinates of the total vertical optical depth at the column bottom, per band.

        Where it is small the flux is set by the extrapolated boundary temperature, so callers check its minimum and
        flag see-through windows. The minimum over g is the conservative reduction.

        Parameters
        ----------
        vmr : dict of str to jax.Array [nlayer]
            Volume mixing ratio per molecule.
        vmr_h2 : jax.Array [nlayer]
            H2 volume mixing ratio, the CIA partner.
        T_art : jax.Array [nlayer]
            Temperature, K.
        mmw_art : jax.Array [nlayer]
            Mean molecular weight, amu.
        vmr_he : jax.Array [nlayer]
            He volume mixing ratio.
        cloud : jax.Array [2], optional
            ``[log10 kappac0, alphac]`` of the exojax power-law cloud.
        lnR0 : float, optional
            ln of the radius multiple at p_ref_bar, at fixed planet mass.

        Returns
        -------
        jax.Array [nband]
            The optical depth.
        """
        vmr_he = _require_he(vmr_he)
        dtau_g = _dtau(vmr, vmr_h2, vmr_he, T_art, mmw_art, _g_layers(T_art, mmw_art, lnR0), cloud)
        return jnp.min(jnp.sum(dtau_g, axis=0), axis=0)

    def _flux_tau(
        vmr: dict[str, Array],
        vmr_h2: Array,
        T_art: Array,
        mmw_art: Array,
        vmr_he: Array | None,
        cloud: Array | None,
        wo_mols: list[str] | None,
        photosphere: bool,
        lnR0: Array | float,
    ) -> tuple[Array, ...]:
        """Return flux and bottom optical depth from one optical-depth build, with the leave-one-out rows when asked."""
        vmr_he = _require_he(vmr_he)
        r_em, _ = _emission_anchor(T_art, mmw_art, lnR0)
        g_em = _g_layers(T_art, mmw_art, lnR0)

        def _finish(dtau_g: Array) -> tuple[Array, Array]:
            """Map an optical depth to (flux, bottom optical depth)."""
            w = None
            if photosphere:
                r_phot, _ = _radius_at(
                    lnp_em, T_art, mmw_art, *_anchor(lnR0), p_ref_used, jnp.exp(_photosphere_lnp(dtau_g, lnp_em))
                )
                w = (r_phot / r_em) ** 2
            return (
                _run_emis_ckd_linsap(art, dtau_g, _boundary_temperature(T_art), nu_grid, ckd_pack.gw, weight_g=w),
                jnp.min(jnp.sum(dtau_g, axis=0), axis=0),
            )

        if wo_mols is None:
            return _finish(_dtau(vmr, vmr_h2, vmr_he, T_art, mmw_art, g_em, cloud))
        (flux, tau), rows = _ckd_dtau_batch(
            art,
            ckd_pack,
            mols,
            molmass,
            opacia,
            opacia_he,
            vmr,
            vmr_h2,
            vmr_he,
            T_art,
            mmw_art,
            g_em,
            list(wo_mols),
            _finish,
            cloud=cloud,
            rayleigh_xs=None,
        )
        nb = tau.shape[0]
        flux_wo = jnp.stack([f for f, _ in rows]) if rows else jnp.zeros((0, nb))
        tau_wo = jnp.stack([t for _, t in rows]) if rows else jnp.zeros((0, nb))
        return flux, tau, flux_wo, tau_wo

    def emission_flux_tau(
        vmr: dict[str, Array],
        vmr_h2: Array,
        T_art: Array,
        mmw_art: Array,
        vmr_he: Array | None = None,
        cloud: Array | None = None,
        wo_mols: list[str] | None = None,
        lnR0: Array | float = 0.0,
    ) -> tuple[Array, ...]:
        """Return the emergent flux and bottom optical depth from one optical-depth build.

        Bitwise what separate ``emission_flux`` and ``tau_bottom`` calls return. Each ``wo_mols`` row is bit-identical
        on the CPU to a from-scratch call on the zeroed profile (``ckd._fold_wo``).

        Parameters
        ----------
        vmr : dict of str to jax.Array [nlayer]
            Volume mixing ratio per molecule.
        vmr_h2 : jax.Array [nlayer]
            H2 volume mixing ratio, the CIA partner.
        T_art : jax.Array [nlayer]
            Temperature, K.
        mmw_art : jax.Array [nlayer]
            Mean molecular weight, amu.
        vmr_he : jax.Array [nlayer]
            He volume mixing ratio, the H2-He CIA partner. Required: the default None raises the explanatory error.
        cloud : jax.Array [2], optional
            ``[log10 kappac0, alphac]`` of the exojax power-law cloud.
        wo_mols : list of str, optional
            Molecules to drop, one row each.
        lnR0 : float, optional
            ln of the radius multiple at p_ref_bar, at fixed planet mass.

        Returns
        -------
        tuple of jax.Array
            ``(flux, tau_bottom)``; with ``wo_mols``, ``(flux, tau_bottom, flux_wo, tau_wo)``.
        """
        return _flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud, wo_mols, photosphere=False, lnR0=lnR0)

    def eclipse_flux_tau(
        vmr: dict[str, Array],
        vmr_h2: Array,
        T_art: Array,
        mmw_art: Array,
        vmr_he: Array | None = None,
        cloud: Array | None = None,
        wo_mols: list[str] | None = None,
        lnR0: Array | float = 0.0,
    ) -> tuple[Array, ...]:
        """Return ``emission_flux_tau`` with the tau = 2/3 photospheric radius folded in.

        The flux is sum_g w_g F_g (R_phot,g / R_em)^2 per band, so eclipse depth = flux / F_star x (R_em / R_star)^2
        with R_em from ``emission_radius``.

        Parameters
        ----------
        vmr : dict of str to jax.Array [nlayer]
            Volume mixing ratio per molecule.
        vmr_h2 : jax.Array [nlayer]
            H2 volume mixing ratio, the CIA partner.
        T_art : jax.Array [nlayer]
            Temperature, K.
        mmw_art : jax.Array [nlayer]
            Mean molecular weight, amu.
        vmr_he : jax.Array [nlayer]
            He volume mixing ratio, the H2-He CIA partner. Required: the default None raises the explanatory error.
        cloud : jax.Array [2], optional
            ``[log10 kappac0, alphac]`` of the exojax power-law cloud.
        wo_mols : list of str, optional
            Molecules to drop, one row each.
        lnR0 : float, optional
            ln of the radius multiple at p_ref_bar, at fixed planet mass.

        Returns
        -------
        tuple of jax.Array
            ``(flux, tau_bottom)``; with ``wo_mols``, ``(flux, tau_bottom, flux_wo, tau_wo)``.
        """
        return _flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud, wo_mols, photosphere=True, lnR0=lnR0)

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
        # the emission anchor, for a consumer's labels and checks
        p_ref_emission_bar=p_ref_em,
        p_ref_bar=p_ref_used,
        molecules=mols,
    )
