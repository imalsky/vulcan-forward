"""Optical depth for the RT: line opacity from the k-tables plus CIA, Rayleigh and the two cloud terms.

Imports exojax, so it follows ``vulcan_chem`` as ``exojax_rt`` does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
from exojax.atm.atmconvert import vmr_to_mmr
from exojax.atm.simple_clouds import powerlaw_clouds

from vulcan_forward import ckd, constants

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import SimpleNamespace

    from exojax.opacity import OpaCIA
    from exojax.rt import ArtEmisPure, ArtTransPure
    from jax import Array

_Finished = TypeVar("_Finished")

# g/mol, for the Rayleigh mmr conversion
_H2_MOLMASS = 2.0 * constants.ATOMIC_MASSES[constants.ATOM_COLS["H"]]
_HE_MOLMASS = constants.ATOMIC_MASSES[constants.ATOM_COLS["He"]]


def _column_ok(T_art: Array, mmw_art: Array, vmr: dict[str, Array]) -> Array:
    """Return whether the column is one the RT can model: a scalar bool, or an error when the call is eager.

    Temperatures inside the opacity window (the k-table and CIA interpolations clamp at their edges), a mean molecular
    weight no lighter than atomic hydrogen, and every mixing ratio in [0, 1]. A NaN fails each test. Nothing can raise
    under jit, so there the caller turns False into a NaN spectrum, which a consumer must check.
    """
    m_h = constants.ATOMIC_MASSES[constants.ATOM_COLS["H"]]
    bounds = {
        "temperature (K, the opacity window)": (T_art, constants.T_OPA_MIN_K, constants.T_OPA_MAX_K),
        "mean molecular weight (amu)": (mmw_art, m_h, jnp.inf),
        **{f"{name} mixing ratio": (v, 0.0, 1.0) for name, v in vmr.items()},
    }
    passed = {name: jnp.all((x >= lo) & (x <= hi)) for name, (x, lo, hi) in bounds.items()}
    ok = jnp.all(jnp.stack(list(passed.values())))
    if not isinstance(ok, jax.core.Tracer) and not bool(ok):
        raise ValueError(
            "the RT column is outside what the model covers: "
            + "; ".join(
                f"{name} spans [{float(jnp.min(bounds[name][0])):g}, {float(jnp.max(bounds[name][0])):g}], "
                f"allowed [{bounds[name][1]:g}, {bounds[name][2]:g}]"
                for name, good in passed.items()
                if not bool(good)
            )
            + ". Reject the column: nothing is clamped."
        )
    return ok


def _accumulate_dtau_ckd(
    art: ArtTransPure | ArtEmisPure,
    pack: SimpleNamespace,
    mols: list[str],
    molmass: dict[str, float],
    opacia: OpaCIA,
    opacia_he: OpaCIA,
    vmr: dict[str, Array],
    vmr_h2: Array,
    vmr_he: Array,
    T_art: Array,
    mmw_art: Array,
    g_prof: Array,
    cloud: Array | None = None,
    rayleigh_xs: tuple[Array, Array] | None = None,
    deck: Array | float | None = None,
) -> Array:
    """Return the optical depth per layer, g-ordinate and band, [nlayer, ng, nband].

    Line opacity comes from each k-table at the layer (T, P), combined by random overlap (``ckd.fold``). The continua
    are smooth across a band, so their band value adds identically to every g-ordinate. Rayleigh is transmission-only:
    it is scattering, and the pure-absorption emission solver must not count it as thermal extinction.
    """
    P = jnp.asarray(art.pressure)
    ok = _column_ok(T_art, mmw_art, {**{key: vmr[key] for key in mols}, "H2": vmr_h2, "He": vmr_he})
    dts = jnp.stack([_ckd_dt_one(art, pack, key, molmass, vmr[key], T_art, mmw_art, g_prof, P) for key in mols])
    cont = _ckd_continuum(
        art, pack, opacia, opacia_he, vmr_h2, vmr_he, T_art, mmw_art, g_prof, cloud, rayleigh_xs, deck
    )
    return ckd.fold(dts, pack.gg, pack.gw) + jnp.where(ok, cont, jnp.nan)[:, None, :]


def _ckd_dt_one(
    art: ArtTransPure | ArtEmisPure,
    pack: SimpleNamespace,
    key: str,
    molmass: dict[str, float],
    vmr_key: Array,
    T_art: Array,
    mmw_art: Array,
    g_prof: Array,
    P: Array,
) -> Array:
    """Return one molecule's optical depth [nlayer, ng, nband]."""
    lk = ckd._interp_logk(pack.logk[key], pack.t_grid, pack.p_grid, T_art, P)  # (nlayer, ng, nb)
    mmr = vmr_to_mmr(vmr_key, molmass[key], mmw_art)
    # layer_optical_depth_ckd multiplies a (nlayer, ng, nband) tensor by dParr[:, None, None] / gravity, so gravity
    # needs the third axis too; the continuum terms stay (nlayer, nband).
    return art.opacity_profile_xs_ckd(jnp.exp(lk), mmr, molmass[key], g_prof[..., None])


def _ckd_continuum(
    art: ArtTransPure | ArtEmisPure,
    pack: SimpleNamespace,
    opacia: OpaCIA,
    opacia_he: OpaCIA,
    vmr_h2: Array,
    vmr_he: Array,
    T_art: Array,
    mmw_art: Array,
    g_prof: Array,
    cloud: Array | None,
    rayleigh_xs: tuple[Array, Array] | None,
    deck: Array | float | None = None,
) -> Array:
    """Return the band continuum [nlayer, nband]: CIA, Rayleigh, the power-law cloud and the grey deck.

    ``cloud`` is ``[log10 kappac0 (cm^2/g at constants.CLOUD_NUC0), alphac]`` for exojax's ``powerlaw_clouds``;
    ``rayleigh_xs`` is ``(xs_h2, xs_he)``, each [nband]; ``deck`` is the log10 pressure (bar) below which layers get the
    opaque grey ``constants.CLOUD_DECK_TAU``, through a logistic step in ln P so the deck stays differentiable. A
    consumer passes either cloud term or both.
    """

    def _cia(opa_: OpaCIA, va: Array, vb: Array) -> Array:
        """Return one CIA pair's optical depth [nlayer, nband]."""
        return art.opacity_profile_cia(opa_.logacia_matrix(T_art), T_art, va, vb, mmw_art[:, None], g_prof)

    cont = _cia(opacia, vmr_h2, vmr_h2) + _cia(opacia_he, vmr_h2, vmr_he)
    if rayleigh_xs is not None:
        xs_h2, xs_he = rayleigh_xs
        nl, nb = art.pressure.shape[0], xs_h2.shape[0]
        cont = cont + art.opacity_profile_xs(
            jnp.broadcast_to(xs_h2[None, :], (nl, nb)), vmr_to_mmr(vmr_h2, _H2_MOLMASS, mmw_art), _H2_MOLMASS, g_prof
        )
        cont = cont + art.opacity_profile_xs(
            jnp.broadcast_to(xs_he[None, :], (nl, nb)), vmr_to_mmr(vmr_he, _HE_MOLMASS, mmw_art), _HE_MOLMASS, g_prof
        )
    if cloud is not None:
        # kappa is per gram of atmosphere (pRT convention), so dtau = kappa dP_cgs / g with no 1/m_u; exojax folds
        # bar_cgs / m_u into its per-molecule opacity_factor instead.
        kappa_c = powerlaw_clouds(pack.nu_bands_j, kappac0=10.0 ** cloud[0], nuc0=constants.CLOUD_NUC0, alphac=cloud[1])
        dP = jnp.asarray(art.dParr)
        cont = cont + kappa_c[None, :] * (dP[:, None] * constants.BAR_CGS / g_prof)
    if deck is not None:
        lnp = jnp.log(jnp.asarray(art.pressure))
        step = jax.nn.sigmoid((lnp - jnp.asarray(deck) * jnp.log(10.0)) / constants.CLOUD_DECK_WIDTH_LNP)  # (nlayer,)
        cont = cont + constants.CLOUD_DECK_TAU * step[:, None]
    return cont


def _ckd_dtau_batch(
    art: ArtTransPure | ArtEmisPure,
    pack: SimpleNamespace,
    mols: list[str],
    molmass: dict[str, float],
    opacia: OpaCIA,
    opacia_he: OpaCIA,
    vmr: dict[str, Array],
    vmr_h2: Array,
    vmr_he: Array,
    T_art: Array,
    mmw_art: Array,
    g_prof: Array,
    wo_mols: list[str],
    finish: Callable[[Array], _Finished],
    cloud: Array | None = None,
    rayleigh_xs: tuple[Array, Array] | None = None,
    deck: Array | float | None = None,
) -> tuple[_Finished, list[_Finished]]:
    """Return the full and leave-one-out observables in one pass over the k-fold.

    ``finish`` maps an optical depth [nlayer, ng, nband] to its observable; returns ``(finish(full), [finish(wo) ...])``
    aligned to ``wo_mols``. The dropped molecule is still folded, as the zero tensor its zeroed VMR produces, so every
    output is bit-identical on the CPU to a from-scratch solve with that VMR zeroed (``ckd._fold_wo``). The per-molecule
    tensors are held for the fold tails, about n x 35 MB at planner defaults.
    """
    bad = [m for m in wo_mols if m not in mols]
    if bad or len(set(wo_mols)) != len(wo_mols):
        raise ValueError(f"wo_mols {list(wo_mols)!r} must be unique members of the RT molecule set {list(mols)!r}")
    P = jnp.asarray(art.pressure)
    ok = _column_ok(T_art, mmw_art, {**{k: vmr[k] for k in mols}, "H2": vmr_h2, "He": vmr_he})
    dts = [_ckd_dt_one(art, pack, k, molmass, vmr[k], T_art, mmw_art, g_prof, P) for k in mols]
    cont = _ckd_continuum(
        art, pack, opacia, opacia_he, vmr_h2, vmr_he, T_art, mmw_art, g_prof, cloud, rayleigh_xs, deck
    )
    cont3 = jnp.where(ok, cont, jnp.nan)[:, None, :]

    def zero_of(i: int) -> Array:
        """Return the optical depth of molecule ``i`` with its VMR zeroed."""
        return _ckd_dt_one(art, pack, mols[i], molmass, jnp.zeros_like(vmr[mols[i]]), T_art, mmw_art, g_prof, P)

    wo_idx = [mols.index(m) for m in wo_mols]
    full_tot, wo = ckd._fold_wo(dts, zero_of, pack.gg, pack.gw, wo_idx, finish=lambda t: finish(t + cont3))
    by_idx = dict(wo)
    return finish(full_tot + cont3), [by_idx[i] for i in wo_idx]
