"""The differentiable map from chemistry parameters to the converged column, on vulcan3.

``build_chem_model`` runs the one-time pre-loop and returns a model whose ``converged_y(theta)`` re-converges the column
for ``theta = [lnZ, c_o, lnKzz, *extras, *tp]``; ``_chem_model`` holds the solves. Rates, structure and condensation
arrays are rebuilt on-graph per proposal; the runner's ``lax.while_loop`` supports jvp but not vjp, so forward mode is
the end-to-end route.

This module owns the first jax import: it sets jax x64 and refuses to follow exojax. ``select_network`` hands the
profile's network to vulcan3. docs/method.md describes the composition map and the limits of a condensing column.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from vulcan_forward import constants

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jax import Array

    Profile = dict[str, Any]

__all__ = [
    "CHEM_EXTRAS",
    "ChemParams",
    "ConvDiag",
    "build_chem_model",
    "bz_margin",
    "params_from_vector",
    "select_network",
]

logger = logging.getLogger(__name__)

if "exojax" in sys.modules:
    raise RuntimeError(
        "vulcan_forward.vulcan_chem must be imported BEFORE exojax: it fixes "
        "jax x64. Import order: vulcan_forward.vulcan_chem, then "
        "vulcan_forward.exojax_rt."
    )

os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from vulcan_forward._chem_layout import CHEM_EXTRAS, ChemParams, ConvDiag, _check_extras, params_from_vector
from vulcan_forward._elemental import _assert_composition_tables, _build_elemental, bz_margin

# The settings build_chem_model needs and has no default for: they name the chemistry, and belong to the caller's case.
_CHEMISTRY_KEYS = {
    "network": "the kinetics network file, a path vulcan3 resolves (e.g. thermo/SNCHO_photo_network_2025.txt)",
    "atom_list": "the network's elements (e.g. ['H', 'O', 'C', 'N', 'S'])",
    "vulcan_cfg_name": "the vulcan3 config that holds the planet, the star and the T-P and Kzz profile (name or path)",
}
# What vulcan3 freezes at its first import, by the environment variable it reads.
_FROZEN_ENV = {"network": "VULCAN3_NETWORK", "atom_list": "VULCAN3_ATOM_LIST"}
# Optional top-level profile keys copied onto the vulcan3 config, with their type; None keeps the config's value.
_OPTIONAL_CFG_KEYS = {"nz": int, "count_min": int, "count_max": int, "dt_max": float, "yconv_min": float}
# Every top-level profile key that is also a vulcan3 config key: stated in cfg_overrides too, it would be set twice.
_TOP_LEVEL_CFG_KEYS = ("use_photo", "yconv_cri", *_OPTIONAL_CFG_KEYS)


def _atom_names(atom_list: str | Iterable[str]) -> list[str]:
    """Return the element names of a comma-separated string or a sequence."""
    return [a.strip() for a in (atom_list.split(",") if isinstance(atom_list, str) else atom_list)]


def select_network(network: str, atom_list: str | Iterable[str]) -> None:
    """Hand the kinetics network and its atom list to vulcan3, which freezes both at its first import.

    Call it before anything imports vulcan3; ``build_chem_model`` calls it with its profile's values. One process runs
    one network, so a later call must name the same pair.

    Parameters
    ----------
    network : str
        The network file, a path vulcan3 resolves, such as ``thermo/SNCHO_photo_network_2025.txt``.
    atom_list : str or sequence of str
        The network's elements, comma-separated or as a sequence.

    Raises
    ------
    RuntimeError
        If the environment, or a vulcan3 that is already imported, holds another network or atom list.
    """
    want = {"network": str(network), "atom_list": ",".join(_atom_names(atom_list))}
    if "vulcan3" in sys.modules:
        from vulcan3.config import default_config
        from vulcan3.paths import resolve_data_path

        # What vulcan3 built at its import where those modules are loaded, else the config it will build them from.
        chem_funs, jax_step = sys.modules.get("vulcan3.chem_funs"), sys.modules.get("vulcan3.jax_step")
        frozen = default_config()
        have = {
            "network": str(chem_funs.NETWORK.network_path if chem_funs is not None else frozen.network),
            "atom_list": ",".join(jax_step.IMPORT_ATOM_LIST if jax_step is not None else frozen.atom_list),
        }
        same_file = (
            Path(resolve_data_path(have["network"])).resolve() == Path(resolve_data_path(want["network"])).resolve()
        )
        if not same_file or have["atom_list"] != want["atom_list"]:
            raise RuntimeError(
                f"vulcan3 is already imported with network {have['network']!r} and atoms {have['atom_list']!r}, and "
                f"this call asks for {want['network']!r} and {want['atom_list']!r}. vulcan3 freezes both at its first "
                "import: call select_network before anything imports vulcan3, and run another network in another "
                "process."
            )
        return
    held = {k: os.environ.get(env) for k, env in _FROZEN_ENV.items()}
    conflict = {_FROZEN_ENV[k]: (held[k], want[k]) for k in want if held[k] not in (None, want[k])}
    if conflict:
        raise RuntimeError(
            "the environment disagrees with the network this call selects: "
            + "; ".join(f"{env} is {got!r}, asked for {asked!r}" for env, (got, asked) in conflict.items())
            + ". Unset the variable: the caller's settings decide."
        )
    for key, env in _FROZEN_ENV.items():
        os.environ[env] = want[key]


def _check_overrides(profile: Profile) -> dict[str, Any]:
    """Return the profile's ``cfg_overrides``, refusing a key the profile also sets or that cannot take effect there."""
    import vulcan3

    overrides = dict(profile.get("cfg_overrides") or {})
    # setattr checks nothing, so a removed or misspelled key is refused first.
    vulcan3.validate_overrides(overrides)
    frozen = sorted(set(overrides) & set(_FROZEN_ENV))
    if frozen:
        raise ValueError(
            f"cfg_overrides sets {frozen}, which vulcan3 freezes at its first import: name them at the top level of "
            "the profile ('network', 'atom_list')."
        )
    twice = sorted(k for k in _TOP_LEVEL_CFG_KEYS if k in overrides and profile.get(k) is not None)
    if twice:
        raise ValueError(
            f"{twice} are set at the top level of the profile and again in cfg_overrides: state each once."
        )
    if "runtime" in overrides and "dt_max" not in overrides and profile.get("dt_max") is None:
        raise ValueError(
            "cfg_overrides sets 'runtime' without 'dt_max': vulcan3 derived dt_max from the config file's runtime when "
            "it loaded the file, so it would not follow. State dt_max too."
        )
    return overrides


def _check_one_planet(cfg: Any, profile: Profile) -> None:
    """Refuse a chemistry planet whose mass disagrees with the RT geometry in the same profile."""
    from vulcan3.phy_const import G_grav

    if "rp_cm" not in profile or "gs_cgs" not in profile:
        logger.info("[chem] no rp_cm and gs_cgs in the profile: the chemistry planet is not compared with an RT one")
        return
    gm_chem = G_grav * float(cfg.Mp)
    gm_rt = float(profile["gs_cgs"]) * float(profile["rp_cm"]) ** 2
    if abs(gm_chem / gm_rt - 1.0) > constants.GM_MATCH_RTOL:
        raise ValueError(
            f"the chemistry and the RT describe different planets: G Mp = {gm_chem:.4e} cm^3 s^-2 from the vulcan3 "
            f"config (Mp = {float(cfg.Mp):.4e} g) against gs_cgs rp_cm^2 = {gm_rt:.4e} from the profile, "
            f"{100.0 * (gm_chem / gm_rt - 1.0):+.1f} % apart (allowed {100.0 * constants.GM_MATCH_RTOL:g} %). Set Mp "
            "in cfg_overrides, or correct gs_cgs and rp_cm."
        )


def _configured_cfg(profile: Profile) -> Any:
    """Load the VULCAN config the profile names and apply the profile to it, refusing what the rebuild cannot follow."""
    missing = [k for k in _CHEMISTRY_KEYS if not profile.get(k)]
    if missing:
        raise ValueError(
            f"profile is missing {missing}, which have no default: "
            + "; ".join(f"{k} ({_CHEMISTRY_KEYS[k]})" for k in missing)
            + "."
        )
    select_network(profile["network"], profile["atom_list"])
    import vulcan3

    overrides = _check_overrides(profile)
    cfg = vulcan3.load_config(profile["vulcan_cfg_name"])
    # vulcan3's pre-loop setup compares these with what it froze at import, so it is the profile's pair it checks.
    cfg.network = str(profile["network"])
    cfg.atom_list = _atom_names(profile["atom_list"])
    cfg.use_print_prog = False
    cfg.use_photo = bool(profile["use_photo"])
    cfg.yconv_cri = float(profile["yconv_cri"])
    # is not None: count_min=0 and yconv_min=0 are legitimate values.
    for key, cast in _OPTIONAL_CFG_KEYS.items():
        if profile.get(key) is not None:
            setattr(cfg, key, cast(profile[key]))
    for _k, _v in overrides.items():
        setattr(cfg, _k, _v)
    _check_one_planet(cfg, profile)

    if list(cfg.T_cross_sp or []):
        raise ValueError(
            f"T_cross_sp={list(cfg.T_cross_sp)} is not supported here: the "
            "per-proposal rebuild (_prep) does not rebuild the T-dependent cross "
            "sections (pv.p_absp_T_cross, pv.p_cross_J_T), so photolysis would use "
            "the baseline temperature. Set T_cross_sp=[] in the config or "
            "cfg_overrides."
        )
    # The chemistry grid must reach the RT top. An explicit cfg_overrides P_t wins: the model-top ladder and the
    # condensation pin bring their own grids.
    if "P_t" not in (profile.get("cfg_overrides") or {}):
        _ptop = profile.get("art_ptop_bar")
        cfg.P_t = float(_ptop if _ptop is not None else constants.ART_PTOP_BAR) * constants.BAR_CGS

    return cfg


def _log_inputs(network_file: Path, cfg: Any, profile: Profile) -> None:
    """Log the files this build reads: vulcan3 looks in the working directory before its packaged data."""
    from vulcan3.paths import resolve_data_path

    logger.info(
        "[chem] config %r: network %s, atmosphere %s, stellar flux %s",
        profile["vulcan_cfg_name"],
        network_file,
        resolve_data_path(cfg.atm_file) if cfg.atm_type == "file" else cfg.atm_type,
        resolve_data_path(cfg.sflux_file) if cfg.use_photo else "not read (photochemistry off)",
    )


def _thermo_dir(network_file: Path) -> Path:
    """Return the folder holding the network's NASA9 Gibbs data, refusing a network that has none beside it.

    The rates and the equilibrium start must read the Gibbs data of the tree the network came from; another tree's
    data would be a different chemistry with no message.
    """
    thermo_dir = Path(network_file).parent
    if not (thermo_dir / "NASA9").is_dir():
        raise FileNotFoundError(
            f"no NASA9 folder beside the network {network_file}: expected {thermo_dir / 'NASA9'}. Keep the network in "
            "a thermo folder that holds its NASA9 Gibbs data."
        )
    return thermo_dir


def _check_stellar_span(cfg: Any, var: Any) -> None:
    """Refuse a stellar spectrum that does not cover every photolysis threshold of the network.

    vulcan3 cuts its photolysis grid to the file's span, as VULCAN 2 does, so a short file switches photolysis off
    outside it without a word.
    """
    from vulcan3.photo_setup import _load_thresholds  # vulcan3's table of dissociation thresholds, nm

    thresholds = _load_thresholds(var.photo_sp)
    need_lo, need_hi = min(thresholds.values()), max(thresholds.values())
    lo, hi = float(var.def_bin_min), float(var.def_bin_max)
    if lo > need_lo or hi < need_hi:
        raise ValueError(
            f"the stellar spectrum {cfg.sflux_file} covers {lo:g}-{hi:g} nm, but this network's photolysis thresholds "
            f"run from {need_lo:g} to {need_hi:g} nm. Photolysis outside the file would be switched off. Use a "
            "spectrum that covers the range."
        )


def _make_kzz_eff(
    kzz_mode: str, profile: Profile, atm: Any, Kzz0: Array
) -> Callable[[float | Array, float | Array], Array]:
    """Return ``Kzz(lnKzz, kzz_slope)`` at the interior interfaces, where the runner reads Kzz, [nz-1].

    "scale" keeps the config's profile shape. "powerlaw" anchors Kzz0 at ``kzz_p_ref_bar``, log-log interpolated on the
    config's profile, and tilts it by (P / P_ref)^-slope, so lnKzz = 0, slope = 0 is a constant Kzz equal to the
    profile's value there.
    """
    from vulcan3.atm_setup import compute_pico

    _p_int_bar = (
        np.asarray(compute_pico(np.asarray(atm.pco, dtype=np.float64)), dtype=np.float64)[1:-1] / constants.BAR_CGS
    )
    _kzz_ref0 = _p_over_ref = None
    if kzz_mode == "powerlaw":
        kzz_p_ref_bar = float(profile["kzz_p_ref_bar"])
        if not (_p_int_bar.min() <= kzz_p_ref_bar <= _p_int_bar.max()):
            raise ValueError(
                f"kzz_p_ref_bar={kzz_p_ref_bar:g} lies outside the Kzz grid "
                f"[{_p_int_bar.min():.3g}, {_p_int_bar.max():.3g}] bar"
            )
        _order = np.argsort(_p_int_bar)
        _kzz_ref0 = float(
            np.exp(
                np.interp(
                    np.log(kzz_p_ref_bar),
                    np.log(_p_int_bar[_order]),
                    np.log(np.asarray(Kzz0, dtype=np.float64)[_order]),
                )
            )
        )
        _p_over_ref = jnp.asarray(_p_int_bar / kzz_p_ref_bar)
        logger.info("[chem] kzz_mode=powerlaw: Kzz0(%g bar) = %.3e cm^2/s, slope from theta", kzz_p_ref_bar, _kzz_ref0)

    def _kzz_eff(lnKzz: float | Array, kzz_slope: float | Array) -> Array:
        """Return the effective Kzz, cm^2/s."""
        if kzz_mode == "powerlaw":
            return _kzz_ref0 * jnp.exp(lnKzz) * _p_over_ref ** (-kzz_slope)  # type: ignore[operator]  # set for powerlaw
        return Kzz0 * jnp.exp(lnKzz)

    return _kzz_eff


def _make_runner(cfg: Any, rs: Any, para: Any) -> Any:
    """Return vulcan3's outer loop on a fresh Rosenbrock solver, with the pre-loop's photolysis statics."""
    import vulcan3.legacy_io as op
    from vulcan3 import op_jax, outer_loop

    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver.photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
    solver.naming_solver(para)
    return integ


def _checked_kzz_mode(profile: Profile, chem_extras: tuple[str, ...]) -> str:
    """Return the profile's Kzz mode, refusing an unknown one and a slope only the power law would read."""
    kzz_mode = str(profile.get("kzz_mode", "scale"))
    if kzz_mode not in ("scale", "powerlaw"):
        raise ValueError(f"kzz_mode={kzz_mode!r}: expected 'scale' or 'powerlaw'")
    if "kzz_p_ref_bar" in profile and kzz_mode != "powerlaw":
        raise ValueError(
            f"kzz_p_ref_bar is set but kzz_mode is {kzz_mode!r}: only the power law reads the reference pressure, so "
            "it would change nothing. Drop the key."
        )
    if kzz_mode == "powerlaw" and "kzz_p_ref_bar" not in profile:
        raise ValueError(
            "kzz_mode 'powerlaw' needs kzz_p_ref_bar, the pressure in bar where the Kzz profile is read and the slope "
            "pivots: it belongs to the case and has no default."
        )
    if "kzz_slope" in chem_extras and kzz_mode != "powerlaw":
        raise ValueError(
            "chem_extras carries 'kzz_slope' but kzz_mode is "
            f"{kzz_mode!r}: the slope is read only by the power law, so "
            "it would be a silent no-op"
        )
    return kzz_mode


def _refuse_surface_h2o(cfg: Any, use_condense: bool) -> None:
    """Refuse the one inert condensation configuration vulcan3's runtime validation cannot see."""
    if use_condense and bool(cfg.use_sat_surfaceH2O):
        raise NotImplementedError(
            "use_condense with use_sat_surfaceH2O=True is unsupported in the "
            "T-varying model: it rewrites the fixed-bottom H2O boundary "
            "condition from the STRUCTURAL temperature at ini time, which a "
            "live T(P) does not rebuild. Disable use_sat_surfaceH2O."
        )


def _checked_conden_spec(cfg: Any, var: Any, atm: Any, sidx: dict[str, int]) -> Any:
    """Return the static condensation metadata ``_prep`` rebuilds the dynamic arrays from, refusing an inert species."""
    from vulcan3 import conden

    conden_spec = conden.make_conden_spec(cfg, var, atm, sidx)
    relax_set = set(cfg.use_relax or [])
    inert = [sp for sp in cfg.condense_sp if sp not in conden_spec.gas_names and sp not in relax_set]
    if inert:
        raise ValueError(
            f"condense_sp entries {inert} have no condensation reaction in "
            f"network {cfg.network!r} and are not in use_relax -- they would "
            "silently not condense. Remove them or add the reaction/relax."
        )
    logger.info(
        "[chem] condensation ON: kinetics rows %s, relax H2O=%s NH3=%s, fix_species=%s; "
        "conden arrays rebuilt on-graph at each proposed T",
        list(conden_spec.gas_names),
        conden_spec.h2o_active,
        conden_spec.nh3_active,
        list(conden_spec.fix_names),
    )
    return conden_spec


def build_chem_model(
    profile: Profile,
    tp_eval: Callable[[tuple[float, ...] | Array, Array], Array] | None = None,
    n_tp_params: int = 0,
    chem_extras: Iterable[str] = (),
) -> SimpleNamespace:
    """Build the chemistry model for one profile.

    Parameters
    ----------
    profile : dict
        Settings, with keys in ``constants.PROFILE_KEYS``. ``network``, ``atom_list``, ``vulcan_cfg_name``,
        ``use_photo`` and ``yconv_cri`` are required.
    tp_eval : callable, optional
        ``tp_eval(theta[k:k+n_tp_params], p_bar)`` returns T(P), K, on the bar grid, with k = 3 + len(chem_extras).
        None: T is the baseline profile plus a uniform offset ``theta[k]``, K.
    n_tp_params : int, optional
        Entries ``tp_eval`` consumes.
    chem_extras : sequence of str, optional
        The optional entries of the vector between ``lnKzz`` and the temperature block, a subset of ``CHEM_EXTRAS``
        in order.

    Returns
    -------
    types.SimpleNamespace
        The solves ``converged_y``, ``converged_y_batch``, ``converged_y_queue``, ``converged_y_jvp``, ``run_diag``; the
        checks ``conv_normal_at_exit``, ``audit_init``, ``prep_pv``, ``co_bz_bound``, ``co_bz_margin``; and the build's
        data ``y0``, ``T_base`` (K), ``p_bar``, ``dz`` (cm), ``sidx``, ``species_masses`` (g/mol), ``compo_array``,
        ``atom_list``, ``nz``, ``ni``, ``chem_extras``, ``kzz_mode``, ``count_max``, ``yconv_min``, ``conden_spec``.

    Raises
    ------
    ValueError
        On an unknown key or mode, unsupported T-dependent cross sections, or a condensing species with no reaction.
        Condensation with ``use_sat_surfaceH2O``, which a live T(P) does not rebuild, raises ``NotImplementedError``.
    """
    t0 = time.time()
    constants.check_profile_keys(profile)
    chem_extras = _check_extras(chem_extras)
    kzz_mode = _checked_kzz_mode(profile, chem_extras)
    cfg = _configured_cfg(profile)

    from vulcan3 import atm_jax, composition, network as net_mod
    from vulcan3.gibbs import load_nasa9
    from vulcan3.ini_abun import ratio_indices
    from vulcan3.jax_step import make_atm_static
    from vulcan3.paths import resolve_data_path
    from vulcan3.state import RunState, legacy_view

    from vulcan_forward._chem_model import _CarrySeed, _ChemSetup, _ChemSolves

    _assert_composition_tables(composition)

    use_condense = bool(cfg.use_condense)
    _refuse_surface_h2o(cfg, use_condense)
    network_file = resolve_data_path(cfg.network)
    thermo_dir = _thermo_dir(network_file)
    _log_inputs(network_file, cfg, profile)

    rs = RunState.with_pre_loop_setup(cfg)
    var, atm, para = legacy_view(rs)
    if cfg.use_photo:
        _check_stellar_span(cfg, var)
    network = net_mod.parse_network(str(network_file))
    nz, ni = atm.Tco.shape[0], network.ni
    sidx = dict(network.species_idx)
    conden_spec = _checked_conden_spec(cfg, var, atm, sidx) if use_condense else None

    p_bar = np.asarray(atm.pco, dtype=np.float64) / constants.BAR_CGS
    nasa9, _ = load_nasa9(network.species, thermo_dir)

    integ = _make_runner(cfg, rs, para)
    logger.info("[chem] setup %.1fs; nz=%d ni=%d photo=%s", time.time() - t0, nz, ni, cfg.use_photo)
    integ.ensure_runner(var, atm)

    atm_static = make_atm_static(atm, ni, nz, cfg=integ.config)
    state0 = integ.pack_state(rs)
    T_base = jnp.asarray(np.asarray(atm.Tco, dtype=np.float64))
    kzz_eff = _make_kzz_eff(kzz_mode, profile, atm, atm_static.Kzz)

    # refresh_static feeds the runner's own hydrostatic-refresh kernel; phys0 and spec_atm feed atm_jax.mol_diff, the
    # on-graph Dzz(T, M) builder.
    refresh_static = integ.build_refresh_static(atm)
    phys0, spec_atm = atm_jax.make_physical_inputs(cfg, var, atm, list(network.species))

    compo = np.asarray(composition.compo_array)
    # The two modes differ only in a warm continuation's guess; the projection puts both on the same targets.
    co_mode = str(profile.get("co_mode", "fixed_O"))
    if co_mode not in ("proxy", "fixed_O"):
        raise ValueError(f"co_mode={co_mode!r}: expected 'proxy' or 'fixed_O'")
    atomic_masses = jnp.asarray(np.asarray(constants.ATOMIC_MASSES, dtype=np.float64))
    species_masses = jnp.asarray(np.asarray(compo, dtype=np.float64)) @ atomic_masses  # (ni,)

    y0 = np.asarray(state0.y, dtype=np.float64)
    elem = _build_elemental(compo, sidx, y0, ni, co_mode == "fixed_O")

    setup = _ChemSetup(
        cfg=cfg,
        network=network,
        nasa9=nasa9,
        remove_list=cfg.remove_list,
        conden_spec=conden_spec,
        tp_eval=tp_eval,
        n_tp_params=n_tp_params,
        chem_extras=chem_extras,
        kzz_eff=kzz_eff,
        elem=elem,
        ratio_idx=ratio_indices([e for e, _ in elem.pairs]),
        T_base=T_base,
        pco=jnp.asarray(np.asarray(atm.pco, dtype=np.float64)),
        p_bar=jnp.asarray(p_bar),
        compo_run=jnp.asarray(np.asarray(integ.compo_arr, dtype=np.float64)),
        state0=state0,
        atm_static=atm_static,
        refresh_static=refresh_static,
        phys0=phys0,
        spec_atm=spec_atm,
        carry=_CarrySeed.from_cfg(cfg),
    )
    solves = _ChemSolves(setup, integ)

    return SimpleNamespace(
        run_diag=solves.run_diag,
        converged_y=solves.converged_y,
        converged_y_batch=solves.converged_y_batch,
        converged_y_queue=solves.converged_y_queue,
        converged_y_jvp=solves.converged_y_jvp,
        conv_normal_at_exit=setup.conv_normal_at_exit,
        audit_init=setup.audit_init,
        conden_spec=conden_spec,
        prep_pv=setup.prep_pv,
        _integ=integ,
        co_bz_bound=elem.margin(y0),  # the build's initial column
        co_bz_margin=elem.margin,
        y0=y0,
        compo_array=compo,
        atom_list=tuple(composition.atom_list),
        T_base=np.asarray(T_base),
        p_bar=p_bar,
        dz=np.asarray(atm.dz, dtype=np.float64),
        sidx=sidx,
        species_masses=species_masses,
        nz=nz,
        ni=ni,
        chem_extras=chem_extras,
        kzz_mode=kzz_mode,
        count_max=int(cfg.count_max),
        yconv_min=float(cfg.yconv_min),
    )
