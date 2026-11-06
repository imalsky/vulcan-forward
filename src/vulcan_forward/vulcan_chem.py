# size-limit-ok: one closure per solve, with the numpy docs of the public ones
"""The differentiable map from chemistry parameters to the converged column, on vulcan3.

``build_chem_model`` runs the one-time pre-loop and returns a model whose ``converged_y(theta)`` re-converges the column
for ``theta = [lnZ, c_o, lnKzz, *extras, *tp]``. Rates, structure and condensation arrays are rebuilt on-graph per
proposal; the runner's ``lax.while_loop`` supports jvp but not vjp, so forward mode is the end-to-end route.

This module owns the first jax import: it sets the import-frozen ``VULCAN3_*`` variables and jax x64, and refuses to
follow exojax. docs/method.md describes the composition map and the limits of a condensing column.
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
    from jax.typing import ArrayLike

    Profile = dict[str, Any]

__all__ = ["CHEM_EXTRAS", "ChemParams", "ConvDiag", "build_chem_model", "bz_margin", "params_from_vector"]

logger = logging.getLogger(__name__)

if "exojax" in sys.modules:
    raise RuntimeError(
        "vulcan_forward.vulcan_chem must be imported BEFORE exojax: it fixes "
        "jax x64 and the VULCAN3_* import-frozen env vars. Import order: "
        "vulcan_forward.vulcan_chem, then vulcan_forward.exojax_rt."
    )

# vulcan3 freezes network and atom_list at its first import, so arriving after it makes the setdefault below a no-op.
_WANT_ENV = {"VULCAN3_NETWORK": constants.DEFAULT_NETWORK, "VULCAN3_ATOM_LIST": constants.DEFAULT_ATOM_LIST}
if "vulcan3" in sys.modules:
    _conflict = {k: (os.environ.get(k), v) for k, v in _WANT_ENV.items() if os.environ.get(k) not in (None, v)}
    if not _conflict:
        _frozen = getattr(sys.modules["vulcan3"], "chem_funs", None)
        _frozen = getattr(getattr(_frozen, "NETWORK", None), "network_path", None)
        if _frozen and Path(_frozen).name != Path(constants.DEFAULT_NETWORK).name:
            _conflict = {"VULCAN3_NETWORK": (_frozen, constants.DEFAULT_NETWORK)}
    if _conflict:
        raise RuntimeError(
            "vulcan3 was imported before vulcan_forward.vulcan_chem, so its "
            "network/atom_list are already frozen and cannot be changed: "
            + "; ".join(f"{k} is {got!r} but this engine needs {want!r}" for k, (got, want) in _conflict.items())
            + ". Import vulcan_forward.vulcan_chem first, or set those env vars "
            "before importing vulcan3."
        )

# setdefault: a caller driving another network sets these itself.
for _k, _v in _WANT_ENV.items():
    os.environ.setdefault(_k, _v)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from vulcan_forward._chem_layout import CHEM_EXTRAS, ChemParams, ConvDiag, _check_extras, params_from_vector
from vulcan_forward._elemental import _assert_composition_tables, _build_elemental, bz_margin


def _configured_cfg(profile: Profile) -> Any:
    """Load the VULCAN config the profile names and apply the profile to it, refusing what the rebuild cannot follow."""
    import vulcan3

    cfg = vulcan3.load_config(profile.get("vulcan_cfg_name") or constants.DEFAULT_CFG_NAME)
    cfg.use_print_prog = False
    cfg.use_photo = bool(profile["use_photo"])
    cfg.yconv_cri = float(profile["yconv_cri"])
    # is not None: count_min=0 and yconv_min=0 are legitimate values.
    for key, cast in (("nz", int), ("count_min", int), ("count_max", int), ("dt_max", float), ("yconv_min", float)):
        if profile.get(key) is not None:
            setattr(cfg, key, cast(profile[key]))
    # setattr checks nothing, so a removed or misspelled key is refused first.
    vulcan3.validate_overrides(profile.get("cfg_overrides") or {})
    for _k, _v in (profile.get("cfg_overrides") or {}).items():
        setattr(cfg, _k, _v)

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
        kzz_p_ref_bar = float(profile.get("kzz_p_ref_bar", constants.KZZ_P_REF_BAR))
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


# size-limit-ok: the closures of one build share its arrays; a split would pass them all as arguments
def build_chem_model(  # noqa: C901  # counts its closures
    profile: Profile,
    tp_eval: Callable[[tuple[float, ...] | Array, Array], Array] | None = None,
    n_tp_params: int = 0,
    chem_extras: Iterable[str] = (),
) -> SimpleNamespace:
    """Build the chemistry model for one profile.

    Parameters
    ----------
    profile : dict
        Settings, with keys in ``constants.PROFILE_KEYS``. ``use_photo`` and ``yconv_cri`` are required.
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
    NotImplementedError
        If condensation meets ``use_sat_surfaceH2O``, which a live T(P) does not rebuild.
    """
    t0 = time.time()
    constants.check_profile_keys(profile)
    chem_extras = _check_extras(chem_extras)
    kzz_mode = str(profile.get("kzz_mode", "scale"))
    if kzz_mode not in ("scale", "powerlaw"):
        raise ValueError(f"kzz_mode={kzz_mode!r}: expected 'scale' or 'powerlaw'")
    if "kzz_slope" in chem_extras and kzz_mode != "powerlaw":
        raise ValueError(
            "chem_extras carries 'kzz_slope' but kzz_mode is "
            f"{kzz_mode!r}: the slope is read only by the power law, so "
            "it would be a silent no-op"
        )
    import vulcan3

    cfg = _configured_cfg(profile)

    import vulcan3.legacy_io as op
    from vulcan3 import (
        atm_jax,
        atm_refresh as atm_refresh_mod,
        composition,
        conden as conden_mod,
        network as net_mod,
        op_jax,
        outer_loop,
        rates_jax,
    )
    from vulcan3.atm_setup import VISCOSITY_TABLE, settling_velocity_jax
    from vulcan3.gibbs import load_nasa9
    from vulcan3.ini_abun import column_atoms, element_vector, eq_seed, ratio_indices
    from vulcan3.jax_step import make_atm_static
    from vulcan3.paths import resolve_data_path
    from vulcan3.phy_const import kb
    from vulcan3.state import RunState, legacy_view

    _assert_composition_tables(composition)

    # vulcan3's runtime validation refuses the inert condensation configurations; this is the one it cannot see.
    use_condense = bool(cfg.use_condense)
    if use_condense and bool(cfg.use_sat_surfaceH2O):
        raise NotImplementedError(
            "use_condense with use_sat_surfaceH2O=True is unsupported in the "
            "T-varying model: it rewrites the fixed-bottom H2O boundary "
            "condition from the STRUCTURAL temperature at ini time, which a "
            "live T(P) does not rebuild. Disable use_sat_surfaceH2O."
        )

    rs = RunState.with_pre_loop_setup(cfg)
    var, atm, para = legacy_view(rs)
    network = net_mod.parse_network(str(resolve_data_path(cfg.network)))
    nz, ni = atm.Tco.shape[0], network.ni
    sidx = dict(network.species_idx)

    # Static condensation metadata; _prep rebuilds the dynamic arrays from it at every proposed T.
    conden_spec = None
    if use_condense:
        conden_spec = conden_mod.make_conden_spec(cfg, var, atm, sidx)
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

    pco = jnp.asarray(np.asarray(atm.pco, dtype=np.float64))
    p_bar = np.asarray(atm.pco, dtype=np.float64) / constants.BAR_CGS
    p_bar_j = jnp.asarray(p_bar)
    thermo_dir = resolve_data_path(cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = Path(vulcan3.__file__).resolve().parent / "thermo"
    nasa9, _ = load_nasa9(network.species, thermo_dir)
    remove_list = cfg.remove_list

    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver.photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
    solver.naming_solver(para)
    logger.info("[chem] setup %.1fs; nz=%d ni=%d photo=%s", time.time() - t0, nz, ni, cfg.use_photo)
    integ.ensure_runner(var, atm)

    # The termination budget and diffusion blend live on the carry, and state0 is packed once under the cold statics,
    # so each solve re-seeds them. Under the hybrid scheme a warm continuation starts from a column converged on the
    # central operator, so it continues on it.
    count_max_v = int(cfg.count_max)
    count_min_v = int(cfg.count_min)
    runtime_v = float(cfg.runtime)
    use_vm_mol_v = bool(cfg.use_vm_mol)
    hybrid_v = use_vm_mol_v and bool(cfg.use_hybrid_vm_mol)
    _warm_note = "; warm continuation pinned to central difference (the converged phase-1 operator)" if hybrid_v else ""
    logger.info("[chem] diffusion scheme: use_vm_mol=%s hybrid=%s%s", use_vm_mol_v, hybrid_v, _warm_note)

    def _seeded_prep(
        theta: ChemParams | Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> tuple[Any, Any]:
        """Return ``_prep`` with the carry's termination budget and diffusion blend seeded for the runner."""
        init, atm_T = _prep(theta, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref, s_o_ref=s_o_ref)
        blend = 1.0 if use_vm_mol_v else 0.0
        if warm_y is not None and hybrid_v:
            blend = 0.0
        return init._replace(
            hybrid_use_vm=jnp.float64(blend),
            count_min_dyn=jnp.int32(count_min_v),
            count_max_dyn=jnp.int32(count_max_v),
            runtime_dyn=jnp.float64(runtime_v),
        ), atm_T

    def _conv_diag(final: Any, tangent_ok: Array | bool = True, tangent_longdy: Array | float = jnp.nan) -> ConvDiag:
        """Read ``ConvDiag`` off the runner's exit carry.

        ``conv_normal`` is ``vulcan3.conv_normal`` AND ``tangent_ok``, the ``run_jvp`` sensitivity certificate.
        ``conv_branch``
        stays the column's own branch, so a consumer can tell an unsettled sensitivity from an uncertified column.
        """
        ok, branch = vulcan3.conv_normal(final, cfg)
        flat = jnp.argmax(final.where_varies_most)
        return ConvDiag(
            accept_count=final.accept_count,
            longdy=final.longdy,
            longdydt=final.longdydt,
            conv_normal=ok & tangent_ok,
            aflux_change=final.aflux_change,
            conv_branch=branch,
            cell_species=(flat % ni).astype(jnp.int32),
            cell_layer=(flat // ni).astype(jnp.int32),
            cell_vmr=final.ymix.reshape(-1)[flat],
            t=final.t,
            dt=final.dt,
            tangent_longdy=jnp.float64(tangent_longdy),
        )

    def _conv_normal_at_exit(final: Any) -> Array:
        """Return the certificate of a raw ``run_diag`` final carry."""
        return _conv_diag(final).conv_normal

    atm_static = make_atm_static(atm, ni, nz, cfg=integ.config)
    state0 = integ.pack_state(rs)
    y0 = state0.y
    Kzz0 = atm_static.Kzz
    pv0 = state0.pv
    T_base = jnp.asarray(np.asarray(atm.Tco, dtype=np.float64))

    _kzz_eff = _make_kzz_eff(kzz_mode, profile, atm, Kzz0)

    # refresh_static feeds the runner's own hydrostatic-refresh kernel; phys0 and spec_atm feed atm_jax.mol_diff, the
    # on-graph Dzz(T, M) builder.
    refresh_static = integ.build_refresh_static(atm)
    phys0, spec_atm = atm_jax.make_physical_inputs(cfg, var, atm, list(network.species))
    use_vm = bool(spec_atm.use_vm_mol and spec_atm.use_moldiff)
    use_set = bool(spec_atm.use_settling and spec_atm.use_moldiff)

    compo = np.asarray(composition.compo_array)
    # The two modes differ only in a warm continuation's guess; the projection puts both on the same targets.
    co_mode = str(profile.get("co_mode", "fixed_O"))
    if co_mode not in ("proxy", "fixed_O"):
        raise ValueError(f"co_mode={co_mode!r}: expected 'proxy' or 'fixed_O'")
    co_fixed_o = co_mode == "fixed_O"
    atomic_masses = jnp.asarray(np.asarray(constants.ATOMIC_MASSES, dtype=np.float64))
    species_masses = jnp.asarray(np.asarray(compo, dtype=np.float64)) @ atomic_masses  # (ni,)
    # The runner's own (ni, n_atoms) table, for atom_ini in the runner's basis.
    compo_run = jnp.asarray(np.asarray(integ.compo_arr, dtype=np.float64))

    _y0_np = np.asarray(y0, dtype=np.float64)
    elem = _build_elemental(compo, sidx, _y0_np, ni, co_fixed_o)
    _targets, _elemental_project, _guess_y0, co_bz_margin = elem.targets, elem.project, elem.guess, elem.margin
    elem_pairs, E_np, R0_ratios, _names = elem.pairs, elem.E_np, elem.R0_ratios, elem.names
    _zk, _ck, _sk = elem.kinds

    # The cold-start seed is the network's Gibbs equilibrium at the proposal's T-P and column ratios. Its tangent is
    # zero by construction (vulcan3 carries the custom_jvp); lnZ and c_o tangents enter through the projection.
    _ratio_idx = ratio_indices([e for e, _ in elem_pairs])

    def _eq_seed(T: Array, ratios: Array, M: Array) -> Array:
        """Return the equilibrium column as absolute densities, [nz, ni]; ``eq_seed`` returns mixing ratios."""
        return eq_seed(T, p_bar_j, element_vector(ratios, _ratio_idx)) * M[:, None]

    co_bz_bound = co_bz_margin(_y0_np)  # the build's initial column

    def _as_params(p: ChemParams | Array) -> ChemParams:
        """Accept named ``ChemParams`` or a positional vector."""
        if isinstance(p, ChemParams):
            return p
        return params_from_vector(p, n_tp_params, has_tp_eval=tp_eval is not None, extras=chem_extras)

    # size-limit-ok: one proposal's state, in program order
    def _prep(
        theta: ChemParams | Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> tuple[Any, Any]:
        """Build the initial runner state and atmosphere for one proposal.

        With ``warm_y``, start from that converged column. The projection fixes each layer's total at M and the
        layer-summed column ratios to H at the theta targets, not the dz-weighted inventory the element budget tracks,
        so two starts at one theta can carry different weighted inventories.
        """
        # One column per solve: a (1, ni) column would broadcast and seed every layer from one layer. Shapes are static,
        # so this runs under jit and vmap.
        if warm_y is not None and jnp.shape(warm_y) != (nz, ni):
            raise ValueError(
                f"warm_y has shape {tuple(jnp.shape(warm_y))}: one converged "
                f"column of shape ({nz}, {ni}) is expected per solve "
                f"(batched callers pass ({nz}, {ni}) per lane, i.e. "
                f"(N, {nz}, {ni}) stacked)."
            )
        _p = _as_params(theta)
        lnZ, c_o, lnKzz, s_o = _p.lnZ, _p.c_o, _p.lnKzz, _p.s_o

        T = T_base + _p.tp[0] if tp_eval is None else tp_eval(_p.tp, p_bar_j)
        M = pco / (kb * T)
        k_arr = rates_jax.build_rate_array(
            network, T, M, nasa9, remove_list, use_lowT_caps=bool(cfg.use_lowT_limit_rates)
        )
        Ti = 0.5 * (T[:-1] + T[1:])
        Kzz_eff = _kzz_eff(lnKzz, _p.kzz_slope)

        if warm_y is None:
            y0p = _eq_seed(T, _targets(lnZ, c_o, s_o), M)
        else:
            y0p = _guess_y0(lnZ, c_o, s_o, warm_y, lnZ_ref, c_o_ref, s_o_ref)

        # atom_ini comes from the repaired column, so the conservation anchor matches the initial gas.
        y0p, _min_adj = _elemental_project(y0p, M, lnZ, c_o, s_o)
        ymix0 = y0p / M[:, None]
        atom_ini_new = jnp.einsum("zi,ia->a", y0p, compo_run)  # runner atom order
        pv_T = pv0._replace(n_0=M, r_Tco=T, Kzz=Kzz_eff, atom_ini=atom_ini_new)

        # Geometry from the runner's own refresh kernel, so the initial carry equals what the in-loop refresh maintains.
        # The runner recomputes vm in-loop from atm.Dzz, so rebuilding Dzz fixes the whole molecular-diffusion channel.
        refresh_lane = refresh_static._replace(Tco=T)
        mu_i, g_i, Hp_i, dz_i, zco_i, dzi_i, Hpi_i = atm_refresh_mod.update_mu_dz_jax(ymix0, refresh_lane)
        Dzz_new, _Dzz_cen, vm_new = atm_jax.mol_diff(phys0._replace(Tco=T), spec_atm, M, g_i, Hp_i, dz_i)
        if not use_vm:
            vm_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        if use_set:
            _na, _a, _b = VISCOSITY_TABLE[spec_atm.atm_base]
            vs_new = settling_velocity_jax(_na, _a, _b, T, g_i, spec_atm.settle_coeff)
        else:
            vs_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        pv_T = pv_T._replace(r_Dzz_top=Dzz_new[-1], y_ini=y0p)

        # The carry holds baseline-T condensation values.
        if conden_spec is not None:
            cprof = conden_mod.build_conden_profile(conden_spec, T, pco, M, Dzz_new)
            pv_T = pv_T._replace(
                c_Dg_per_re=cprof.Dg_per_re,
                c_sat_n_per_re=cprof.sat_n_per_re,
                c_h2o_Dg=cprof.h2o_Dg,
                c_h2o_sat=cprof.h2o_sat,
                c_nh3_Dg=cprof.nh3_Dg,
                c_nh3_sat=cprof.nh3_sat,
                c_nh3_conden_top=cprof.nh3_conden_top,
                fix_species_sat_mix=cprof.fix_species_sat_mix,
            )
        atm_T = atm_static._replace(
            Tco=T, Ti=Ti, M=M, Kzz=Kzz_eff, Dzz=Dzz_new, vm=vm_new, vs=vs_new, g=g_i, dzi=dzi_i, Hpi=Hpi_i
        )

        # y_prev is the revert target on a rejected step and the state the element budget differences against; state0
        # carries the baseline column there. budget_ref starts from this theta's column, so the proposal's composition
        # change is not charged to the solver's conservation.
        init = state0._replace(
            y=y0p,
            y_prev=y0p,
            ymix=ymix0,
            k_arr=k_arr,
            pv=pv_T,
            mu=mu_i,
            g=g_i,
            Hp=Hp_i,
            dz=dz_i,
            zco=zco_i,
            dzi=dzi_i,
            Hpi=Hpi_i,
            vs=vs_new,
            budget_ref=column_atoms(y0p, dz_i, compo_run),
            budget_err=jnp.zeros_like(state0.budget_err),
            budget_drift=jnp.zeros_like(state0.budget_drift),
        )
        return init, atm_T

    def run_diag(
        theta: ChemParams | Array,
        return_atm: bool = False,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> tuple[Any, ...]:
        """Solve for a diagnostic: return the final carry and the initial state.

        Parameters
        ----------
        theta : ChemParams or jax.Array
            The parameters.
        return_atm : bool, optional
            Also return the theta-dependent ``AtmStatic`` the runner was driven with, the operating point a reverse-mode
            adjoint must linearize around.
        warm_y : jax.Array [nz, ni], optional
            A converged column to continue from; None starts cold from the equilibrium seed.
        lnZ_ref : float, optional
            The ln metallicity multiple ``warm_y`` was converged at.
        c_o_ref : float, optional
            The ln C/O multiple it was converged at.
        s_o_ref : float, optional
            The ln S/O multiple it was converged at.

        Returns
        -------
        tuple
            ``(final, init)``, or ``(final, init, atm)`` with ``return_atm``.
        """
        init, atm_T = _seeded_prep(theta, warm_y, lnZ_ref, c_o_ref, s_o_ref)
        final = integ.runner(init, atm_T)
        return (final, init, atm_T) if return_atm else (final, init)

    def prep_pv(theta: ChemParams | Array) -> Any:
        """Return the initial-carry ``ProfileVars`` for ``theta``, without a solve, for tests and diagnostics.

        Parameters
        ----------
        theta : ChemParams or jax.Array
            The parameters.

        Returns
        -------
        ProfileVars
            The per-proposal arrays (n_0, Kzz, atom_ini and, with condensation on, the rebuilt c_* arrays).
        """
        init, _atm_T = _prep(theta)
        return init.pv

    def converged_y(
        theta: ChemParams | Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        return_conv_diag: bool = False,
        s_o_ref: Array | float = 0.0,
    ) -> Array | tuple[Array, ConvDiag]:
        """Return the converged column, forward-mode differentiable in ``theta``.

        Under the hybrid vm_mol scheme a warm continuation runs on the central operator instead of re-entering upwind
        phase 0.

        Parameters
        ----------
        theta : ChemParams or jax.Array
            The parameters.
        warm_y : jax.Array [nz, ni], optional
            A converged column to continue from; None starts cold from the equilibrium seed.
        lnZ_ref : float, optional
            The ln metallicity multiple ``warm_y`` was converged at.
        c_o_ref : float, optional
            The ln C/O multiple it was converged at.
        return_conv_diag : bool, optional
            Also return the ``ConvDiag``, a free read off the primal carry whose float fields carry a tangent.
        s_o_ref : float, optional
            The ln S/O multiple it was converged at.

        Returns
        -------
        jax.Array [nz, ni], or tuple
            Absolute number densities, cm^-3; with ``return_conv_diag``, ``(y, ConvDiag)``.
        """
        init, atm_T = _seeded_prep(theta, warm_y, lnZ_ref, c_o_ref, s_o_ref)
        final = integ.runner(init, atm_T)
        if return_conv_diag:
            return final.y, _conv_diag(final)
        return final.y

    def _ref_leaves(
        n: int, lnZ_ref: Array | float, c_o_ref: Array | float, s_o_ref: Array | float
    ) -> tuple[Array, ...]:
        """Broadcast the reference composition to per-lane leaves [n] for the vmapped prep."""
        return tuple(jnp.broadcast_to(jnp.asarray(r, dtype=jnp.float64), (n,)) for r in (lnZ_ref, c_o_ref, s_o_ref))

    def converged_y_batch(
        thetas: Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        return_conv_diag: bool = False,
        s_o_ref: Array | float = 0.0,
    ) -> Array | tuple[Array, ConvDiag]:
        """Return the converged columns of a stack of thetas, on the batched runner.

        One while loop sits above the lane vmap (``OuterLoop.run_batch``): photolysis and the geometry refresh follow
        the loop's tick, not the lane's accept count, so a lane is not bit-identical to its solo ``converged_y``; the
        two agree at the convergence scale. A lane's result does not depend on the other lanes. A plain ``jax.jvp``
        through this is supported; a tangent-certified derivative is ``converged_y_jvp``.

        Parameters
        ----------
        thetas : jax.Array [N, n_theta]
            The parameters.
        warm_y : jax.Array [N, nz, ni], optional
            One converged column per lane; None starts cold from the equilibrium seed.
        lnZ_ref : float or jax.Array [N], optional
            The ln metallicity multiple each ``warm_y`` was converged at; a scalar is shared. Never differentiated.
        c_o_ref : float or jax.Array [N], optional
            The ln C/O multiple, likewise.
        return_conv_diag : bool, optional
            Also return the per-lane ``ConvDiag``.
        s_o_ref : float or jax.Array [N], optional
            The ln S/O multiple, likewise.

        Returns
        -------
        jax.Array [N, nz, ni], or tuple
            Absolute number densities, cm^-3; with ``return_conv_diag``, ``(y, ConvDiag)``.
        """
        refs = _ref_leaves(int(jnp.shape(thetas)[0]), lnZ_ref, c_o_ref, s_o_ref)

        # warm_y=None is an empty pytree node, so the same vmap covers the cold seed.
        init_b, atm_b = jax.vmap(
            _seeded_prep,
            out_axes=(0, outer_loop.ATM_STATIC_BATCH_AXES),
        )(thetas, warm_y, *refs)
        final_b = integ.run_batch(init_b, atm_b)
        if return_conv_diag:
            return final_b.y, jax.vmap(_conv_diag)(final_b)
        return final_b.y

    # run_queue keys its compiled program on the init_fn and out_fn objects, so fresh closures per call would recompile.
    # The pair depends only on static choices: the reference composition rides the jobs pytree.
    _queue_fns: dict[tuple[bool, bool], tuple[Callable[..., Any], Callable[..., Any]]] = {}

    # size-limit-ok: most of the length is the numpy docs
    def converged_y_queue(
        thetas: Array,
        n_lanes: int,
        *,
        chunk: int | None = None,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        active: ArrayLike | None = None,
        s_o_ref: Array | float = 0.0,
    ) -> tuple[Array, ConvDiag]:
        """Return ``converged_y_batch`` on ``n_lanes`` lanes, refilled from the job queue.

        A lane that certifies is written out and takes the next theta inside the same while loop
        (``OuterLoop.run_queue``), so wall time follows total work over lanes. A refilled theta enters at the tick its
        lane was freed at, which moves the cadence as the batch already does against the solo solve. With ``n_lanes >=
        N`` nothing is refilled, but the result is not bitwise the batch's: the seed is compiled inside the loop.

        Parameters
        ----------
        thetas : jax.Array [N, n_theta]
            The parameters.
        n_lanes : int
            The lane count.
        chunk : int, optional
            Lanes refilled per pass; None keeps vulcan3's.
        warm_y : jax.Array [N, nz, ni], optional
            One converged column per lane; None starts cold from the equilibrium seed.
        lnZ_ref : float or jax.Array [N], optional
            The ln metallicity multiple each ``warm_y`` was converged at; a scalar is shared. Never differentiated.
        c_o_ref : float or jax.Array [N], optional
            The ln C/O multiple, likewise.
        active : array of bool [N], optional
            False marks a placeholder: its lane starts done, takes no step and goes to the next job; the caller discards
            its result. None is the program without the mask.
        s_o_ref : float or jax.Array [N], optional
            The ln S/O multiple, likewise.

        Returns
        -------
        tuple
            ``(y [N, nz, ni], ConvDiag stacked over N)``.

        Raises
        ------
        ValueError
            If ``active`` does not have shape [N].
        """
        masked = active is not None
        key = (warm_y is not None, masked)
        fns = _queue_fns.get(key)
        if fns is None:
            if masked:

                def init_fn(job: tuple[Any, ...]) -> tuple[Any, Any]:
                    """Prep one job for its lane."""
                    *core, act = job
                    init, atm_T = _seeded_prep(*core)
                    return init._replace(is_done=init.is_done | jnp.logical_not(act)), atm_T
            else:

                def init_fn(job: tuple[Any, ...]) -> tuple[Any, Any]:
                    """Prep one job for its lane."""
                    return _seeded_prep(*job)

            def out_fn(final: Any) -> tuple[Array, ConvDiag]:
                """Write one finished lane out.

                The ``ConvDiag`` is a report, never differentiated: with a tangent it would keep the certificate ring's
                tangent alive through the queue's refill cond.
                """
                return final.y, jax.tree_util.tree_map(jax.lax.stop_gradient, _conv_diag(final))

            fns = _queue_fns[key] = (init_fn, out_fn)
        init_fn, out_fn = fns

        n = int(jnp.shape(thetas)[0])
        jobs = (thetas, warm_y, *_ref_leaves(n, lnZ_ref, c_o_ref, s_o_ref))
        if masked:
            act = jnp.asarray(active, dtype=bool)
            if act.shape != (n,):
                raise ValueError(f"converged_y_queue: active must have shape ({n},), got {act.shape}")
            jobs = (*jobs, act)
        chunk_kw = {} if chunk is None else {"chunk": int(chunk)}
        (y, cd), _n_iter = integ.run_queue(init_fn, jobs, int(n_lanes), out_fn, **chunk_kw)
        return y, cd

    def converged_y_jvp(
        theta: ChemParams | Array,
        tangent: ChemParams | Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> tuple[Array, Array, ConvDiag]:
        """Return a forward-mode sensitivity certified by the solver.

        ``dy`` is read when both the column and the tangent pass the certificate (``OuterLoop.run_jvp``): the tangent is
        held, per cell, to the same change-over-lookback tolerance as ``y``, so pass ``tangent`` in
        finite-difference-step units (``e_i * h_i``) and divide ``dy`` by ``h_i``. A plain ``jax.jvp(converged_y)``
        stops when the column certifies, which from a converged warm start is at ``count_min`` with the tangent
        unrelaxed.

        A stacked ``tangent`` rides vulcan3's tangent axis, so the primal is integrated once for all D directions;
        ``jax.vmap``
        of this call would integrate it D times. The run stops when every direction has settled.

        Parameters
        ----------
        theta : ChemParams or jax.Array
            The parameters.
        tangent : ChemParams or jax.Array [n_theta] or [D, n_theta]
            The direction or directions.
        warm_y : jax.Array [nz, ni], optional
            A converged column to continue from; None starts cold from the equilibrium seed.
        lnZ_ref : float, optional
            The ln metallicity multiple ``warm_y`` was converged at.
        c_o_ref : float, optional
            The ln C/O multiple it was converged at.
        s_o_ref : float, optional
            The ln S/O multiple it was converged at.

        Returns
        -------
        tuple
            ``(y, dy, ConvDiag)``: ``dy`` is [nz, ni] or [D, nz, ni], and ``ConvDiag.tangent_longdy`` the worst
            direction's.
        """

        def _seeded(th: ChemParams | Array) -> tuple[Any, Any]:
            """Return the seeded prep of one theta, for the linearization."""
            return _seeded_prep(th, warm_y, lnZ_ref, c_o_ref, s_o_ref)

        def _leaves(p: ChemParams | Array) -> ChemParams | Array:
            """Return theta or a tangent as float64 leaves, named or positional."""
            if isinstance(p, ChemParams):
                return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), p)
            return jnp.asarray(p, dtype=jnp.float64)

        def _stack_dir(*ds: Array) -> Array:
            """Stack one leaf over directions; float0 placeholders are direction-independent."""
            if getattr(ds[0], "dtype", None) == jax.dtypes.float0:
                return ds[0]
            return jnp.stack(ds)

        th, tan = _leaves(theta), _leaves(tangent)
        if not isinstance(tan, ChemParams) and jnp.ndim(tan) == 2:
            # linearize: the primal build runs once.
            (init, atm_T), lin = jax.linearize(_seeded, th)
            dinit, datm = jax.tree_util.tree_map(_stack_dir, *[lin(v) for v in tan])
        else:
            (init, atm_T), (dinit, datm) = jax.jvp(_seeded, (th,), (tan,))
        final, dfinal, tl, ok = integ.run_jvp(init, atm_T, dinit, datm)
        return final.y, dfinal.y, _conv_diag(final, tangent_ok=ok, tangent_longdy=jnp.max(tl))

    def audit_init(
        theta: ChemParams | Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> dict[str, Any]:
        """Audit the initial column built for ``theta``, on the host and off any AD path.

        Parameters
        ----------
        theta : ChemParams or jax.Array
            The parameters.
        warm_y : jax.Array [nz, ni], optional
            A converged column to continue from; None starts cold from the equilibrium seed.
        lnZ_ref : float, optional
            The ln metallicity multiple ``warm_y`` was converged at.
        c_o_ref : float, optional
            The ln C/O multiple it was converged at.
        s_o_ref : float, optional
            The ln S/O multiple it was converged at.

        Returns
        -------
        dict
            ``density_closure_max_rel``, the achieved, baseline and target ``*_ratios_to_H``, ``dln_CO_achieved`` and
            with sulfur ``dln_SO_achieved``, ``ratio_max_rel_err``, ``min_repair_factor`` (must be > 0) and
            ``atom_ini_max_rel_err``.
        """
        _p = _as_params(theta)
        lnZ, c_o, s_o = float(_p.lnZ), float(_p.c_o), float(_p.s_o)
        init, _atm_T = _prep(theta, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref, s_o_ref=s_o_ref)
        y = np.asarray(init.y, dtype=np.float64)
        Mn = np.asarray(init.pv.n_0, dtype=np.float64)
        A = (y @ E_np).sum(axis=0)
        ratios = A[1:] / A[0]
        iC, iO = _names.index("C"), _names.index("O")
        tg = R0_ratios * np.exp(lnZ * _zk + c_o * _ck + s_o * _sk)
        out = {
            "density_closure_max_rel": float(np.max(np.abs(y.sum(axis=1) - Mn) / Mn)),
            "ratios_to_H": dict(zip(_names, ratios.tolist(), strict=True)),
            "baseline_ratios_to_H": dict(zip(_names, R0_ratios.tolist(), strict=True)),
            "dln_CO_achieved": float(np.log((ratios[iC] / ratios[iO]) / (R0_ratios[iC] / R0_ratios[iO]))),
            "target_ratios_to_H": dict(zip(_names, tg.tolist(), strict=True)),
            "ratio_max_rel_err": float(np.max(np.abs(ratios / tg - 1.0))),
        }
        if "S" in _names:
            iS = _names.index("S")
            out["dln_SO_achieved"] = float(np.log((ratios[iS] / ratios[iO]) / (R0_ratios[iS] / R0_ratios[iO])))
        # Project the raw guess: projecting the already-repaired y would always report a factor near 1.
        if warm_y is None:
            y_guess = _eq_seed(init.pv.r_Tco, jnp.asarray(tg), jnp.asarray(Mn))
        else:
            y_guess = _guess_y0(lnZ, c_o, s_o, warm_y, lnZ_ref, c_o_ref, s_o_ref)
        _yg, min_adj = _elemental_project(y_guess, jnp.asarray(Mn), lnZ, c_o, s_o)
        out["min_repair_factor"] = float(min_adj)
        ai = np.asarray(init.pv.atom_ini, dtype=np.float64)
        a_run = y @ np.asarray(integ.compo_arr, dtype=np.float64)
        out["atom_ini_max_rel_err"] = float(np.max(np.abs(a_run.sum(axis=0) - ai) / ai))
        return out

    return SimpleNamespace(
        run_diag=run_diag,
        converged_y=converged_y,
        converged_y_batch=converged_y_batch,
        converged_y_queue=converged_y_queue,
        converged_y_jvp=converged_y_jvp,
        conv_normal_at_exit=_conv_normal_at_exit,
        audit_init=audit_init,
        conden_spec=conden_spec,
        prep_pv=prep_pv,
        _integ=integ,
        co_bz_bound=co_bz_bound,
        co_bz_margin=co_bz_margin,
        y0=np.asarray(y0, dtype=np.float64),
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
