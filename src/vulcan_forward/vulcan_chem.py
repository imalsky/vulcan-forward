"""The differentiable physics-parameters -> converged-VMR map on VULCAN-JAX.

``build_chem_model(profile)`` runs the one-time pre-loop + a warm-up convergence
(compiles and caches the JIT'd inner runner), then returns a model whose
``converged_y(theta)`` re-converges the column as a function of
``theta = [lnZ, c_o, lnKzz, T...]``.

Abundance knobs: the initial column is renormalized to sum_i n_i = M per layer
and repaired (fixed Newton-style iterations on He/H2O/CO/N2/H2S) so the column
elemental ratios hit the theta targets exactly (He/H fixed; O/N/S x Z; C x Z
e^{c_o}). ``pv.atom_ini`` is rebuilt from the repaired column. Residuals ~1e-8
relative; measure with ``audit_init``.

Rate constants and the T/composition-dependent structure (Dzz + vm/vs, pv.Kzz,
the initial carry geometry) are rebuilt on-graph per proposal. Condensation
follows the live T(P) too: ``_prep`` rebuilds every T-dependent conden array
per proposal, and unsupported conden configs refuse at build. The cold-trap
index and active-layer set are discrete, so a jvp through a condensing state
is valid only away from those switches.

KNOWN LIMITATION -- conden-on does NOT reduce to conden-off when nothing
condenses: the fix_species pin freezes the reservoirs at their
stop_conden_time state. Enable condensation only where the species
genuinely condenses.

The photolysis cross-section T-interpolation stays frozen by design. The
runner's lax.while_loop supports jvp/jacfwd but NOT vjp; forward mode is the
end-to-end route.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from vulcan_forward import constants

# This module must own the first jax import: it fixes the VULCAN_JAX_* import-frozen
# env vars and jax x64. If exojax got in first those knobs are already baked wrong.
if "exojax" in sys.modules:
    raise RuntimeError(
        "vulcan_forward.vulcan_chem must be imported BEFORE exojax: it fixes "
        "jax x64 and the VULCAN_JAX_* import-frozen env vars. Import order: "
        "vulcan_forward.vulcan_chem, then vulcan_forward.exojax_rt.")

# VULCAN-JAX freezes network/atom_list at ITS first import, so arriving after
# it makes the setdefault below a no-op and the run fails much later. Raise
# only when the frozen choice actually conflicts with what this engine needs.
_WANT_ENV = {"VULCAN_JAX_NETWORK": constants.DEFAULT_NETWORK,
             "VULCAN_JAX_ATOM_LIST": constants.DEFAULT_ATOM_LIST}
if "vulcan_jax" in sys.modules:
    _conflict = {k: (os.environ.get(k), v) for k, v in _WANT_ENV.items()
                 if os.environ.get(k) not in (None, v)}
    # With the env vars UNSET -- the default -- comparing them proves nothing:
    # vulcan_jax has already frozen whatever ITS config named. Read the frozen
    # state itself, which is what the env vars were only ever describing.
    if not _conflict:
        _frozen = getattr(sys.modules["vulcan_jax"], "chem_funs", None)
        _frozen = getattr(getattr(_frozen, "_NETWORK", None), "network_path", None)
        if _frozen and os.path.basename(_frozen) != os.path.basename(
                constants.DEFAULT_NETWORK):
            _conflict = {"VULCAN_JAX_NETWORK": (_frozen,
                                                constants.DEFAULT_NETWORK)}
    if _conflict:
        raise RuntimeError(
            "vulcan_jax was imported before vulcan_forward.vulcan_chem, so its "
            "network/atom_list are already frozen and cannot be changed: "
            + "; ".join(f"{k} is {got!r} but this engine needs {want!r}"
                        for k, (got, want) in _conflict.items())
            + ". Import vulcan_forward.vulcan_chem first, or set those env vars "
              "before importing vulcan_jax.")

# --- env setup MUST happen before importing vulcan_jax / jax ------------------
# setdefault, not assignment: a caller driving a non-default network (a batched
# emulator, say) sets these itself, and silently overwriting that choice would
# model different chemistry than was asked for.
for _k, _v in _WANT_ENV.items():
    os.environ.setdefault(_k, _v)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Column-repair pairs for the exact-elemental mode: element -> adjuster species,
# the runner's own atom-conservation reservoirs (abundant carriers in H2-dominated
# gas, so the linear repair stays tiny and well-conditioned). H is the reference
# element; He preserves the baseline He/H.
_ELEMENTAL_REPAIR = (("He", "He"), ("O", "H2O"), ("C", "CO"), ("N", "N2"), ("S", "H2S"))
# Renorm+repair iterations: the residual contracts geometrically
# (~1e-2 -> ~1e-8 by three passes; see audit_init).
_ELEMENTAL_REPAIR_ITERS = 3

# Tolerance (g/mol) for checking constants.ATOMIC_MASSES against the package's
# composition-table mass column, which carries mild rounding (O listed as 16.0 vs
# 15.999; the electron as 1e-3 vs 5.4858e-4). Real drift -- a swapped or wrong
# element column -- differs by >= 1 g/mol and is caught cleanly.
_ATOMIC_MASS_TOL_G_MOL = 0.01


def _assert_composition_tables(composition) -> None:
    """Verify constants.ATOM_COLS / constants.ATOMIC_MASSES against the composition
    metadata vulcan_jax actually loaded.

    Both config tables are positional hardcoded mirrors of the package's
    composition table (whose atom-column order comes from the com_file header at
    import). A package change that reorders or extends the atom columns would
    silently corrupt the Z / C-O masks, the exact-elemental repair matrix, and
    the mean-molecular-weight masses, so any mismatch raises here at build time.
    """
    atom_list = tuple(composition.atom_list)
    if len(constants.ATOMIC_MASSES) != len(atom_list):
        raise RuntimeError(
            f"constants.ATOMIC_MASSES has {len(constants.ATOMIC_MASSES)} entries but "
            f"vulcan_jax.composition.atom_list has {len(atom_list)} columns "
            f"{atom_list}: the hardcoded mass table no longer mirrors the "
            "package's composition table.")
    for elem, col in constants.ATOM_COLS.items():
        if atom_list[col] != elem:
            raise RuntimeError(
                f"constants.ATOM_COLS[{elem!r}] = {col} but "
                f"vulcan_jax.composition.atom_list[{col}] is {atom_list[col]!r} "
                f"(full order {atom_list}): the hardcoded element-column map no "
                "longer matches the package's atom order.")
    compo, compo_row = composition.compo, composition.compo_row
    compo_arr = np.asarray(composition.compo_array, dtype=np.float64)
    for col, atom in enumerate(atom_list):
        if atom in compo_row:
            table_mass = float(compo[compo_row.index(atom)]["mass"])
            if abs(table_mass - float(constants.ATOMIC_MASSES[col])) > _ATOMIC_MASS_TOL_G_MOL:
                raise RuntimeError(
                    f"constants.ATOMIC_MASSES[{col}] = {constants.ATOMIC_MASSES[col]} "
                    f"for {atom!r} but the package composition table's {atom!r} "
                    f"row has mass {table_mass}: the hardcoded mass no longer "
                    "matches the package's mass data.")
        elif compo_arr[:, col].sum() != 0.0:
            raise RuntimeError(
                f"composition atom column {col} ({atom!r}) is carried by network "
                "species but has no monatomic table row to verify "
                "constants.ATOMIC_MASSES against; extend the check before trusting "
                "the hardcoded mass.")


class ChemParams(NamedTuple):
    """Named chemistry parameters -- the engine's primitive.

    ``lnZ`` scales the metals (ln of the multiple of the baseline metallicity),
    ``c_o`` moves carbon at fixed oxygen (ln of the C/O multiple), ``lnKzz``
    scales the whole eddy-diffusion profile, and ``tp`` carries the temperature
    parameters. What ``tp`` MEANS is fixed when the model is built: with a
    ``tp_eval`` hook it is that hook's parameter block; without one it is a
    single uniform temperature offset in K, added to the structural profile.

    Prefer this over a bare vector in new code, because the field names carry
    the meaning. The positional vector form is still accepted everywhere (see
    ``params_from_vector``) and remains the right shape for a sampler or for
    forward-mode AD, where the parameters ARE a vector and the tangent has to
    match it. This type is a NamedTuple, so it is also a JAX pytree.
    """
    lnZ: float = 0.0
    c_o: float = 0.0
    lnKzz: float = 0.0
    tp: tuple = ()

    def to_vector(self):
        """The equivalent positional vector ``[lnZ, c_o, lnKzz, *tp]``."""
        return jnp.asarray([self.lnZ, self.c_o, self.lnKzz, *self.tp],
                           dtype=jnp.float64)


def params_from_vector(theta, n_tp_params: int, *, has_tp_eval: bool):
    """Read a positional ``theta`` into named :class:`ChemParams`.

    The layout is the retrieval framework's, kept for compatibility:
    ``theta[0:3]`` is ``[lnZ, c_o, lnKzz]`` and the tail is the temperature
    block -- ``theta[3:3+n_tp_params]`` when the model was built with a
    ``tp_eval`` hook, else the single element ``theta[3]``.
    """
    v = jnp.asarray(theta, dtype=jnp.float64)
    tp = v[3:3 + n_tp_params] if has_tp_eval else v[3:4]
    return ChemParams(lnZ=v[0], c_o=v[1], lnKzz=v[2], tp=tp)


def bz_margin(y, n_o, carbon_mask, o_only_mask) -> float:
    """Positivity margin of the fixed-O C/O seed map on a column ``y`` (nz, ni).

    The map scales every C-bearing species by e^c and compensates the oxygen
    they drag along by scaling the O-only carriers by
    b_z = 1 + (1 - e^c) OC_z / OO_z, so b_z > 0 iff c < ln(1 + OO_z/OC_z).
    Returns the worst layer's bound, ln(1 + min_z OO_z/OC_z): 0 where a layer
    has no O-only carriers left, NaN where a layer has no oxygen in either
    group -- gate with ``not margin > x`` so NaN refuses. ``n_o`` is the O
    atoms per species, the masks the C-bearing / O-only species (ni,).
    """
    y = np.asarray(y, dtype=np.float64)
    n_o = np.asarray(n_o, dtype=np.float64)
    oc_z = (y * (n_o * np.asarray(carbon_mask, dtype=np.float64))[None, :]).sum(axis=1)
    oo_z = (y * (n_o * np.asarray(o_only_mask, dtype=np.float64))[None, :]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log(1.0 + np.min(oo_z / oc_z)))


class ConvDiag(NamedTuple):
    """Per-solve convergence diagnostics read off the runner's final carry.

    Every field rides the runner's primal carry, so reading them is free inside a
    forward-mode jvp chain. ``accept_count`` and ``count_since_new_min`` are
    integer-valued (no tangent); ``longdy``/``longdydt`` are floats and DO carry a
    tangent -- callers on an AD path must ``stop_gradient`` them.

    ``accept_count`` alone is NOT a convergence test: the stall fallback (and the
    hybrid vm_mol phase flip) can terminate the runner with accept_count well under
    the cap on a state whose tangent has not settled. ``conv_normal`` is the
    runner's canonical two-branch certification (tight yconv_cri/slope_cri OR
    loose yconv_min/slope_min, AND the photo-flux gate, the geometry term and
    the column element budget) recomputed at the exit state: False on an exit
    that only certified via the stall fallback or that exhausted a budget.
    """

    accept_count: jnp.ndarray         # () int32   accepted steps taken
    longdy: jnp.ndarray               # () float64 runner's convergence metric at exit
    longdydt: jnp.ndarray             # () float64 longdy per lookback time
    count_since_new_min: jnp.ndarray  # () int32   steps since longdy improved >= 5%
    conv_normal: jnp.ndarray          # () bool    canonical certification at exit
    aflux_change: jnp.ndarray         # () float64 max relative actinic-flux change at exit
    conv_branch: jnp.ndarray          # () int32   certified branch: 1 tight (yconv_cri,
    #                                               slope_cri), 2 loose (yconv_min,
    #                                               slope_min), 0 not certified
    cell_species: jnp.ndarray         # () int32   species index of the cell that sets longdy
    cell_layer: jnp.ndarray           # () int32   layer index of that cell (0 = bottom)
    cell_vmr: jnp.ndarray             # () float64 mixing ratio of that cell at exit
    t: jnp.ndarray                    # () float64 integration time at exit (s)
    dt: jnp.ndarray                   # () float64 step size at exit (s)
    tangent_longdy: jnp.ndarray       # () float64 converged_y_jvp only: the tangent's
    #                                               longdy at exit (NaN on the primal path)
    budget_drift_max: jnp.ndarray     # () float64 max |X/H drift| of the column budget at
    #                                               exit (C23)
    budget_drift_atom: jnp.ndarray    # () int32   index into the runner's atom order
    #                                               (``atom_order``) of that maximum


_SCRATCH_ROOT: str | None = None


def _redirect_output_dirs(cfg) -> None:
    """Point VULCAN-JAX's output directory away from the caller's CWD.

    ``op.Output``'s constructor creates ``cfg.output_dir``, and the shipped
    VULCAN-JAX configs make it RELATIVE -- so building a model would create
    ./output wherever the caller happens to be standing. A library has no
    business writing into a caller's working directory. This engine never
    writes .vul output (the Output object exists only to satisfy
    ``OuterLoop``'s signature), so it is redirected to one per-process temp
    directory. An absolute path a caller set deliberately is left alone.
    """
    global _SCRATCH_ROOT
    val = getattr(cfg, "output_dir", None)
    if val is None or os.path.isabs(str(val)):
        return
    if _SCRATCH_ROOT is None:
        _SCRATCH_ROOT = tempfile.mkdtemp(prefix="vulcan_forward_")
    cfg.output_dir = os.path.join(_SCRATCH_ROOT, "output_dir")


def build_chem_model(profile: dict, tp_eval=None, n_tp_params: int = 0) -> SimpleNamespace:
    """Build the chemistry model and the differentiable converged_y(theta).

    Parameters
    ----------
    profile : dict
        A caller-owned settings dict (keys: ``constants.PROFILE_KEYS``; an
        unknown key raises). ``use_photo`` and ``yconv_cri`` are required.
        ``profile["skip_warmup"]`` (default False) skips the build-time
        warm-up SOLVE and keeps only its runner-closure half: bit-identical
        for consumers that never read ``baseline_conv_normal`` (which is then
        None = not evaluated). Inference profiles must leave it False.
    tp_eval : callable or None, optional
        Temperature-profile hook. When ``None`` (default) the
        temperature is the validated uniform shift ``T = T_base + theta[3]`` (theta[3]
        is a bulk offset). When supplied,
        ``tp_eval(theta[3:3+n_tp_params], p_bar)`` returns the full (nz,) T-P profile
        (bar-indexed) that replaces the scalar shift -- used by the retrieval framework
        to retrieve an ExoJax Guillot/power-law T-P. Either way the rate table AND the
        T/composition-dependent atmospheric structure are rebuilt on-graph.
    n_tp_params : int, optional
        Number of T-P parameters consumed from ``theta[3:]`` when ``tp_eval`` is given.

    Returns
    -------
    SimpleNamespace with fields (the full list is the namespace at the end):
        converged_y(theta, ...) -> (nz, ni) number densities, differentiable
        audit_init(theta) -> host-side dict of elemental/density residuals at init
        T_base   : (nz,) baseline temperature (np.float64)
        p_bar    : (nz,) pressure grid in bar (np.float64)
        sidx     : dict species-name -> column index
        species_masses : (ni,) jnp molar mass per species (g/mol)
        nz, ni   : ints
    """
    t0 = time.time()
    constants.check_profile_keys(profile)
    import vulcan_jax

    # Baseline VULCAN config, loaded by name from vulcan_jax/configs/*.yaml
    # (overridable per profile; the case presets set this). Env VULCAN_JAX_* was
    # set above, so this first vulcan_jax import freezes the SNCHO network.
    cfg = vulcan_jax.load_config(profile.get("vulcan_cfg_name") or constants.DEFAULT_CFG_NAME)
    cfg.use_print_prog = False
    cfg.use_photo = bool(profile["use_photo"])
    cfg.yconv_cri = float(profile["yconv_cri"])
    # `is not None`, never truthiness: 0 is a legitimate value for several of
    # these (count_min=0 no floor, yconv_min=0 closes the OR-branch) and must
    # not silently fall back to cfg.
    if profile.get("nz") is not None:
        cfg.nz = int(profile["nz"])
    if profile.get("count_min") is not None:
        cfg.count_min = int(profile["count_min"])
    if profile.get("count_max") is not None:
        cfg.count_max = int(profile["count_max"])
    if profile.get("dt_max") is not None:   # physical step-size cap (prevents the dt-balloon
        cfg.dt_max = float(profile["dt_max"])  # non-convergence at high Kzz; see config_schema)
    if profile.get("yconv_min") is not None:  # close the loose convergence OR-branch (default 0.1)
        cfg.yconv_min = float(profile["yconv_min"])
    # Generic cfg overrides, applied BEFORE the pre-loop build so they reach
    # make_atm_static / OuterLoop exactly like use_photo does. setattr checks
    # nothing, so a removed or misspelled key is refused first.
    vulcan_jax.validate_overrides(profile.get("cfg_overrides") or {})
    for _k, _v in (profile.get("cfg_overrides") or {}).items():
        setattr(cfg, _k, _v)
    # Warm-continuation step cap for the MUTATION path only: a proposal still
    # unconverged at warm_count_max is headed for rejection, so cut the loop
    # there instead of dragging the lockstep batch to the cold cap. The cap
    # rides the runner CARRY (`count_max_dyn`, seeded by `_runner_carry_seed`),
    # so the batched entry points take it per lane through `warm_cap=True` and
    # share one compiled runner with the cold call. Checked after the
    # overrides: a `cfg_overrides` count_max is the cold cap too.
    _wcm = profile.get("warm_count_max")
    warm_count_max = int(_wcm) if _wcm is not None else int(cfg.count_max)
    if warm_count_max > int(cfg.count_max):
        raise ValueError(
            f"warm_count_max={warm_count_max} exceeds count_max={int(cfg.count_max)}: "
            "the warm mutation cap must be at most the cold cap (it exists to REJECT "
            "doomed proposals earlier, not to extend them)")
    # The chemistry grid must reach the RT top (interp_map refuses a clamped
    # top). An explicit cfg_overrides P_t wins: the model-top ladder and the
    # condensation pin bring their own grids.
    if "P_t" not in (profile.get("cfg_overrides") or {}):
        _ptop = profile.get("art_ptop_bar")
        cfg.P_t = float(_ptop if _ptop is not None else constants.ART_PTOP_BAR) * 1.0e6

    from vulcan_jax.state import RunState, legacy_view
    from vulcan_jax import network as net_mod, composition, rates_jax
    from vulcan_jax import atm_jax, atm_refresh as atm_refresh_mod
    from vulcan_jax import conden as conden_mod
    from vulcan_jax.atm_setup import _VISCOSITY_TABLE, settling_velocity_jax
    from vulcan_jax.jax_step import make_atm_static
    from vulcan_jax.gibbs import load_nasa9
    from vulcan_jax.ini_abun import (column_atoms, element_vector, eq_seed,
                                     ratio_indices)
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.phy_const import kb
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop

    _assert_composition_tables(composition)

    # Condensation with a live T(P) needs configuration that can actually
    # condense; refuse the silently-inert combinations upfront (standing rule:
    # loud errors, no silent fallbacks). The dynamic rebuild itself happens in
    # _prep below via conden.build_conden_profile.
    use_condense = bool(getattr(cfg, "use_condense", False))
    if use_condense:
        if not bool(getattr(cfg, "use_moldiff", True)):
            raise ValueError(
                "use_condense=True requires use_moldiff=True: the condensation "
                "growth term Dg IS the species' molecular-diffusion coefficient "
                "(op.conden's continuum-regime rate), so with molecular diffusion "
                "off every condensation rate would silently be zero.")
        if bool(getattr(cfg, "use_sat_surfaceH2O", False)):
            raise NotImplementedError(
                "use_condense with use_sat_surfaceH2O=True is unsupported in the "
                "T-varying model: it rewrites the fixed-bottom H2O boundary "
                "condition from the STRUCTURAL temperature at ini time, which a "
                "live T(P) does not rebuild. Disable use_sat_surfaceH2O.")
        if not list(getattr(cfg, "condense_sp", []) or []):
            raise ValueError(
                "use_condense=True with an empty condense_sp: nothing would "
                "condense. List the condensable gas species (network "
                "condensation reactions and/or use_relax species).")

    rs = RunState.with_pre_loop_setup(cfg)
    var, atm, para = legacy_view(rs)
    network = net_mod.parse_network(str(resolve_data_path(cfg.network)))
    nz, ni = atm.Tco.shape[0], network.ni
    sidx = dict(network.species_idx)

    # Static condensation metadata (species identity, particle coefficients,
    # relax/fix flags) -- extracted once; _prep rebuilds the dynamic arrays
    # from it at every proposed T. None when condensation is off.
    conden_spec = None
    if use_condense:
        conden_spec = conden_mod.make_conden_spec(cfg, var, atm, sidx)
        relax_set = set(getattr(cfg, "use_relax", []) or [])
        inert = [sp for sp in cfg.condense_sp
                 if sp not in conden_spec.gas_names and sp not in relax_set]
        if inert:
            raise ValueError(
                f"condense_sp entries {inert} have no condensation reaction in "
                f"network {cfg.network!r} and are not in use_relax -- they would "
                "silently not condense. Remove them or add the reaction/relax.")
        print(f"[chem] condensation ON: kinetics rows {list(conden_spec.gas_names)}, "
              f"relax H2O={conden_spec.h2o_active} NH3={conden_spec.nh3_active}, "
              f"fix_species={list(conden_spec.fix_names)}; conden arrays rebuilt "
              "on-graph at each proposed T", flush=True)

    pco = jnp.asarray(np.asarray(atm.pco, dtype=np.float64))
    p_bar = np.asarray(atm.pco, dtype=np.float64) / 1.0e6
    p_bar_j = jnp.asarray(p_bar)   # bar-indexed grid for the optional tp_eval hook

    thermo_dir = resolve_data_path(cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = Path(vulcan_jax.__file__).resolve().parent / "thermo"
    nasa9, _ = load_nasa9(network.species, thermo_dir)
    remove_list = getattr(cfg, "remove_list", None)

    # --- one warm-up run: compiles/caches integ._runner and confirms the primal converges
    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    _redirect_output_dirs(cfg)
    integ = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
    solver.naming_solver(para)
    skip_warmup = bool(profile.get("skip_warmup", False))
    print(f"[chem] setup {time.time() - t0:.1f}s; nz={nz} ni={ni} photo={cfg.use_photo}; "
          + ("runner closure only (skip_warmup)" if skip_warmup
             else "warming up runner ..."), flush=True)
    if skip_warmup:
        # The warm-up's converged column is never consumed by this module
        # (state0 packs from the pre-loop ``rs`` below), so a forward-only
        # consumer can skip the full solve and keep just the runner-closure
        # half of the warm-up. Host-side construction only, no solve; the
        # XLA compile then happens on the first real solve, which the
        # persistent compile cache already covers. baseline_conv_normal is
        # None (= not evaluated, distinct from failed); the inference path
        # must not set this flag.
        integ._ensure_runner(var, atm)
        baseline_conv_normal = None
    else:
        tw = time.time()
        rs_warmup = integ(rs)
        # Check the warm-up exit: the runner's own end classification, so
        # end_case 1 is a run that stopped certified (past the ready gate,
        # both branches' terms, the flux gate, C21 and C23) on a finite
        # column. A diagnostic only (notes §2): nothing consumes the warm-up
        # column (state0 packs from the pre-loop ``rs``) and every solve
        # certifies itself, so False flags a configuration that may not
        # converge. Exported as ``baseline_conv_normal``; vulcan-retrieval
        # warns on it.
        _w_end = int(rs_warmup.params.end_case)
        baseline_conv_normal = _w_end == 1
        if not baseline_conv_normal:
            print(f"[chem] WARNING: the warm-up solve did not end certified "
                  f"(end_case={_w_end}, termination_reason="
                  f"{int(rs_warmup.params.termination_reason)}, longdy="
                  f"{float(rs_warmup.step.longdy):.3e}). Its column is not "
                  f"used, but this configuration may not converge.", flush=True)
        print(f"[chem] warm-up converge {time.time() - tw:.1f}s "
              f"(certified: {baseline_conv_normal})", flush=True)

    # --- runner-carry budget/scheme seeding --------------------------------
    # The termination budget and diffusion-scheme blend live on the CARRY, not
    # the statics, and state0 is packed ONCE under the COLD statics -- so every
    # per-proposal solve must re-seed them for the runner that consumes it.
    # Otherwise a warm-capped solve runs to the cold count_max, and under the hybrid
    # default every warm continuation restarts in upwind phase 0 and exhausts
    # the warm cap before the phase flip can certify. Warm continuations start
    # from a column already converged on the central operator (a completed
    # hybrid run ends in phase 1), so they continue on it; pure-upwind configs
    # keep upwind (their steady state IS the upwind fixed point).
    cold_count_max = int(cfg.count_max)
    count_min_v = int(cfg.count_min)
    runtime_v = float(cfg.runtime)
    use_vm_mol_v = bool(cfg.use_vm_mol)
    hybrid_v = use_vm_mol_v and bool(getattr(cfg, "use_hybrid_vm_mol", False))
    _warm_note = ("; warm continuation pinned to central difference (the "
                  "converged phase-1 operator)" if hybrid_v else "")
    print(f"[chem] diffusion scheme: use_vm_mol={use_vm_mol_v} "
          f"hybrid={hybrid_v}{_warm_note}", flush=True)

    def _runner_carry_seed(init, *, warm_continuation, warm_cap):
        """Re-seed the carry's live termination budget + diffusion blend for the
        runner about to consume ``init`` (see the block comment above)."""
        blend = 1.0 if use_vm_mol_v else 0.0
        if warm_continuation and hybrid_v:
            blend = 0.0   # continue on the converged (phase-1, central) operator
        return init._replace(
            hybrid_use_vm=jnp.float64(blend),
            count_min_dyn=jnp.int32(count_min_v),
            count_max_dyn=jnp.int32(warm_count_max if warm_cap else cold_count_max),
            runtime_dyn=jnp.float64(runtime_v),
        )

    def _conv_diag(final, tangent_ok=True, tangent_longdy=jnp.nan):
        """ConvDiag read off the runner's exit carry. ``conv_normal`` is
        vulcan-jax's ``conv_normal`` certificate (tight OR loose branch, the
        photo-flux gate, the geometry and element-budget terms; not the ready
        gate, hybrid-phase or non-finite exit) AND the supplied tangent
        certificate. True only for a certified exit; False when the exit came
        from the stall fallback or exhausted a count/runtime budget. The
        controlling cell is the argmax
        of the masked per-cell ratio the runner maximised for longdy
        (``where_varies_most`` rides the carry). ``tangent_ok`` is the
        solver's sensitivity certificate on the ``converged_y_jvp`` path
        (``OuterLoop.run_jvp``); the primal path passes the default. It
        enters ``conv_normal`` only: ``conv_branch`` stays the COLUMN's own
        branch, so a consumer can tell an unsettled sensitivity (branch
        set, ``tangent_longdy`` above the gate) from an uncertified column."""
        ok, branch = vulcan_jax.conv_normal(final, cfg)
        flat = jnp.argmax(final.where_varies_most)
        return ConvDiag(
            accept_count=final.accept_count,
            longdy=final.longdy,
            longdydt=final.longdydt,
            count_since_new_min=final.count_since_new_min,
            conv_normal=ok & tangent_ok,
            aflux_change=final.aflux_change,
            conv_branch=branch,
            cell_species=(flat % ni).astype(jnp.int32),
            cell_layer=(flat // ni).astype(jnp.int32),
            cell_vmr=final.ymix.reshape(-1)[flat],
            t=final.t,
            dt=final.dt,
            tangent_longdy=jnp.float64(tangent_longdy),
            budget_drift_max=jnp.max(jnp.abs(final.budget_drift)),
            budget_drift_atom=jnp.argmax(jnp.abs(final.budget_drift)).astype(jnp.int32),
        )

    def _conv_normal_at_exit(final):
        """The canonical certification alone (see ``_conv_diag``)."""
        return _conv_diag(final).conv_normal

    atm_static = make_atm_static(atm, ni, nz, cfg=integ._cfg)
    state0 = integ._pack_state_from_runstate(rs)
    y0 = state0.y
    Kzz0 = atm_static.Kzz
    pv0 = state0.pv
    T_base = jnp.asarray(np.asarray(atm.Tco, dtype=np.float64))

    # --- on-graph atmosphere rebuild inputs -------------------------------
    # refresh_static packs the runner's own hydrostatic-refresh kernel inputs (pico,
    # gs, Rp, pref anchor, species masses); update_mu_dz_jax(ymix, st) is exactly what
    # the runner fires in-loop every update_frq accepted steps, so seeding the initial
    # carry with it makes step 1 consistent with what the loop maintains thereafter.
    # phys0/spec_atm feed atm_jax._mol_diff, the committed on-graph Dzz(T, M) builder
    # (field-for-field equal to the host make_atm_static for this atm_type; validated
    # in VULCAN-JAX tests/test_atm_jax.py).
    refresh_static = integ._build_refresh_static(atm)
    phys0, spec_atm = atm_jax.make_physical_inputs(cfg, var, atm, list(network.species))
    use_vm = bool(spec_atm.use_vm_mol and spec_atm.use_moldiff)
    use_set = bool(spec_atm.use_settling and spec_atm.use_moldiff)

    # --- composition masks for the y0 knobs -------------------------------
    compo = np.asarray(composition.compo_array)
    metal_cols = [constants.ATOM_COLS[a] for a in ("O", "C", "N", "S")]
    # Scales every C/N/O/S-bearing species; NOT an exact elemental direction
    # (bound H scales along, ~0.6% per e-fold of Z at 10x solar). It is only
    # the initial guess: the elemental repair removes the leakage.
    metal_mask = jnp.asarray((compo[:, metal_cols].sum(axis=1) > 0).astype(np.float64))
    carbon_mask = jnp.asarray(                                   # C/O proxy
        (compo[:, constants.ATOM_COLS["C"]] > 0).astype(np.float64))
    # fixed-O C/O mode ("co_mode": "fixed_O"): every C atom lives in a C-bearing species,
    # and the O-carriers holding no C (H2O, OH, O2, SO, SO2, NO, ...) are disjoint from
    # them -- the two masks partition all O between "dragged along by C-carriers" and
    # "free to compensate".
    nO_per_species = jnp.asarray(np.asarray(compo[:, constants.ATOM_COLS["O"]], dtype=np.float64))
    o_only_mask = jnp.asarray(((compo[:, constants.ATOM_COLS["O"]] > 0)
                               & (compo[:, constants.ATOM_COLS["C"]] == 0)).astype(np.float64))
    co_mode = str(profile.get("co_mode", "proxy"))
    if co_mode not in ("proxy", "fixed_O"):
        raise ValueError(f"co_mode={co_mode!r}: expected 'proxy' or 'fixed_O'")
    co_fixed_o = co_mode == "fixed_O"
    atomic_masses = jnp.asarray(np.asarray(constants.ATOMIC_MASSES, dtype=np.float64))
    species_masses = jnp.asarray(np.asarray(compo, dtype=np.float64)) @ atomic_masses  # (ni,)
    # runner's own (ni, n_atoms) composition table, columns in its internal _atom_order --
    # used to rebuild the conserved atom totals (atom_ini) in the runner's exact basis.
    compo_run = jnp.asarray(np.asarray(integ._compo_arr, dtype=np.float64))

    # --- exact-elemental targets + repair tables ----------------------------
    # Baseline column-integrated elemental totals from the pristine y0 (which sums to
    # M_base per layer by construction: equilibrium mixing ratios x layer density).
    # Targets are RATIOS to elemental H; absolute densities follow from sum_i n_i = M.
    elem_pairs = [(e, sp) for e, sp in _ELEMENTAL_REPAIR
                  if sp in sidx and compo[:, constants.ATOM_COLS[e]].sum() > 0]
    _y0_np = np.asarray(y0, dtype=np.float64)
    _elem_cols = [constants.ATOM_COLS["H"]] + [constants.ATOM_COLS[e] for e, _ in elem_pairs]
    # (ni, 1+nrep) atoms-per-molecule for [H, He, O, C, N, S]-as-present
    E_mat = jnp.asarray(np.asarray(compo[:, _elem_cols], dtype=np.float64))
    rep_cols = np.asarray([sidx[sp] for _, sp in elem_pairs], dtype=np.int64)
    A0 = _y0_np @ np.asarray(compo[:, _elem_cols], dtype=np.float64)  # per-layer (nz, 1+nrep)
    A0 = A0.sum(axis=0)                                               # column totals
    missing = [sp for _, sp in _ELEMENTAL_REPAIR if sp not in sidx]
    if not elem_pairs:
        raise RuntimeError("elemental mode: no repair species found in the network")
    R0_ratios = A0[1:] / A0[0]
    # per-element theta-scaling kind: He fixed; O/N/S x Z; C x Z e^{c_o}
    _zk = np.asarray([0.0 if e == "He" else 1.0 for e, _ in elem_pairs])
    _ck = np.asarray([1.0 if e == "C" else 0.0 for e, _ in elem_pairs])
    zscale_kind = jnp.asarray(_zk)
    cscale_kind = jnp.asarray(_ck)
    R0_j = jnp.asarray(R0_ratios)
    _names = [e for e, _ in elem_pairs]
    print("[chem] elemental mode: exact column ratios to H via repair species "
          f"{[sp for _, sp in elem_pairs]}"
          + (f" (absent: {missing})" if missing else "")
          + "; baseline C/O = "
          f"{A0[1 + _names.index('C')] / A0[1 + _names.index('O')]:.4f}",
          flush=True)

    _nC = np.asarray(compo[:, constants.ATOM_COLS["C"]], dtype=np.float64)
    _nO = np.asarray(compo[:, constants.ATOM_COLS["O"]], dtype=np.float64)
    _mC = np.asarray(carbon_mask)
    _mOo = np.asarray(o_only_mask)

    # --- cold-start seed: the network's own Gibbs equilibrium at the proposal's
    # own T-P and column elemental ratios (the upstream VULCAN start). End-to-end
    # JAX (vulcan_jax.ini_abun.eq_seed): no host callback, so it jits and vmaps
    # with the rest of the solve. Its tangent is zero by construction (vulcan-jax
    # carries the custom_jvp); lnZ / c_o tangents enter through the exact
    # elemental projection below. Only cold solves (warm_y=None) use it.
    _ratio_idx = ratio_indices([e for e, _ in elem_pairs])
    _p_bar_seed = jnp.asarray(np.asarray(pco, dtype=np.float64) / 1.0e6)

    def _eq_seed(T, ratios, M):
        """The equilibrium column as ABSOLUTE densities (nz, ni): eq_seed
        returns mixing ratios, and every caller here works in densities."""
        return eq_seed(T, _p_bar_seed,
                       element_vector(ratios, _ratio_idx)) * M[:, None]

    def co_bz_margin(y):
        """Positivity margin of the fixed-O C/O knob on the column ``y`` (nz, ni):
        see :func:`bz_margin`. Evaluate it on the column the tangent actually
        starts from (a warm converged column, not only the build's initial
        one). inf in proxy mode (no b_z compensation)."""
        if not co_fixed_o:
            return float("inf")
        return bz_margin(y, _nO, _mC, _mOo)

    co_bz_bound = co_bz_margin(_y0_np)   # the build's initial column
    if co_fixed_o:
        # Build-time diagnostics for the fixed-O C/O knob: baseline C/O, how much of the
        # column's O sits in C-carriers (sets the b_z compensation), and the worst-layer
        # O-only share (b_z blows up where O-only carriers vanish).
        _y0n = _y0_np
        _C_tot = float((_y0n * _nC[None, :]).sum())
        _O_tot = float((_y0n * _nO[None, :]).sum())
        _OC_z = (_y0n * (_nO * _mC)[None, :]).sum(axis=1)
        _OO_z = (_y0n * (_nO * _mOo)[None, :]).sum(axis=1)
        print(f"[chem] fixed-O C/O knob: baseline C/O = {_C_tot/_O_tot:.4f} "
              f"(ln = {np.log(_C_tot/_O_tot):+.4f}); O-in-C-carriers share "
              f"median {np.median(_OC_z/(_OC_z+_OO_z)):.3f}, max {np.max(_OC_z/(_OC_z+_OO_z)):.3f} "
              f"(b_z stays positive for c_o < {co_bz_bound:.2f})", flush=True)

    rep_cols_j = jnp.asarray(rep_cols)

    def _elemental_project(y_in, M, lnZ, c_o):
        """Renormalize to sum_i n_i = M and repair the column elemental ratios exactly.

        y_in : (nz, ni) guessed absolute densities. Returns (y_out, min_adj) where
        y_out rows sum to M and the column ratios-to-H equal the theta targets to the
        fixed-iteration residual (~1e-8 rel; audit_init measures it), and min_adj is
        the smallest per-species repair factor (must stay > 0 for a physical column;
        it is ~1 +/- the mask-leakage scale everywhere in the shipped prior boxes).
        """
        targets = R0_j * jnp.exp(lnZ * zscale_kind + c_o * cscale_kind)  # (nrep,)
        y = y_in * (M / jnp.sum(y_in, axis=1))[:, None]
        min_adj = jnp.asarray(1.0, dtype=jnp.float64)
        for _ in range(_ELEMENTAL_REPAIR_ITERS):
            A = jnp.einsum("zi,ie->e", y, E_mat)                # [H, e1..] column totals
            col_tot = jnp.sum(y[:, rep_cols_j], axis=0)         # (nrep,) adjuster columns
            B = E_mat[rep_cols_j, :].T * col_tot[None, :]       # (1+nrep, nrep)
            Msys = B[1:, :] - targets[:, None] * B[0:1, :]
            rhs = targets * A[0] - A[1:]
            alpha = jnp.linalg.solve(Msys, rhs)                 # (nrep,) additive factors
            min_adj = jnp.minimum(min_adj, jnp.min(1.0 + alpha))
            scale_vec = jnp.ones(ni, dtype=jnp.float64).at[rep_cols_j].set(1.0 + alpha)
            y = y * scale_vec[None, :]
            y = y * (M / jnp.sum(y, axis=1))[:, None]
        return y, min_adj

    def _guess_y0(lnZ, c_o, warm_y, lnZ_ref, c_o_ref):
        """Mask-scaled continuation GUESS from the converged column ``warm_y``
        (shared by _prep and audit_init): only the increments (lnZ - lnZ_ref,
        c_o - c_o_ref) are applied, so a large absolute perturbation is reached
        by small steps from a nearby converged state."""
        c_o_inc = c_o - c_o_ref     # incremental C/O relative to the warm state
        base = warm_y
        if co_fixed_o:
            # c_o == delta ln(C/O) at fixed O, exactly, layer by layer: scale
            # C-bearing species by e^c; compensate the O they drag along by
            # scaling O-only carriers by b_z = 1 + (1 - e^c)*O_Ccarriers/O_Oonly,
            # keeping each layer's O total invariant. Smooth in c_o -> AD-safe;
            # b_z > 0 within the range printed at build.
            OC_z = (base * (nO_per_species * carbon_mask)[None, :]).sum(axis=1)
            OO_z = (base * (nO_per_species * o_only_mask)[None, :]).sum(axis=1)
            b_z = 1.0 + (1.0 - jnp.exp(c_o_inc)) * OC_z / OO_z                # (nz,)
            cofac = jnp.where(carbon_mask[None, :] > 0, jnp.exp(c_o_inc), 1.0)  # (1, ni)
            cofac = jnp.where(o_only_mask[None, :] > 0, b_z[:, None], cofac)  # (nz, ni)
            y0p = base * jnp.exp((lnZ - lnZ_ref) * metal_mask)[None, :] * cofac
        else:
            scale = jnp.exp((lnZ - lnZ_ref) * metal_mask + c_o_inc * carbon_mask)  # (ni,)
            y0p = base * scale[None, :]
        return y0p

    def _as_params(p):
        """Accept named :class:`ChemParams` or a positional vector."""
        if isinstance(p, ChemParams):
            return p
        return params_from_vector(p, n_tp_params, has_tp_eval=tp_eval is not None)

    def _prep(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Build the perturbed initial runner state + atm from ChemParams (or
        the positional vector [lnZ, c_o, lnKzz, T...]).

        Continuation: pass warm_y = a previously-CONVERGED y (with its lnZ_ref /
        c_o_ref) to warm-start from there. The guess is projected onto the
        exact theta targets, so the conserved inventory is path-independent."""
        # One column per solve, always: the batched entry points map this over
        # the leading axis, so warm_i is (nz, ni) there too. Any other trailing
        # shape would BROADCAST silently -- a (1, ni) column would seed every
        # layer from one layer. Shapes are static, so this check runs under jit
        # and under vmap.
        if warm_y is not None and jnp.shape(warm_y) != (nz, ni):
            raise ValueError(
                f"warm_y has shape {tuple(jnp.shape(warm_y))}: one converged "
                f"column of shape ({nz}, {ni}) is expected per solve "
                f"(batched callers pass ({nz}, {ni}) per lane, i.e. "
                f"(N, {nz}, {ni}) stacked).")
        _p = _as_params(theta)
        lnZ, c_o, lnKzz = _p.lnZ, _p.c_o, _p.lnKzz

        # Temperature: uniform T shift by default; with a tp_eval hook the full
        # differentiable T-P profile. Either way the rate table is rebuilt
        # on-graph (rates_jax) with n_0 = pco/(kb T).
        if tp_eval is None:
            T = T_base + _p.tp[0]
        else:
            T = tp_eval(_p.tp, p_bar_j)
        M = pco / (kb * T)
        # Honor cfg.use_lowT_limit_rates: build_rate_array defaults it off, and
        # silently ignoring a set config flag violates the loud-errors rule.
        k_arr = rates_jax.build_rate_array(
            network, T, M, nasa9, remove_list,
            use_lowT_caps=bool(cfg.use_lowT_limit_rates))
        Ti = 0.5 * (T[:-1] + T[1:])
        Kzz_eff = Kzz0 * jnp.exp(lnKzz)

        if warm_y is None:
            ratios = R0_j * jnp.exp(lnZ * zscale_kind + c_o * cscale_kind)
            y0p = _eq_seed(T, ratios, M)
        else:
            y0p = _guess_y0(lnZ, c_o, warm_y, lnZ_ref, c_o_ref)

        # Exact construction: sum_i n_i = M per layer AND exact column elemental
        # ratios; atom_ini rebuilt from the repaired column so the conservation
        # anchor matches the actual initial gas.
        y0p, _min_adj = _elemental_project(y0p, M, lnZ, c_o)
        ymix0 = y0p / M[:, None]
        atom_ini_new = jnp.einsum("zi,ia->a", y0p, compo_run)  # runner atom order
        pv_T = pv0._replace(n_0=M, r_Tco=T, Kzz=Kzz_eff, atom_ini=atom_ini_new)

        # --- atmospheric structure at the proposed T + composition --------
        # Hydrostatic geometry via the runner's OWN refresh kernel (so the initial
        # carry equals what the in-loop refresh maintains); Dzz/vm/vs via the
        # committed on-graph builder at the proposed (T, M). The runner splices the
        # carry geometry into every step and recomputes vm in-loop from atm.Dzz, so
        # rebuilding Dzz here fixes the whole molecular-diffusion channel.
        refresh_lane = refresh_static._replace(Tco=T)
        mu_i, g_i, Hp_i, dz_i, zco_i, dzi_i, Hpi_i = atm_refresh_mod.update_mu_dz_jax(
            ymix0, refresh_lane)
        Dzz_new, _Dzz_cen, vm_new = atm_jax._mol_diff(
            phys0._replace(Tco=T), spec_atm, M, g_i, Hp_i, dz_i)
        if not use_vm:
            vm_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        if use_set:
            _na, _a, _b = _VISCOSITY_TABLE[spec_atm.atm_base]
            vs_new = settling_velocity_jax(_na, _a, _b, T, g_i, spec_atm.settle_coeff)
        else:
            vs_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        # y_ini is kept for the end-of-run print; the element-budget
        # certificate's reference is `budget_ref`, seeded on the init below
        # from THIS theta's starting column on ITS own grid, with the per-step
        # accumulator zeroed -- as atom_ini is re-anchored above. The baseline
        # column would charge the proposal's own composition change to the
        # solver's conservation.
        pv_T = pv_T._replace(r_Dzz_top=Dzz_new[-1], y_ini=y0p)

        # --- condensation at the proposed T ---------------------------------
        # Rebuild every T/structure-dependent condensation array from the SAME
        # live temperature and structure the chemistry uses (saturation number
        # densities, Dg growth terms from the live Dzz, relax inputs, NH3
        # cold-trap argmin, fix-species sat-mix rows) and splice them into the
        # ProfileVars carry the runner reads each step. No baseline-frozen
        # condensation table survives into a live-T solve.
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
        atm_T = atm_static._replace(Tco=T, Ti=Ti, M=M, Kzz=Kzz_eff, Dzz=Dzz_new,
                                    vm=vm_new, vs=vs_new, g=g_i, dzi=dzi_i, Hpi=Hpi_i)

        # y_prev is the runner's revert target on a rejected step AND the state
        # the C23 per-step accumulation differences against; state0 carries the
        # BASELINE column there, so it must be re-seeded with this theta's own.
        init = state0._replace(y=y0p, y_prev=y0p, ymix=ymix0, k_arr=k_arr, pv=pv_T,
                               mu=mu_i, g=g_i, Hp=Hp_i, dz=dz_i, zco=zco_i,
                               dzi=dzi_i, Hpi=Hpi_i, vs=vs_new,
                               budget_ref=column_atoms(y0p, dz_i, compo_run),
                               budget_err=jnp.zeros_like(state0.budget_err),
                               budget_drift=jnp.zeros_like(state0.budget_drift))
        return init, atm_T

    def run_diag(theta, return_atm=False, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Diagnostic solve: returns (final_runner_state, init_state).

        Lets a caller inspect convergence and conserved-total drift. Not on any
        AD path. ``return_atm=True`` additionally returns the theta-dependent
        AtmStatic the runner was actually driven with -- the operating point a
        reverse-mode adjoint must linearize around (the setup-time baseline is
        WRONG whenever theta carries a T-P or Kzz offset). Cold by default;
        ``warm_y`` / ``lnZ_ref`` / ``c_o_ref`` start it as ``converged_y``'s
        continuation does."""
        init, atm_T = _prep(theta, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
        init = _runner_carry_seed(init, warm_continuation=warm_y is not None,
                                  warm_cap=False)
        final = integ._runner(init, atm_T)
        return (final, init, atm_T) if return_atm else (final, init)

    def prep_pv(theta):
        """The initial-carry ProfileVars for ``theta`` -- the per-proposal arrays
        (n_0, Kzz, atom_ini, and with condensation on the live-rebuilt c_* conden
        arrays + fix_species_sat_mix) WITHOUT running the solver. Pure function
        of theta; jit/vmap/jvp-traceable. Diagnostics/tests only."""
        init, _atm_T = _prep(theta)
        return init.pv

    def converged_y(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0,
                    warm_cap=False, return_conv_diag=False):
        """Converged ABSOLUTE number densities y (nz, ni), with optional
        continuation warm-start (warm_y at lnZ_ref / c_o_ref). Forward-mode
        differentiable w.r.t. theta.

        The carry's termination budget + diffusion blend are re-seeded per
        solve; under the hybrid vm_mol default a warm continuation runs on the
        central operator instead of re-entering upwind phase 0.

        ``warm_cap=True`` caps the solve at ``warm_count_max`` (the SMC mutation
        path; the cap rides the carry). ``return_conv_diag=True`` returns ``(y, ConvDiag)`` -- free
        reads off the primal carry; ``conv_normal`` is the canonical
        certification recomputed at the exit, so a stall or budget exit reads
        False even when ``longdy < yconv_min``. ConvDiag's integer fields
        carry no tangent -- AD callers stop_gradient them."""
        init, atm_T = _prep(theta, warm_y=warm_y,
                            lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
        init = _runner_carry_seed(init, warm_continuation=warm_y is not None,
                                  warm_cap=warm_cap)
        final = integ._runner(init, atm_T)
        if return_conv_diag:
            return final.y, _conv_diag(final)
        return final.y

    def _ref_leaves(n, lnZ_ref, c_o_ref):
        """The reference composition as PER-LANE (n,) leaves for the vmapped
        prep. A scalar broadcasts, so a shared reference still works; the
        retrieval's warm mutation passes each particle its OWN carried
        (lnZ, c_o). References are CONSTANTS of the map -- a caller
        differentiates with respect to theta, never to these."""
        return (jnp.broadcast_to(jnp.asarray(lnZ_ref, dtype=jnp.float64), (n,)),
                jnp.broadcast_to(jnp.asarray(c_o_ref, dtype=jnp.float64), (n,)))

    def converged_y_batch(thetas, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0,
                          warm_cap=False, return_conv_diag=False):
        """Converged ABSOLUTE number densities for a STACK of thetas (N, n_theta)
        -> y (N, nz, ni), with the optional per-lane continuation warm-start
        ``warm_y`` (N, nz, ni) at lnZ_ref / c_o_ref and, with
        ``return_conv_diag=True``, the per-lane ``(y, ConvDiag)`` of
        ``converged_y``. The results are the BATCHED runner's
        (``OuterLoop.run_batch``, one while loop ABOVE the lane vmap):
        photolysis and the geometry refresh follow the loop's iteration tick,
        not the lane's accept count, so a lane is NOT bit-identical to its solo
        ``converged_y`` -- the two agree at the convergence scale. A lane's
        result does not depend on the other lanes -- each freezes at its own
        exit. A plain ``jax.jvp`` through this entry point is supported (the
        stop test reads the primal only); a tangent-CERTIFIED derivative is
        ``converged_y_jvp``.

        ``lnZ_ref`` / ``c_o_ref`` may be scalars or ``(N,)`` arrays: the
        mutation path gives every lane the reference its carried column was
        converged at. ``warm_cap=True`` caps every lane at ``warm_count_max``
        -- the mutation-path semantics of ``converged_y(..., warm_cap=True)``,
        carried by ``count_max_dyn`` (the runner reads the budget off the
        carry, so the cap needs no second runner).
        """
        lnZ_r, c_o_r = _ref_leaves(int(jnp.shape(thetas)[0]), lnZ_ref, c_o_ref)

        def prep_one(theta_i, warm_i, lnZ_i, c_o_i):
            init, atm_T = _prep(theta_i, warm_y=warm_i,
                                lnZ_ref=lnZ_i, c_o_ref=c_o_i)
            return _runner_carry_seed(init, warm_continuation=warm_y is not None,
                                      warm_cap=warm_cap), atm_T

        # The AtmStatic toggles are unbatched Python bools (the runner's own
        # lane vmap broadcasts them), so they take out_axes None and every
        # array leaf takes 0 -- exactly `_ATM_STATIC_BATCH_AXES`. warm_y=None
        # is an empty pytree node, so the same vmap covers the cold seed.
        init_b, atm_b = jax.vmap(
            prep_one, out_axes=(0, outer_loop._ATM_STATIC_BATCH_AXES),
        )(thetas, warm_y, lnZ_r, c_o_r)
        final_b = integ.run_batch(init_b, atm_b)
        if return_conv_diag:
            return final_b.y, jax.vmap(_conv_diag)(final_b)
        return final_b.y

    # `run_queue` keys its compiled program on the init_fn / out_fn OBJECTS, so
    # fresh closures per call would recompile and grow its cache once per call.
    # The pair depends only on this key, which holds STATIC choices only: the
    # reference composition rides the jobs pytree (gathered per job), so its
    # VALUES never enter a closure and never cost a compile.
    _queue_fns = {}

    def converged_y_queue(thetas, n_lanes, *, chunk=8, warm_y=None,
                          lnZ_ref=0.0, c_o_ref=0.0, warm_cap=False):
        """``converged_y_batch`` on ``n_lanes`` lanes with refill from the job
        queue (vulcan-jax ``OuterLoop.run_queue``): a lane that certifies is
        written out and takes the next theta inside the same while loop, so
        wall time follows total work / lanes instead of the slowest theta.

        Same map and the same convergence-scale contract as the plain batch --
        a refilled theta enters at the tick its lane was freed at, which moves
        the photolysis / geometry cadence the way the batch already moves it
        against the solo solve. With ``n_lanes >= N`` nothing is refilled and
        every theta runs the plain batch's ticks (same accept_count, same
        certificate), but the answer is still not BITWISE the batch's: the
        seed is built inside ``run_queue``'s jitted loop and outside it in
        ``converged_y_batch``, and those two compilations of ``_prep`` differ
        by a ulp in y_ini, which the trajectory amplifies. Returns
        ``(y (N, nz, ni), ConvDiag stacked over N)``; the ConvDiag is not
        optional here (it rides the per-job write-out).

        ``lnZ_ref`` / ``c_o_ref`` (scalar or ``(N,)``) and ``warm_cap`` mean
        what they mean on ``converged_y_batch``: the references ride the jobs
        pytree, so each job is prepped at its own, and the cap rides the
        carry."""
        key = (warm_y is not None, bool(warm_cap))
        fns = _queue_fns.get(key)
        if fns is None:
            warm_cont, warm_cap_k = key

            def init_fn(job):
                theta_i, warm_i, lnZ_i, c_o_i = job
                init, atm_T = _prep(theta_i, warm_y=warm_i,
                                    lnZ_ref=lnZ_i, c_o_ref=c_o_i)
                return _runner_carry_seed(init, warm_continuation=warm_cont,
                                          warm_cap=warm_cap_k), atm_T

            def out_fn(final):
                # The ConvDiag is a report, never differentiated. With a
                # tangent it would keep the certificate ring's tangent alive
                # through the queue's refill cond (vulcan-jax `run_queue` zeroes
                # the ring's at its lane step for the same reason).
                return final.y, jax.tree_util.tree_map(jax.lax.stop_gradient,
                                                       _conv_diag(final))

            fns = _queue_fns[key] = (init_fn, out_fn)
        init_fn, out_fn = fns

        # warm_y=None is an empty pytree node, so the same jobs pytree covers
        # the cold seed.
        lnZ_r, c_o_r = _ref_leaves(int(jnp.shape(thetas)[0]), lnZ_ref, c_o_ref)
        (y, cd), _n_iter = integ.run_queue(
            init_fn, (thetas, warm_y, lnZ_r, c_o_r), int(n_lanes), out_fn,
            chunk=int(chunk))
        return y, cd

    def converged_y_jvp(theta, tangent, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Forward-mode sensitivity certified by the solver: ``(y, dy, ConvDiag)``.

        ``dy`` is the tangent of ``converged_y`` along ``tangent`` (same shape
        as ``theta``), read when BOTH the column and the tangent pass the
        runner's certificate (vulcan-jax ``OuterLoop.run_jvp``): the tangent
        is held, per cell, to the same change-over-lookback tolerance as
        ``y``, so pass ``tangent`` in the units of a finite-difference step
        (``e_i * h_i``) and divide ``dy`` by ``h_i`` afterwards. A plain
        ``jax.jvp(converged_y)`` stops when the column certifies, which from a
        converged warm start is at ``count_min`` with the tangent unrelaxed.
        ``ConvDiag.conv_normal`` includes the tangent term; ``tangent_longdy``
        is the tangent's longdy at exit.

        SEVERAL DIRECTIONS AT ONCE: pass ``tangent`` as a stack ``(D, n_theta)``
        against the single ``(n_theta,)``. The directions ride vulcan-jax's own
        tangent axis, so the solver integrates the primal ONCE for all D
        (``jax.vmap`` of this call would integrate it D times -- the tangent
        certificate is in the loop predicate). ``dy`` comes back ``(D, nz, ni)``;
        the run stops only when EVERY direction has settled, so ``tangent_ok``
        is the AND over directions and ``tangent_longdy`` the worst of them."""
        def _seeded(th):
            init, atm_T = _prep(th, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
            return _runner_carry_seed(init, warm_continuation=warm_y is not None,
                                      warm_cap=False), atm_T
        def _leaves(p):   # named or positional, like converged_y
            if isinstance(p, ChemParams):
                return jax.tree_util.tree_map(
                    lambda x: jnp.asarray(x, dtype=jnp.float64), p)
            return jnp.asarray(p, dtype=jnp.float64)
        def _stack_dir(*ds):
            # float0 placeholders (the carry's int/bool leaves) are
            # direction-independent; nothing downstream reads them.
            if getattr(ds[0], "dtype", None) == jax.dtypes.float0:
                return ds[0]
            return jnp.stack(ds)

        th, tan = _leaves(theta), _leaves(tangent)
        if not isinstance(tangent, ChemParams) and jnp.ndim(tan) == 2:
            # linearize: the primal build runs once, each direction only pushes
            # its tangent through it.
            (init, atm_T), lin = jax.linearize(_seeded, th)
            dinit, datm = jax.tree_util.tree_map(_stack_dir, *[lin(v) for v in tan])
        else:
            (init, atm_T), (dinit, datm) = jax.jvp(_seeded, (th,), (tan,))
        final, dfinal, tl, ok = integ.run_jvp(init, atm_T, dinit, datm)
        return final.y, dfinal.y, _conv_diag(final, tangent_ok=ok,
                                             tangent_longdy=jnp.max(tl))

    def audit_init(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Host-side audit of the initial column built for ``theta`` (not on any AD path).

        Returns a dict with the quantities the science review asked to see verified at
        every retrieval point: relative density-closure error max_z |sum_i n_i - M|/M,
        the achieved-vs-target column elemental ratios, the achieved dln(C/O) vs
        theta, the smallest elemental-repair factor (must be > 0), and the atom_ini
        consistency |atoms(y_init) - atom_ini|/atom_ini in the runner's atom basis.
        """
        th = _as_params(theta).to_vector()
        init, _atm_T = _prep(theta, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
        y = np.asarray(init.y, dtype=np.float64)
        Mn = np.asarray(init.pv.n_0, dtype=np.float64)
        A = (y @ np.asarray(compo[:, _elem_cols], dtype=np.float64)).sum(axis=0)
        ratios = A[1:] / A[0]
        names = [e for e, _ in elem_pairs]
        out = {
            "density_closure_max_rel": float(np.max(np.abs(y.sum(axis=1) - Mn) / Mn)),
            "ratios_to_H": dict(zip(names, ratios.tolist())),
            "baseline_ratios_to_H": dict(zip(names, (A0[1:] / A0[0]).tolist())),
        }
        if "C" in names and "O" in names:
            r_now = ratios[names.index("C")] / ratios[names.index("O")]
            r_base = (A0[1:] / A0[0])[names.index("C")] / (A0[1:] / A0[0])[names.index("O")]
            out["dln_CO_achieved"] = float(np.log(r_now / r_base))
        tg = np.asarray(R0_j) * np.exp(float(th[0]) * np.asarray(zscale_kind)
                                       + float(th[1]) * np.asarray(cscale_kind))
        out["target_ratios_to_H"] = dict(zip(names, tg.tolist()))
        out["ratio_max_rel_err"] = float(np.max(np.abs(ratios / tg - 1.0)))
        # Re-run the projection from the raw GUESS to expose the actual repair
        # magnitude (projecting the already-repaired y would always report ~1).
        if warm_y is None:
            y_guess = _eq_seed(init.pv.r_Tco, jnp.asarray(tg),
                               jnp.asarray(Mn))
        else:
            y_guess = _guess_y0(th[0], th[1], warm_y, lnZ_ref, c_o_ref)
        _yg, min_adj = _elemental_project(y_guess, jnp.asarray(Mn), th[0], th[1])
        out["min_repair_factor"] = float(min_adj)
        ai = np.asarray(init.pv.atom_ini, dtype=np.float64)
        a_run = y @ np.asarray(integ._compo_arr, dtype=np.float64)
        out["atom_ini_max_rel_err"] = float(np.max(np.abs(a_run.sum(axis=0) - ai) / ai))
        return out

    return SimpleNamespace(
        run_diag=run_diag,
        converged_y=converged_y,
        converged_y_batch=converged_y_batch,   # PRIMAL batched twin (run_batch)
        converged_y_queue=converged_y_queue,   # the same batch on n_lanes lanes
        #                                        with refill (run_queue)
        converged_y_jvp=converged_y_jvp,
        conv_normal_at_exit=_conv_normal_at_exit,  # certify a raw run_diag final
        #                                            carry (gate on conv_normal,
        #                                            never longdy alone)
        audit_init=audit_init,
        baseline_conv_normal=baseline_conv_normal,  # warm-up exit certified?
        #                                             (the retrieval warns on False;
        #                                             None = skip_warmup, not
        #                                             evaluated)
        conden_spec=conden_spec,   # static conden metadata (None when conden off)
        prep_pv=prep_pv,           # theta -> initial ProfileVars (no solve; tests)
        _integ=integ,              # the OuterLoop (baked statics access; tests only)
        co_bz_bound=co_bz_bound,   # fixed-O knob validity: b_z > 0 iff c_o < this (build column)
        co_bz_margin=co_bz_margin, # the same margin on any column, e.g. the warm converged one
        y0=np.asarray(y0, dtype=np.float64),   # baked baseline column (warm-start fallback)
        compo_array=compo,
        atom_list=tuple(composition.atom_list),   # compo_array column order
        atom_order=tuple(integ._atom_order),      # runner's atom basis; ConvDiag's
        #                                           budget_drift_atom indexes this
        T_base=np.asarray(T_base),
        p_bar=p_bar,
        dz=np.asarray(atm.dz, dtype=np.float64),   # layer thickness (cm), n0*dz weights
        sidx=sidx,
        species_masses=species_masses,
        nz=nz, ni=ni,
        count_max=int(cfg.count_max),   # the resolved (profile-overridden or module-default) cap
        warm_count_max=warm_count_max,  # mutation-path cap (warm_cap=True; == count_max when unset)
        yconv_min=float(cfg.yconv_min), # loose convergence gate: a converged solve has longdy<this
    )
