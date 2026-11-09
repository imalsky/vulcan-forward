"""The model ``build_chem_model`` assembles: what one build fixes for every proposal, and the solves that run them.

``_ChemSetup`` rebuilds rates, composition, geometry and diffusion on-graph for one theta. ``_ChemSolves`` drives
vulcan3's runner from that state, solo, batched, queued or with a tangent. Imported inside ``build_chem_model``: this
module imports vulcan3, which freezes its network at the first import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import vulcan3
from vulcan3 import atm_jax, atm_refresh, conden, outer_loop, rates_jax
from vulcan3.atm_setup import VISCOSITY_TABLE, settling_velocity_jax
from vulcan3.ini_abun import column_atoms, element_vector, eq_seed
from vulcan3.phy_const import kb

from vulcan_forward._chem_layout import ChemParams, ConvDiag, params_from_vector

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import SimpleNamespace

    from jax import Array
    from jax.typing import ArrayLike

logger = logging.getLogger(__name__)


class _CarrySeed(NamedTuple):
    """The termination budget and diffusion blend, which each solve re-seeds on the carry.

    They live on the carry, and state0 is packed once under the cold statics. Under the hybrid scheme a warm
    continuation starts from a column converged on the central operator, so it continues on it.
    """

    count_min: int
    count_max: int
    runtime: float
    use_vm_mol: bool
    hybrid: bool

    @classmethod
    def from_cfg(cls, cfg: Any) -> _CarrySeed:
        """Read the seed from the config, and log the diffusion scheme it selects."""
        use_vm_mol = bool(cfg.use_vm_mol)
        hybrid = use_vm_mol and bool(cfg.use_hybrid_vm_mol)
        note = "; warm continuation pinned to central difference (the converged phase-1 operator)" if hybrid else ""
        logger.info("[chem] diffusion scheme: use_vm_mol=%s hybrid=%s%s", use_vm_mol, hybrid, note)
        return cls(int(cfg.count_min), int(cfg.count_max), float(cfg.runtime), use_vm_mol, hybrid)


@dataclass(frozen=True, eq=False)
class _ChemSetup:
    """The arrays one build fixes, and what they give for each proposal: the initial state and the exit report.

    ``OuterLoop.run_queue`` keys its compiled programs on ``queue_init`` and ``queue_out``, which compare equal, so a
    repeat call reuses the program; ``eq=False`` hashes them by identity. JAX's trace caches keep these methods until
    the compiled program dies, and the runner holds that program, so nothing here may reach the runner: a dropped model
    would never be freed.
    """

    cfg: Any
    network: Any
    nasa9: Any
    remove_list: Any
    conden_spec: Any
    tp_eval: Callable[[tuple[float, ...] | Array, Array], Array] | None
    n_tp_params: int
    chem_extras: tuple[str, ...]
    kzz_eff: Callable[[float | Array, float | Array], Array]
    elem: SimpleNamespace
    ratio_idx: Any
    T_base: Array
    pco: Array
    p_bar: Array
    compo_run: Array  # the runner's own (ni, n_atoms) table, for atom_ini in the runner's basis
    state0: Any
    atm_static: Any
    refresh_static: Any
    phys0: Any
    spec_atm: Any
    carry: _CarrySeed

    def _eq_seed(self, T: Array, ratios: Array, M: Array) -> Array:
        """Return the equilibrium column as absolute densities, [nz, ni]; ``eq_seed`` returns mixing ratios.

        The cold-start seed is the network's Gibbs equilibrium at the proposal's T-P and column ratios. Its tangent is
        zero by construction (vulcan3 carries the custom_jvp); lnZ and c_o tangents enter through the projection.
        """
        return eq_seed(T, self.p_bar, element_vector(ratios, self.ratio_idx)) * M[:, None]

    def _as_params(self, p: ChemParams | Array) -> ChemParams:
        if isinstance(p, ChemParams):
            return p
        return params_from_vector(p, self.n_tp_params, has_tp_eval=self.tp_eval is not None, extras=self.chem_extras)

    def _prep(
        self,
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
        spec_atm, state0 = self.spec_atm, self.state0
        nz, ni = state0.y.shape
        # One column per solve: a (1, ni) column would broadcast and seed every layer from one layer. Shapes are static,
        # so this runs under jit and vmap.
        if warm_y is not None and jnp.shape(warm_y) != (nz, ni):
            raise ValueError(
                f"warm_y has shape {tuple(jnp.shape(warm_y))}: one converged "
                f"column of shape ({nz}, {ni}) is expected per solve "
                f"(batched callers pass ({nz}, {ni}) per lane, i.e. "
                f"(N, {nz}, {ni}) stacked)."
            )
        _p = self._as_params(theta)
        lnZ, c_o, lnKzz, s_o = _p.lnZ, _p.c_o, _p.lnKzz, _p.s_o

        T = self.T_base + _p.tp[0] if self.tp_eval is None else self.tp_eval(_p.tp, self.p_bar)
        M = self.pco / (kb * T)
        k_arr = rates_jax.build_rate_array(
            self.network, T, M, self.nasa9, self.remove_list, use_lowT_caps=bool(self.cfg.use_lowT_limit_rates)
        )
        Ti = 0.5 * (T[:-1] + T[1:])
        Kzz_eff = self.kzz_eff(lnKzz, _p.kzz_slope)

        if warm_y is None:
            y0p = self._eq_seed(T, self.elem.targets(lnZ, c_o, s_o), M)
        else:
            y0p = self.elem.guess(lnZ, c_o, s_o, warm_y, lnZ_ref, c_o_ref, s_o_ref)

        # atom_ini comes from the repaired column, so the conservation anchor matches the initial gas.
        y0p, _min_adj = self.elem.project(y0p, M, lnZ, c_o, s_o)
        ymix0 = y0p / M[:, None]
        atom_ini_new = jnp.einsum("zi,ia->a", y0p, self.compo_run)  # runner atom order
        pv_T = state0.pv._replace(n_0=M, r_Tco=T, Kzz=Kzz_eff, atom_ini=atom_ini_new)

        # Geometry from the runner's own refresh kernel, so the initial carry equals what the in-loop refresh maintains.
        # The runner recomputes vm in-loop from atm.Dzz, so rebuilding Dzz fixes the whole molecular-diffusion channel.
        refresh_lane = self.refresh_static._replace(Tco=T)
        mu_i, g_i, Hp_i, dz_i, zco_i, dzi_i, Hpi_i = atm_refresh.update_mu_dz_jax(ymix0, refresh_lane)
        Dzz_new, _Dzz_cen, vm_new = atm_jax.mol_diff(self.phys0._replace(Tco=T), spec_atm, M, g_i, Hp_i, dz_i)
        if not (spec_atm.use_vm_mol and spec_atm.use_moldiff):
            vm_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        if spec_atm.use_settling and spec_atm.use_moldiff:
            _na, _a, _b = VISCOSITY_TABLE[spec_atm.atm_base]
            vs_new = settling_velocity_jax(_na, _a, _b, T, g_i, spec_atm.settle_coeff)
        else:
            vs_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        pv_T = pv_T._replace(r_Dzz_top=Dzz_new[-1], y_ini=y0p)

        # The carry holds baseline-T condensation values.
        if self.conden_spec is not None:
            cprof = conden.build_conden_profile(self.conden_spec, T, self.pco, M, Dzz_new)
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
        atm_T = self.atm_static._replace(
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
            budget_ref=column_atoms(y0p, dz_i, self.compo_run),
            budget_err=jnp.zeros_like(state0.budget_err),
            budget_drift=jnp.zeros_like(state0.budget_drift),
        )
        return init, atm_T

    def seeded_prep(
        self,
        theta: ChemParams | Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> tuple[Any, Any]:
        """Return ``_prep`` with the carry's termination budget and diffusion blend seeded for the runner."""
        init, atm_T = self._prep(theta, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref, s_o_ref=s_o_ref)
        carry = self.carry
        blend = 1.0 if carry.use_vm_mol else 0.0
        if warm_y is not None and carry.hybrid:
            blend = 0.0
        return init._replace(
            hybrid_use_vm=jnp.float64(blend),
            count_min_dyn=jnp.int32(carry.count_min),
            count_max_dyn=jnp.int32(carry.count_max),
            runtime_dyn=jnp.float64(carry.runtime),
        ), atm_T

    def conv_diag(
        self, final: Any, tangent_ok: Array | bool = True, tangent_longdy: Array | float = jnp.nan
    ) -> ConvDiag:
        """Read ``ConvDiag`` off the runner's exit carry.

        ``conv_normal`` is ``vulcan3.conv_normal`` AND ``tangent_ok``, the ``run_jvp`` sensitivity certificate.
        ``conv_branch`` stays the column's own branch, so a consumer can tell an unsettled sensitivity from an
        uncertified column.
        """
        ok, branch = vulcan3.conv_normal(final, self.cfg)
        flat = jnp.argmax(final.where_varies_most)
        ni = self.state0.y.shape[1]
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
            termination_reason=final.termination_reason,
            last_delta=final.delta,
        )

    def conv_normal_at_exit(self, final: Any) -> Array:
        """Return the certificate of a raw ``run_diag`` final carry."""
        return self.conv_diag(final).conv_normal

    def queue_init(self, job: tuple[Any, ...]) -> tuple[Any, Any]:
        """Prep one job for its lane."""
        return self.seeded_prep(*job)

    def queue_init_masked(self, job: tuple[Any, ...]) -> tuple[Any, Any]:
        """Prep one job for its lane; a masked job's lane starts done."""
        *core, act = job
        init, atm_T = self.seeded_prep(*core)
        return init._replace(is_done=init.is_done | jnp.logical_not(act)), atm_T

    def queue_out(self, final: Any) -> tuple[Array, ConvDiag]:
        """Write one finished lane out.

        The ``ConvDiag`` is a report, never differentiated: with a tangent it would keep the certificate ring's tangent
        alive through the queue's refill cond.
        """
        return final.y, jax.tree_util.tree_map(jax.lax.stop_gradient, self.conv_diag(final))

    def prep_pv(self, theta: ChemParams | Array) -> Any:
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
        init, _atm_T = self._prep(theta)
        return init.pv

    def audit_init(
        self,
        theta: ChemParams | Array,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> dict[str, Any]:
        """Audit the initial column built for ``theta``, on the host and off any AD path.

        Takes ``converged_y``'s warm arguments. Returns ``density_closure_max_rel``, the achieved, baseline and target
        ``*_ratios_to_H``, ``dln_CO_achieved`` and with sulfur ``dln_SO_achieved``, ``ratio_max_rel_err``,
        ``min_repair_factor`` (must be > 0) and ``atom_ini_max_rel_err``.
        """
        elem = self.elem
        names, baseline = elem.names, elem.R0_ratios
        z_kind, c_kind, s_kind = elem.kinds
        _p = self._as_params(theta)
        lnZ, c_o, s_o = float(_p.lnZ), float(_p.c_o), float(_p.s_o)
        init, _atm_T = self._prep(theta, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref, s_o_ref=s_o_ref)
        y = np.asarray(init.y, dtype=np.float64)
        Mn = np.asarray(init.pv.n_0, dtype=np.float64)
        A = (y @ elem.E_np).sum(axis=0)
        ratios = A[1:] / A[0]
        iC, iO = names.index("C"), names.index("O")
        tg = baseline * np.exp(lnZ * z_kind + c_o * c_kind + s_o * s_kind)
        out = {
            "density_closure_max_rel": float(np.max(np.abs(y.sum(axis=1) - Mn) / Mn)),
            "ratios_to_H": dict(zip(names, ratios.tolist(), strict=True)),
            "baseline_ratios_to_H": dict(zip(names, baseline.tolist(), strict=True)),
            "dln_CO_achieved": float(np.log((ratios[iC] / ratios[iO]) / (baseline[iC] / baseline[iO]))),
            "target_ratios_to_H": dict(zip(names, tg.tolist(), strict=True)),
            "ratio_max_rel_err": float(np.max(np.abs(ratios / tg - 1.0))),
        }
        if "S" in names:
            iS = names.index("S")
            out["dln_SO_achieved"] = float(np.log((ratios[iS] / ratios[iO]) / (baseline[iS] / baseline[iO])))
        # Project the raw guess: projecting the already-repaired y would always report a factor near 1.
        if warm_y is None:
            y_guess = self._eq_seed(init.pv.r_Tco, jnp.asarray(tg), jnp.asarray(Mn))
        else:
            y_guess = elem.guess(lnZ, c_o, s_o, warm_y, lnZ_ref, c_o_ref, s_o_ref)
        _yg, min_adj = elem.project(y_guess, jnp.asarray(Mn), lnZ, c_o, s_o)
        out["min_repair_factor"] = float(min_adj)
        ai = np.asarray(init.pv.atom_ini, dtype=np.float64)
        a_run = y @ np.asarray(self.compo_run, dtype=np.float64)
        out["atom_ini_max_rel_err"] = float(np.max(np.abs(a_run.sum(axis=0) - ai) / ai))
        return out


def _ref_leaves(n: int, lnZ_ref: Array | float, c_o_ref: Array | float, s_o_ref: Array | float) -> tuple[Array, ...]:
    """Broadcast the reference composition to per-lane leaves [n] for the vmapped prep."""
    return tuple(jnp.broadcast_to(jnp.asarray(r, dtype=jnp.float64), (n,)) for r in (lnZ_ref, c_o_ref, s_o_ref))


def _float64_leaves(p: ChemParams | Array) -> ChemParams | Array:
    """Return theta or a tangent as float64 leaves, named or positional."""
    if isinstance(p, ChemParams):
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), p)
    return jnp.asarray(p, dtype=jnp.float64)


def _stack_dir(*ds: Array) -> Array:
    """Stack one leaf over directions; float0 placeholders are direction-independent."""
    if getattr(ds[0], "dtype", None) == jax.dtypes.float0:
        return ds[0]
    return jnp.stack(ds)


@dataclass(frozen=True, eq=False)
class _ChemSolves:
    """The routes from a proposal's start through vulcan3's runner to the converged column."""

    setup: _ChemSetup
    integ: Any

    def run_diag(
        self,
        theta: ChemParams | Array,
        return_atm: bool = False,
        warm_y: Array | None = None,
        lnZ_ref: Array | float = 0.0,
        c_o_ref: Array | float = 0.0,
        s_o_ref: Array | float = 0.0,
    ) -> tuple[Any, ...]:
        """Solve for a diagnostic: return ``(final carry, initial state)``; the warm arguments are ``converged_y``'s.

        ``return_atm`` adds the theta-dependent ``AtmStatic`` the runner was driven with, the operating point a
        reverse-mode adjoint must linearize around, as a third entry.
        """
        init, atm_T = self.setup.seeded_prep(theta, warm_y, lnZ_ref, c_o_ref, s_o_ref)
        final = self.integ.runner(init, atm_T)
        return (final, init, atm_T) if return_atm else (final, init)

    def converged_y(
        self,
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
        init, atm_T = self.setup.seeded_prep(theta, warm_y, lnZ_ref, c_o_ref, s_o_ref)
        final = self.integ.runner(init, atm_T)
        if return_conv_diag:
            return final.y, self.setup.conv_diag(final)
        return final.y

    def converged_y_batch(
        self,
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
            self.setup.seeded_prep,
            out_axes=(0, outer_loop.ATM_STATIC_BATCH_AXES),
        )(thetas, warm_y, *refs)
        final_b = self.integ.run_batch(init_b, atm_b)
        if return_conv_diag:
            return final_b.y, jax.vmap(self.setup.conv_diag)(final_b)
        return final_b.y

    def converged_y_queue(
        self,
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
        init_fn = self.setup.queue_init_masked if masked else self.setup.queue_init

        n = int(jnp.shape(thetas)[0])
        jobs = (thetas, warm_y, *_ref_leaves(n, lnZ_ref, c_o_ref, s_o_ref))
        if masked:
            act = jnp.asarray(active, dtype=bool)
            if act.shape != (n,):
                raise ValueError(f"converged_y_queue: active must have shape ({n},), got {act.shape}")
            jobs = (*jobs, act)
        chunk_kw = {} if chunk is None else {"chunk": int(chunk)}
        (y, cd), _n_iter = self.integ.run_queue(init_fn, jobs, int(n_lanes), self.setup.queue_out, **chunk_kw)
        return y, cd

    def converged_y_jvp(
        self,
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
        ``jax.vmap`` of this call would integrate it D times. The run stops when every direction has settled.

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
            return self.setup.seeded_prep(th, warm_y, lnZ_ref, c_o_ref, s_o_ref)

        th, tan = _float64_leaves(theta), _float64_leaves(tangent)
        if not isinstance(tan, ChemParams) and jnp.ndim(tan) == 2:
            # linearize: the primal build runs once.
            (init, atm_T), lin = jax.linearize(_seeded, th)
            dinit, datm = jax.tree_util.tree_map(_stack_dir, *[lin(v) for v in tan])
        else:
            (init, atm_T), (dinit, datm) = jax.jvp(_seeded, (th,), (tan,))
        final, dfinal, tl, ok = self.integ.run_jvp(init, atm_T, dinit, datm)
        return final.y, dfinal.y, self.setup.conv_diag(final, tangent_ok=ok, tangent_longdy=jnp.max(tl))
