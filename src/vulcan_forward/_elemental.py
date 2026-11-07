"""The column elemental map: composition-table checks, the composition masks and the elemental repair.

``_build_elemental`` holds the theta targets, the projection onto them and the warm-start guess; docs/method.md
describes them.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from vulcan_forward import constants

if TYPE_CHECKING:
    from types import ModuleType

    from jax import Array
    from jax.typing import ArrayLike

logger = logging.getLogger(__name__)

# Element -> adjuster species, scaled by one common factor per element: its dominant carriers in H2-dominated gas, so
# the linear repair stays small and well-conditioned. Carbon needs CO and CH4 together: below 900 K CH4 holds the carbon
# and CO only 1e-3 of it, so a CO-only repair of a 1% carbon error drives CO negative. H is the reference element.
_ELEMENTAL_REPAIR = (("He", ("He",)), ("O", ("H2O",)), ("C", ("CO", "CH4")), ("N", ("N2",)), ("S", ("H2S",)))
# Renormalize-and-repair passes: one takes a warm guess's 1e-1 residual to 1e-10, two reach machine precision.
_ELEMENTAL_REPAIR_ITERS = 3

# g/mol. The package's mass column carries mild rounding (O as 16.0, the electron as 1e-3); a swapped element differs by
# at least 1 g/mol.
_ATOMIC_MASS_TOL_G_MOL = 0.01


def _assert_composition_tables(composition: ModuleType) -> None:
    """Check ``constants.ATOM_COLS`` and ``constants.ATOMIC_MASSES`` against the table vulcan3 loaded.

    Both are positional mirrors of the package's composition table, whose column order comes from the com_file header at
    import. A reordered table would silently corrupt the masks, the repair matrix and the mean molecular weight.
    """
    atom_list = tuple(composition.atom_list)
    if len(constants.ATOMIC_MASSES) != len(atom_list):
        raise RuntimeError(
            f"constants.ATOMIC_MASSES has {len(constants.ATOMIC_MASSES)} entries but "
            f"vulcan3.composition.atom_list has {len(atom_list)} columns "
            f"{atom_list}: the hardcoded mass table no longer mirrors the "
            "package's composition table."
        )
    for elem, col in constants.ATOM_COLS.items():
        if atom_list[col] != elem:
            raise RuntimeError(
                f"constants.ATOM_COLS[{elem!r}] = {col} but "
                f"vulcan3.composition.atom_list[{col}] is {atom_list[col]!r} "
                f"(full order {atom_list}): the hardcoded element-column map no "
                "longer matches the package's atom order."
            )
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
                    "matches the package's mass data."
                )
        elif compo_arr[:, col].sum() != 0.0:
            raise RuntimeError(
                f"composition atom column {col} ({atom!r}) is carried by network "
                "species but has no monatomic table row to verify "
                "constants.ATOMIC_MASSES against; extend the check before trusting "
                "the hardcoded mass."
            )


def bz_margin(y: ArrayLike, n_o: ArrayLike, carbon_mask: ArrayLike, o_only_mask: ArrayLike) -> float:
    """Return the positivity margin of the fixed-O C/O map on a column.

    The map scales every C-bearing species by e^c and compensates the oxygen they carry by scaling the O-only carriers
    by b_z = 1 + (1 - e^c) OC_z / OO_z, so b_z > 0 iff c < ln(1 + OO_z/OC_z).

    Parameters
    ----------
    y : array-like [nz, ni]
        The column.
    n_o : array-like [ni]
        O atoms per species.
    carbon_mask : array-like [ni]
        The C-bearing species.
    o_only_mask : array-like [ni]
        The O-only species.

    Returns
    -------
    float
        ln(1 + min_z OO_z/OC_z): 0 where a layer has no O-only carriers, NaN where a layer has no oxygen in either
        group, so gate with ``not margin > x``.
    """
    y = np.asarray(y, dtype=np.float64)
    n_o = np.asarray(n_o, dtype=np.float64)
    oc_z = (y * (n_o * np.asarray(carbon_mask, dtype=np.float64))[None, :]).sum(axis=1)
    oo_z = (y * (n_o * np.asarray(o_only_mask, dtype=np.float64))[None, :]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log(1.0 + np.min(oo_z / oc_z)))


def _build_elemental(
    compo: np.ndarray, sidx: dict[str, int], y0_np: np.ndarray, ni: int, co_fixed_o: bool
) -> SimpleNamespace:
    """Build the theta targets and the projection for one network, from the baseline column ``y0_np``.

    Targets are ratios to elemental H; absolute densities follow from sum_i n_i = M. Returns a namespace of closures
    that share the tables built here.
    """
    metal_cols = [constants.ATOM_COLS[a] for a in ("O", "C", "N", "S")]
    # Not an exact elemental direction (bound H scales along, 0.6% per e-fold of Z at 10x solar): the repair removes it.
    metal_mask = jnp.asarray((compo[:, metal_cols].sum(axis=1) > 0).astype(np.float64))
    carbon_mask = jnp.asarray((compo[:, constants.ATOM_COLS["C"]] > 0).astype(np.float64))
    sulfur_mask = jnp.asarray((compo[:, constants.ATOM_COLS["S"]] > 0).astype(np.float64))
    # The two masks partition all O between what C-carriers hold and what is free to compensate.
    nO_per_species = jnp.asarray(np.asarray(compo[:, constants.ATOM_COLS["O"]], dtype=np.float64))
    o_only_mask = jnp.asarray(
        ((compo[:, constants.ATOM_COLS["O"]] > 0) & (compo[:, constants.ATOM_COLS["C"]] == 0)).astype(np.float64)
    )
    elem_pairs = [
        (e, tuple(sp for sp in sps if sp in sidx))
        for e, sps in _ELEMENTAL_REPAIR
        if any(sp in sidx for sp in sps) and compo[:, constants.ATOM_COLS[e]].sum() > 0
    ]
    _elem_cols = [constants.ATOM_COLS["H"]] + [constants.ATOM_COLS[e] for e, _ in elem_pairs]
    # (ni, 1 + nrep) atoms per molecule, for H and the elements present.
    E_np = np.asarray(compo[:, _elem_cols], dtype=np.float64)
    E_mat = jnp.asarray(E_np)
    # The group totals are explicit adds: a contraction over species would let XLA reorder the sum between routes.
    rep_cols = np.asarray([sidx[sp] for _, sps in elem_pairs for sp in sps], dtype=np.int64)
    rep_group = np.asarray([k for k, (_, sps) in enumerate(elem_pairs) for _ in sps], dtype=np.int64)
    rep_slots = [np.flatnonzero(rep_group == k).tolist() for k in range(len(elem_pairs))]
    A0 = y0_np @ E_np  # (nz, 1 + nrep), layer sums with no dz weight
    A0 = A0.sum(axis=0)
    missing = [sp for _, sps in _ELEMENTAL_REPAIR for sp in sps if sp not in sidx]
    if not elem_pairs:
        raise RuntimeError("elemental mode: no repair species found in the network")
    R0_ratios = A0[1:] / A0[0]
    # He fixed; O and N x Z; C x Z e^c_o; S x Z e^s_o.
    _zk = np.asarray([0.0 if e == "He" else 1.0 for e, _ in elem_pairs])
    _ck = np.asarray([1.0 if e == "C" else 0.0 for e, _ in elem_pairs])
    _sk = np.asarray([1.0 if e == "S" else 0.0 for e, _ in elem_pairs])
    zscale_kind = jnp.asarray(_zk)
    cscale_kind = jnp.asarray(_ck)
    sscale_kind = jnp.asarray(_sk)
    R0_j = jnp.asarray(R0_ratios)
    _names = [e for e, _ in elem_pairs]

    def _targets(lnZ: Array | float, c_o: Array | float, s_o: Array | float) -> Array:
        """Return the column ratios to H for one composition, [nrep]."""
        return R0_j * jnp.exp(lnZ * zscale_kind + c_o * cscale_kind + s_o * sscale_kind)

    logger.info(
        "[chem] elemental mode: exact column ratios to H via repair species %s%s; baseline C/O = %.4f",
        [sp for _, sp in elem_pairs],
        f" (absent: {missing})" if missing else "",
        A0[1 + _names.index("C")] / A0[1 + _names.index("O")],
    )

    def co_bz_margin(y: ArrayLike) -> float:
        """Return ``bz_margin`` on the column ``y``; inf in proxy mode, which has no b_z compensation."""
        if not co_fixed_o:
            return float("inf")
        return bz_margin(y, nO_per_species, carbon_mask, o_only_mask)

    rep_cols_j = jnp.asarray(rep_cols)
    rep_group_j = jnp.asarray(rep_group)

    def _elemental_project(
        y_in: Array, M: Array, lnZ: Array | float, c_o: Array | float, s_o: Array | float
    ) -> tuple[Array, Array]:
        """Renormalize to sum_i n_i = M and repair the column ratios to the theta targets.

        Returns ``(y, min_adj)``: the residual is ~1e-15 relative (``audit_init`` measures it), and ``min_adj`` is the
        smallest repair factor, which must stay > 0 for a physical column.
        """
        targets = _targets(lnZ, c_o, s_o)  # (nrep,)
        y = y_in * (M / jnp.sum(y_in, axis=1))[:, None]
        min_adj = jnp.asarray(1.0, dtype=jnp.float64)
        for _ in range(_ELEMENTAL_REPAIR_ITERS):
            A = jnp.einsum("zi,ie->e", y, E_mat)  # [H, e1..] column totals
            col_tot = jnp.sum(y[:, rep_cols_j], axis=0)  # adjuster columns
            Bs = E_mat[rep_cols_j, :].T * col_tot[None, :]  # per adjuster species
            cols = []
            for slots in rep_slots:  # (1+nrep, nrep) per group
                b = Bs[:, slots[0]]
                for j in slots[1:]:
                    b = b + Bs[:, j]
                cols.append(b)
            B = jnp.stack(cols, axis=1)
            Msys = B[1:, :] - targets[:, None] * B[0:1, :]
            rhs = targets * A[0] - A[1:]
            alpha = jnp.linalg.solve(Msys, rhs)  # (nrep,) additive factors
            min_adj = jnp.minimum(min_adj, jnp.min(1.0 + alpha))
            scale_vec = jnp.ones(ni, dtype=jnp.float64).at[rep_cols_j].set(1.0 + alpha[rep_group_j])
            y = y * scale_vec[None, :]
            y = y * (M / jnp.sum(y, axis=1))[:, None]
        return y, min_adj

    def _guess_y0(
        lnZ: Array | float,
        c_o: Array | float,
        s_o: Array | float,
        warm_y: Array,
        lnZ_ref: Array | float,
        c_o_ref: Array | float,
        s_o_ref: Array | float,
    ) -> Array:
        """Return the mask-scaled guess from a converged column, applying only the composition increments.

        A large perturbation is reached by small steps from a nearby converged state. The sulfur increment scales the
        S-bearing species and the projection repairs the oxygen they carry.
        """
        c_o_inc = c_o - c_o_ref
        base = warm_y * jnp.exp((s_o - s_o_ref) * sulfur_mask)[None, :]
        if co_fixed_o:
            # Layer by layer at fixed O: b_z (see bz_margin) keeps each layer's O total invariant, smooth in c_o.
            OC_z = (base * (nO_per_species * carbon_mask)[None, :]).sum(axis=1)
            OO_z = (base * (nO_per_species * o_only_mask)[None, :]).sum(axis=1)
            b_z = 1.0 + (1.0 - jnp.exp(c_o_inc)) * OC_z / OO_z  # (nz,)
            cofac = jnp.where(carbon_mask[None, :] > 0, jnp.exp(c_o_inc), 1.0)  # (1, ni)
            cofac = jnp.where(o_only_mask[None, :] > 0, b_z[:, None], cofac)  # (nz, ni)
            y0p = base * jnp.exp((lnZ - lnZ_ref) * metal_mask)[None, :] * cofac
        else:
            scale = jnp.exp((lnZ - lnZ_ref) * metal_mask + c_o_inc * carbon_mask)  # (ni,)
            y0p = base * scale[None, :]
        return y0p

    return SimpleNamespace(
        targets=_targets,
        project=_elemental_project,
        guess=_guess_y0,
        margin=co_bz_margin,
        pairs=elem_pairs,
        E_np=E_np,
        R0_ratios=R0_ratios,
        names=_names,
        kinds=(_zk, _ck, _sk),
    )
