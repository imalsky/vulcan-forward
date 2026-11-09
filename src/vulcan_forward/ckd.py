"""Correlated-k core: (T, P) interpolation of log k and the random-overlap mixture of species.

The tables come from ``exomolop``; docs/method.md gives the method and its sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jax import Array

__all__ = ["fold", "overlap"]

_Finished = TypeVar("_Finished")


def _interp_logk(logk: Array, t_grid: Array, p_grid: Array, T: Array, P: Array) -> Array:
    """Interpolate log k bilinearly in (T, log P), clamped at the edges: [nT, nP, ng, nband] to [nlayer, ng, nband]."""
    lt = jnp.clip(jnp.interp(T, t_grid, jnp.arange(t_grid.size, dtype=T.dtype)), 0.0, t_grid.size - 1.0)
    lp = jnp.clip(
        jnp.interp(jnp.log(P), jnp.log(p_grid), jnp.arange(p_grid.size, dtype=P.dtype)), 0.0, p_grid.size - 1.0
    )
    i0 = jnp.floor(lt).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, t_grid.size - 1)
    ft = (lt - i0)[:, None, None]
    j0 = jnp.floor(lp).astype(jnp.int32)
    j1 = jnp.minimum(j0 + 1, p_grid.size - 1)
    fp = (lp - j0)[:, None, None]
    a = logk[i0, j0] * (1 - ft) + logk[i1, j0] * ft
    b = logk[i0, j1] * (1 - ft) + logk[i1, j1] * ft
    return a * (1 - fp) + b * fp


def overlap(a: Array, b: Array, gg: Array, gw: Array) -> Array:
    """Mix two g-tables by random-overlap resort-rebin.

    Forms every pairwise sum with weight w_i w_j, sorts, accumulates the weight into a cumulative g, and re-interpolates
    onto the ng-point grid (Amundsen et al. 2017). The nodes and weights are the table's split quadrature, not plain
    Gauss-Legendre.

    Parameters
    ----------
    a : jax.Array [nlayer, ng, nband]
        The first table.
    b : jax.Array [nlayer, ng, nband]
        The second table.
    gg : jax.Array [ng]
        Quadrature nodes.
    gw : jax.Array [ng]
        Quadrature weights.

    Returns
    -------
    jax.Array [nlayer, ng, nband]
        The mixture.
    """
    nl, ng, nb = a.shape
    at = jnp.swapaxes(a, 1, 2)  # (nl, nb, ng)
    bt = jnp.swapaxes(b, 1, 2)
    s = (at[:, :, :, None] + bt[:, :, None, :]).reshape(nl, nb, ng * ng)
    w = jnp.broadcast_to((gw[:, None] * gw[None, :]).reshape(ng * ng), (nl, nb, ng * ng))
    # One stable two-operand sort along the last axis carries the weights with the sums. On the GPU it is ~4x faster
    # than an argsort along the middle axis plus two gathers, and bitwise the same. An ulp in the cumulative g moves the
    # spectrum by up to ~1e-9, so the weights stay a sorted payload rather than being re-gathered.
    ss, ws = jax.lax.sort((s, w), dimension=2, num_keys=1, is_stable=True)
    cg = jnp.cumsum(ws, axis=2) - 0.5 * ws
    f = jax.vmap(jax.vmap(lambda c, v: jnp.interp(gg, c, v)))
    return jnp.swapaxes(f(cg, ss), 1, 2)  # (nl, ng, nb)


def fold(dts: Array, gg: Array, gw: Array) -> Array:
    """Left-fold a stack of g-tables with ``overlap``, in stack order.

    The op sequence is that of a Python loop over the molecules, so primal, jvp and vjp match the loop bitwise on the
    CPU; other backends round inside their fusions differently. Under one ``lax.scan`` a reverse-mode gradient keeps a
    third of the loop's memory and runs faster.

    Parameters
    ----------
    dts : jax.Array [nmol, nlayer, ng, nband]
        The per-molecule tables.
    gg : jax.Array [ng]
        Quadrature nodes.
    gw : jax.Array [ng]
        Quadrature weights.

    Returns
    -------
    jax.Array [nlayer, ng, nband]
        The mixture.
    """
    tot, _ = jax.lax.scan(lambda t, dt: (overlap(t, dt, gg, gw), None), dts[0], dts[1:])
    return tot


def _fold_wo(
    dts: list[Array],
    zero_of: Callable[[int], Array],
    gg: Array,
    gw: Array,
    wo_idx: Iterable[int],
    finish: Callable[[Array], _Finished],
) -> tuple[Array, list[tuple[int, _Finished]]]:
    """Left-fold ``dts``; for each i in ``wo_idx`` also fold with ``dts[i]`` replaced by ``zero_of(i)``.

    Returns ``(full_total, [(i, finish(total_wo)), ...])`` in ascending i. Each wo result is the exact op sequence of a
    naive left fold: ``overlap`` is neither associative nor an identity on a zero operand, so a dropped absorber is
    still folded, as the zero tensor its zeroed VMR produces. Only the prefix ``dts[0..i-1]`` is shared (all n species
    as wo: (n-1)(n/2+2) folds instead of (n+1)(n-1)). ``finish`` runs before the next fold, so one wo total is alive at
    a time.
    """
    n = len(dts)
    wo_idx = set(wo_idx)
    out = []
    prefix = None
    for i in range(n):
        if i in wo_idx:
            z = zero_of(i)
            t = z if prefix is None else overlap(prefix, z, gg, gw)
            for j in range(i + 1, n):
                t = overlap(t, dts[j], gg, gw)
            out.append((i, finish(t)))
        prefix = dts[i] if prefix is None else overlap(prefix, dts[i], gg, gw)
    return prefix, out  # type: ignore[return-value]  # dts is never empty: the caller folds at least one
