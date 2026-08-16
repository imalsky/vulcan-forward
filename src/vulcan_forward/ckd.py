"""Correlated-k core: quadrature, band grid, (T, P) interpolation, overlap.

WHY THIS EXISTS
---------------
exojax computes a cross section directly ON the output wavenumber grid; there
is no internal high-resolution grid, and its own ``wavenumber_grid`` warns for
any grid below R = 700,000. A JWST-band model sampled at R ~ 1,500 is therefore
a strided sample of a spectrum that was never resolved, and the sampling error
does not average out -- it biases both observables (measurements: README,
"Opacity: correlated-k, not sampled line-by-line"). Correlated-k does the
expensive integration once, offline, and compresses each band to a few
g-ordinates.

The k-tables themselves come from ExoMolOP (``vulcan_forward.exomolop``); this
module holds the source-independent machinery that consumes them.

MIXTURES
--------
exojax 2.2.3 ships no overlap treatment: ``opacity_profile_xs_ckd`` takes one
species. Tables are per species, because the composition changes every run, so
they are combined here by random-overlap resort-rebin, the standard approach
(petitRADTRANS, PICASO, HELIOS). It is written in JAX and is differentiable,
so the forward-mode tangents the retrieval and the Fisher forecast rely on
still pass through.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def gauss_legendre(ng: int):
    """g-ordinates and weights on [0, 1]."""
    g, w = np.polynomial.legendre.leggauss(int(ng))
    return 0.5 * (g + 1.0), 0.5 * w


def band_edges(nu_min: float, nu_max: float, r_band: float) -> np.ndarray:
    """Log-uniform band edges at resolution ``r_band``, inclusive of nu_max."""
    e = np.exp(np.arange(np.log(nu_min), np.log(nu_max) + 0.5 / r_band,
                         1.0 / r_band))
    e = e[e <= nu_max * (1.0 + 1e-12)]
    if e.size < 2:
        raise ValueError(
            f"band_edges({nu_min}, {nu_max}, {r_band}) produced {e.size} "
            "edges: the band is narrower than one resolution element")
    return e


def _interp_logk(logk, t_grid, p_grid, T, P):
    """Bilinear interpolation of log k in (T, log P).

    logk : (nT, nP, ng, nband); T, P : (nlayer,). Returns (nlayer, ng, nband).
    Linear in T and in log P, clamped at the edges. Differentiable in T, which
    is what the Fisher forecast and the retrieval tangents need.
    """
    lt = jnp.clip(jnp.interp(T, t_grid, jnp.arange(t_grid.size, dtype=T.dtype)),
                  0.0, t_grid.size - 1.0)
    lp = jnp.clip(jnp.interp(jnp.log(P), jnp.log(p_grid),
                             jnp.arange(p_grid.size, dtype=P.dtype)),
                  0.0, p_grid.size - 1.0)
    i0 = jnp.floor(lt).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, t_grid.size - 1)
    ft = (lt - i0)[:, None, None]
    j0 = jnp.floor(lp).astype(jnp.int32)
    j1 = jnp.minimum(j0 + 1, p_grid.size - 1)
    fp = (lp - j0)[:, None, None]
    a = logk[i0, j0] * (1 - ft) + logk[i1, j0] * ft
    b = logk[i0, j1] * (1 - ft) + logk[i1, j1] * ft
    return a * (1 - fp) + b * fp


def overlap(a, b, gg, gw):
    """Random-overlap resort-rebin of two (nlayer, ng, nband) g-tables.

    Forms every pairwise sum with weight w_i w_j, sorts, accumulates the weight
    into a cumulative g, and re-interpolates onto the ng-point grid. This is
    the standard treatment of overlapping absorbers in correlated-k; it assumes
    the two absorbers' line positions are uncorrelated within a band, which is
    the usual and well-tested assumption for unrelated molecules.

    The nodes/weights are ARGUMENTS, not a module assumption: ExoMolOP tables
    carry petitRADTRANS' split quadrature (8 points on [0, 0.9] + 8 on
    [0.9, 1]), not plain Gauss-Legendre, and they flow through unchanged.
    """
    nl, ng, nb = a.shape
    s = (a[:, :, None, :] + b[:, None, :, :]).reshape(nl, ng * ng, nb)
    w = jnp.broadcast_to((gw[:, None] * gw[None, :])[None, :, :, None],
                         (nl, ng, ng, nb)).reshape(nl, ng * ng, nb)
    idx = jnp.argsort(s, axis=1)
    ss = jnp.take_along_axis(s, idx, axis=1)
    ws = jnp.take_along_axis(w, idx, axis=1)
    cg = jnp.cumsum(ws, axis=1) - 0.5 * ws
    f = jax.vmap(jax.vmap(lambda c, v: jnp.interp(gg, c, v),
                          in_axes=(1, 1), out_axes=1),
                 in_axes=(0, 0), out_axes=0)
    return f(cg, ss)


def _fold_wo(dts, zero_of, gg, gw, wo_idx, finish=None):
    """Left-fold ``dts`` with ``overlap``; for each i in ``wo_idx`` also produce
    the same fold with ``dts[i]`` replaced by ``zero_of(i)``, reusing the
    running prefix. Returns ``(full_total, [(i, finish(total_wo)), ...])`` with
    the wo list in ascending-i order.

    Bit-identity contract: every wo result is the EXACT op sequence of a naive
    left fold over ``[dts[0], .., zero_of(i), .., dts[-1]]`` — the fold order is
    load-bearing (``overlap`` is a resort-rebin, neither associative nor an
    exact identity on a zero operand), and a dropped absorber is still folded,
    as the zero tensor its zeroed VMR produces. Only the shared prefix
    ``dts[0..i-1]`` is computed once instead of per wo (~2x fewer folds).

    ``finish`` maps each wo total to its observable before the next fold starts,
    so at most one (nlayer, ng, nband) wo total is alive at a time.
    """
    if finish is None:
        finish = lambda t: t  # noqa: E731
    n = len(dts)
    wo_idx = set(wo_idx)
    bad = [i for i in wo_idx if not 0 <= i < n]
    if bad:
        raise ValueError(f"_fold_wo: wo indices {sorted(bad)} outside 0..{n-1}")
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
    return prefix, out
