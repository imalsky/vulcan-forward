"""Correlated-k unit contracts.

These run without exojax on purpose: ``vulcan_forward.ckd`` never imports it,
so the (T, P) interpolation and the overlap are testable in a bare
environment. The one exojax-backed test (the interpolation oracle) skips via
``find_spec``, never ``pytest.importorskip("exojax")``, which poisons the
import-order contract for every later module in the session.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import jax

# x64 mirrors production; without it the 1e-12 oracle agreement below would
# read ~1e-6 and fail loudly rather than pass silently, but set it anyway.
jax.config.update("jax_enable_x64", True)

HAVE_EXOJAX = importlib.util.find_spec("exojax") is not None
if HAVE_EXOJAX and importlib.util.find_spec("vulcan_jax") is not None:
    # import-order contract: vulcan_chem before anything exojax, in every
    # collection order (test_contract's geometry test imports exojax_rt)
    from vulcan_forward import vulcan_chem  # noqa: F401

from vulcan_forward import ckd  # noqa: E402


def gauss_legendre(ng):
    """g-ordinates and weights on [0, 1]: a synthetic quadrature for these
    unit tests (the engine takes the tables' own)."""
    g, w = np.polynomial.legendre.leggauss(int(ng))
    return 0.5 * (g + 1.0), 0.5 * w


def _mean_conservation_error(ng, sigma, seed):
    """Relative error in the g-weighted mean optical depth after resort-rebin."""
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    nl, nb = 4, 6
    g, w = gauss_legendre(ng)
    a = np.sort(rng.lognormal(-2.0, sigma, size=(nl, ng, nb)), axis=1)
    b = np.sort(rng.lognormal(-1.0, sigma, size=(nl, ng, nb)), axis=1)
    out = np.asarray(ckd.overlap(jnp.asarray(a), jnp.asarray(b),
                                 jnp.asarray(g), jnp.asarray(w)))
    got = np.einsum("g,lgb->lb", w, out)
    want = np.einsum("g,lgb->lb", w, a) + np.einsum("g,lgb->lb", w, b)
    return np.abs(got / want - 1.0)


def test_overlap_nearly_conserves_the_band_mean_and_converges_in_ng():
    """Random overlap redistributes optical depth across g-ordinates and then
    re-interpolates ng*ng combinations back onto ng ordinates, so the
    g-weighted mean is conserved only to the accuracy of that rebin.

    It has to be small, because the transit radius goes as ln(tau): a 1 percent
    error in mean optical depth is 0.01 scale heights, about 8 km on WASP-39 b,
    about 3 ppm of transit depth. And the rebin, the only approximation in the
    mixture treatment, must shrink as ng rises, or the quadrature choice would
    be unfalsifiable.
    """
    err = _mean_conservation_error(16, 1.0, 0)  # ExoMolOP tables carry ng=16
    assert np.median(err) < 0.01
    assert err.max() < 0.05
    coarse = np.median(_mean_conservation_error(8, 1.0, 1))
    fine = np.median(_mean_conservation_error(32, 1.0, 1))
    assert fine < coarse


def test_overlap_returns_a_monotone_g_ordering():
    """k(g) is a sorted distribution by construction; a non-monotone result
    would break the quadrature."""
    import jax.numpy as jnp
    rng = np.random.default_rng(1)
    ng = 16
    g, w = gauss_legendre(ng)
    a = np.sort(rng.lognormal(0.0, 2.0, size=(2, ng, 3)), axis=1)
    b = np.sort(rng.lognormal(0.0, 2.0, size=(2, ng, 3)), axis=1)
    out = np.asarray(ckd.overlap(jnp.asarray(a), jnp.asarray(b),
                                 jnp.asarray(g), jnp.asarray(w)))
    assert np.all(np.diff(out, axis=1) >= -1e-12)


def test_overlap_is_differentiable():
    """The retrieval and the Fisher forecast push forward-mode tangents through
    the whole opacity path, so the sort and the rebin must carry them."""
    import jax
    import jax.numpy as jnp
    ng = 8
    g, w = gauss_legendre(ng)
    gg, gw = jnp.asarray(g), jnp.asarray(w)
    a = jnp.asarray(np.sort(np.linspace(0.1, 2.0, ng))[None, :, None])
    b = jnp.asarray(np.sort(np.linspace(0.2, 1.0, ng))[None, :, None])

    def f(scale):
        return jnp.sum(ckd.overlap(a * scale, b, gg, gw))

    val, tan = jax.jvp(f, (1.0,), (1.0,))
    assert np.isfinite(float(val)) and np.isfinite(float(tan))
    assert float(tan) > 0.0


def test_fold_wo_is_bit_identical_to_naive_refolds():
    """The prefix-memoized leave-one-out fold must be the EXACT op sequence of
    a naive left fold per output -- overlap is a resort-rebin, so it is not
    associative and a zero operand is not guaranteed to pass through as an
    exact identity; any reordering would move the spectrum at the rebin-error
    scale the pRT verification tolerances live at. Grouped contract: full ==
    naive fold, every wo row == naive refold with the zero operand, order is
    load-bearing, empty wo_idx returns the full fold only, out-of-range
    indices refuse."""
    import jax.numpy as jnp
    rng = np.random.default_rng(3)
    nl, ng, nb, n = 3, 8, 4, 5
    g, w = gauss_legendre(ng)
    gg, gw = jnp.asarray(g), jnp.asarray(w)
    dts = [jnp.asarray(np.sort(rng.lognormal(-1.0, 1.0, size=(nl, ng, nb)),
                               axis=1)) for _ in range(n)]
    zero = jnp.zeros((nl, ng, nb))

    def naive(seq):
        tot = None
        for d in seq:
            tot = d if tot is None else ckd.overlap(tot, d, gg, gw)
        return tot

    wo_idx = [0, 2, 4]
    full, wo = ckd._fold_wo(dts, lambda i: zero, gg, gw, wo_idx)
    assert np.array_equal(np.asarray(full), np.asarray(naive(dts)))
    assert [i for i, _ in wo] == wo_idx
    for i, got in wo:
        want = naive([zero if j == i else dts[j] for j in range(n)])
        assert np.array_equal(np.asarray(got), np.asarray(want)), i
    # order is load-bearing: permuting the operands changes bits
    perm, _ = ckd._fold_wo(dts[::-1], lambda i: zero, gg, gw, [])
    assert not np.array_equal(np.asarray(perm), np.asarray(full))
    # empty wo_idx: full only; finish is applied per wo row
    full2, none = ckd._fold_wo(dts, lambda i: zero, gg, gw, [])
    assert none == [] and np.array_equal(np.asarray(full2), np.asarray(full))
    _, summed = ckd._fold_wo(dts, lambda i: zero, gg, gw, [1],
                             finish=lambda t: float(jnp.sum(t)))
    assert isinstance(summed[0][1], float)
    with pytest.raises(ValueError, match="outside"):
        ckd._fold_wo(dts, lambda i: zero, gg, gw, [n])


def test_fold_is_bit_identical_to_the_python_loop_fold():
    """``fold`` runs the molecule folds under one ``lax.scan`` so a gradient
    keeps a third of the unrolled loop's memory. It must stay the naive left
    fold's op sequence: the primal, a jvp and a vjp bitwise equal (both
    jitted, as production runs them), with enough molecules for several scan
    steps. Bitwise on the CPU; on the GH200 the two differ in the last bits
    (XLA fuses a scan body and straight-line code differently and rounds
    inside a fusion accordingly), so there the bar is rounding."""
    import jax.numpy as jnp
    rng = np.random.default_rng(4)
    nl, ng, nb, n = 3, 8, 4, 5
    g, w = gauss_legendre(ng)
    gg, gw = jnp.asarray(g), jnp.asarray(w)
    dts = jnp.asarray(np.sort(rng.lognormal(-1.0, 1.0, size=(n, nl, ng, nb)),
                              axis=2))
    tan = jnp.asarray(rng.normal(size=dts.shape))
    cot = jnp.asarray(rng.normal(size=(nl, ng, nb)))

    def naive(d):
        tot = d[0]
        for i in range(1, n):
            tot = ckd.overlap(tot, d[i], gg, gw)
        return tot

    def scan(d):
        return ckd.fold(d, gg, gw)

    def jvp_vjp(f):
        p, t = jax.jit(lambda d: jax.jvp(f, (d,), (tan,)))(dts)
        v = jax.jit(lambda d: jax.vjp(f, d)[1](cot)[0])(dts)
        return p, t, v

    rtol = 0.0 if jax.default_backend() == "cpu" else 1e-12
    for got, want in zip(jvp_vjp(scan), jvp_vjp(naive)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=rtol, atol=0.0, equal_nan=False)
    # a single molecule is its own fold
    assert np.array_equal(np.asarray(ckd.fold(dts[:1], gg, gw)),
                          np.asarray(dts[0]))


def test_interp_logk_clamps_outside_the_table_rather_than_extrapolating():
    """Extrapolating log k off the end of a k-table produces nonsense opacity;
    clamping is the documented behaviour and the caller validates the span."""
    import jax.numpy as jnp
    t = jnp.asarray(np.linspace(500.0, 2000.0, 4))
    p = jnp.asarray(np.logspace(-6, 1, 3))
    logk = jnp.asarray(np.arange(4 * 3 * 2 * 1, dtype=float).reshape(4, 3, 2, 1))
    lo = np.asarray(ckd._interp_logk(logk, t, p, jnp.asarray([100.0]),
                                     jnp.asarray([1e-9])))
    hi = np.asarray(ckd._interp_logk(logk, t, p, jnp.asarray([9000.0]),
                                     jnp.asarray([1e3])))
    assert np.allclose(lo[0], np.asarray(logk)[0, 0])
    assert np.allclose(hi[0], np.asarray(logk)[-1, -1])


@pytest.mark.skipif(not HAVE_EXOJAX, reason="exojax not installed (light CI)")
def test_interp_logk_matches_exojax_interpolate_log_k_2d():
    """Independent implementation of the same algorithm. exojax's
    ``interpolate_log_k_2d`` (opacity/ckd/core.py) is bilinear in (T, log P)
    on log k with jnp.interp clamping, on the same (nT, nP, ng, nband)
    layout, but interpolates over T per P-column and then over log P, where
    ours forms fractional indices and blends four corners. Agreement is
    float64 rounding: measured max |dlogk| 2.8e-14 (77% bit-identical) at
    off-grid and out-of-range points; 1e-12 is 35x that and nine orders below
    the float32 error the x64 contract exists to prevent."""
    import jax.numpy as jnp
    from exojax.opacity.ckd.core import interpolate_log_k_2d

    rng = np.random.default_rng(0)
    t = jnp.asarray(np.linspace(300.0, 3000.0, 6))
    p = jnp.asarray(np.logspace(-5, 2, 5))
    logk = jnp.asarray(rng.normal(-60.0, 20.0, size=(6, 5, 4, 7)))
    T = jnp.asarray([100.0, 350.0, 1234.5, 2999.0, 3000.0, 5000.0])
    P = jnp.asarray([1e-9, 1e-5, 3.3e-3, 0.7, 99.9, 1e3])
    ours = np.asarray(ckd._interp_logk(logk, t, p, T, P))
    ref = np.stack([np.asarray(interpolate_log_k_2d(logk, t, p, T[i], P[i]))
                    for i in range(6)])
    assert ours.dtype == np.float64
    assert np.allclose(ours, ref, rtol=0.0, atol=1e-12)
