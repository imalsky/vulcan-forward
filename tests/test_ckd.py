"""Correlated-k unit contracts.

These run without exojax on purpose: ``vulcan_forward.ckd`` never imports it,
so the quadrature, the band grid, the (T, P) interpolation and the overlap are
all testable in a bare environment. (Never ``pytest.importorskip("exojax")``
in this repo -- it poisons the import-order contract for every later module in
the session.)
"""
from __future__ import annotations

import numpy as np
import pytest

from vulcan_forward import ckd


def test_gauss_legendre_weights_sum_to_one():
    """The g-ordinates partition [0, 1]; the weights are a probability measure
    over the band, so they must sum to exactly 1 or every k-table is
    misnormalized."""
    for ng in (4, 8, 16, 32):
        g, w = ckd.gauss_legendre(ng)
        assert g.shape == w.shape == (ng,)
        assert np.all((g > 0.0) & (g < 1.0))
        assert np.all(np.diff(g) > 0.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_gauss_legendre_integrates_a_linear_k_exactly():
    """Sanity that the quadrature is on [0, 1] and not [-1, 1]."""
    g, w = ckd.gauss_legendre(8)
    assert float(w @ g) == pytest.approx(0.5, abs=1e-12)


def test_band_edges_are_log_uniform_at_the_requested_resolution():
    e = ckd.band_edges(667.0, 10000.0, 500.0)
    r = np.diff(np.log(e))
    assert np.allclose(r, 1.0 / 500.0, rtol=1e-9)
    assert e[0] == pytest.approx(667.0)
    assert e[-1] <= 10000.0 * (1.0 + 1e-12)
    # a band grid that silently dropped most of the range would be a very
    # quiet way to model the wrong spectrum
    assert e[-1] > 10000.0 * np.exp(-1.0 / 500.0)


def test_band_edges_refuses_a_band_narrower_than_one_element():
    with pytest.raises(ValueError, match="narrower than one resolution"):
        ckd.band_edges(1000.0, 1000.5, 100.0)


def _grey(nl, ng, nb, value):
    import jax.numpy as jnp
    return jnp.full((nl, ng, nb), value)


def test_overlap_of_two_grey_absorbers_is_their_sum():
    """A grey absorber has the same k at every g. Overlapping two of them must
    reproduce the plain sum exactly; any spread means the resort-rebin is
    inventing structure."""
    import jax.numpy as jnp
    ng = 16
    g, w = ckd.gauss_legendre(ng)
    gg, gw = jnp.asarray(g), jnp.asarray(w)
    out = np.asarray(ckd.overlap(_grey(3, ng, 5, 0.25), _grey(3, ng, 5, 0.75),
                                 gg, gw))
    assert np.allclose(out, 1.0, atol=1e-10)


def _mean_conservation_error(ng, sigma, seed):
    """Relative error in the g-weighted mean optical depth after resort-rebin."""
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    nl, nb = 4, 6
    g, w = ckd.gauss_legendre(ng)
    a = np.sort(rng.lognormal(-2.0, sigma, size=(nl, ng, nb)), axis=1)
    b = np.sort(rng.lognormal(-1.0, sigma, size=(nl, ng, nb)), axis=1)
    out = np.asarray(ckd.overlap(jnp.asarray(a), jnp.asarray(b),
                                 jnp.asarray(g), jnp.asarray(w)))
    got = np.einsum("g,lgb->lb", w, out)
    want = np.einsum("g,lgb->lb", w, a) + np.einsum("g,lgb->lb", w, b)
    return np.abs(got / want - 1.0)


def test_overlap_nearly_conserves_the_band_mean_optical_depth():
    """Random overlap redistributes optical depth across g-ordinates and then
    re-interpolates ng*ng combinations back onto ng ordinates, so the
    g-weighted mean is conserved only to the accuracy of that rebin.

    It has to be small, because the transit radius goes as ln(tau): a 1 percent
    error in mean optical depth is 0.01 scale heights, about 8 km on WASP-39 b,
    about 3 ppm of transit depth. That is consistent with the 4.7-7.3 ppm
    measured end to end against the R = 700,000 line-by-line spectrum.
    """
    err = _mean_conservation_error(16, 1.0, 0)  # ExoMolOP tables carry ng=16
    assert np.median(err) < 0.01
    assert err.max() < 0.05


def test_overlap_rebin_error_falls_as_ng_rises():
    """The rebin is the only approximation in the mixture treatment, so it must
    converge; if it did not, raising ng would not buy accuracy and the
    quadrature choice would be unfalsifiable."""
    coarse = np.median(_mean_conservation_error(8, 1.0, 1))
    fine = np.median(_mean_conservation_error(32, 1.0, 1))
    assert fine < coarse


def test_overlap_returns_a_monotone_g_ordering():
    """k(g) is a sorted distribution by construction; a non-monotone result
    would break the quadrature."""
    import jax.numpy as jnp
    rng = np.random.default_rng(1)
    ng = 16
    g, w = ckd.gauss_legendre(ng)
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
    g, w = ckd.gauss_legendre(ng)
    gg, gw = jnp.asarray(g), jnp.asarray(w)
    a = jnp.asarray(np.sort(np.linspace(0.1, 2.0, ng))[None, :, None])
    b = jnp.asarray(np.sort(np.linspace(0.2, 1.0, ng))[None, :, None])

    def f(scale):
        return jnp.sum(ckd.overlap(a * scale, b, gg, gw))

    val, tan = jax.jvp(f, (1.0,), (1.0,))
    assert np.isfinite(float(val)) and np.isfinite(float(tan))
    assert float(tan) > 0.0


def test_interp_logk_is_exact_on_its_own_nodes():
    import jax.numpy as jnp
    rng = np.random.default_rng(2)
    nt, npr, ng, nb = 5, 4, 3, 2
    t = jnp.asarray(np.linspace(300.0, 2900.0, nt))
    p = jnp.asarray(np.logspace(-8, 2, npr))
    logk = jnp.asarray(rng.normal(size=(nt, npr, ng, nb)))
    # one query layer per (T, P) node: interpolation must return that node
    ti = np.repeat(np.arange(nt), npr)
    pi = np.tile(np.arange(npr), nt)
    got = np.asarray(ckd._interp_logk(logk, t, p,
                                      jnp.asarray(np.asarray(t)[ti]),
                                      jnp.asarray(np.asarray(p)[pi])))
    for n, (i, j) in enumerate(zip(ti, pi)):
        assert np.allclose(got[n], np.asarray(logk)[i, j], atol=1e-9), (i, j)


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
    g, w = ckd.gauss_legendre(ng)
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
