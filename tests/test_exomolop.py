"""Contracts for the ExoMolOP k-table ingestion.

Most of these build tiny synthetic HDF5 files rather than touching the real
389 MB tables, so the suite stays fast and runs with no data installed (this
package must be importable and testable empty -- see test_contract.py). The
two tests that DO need a real table skip when it is absent.

The thing most worth pinning here is the refusal to mix tables that disagree
on their band grid or quadrature. ``ckd.overlap`` combines species ordinate by
ordinate; two tables on different g-nodes would be added as though they were
the same distribution and would return a plausible, wrong spectrum.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import jax

# x64 mirrors production (vulcan_chem / exojax_rt enable it at import). Without
# it load_tables refuses rather than silently handing back float32 tables --
# which is itself pinned, in test_load_refuses_without_x64.
jax.config.update("jax_enable_x64", True)

from vulcan_forward import exomolop, paths   # noqa: E402

HAVE_H5PY = importlib.util.find_spec("h5py") is not None
pytestmark = pytest.mark.skipif(not HAVE_H5PY, reason="h5py not installed")

NT, NP, NB, NG = 4, 3, 20, 8


def _write(path, *, nb=NB, ng=NG, nu0=100.0, r=1000.0, kscale=1.0,
           zeros=False, g_shift=0.0):
    """A miniature file in ExoMolOP's petitRADTRANS layout."""
    import h5py
    edges = nu0 * np.exp(np.arange(nb + 1) / r)
    t = np.linspace(300.0, 3000.0, NT)
    p = np.logspace(-5, 2, NP)
    g = np.linspace(0.02, 0.98, ng) + g_shift
    w = np.full(ng, 1.0 / ng)
    rng = np.random.default_rng(0)
    # ExoMolOP order: (n_P, n_T, n_band, n_g), k ascending in g
    k = np.sort(rng.lognormal(-50.0, 1.0, size=(NP, NT, nb, ng)),
                axis=-1) * kscale
    if zeros:
        k[0, 0, 0, :] = 0.0
    with h5py.File(path, "w") as f:
        f["kcoeff"] = k
        f["bin_edges"] = edges
        f["bin_centers"] = np.sqrt(edges[:-1] * edges[1:])
        f["t"] = t
        f["p"] = p
        f["samples"] = g
        f["weights"] = w
        f["mol_mass"] = np.array([18])
        f["ngauss"] = ng
    return k, edges, t, p, g, w


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    (tmp_path / "exomolop").mkdir()
    return tmp_path


def test_paths_come_from_the_data_root_not_the_package(data_root):
    """Library contract: never resolve data from the package's own __file__."""
    assert exomolop.table_dir() == data_root / "exomolop"
    assert exomolop.table_path("H2O").name == "H2O.ktable.h5"


def test_missing_table_raises_with_the_fetch_command(data_root):
    """Standing fail-loud rule: name the remedy, and never download 389 MB
    behind the caller's back."""
    with pytest.raises(FileNotFoundError) as e:
        exomolop.load_tables(["H2O"], 100.0, 200.0)
    msg = str(e.value)
    assert "vulcan_forward.fetch_exomolop" in msg
    assert "--molecules H2O" in msg


def test_unknown_molecule_raises_before_touching_the_disk(data_root):
    with pytest.raises(KeyError):
        exomolop.load_tables(["NOT_A_MOLECULE"], 100.0, 200.0)


def test_layout_is_transposed_to_the_engine_order(data_root):
    """ExoMolOP stores (n_P, n_T, n_band, n_g); everything downstream indexes
    (n_T, n_P, n_g, n_band). A wrong transpose still returns finite numbers of
    the right total size, so it has to be pinned elementwise."""
    k, edges, t, p, g, w = _write(exomolop.table_path("H2O"))
    pack = exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)
    got = np.asarray(pack.logk["H2O"])
    assert got.shape == (NT, NP, NG, NB)
    want = np.log(k.transpose(1, 0, 3, 2))
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(pack.t_grid), t)
    assert np.allclose(np.asarray(pack.p_grid), p)
    assert np.allclose(np.asarray(pack.gg), g)
    assert np.allclose(np.asarray(pack.gw), w)
    assert pack.ng == NG


def test_only_bands_fully_inside_the_request_are_kept(data_root):
    _, edges, *_ = _write(exomolop.table_path("H2O"))
    lo, hi = edges[5], edges[12]
    pack = exomolop.load_tables(["H2O"], lo, hi, verbose=False)
    assert pack.band_edges[0] >= lo and pack.band_edges[-1] <= hi
    assert pack.nu_bands.size == 7
    # centres are the geometric means of the kept edges
    assert np.allclose(pack.nu_bands,
                       np.sqrt(pack.band_edges[:-1] * pack.band_edges[1:]))


def test_a_request_outside_the_table_raises(data_root):
    _, edges, *_ = _write(exomolop.table_path("H2O"))
    with pytest.raises(ValueError, match="no band inside"):
        exomolop.load_tables(["H2O"], edges[-1] * 10, edges[-1] * 20,
                             verbose=False)


def test_zero_cross_sections_floor_instead_of_going_to_minus_infinity(data_root):
    """Their tables carry exact zeros where a species has no lines in a band.
    log(0) would poison the bilinear (T, P) interpolation for every layer, not
    just that band, so it must floor."""
    _, edges, *_ = _write(exomolop.table_path("H2O"), zeros=True)
    pack = exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)
    lk = np.asarray(pack.logk["H2O"])
    assert np.all(np.isfinite(lk))
    assert lk.min() == pytest.approx(np.log(exomolop.K_FLOOR))


@pytest.mark.parametrize("kw,what", [
    (dict(g_shift=0.01), "g ordinates"),
    (dict(nb=NB - 1), "band grid"),
    (dict(nu0=101.0), "band grid"),
])
def test_tables_that_disagree_on_grid_or_quadrature_are_refused(data_root, kw,
                                                                what):
    """The mixing is ordinate by ordinate, so tables from different releases
    cannot be combined -- and the failure would be silent, not a crash."""
    _write(exomolop.table_path("H2O"))
    _write(exomolop.table_path("CO2"), **kw)
    with pytest.raises(ValueError, match="disagree on the"):
        exomolop.load_tables(["H2O", "CO2"], 1.0, 1.0e6, verbose=False)


def test_matching_tables_combine(data_root):
    _, edges, *_ = _write(exomolop.table_path("H2O"))
    _write(exomolop.table_path("CO2"), kscale=3.0)
    pack = exomolop.load_tables(["H2O", "CO2"], edges[0], edges[-1],
                                verbose=False)
    assert set(pack.logk) == {"H2O", "CO2"}


def test_available_lists_what_is_installed(data_root):
    assert exomolop.available() == []
    _write(exomolop.table_path("H2O"))
    _write(exomolop.table_path("CO2"))
    assert exomolop.available() == ["CO2", "H2O"]


# --------------------------------------------------------------------------
# Against a REAL table, when one is installed.
# --------------------------------------------------------------------------

def _need_real():
    """Resolve at CALL time, not import time: test_contract.py deliberately
    clears the data root to prove the library imports with no data installed,
    so a module-level skipif marker would be evaluated against whichever
    environment happened to exist when this module was collected."""
    try:
        p = exomolop.table_path("H2O")
    except RuntimeError:                    # no data root configured
        pytest.skip("no $VULCAN_FORWARD_DATA configured")
    if not p.exists():
        pytest.skip(f"no ExoMolOP H2O table at {p} "
                    "(python -m vulcan_forward.fetch_exomolop)")


def test_real_table_uses_the_petitradtrans_split_quadrature():
    """Their quadrature is NOT plain Gauss-Legendre: 8 points on [0, 0.9] plus
    8 on [0.9, 1.0]. ckd.overlap takes the nodes as arguments so nothing had to
    change to accept it, but if a future release switched to a flat rule the
    weights would stop splitting 0.9/0.1 and this is where we would find out.
    """
    _need_real()
    pack = exomolop.load_tables(["H2O"], 2000.0, 2100.0, verbose=False)
    w = np.asarray(pack.gw)
    g = np.asarray(pack.gg)
    assert pack.ng == 16
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert w[:8].sum() == pytest.approx(0.9, abs=1e-6)
    assert w[8:].sum() == pytest.approx(0.1, abs=1e-6)
    assert np.all((g > 0.0) & (g < 1.0))
    assert np.all(np.diff(g) > 0.0)


def test_real_table_is_a_cross_section_per_molecule_not_per_gram():
    """Units were verified empirically, not assumed, and they are load-bearing:
    ``opacity_profile_xs_ckd`` multiplies by mmr and divides by the molecular
    mass, so a cm^2/g table would come out ~3.3e22 times too large. An H2O
    band-mean cross section at 1200 K and 0.1 bar is of order 1e-23 to 1e-19
    cm^2/molecule; per gram it would be 1e-1 to 1e3.
    """
    _need_real()
    pack = exomolop.load_tables(["H2O"], 2400.0, 2600.0, verbose=False)
    lk = np.asarray(pack.logk["H2O"])
    t = np.asarray(pack.t_grid)
    p = np.asarray(pack.p_grid)
    it = int(np.argmin(np.abs(t - 1200.0)))
    ip = int(np.argmin(np.abs(p - 0.1)))
    kbar = np.exp(lk[it, ip]).T @ np.asarray(pack.gw)     # (nband,)
    med = float(np.median(kbar))
    assert 1e-26 < med < 1e-17, med


def test_load_refuses_without_x64(data_root, monkeypatch):
    """float32 k-tables are a ~1e-5 relative error on every cross section that
    would surface as an unexplained spectrum shift, never as a crash. Refuse."""
    _, edges, *_ = _write(exomolop.table_path("H2O"))

    class _Off:
        jax_enable_x64 = False

    # load_tables does `from jax import config`, i.e. getattr(jax, "config"),
    # so swapping the attribute is enough; jax's real Config has no setter and
    # flipping x64 globally mid-session would poison every later test.
    monkeypatch.setattr(jax, "config", _Off())
    with pytest.raises(RuntimeError, match="x64"):
        exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)


def test_tables_are_float64(data_root):
    _, edges, *_ = _write(exomolop.table_path("H2O"))
    pack = exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)
    assert pack.logk["H2O"].dtype == np.float64
