"""Contracts for the ExoMolOP k-table ingestion and the fetcher.

Most tests build tiny synthetic HDF5 files rather than reading the real 389 MB tables, so they run with no data
installed; the tests that need a real table skip when it is absent. The refusal most worth pinning is to mix tables
that disagree on their band grid or quadrature: ``ckd.overlap`` combines species ordinate by ordinate, so two tables
on different g-nodes would be added as one distribution and return a plausible, wrong spectrum.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

import jax

# x64, as vulcan_chem and exojax_rt enable it at import. Without it load_tables refuses rather than hand back float32
# tables, which test_load_refuses_without_x64 pins.
jax.config.update("jax_enable_x64", True)

from vulcan_forward import constants, exomolop, fetch_exomolop, paths

HAVE_H5PY = importlib.util.find_spec("h5py") is not None
pytestmark = pytest.mark.skipif(not HAVE_H5PY, reason="h5py not installed")

DATA = Path(__file__).parent / "data"
NT, NP, NB, NG = 4, 3, 20, 8
ROUND_TOL = 1e-12  # float64 rounding bar
SPLIT_TOL = 1e-6  # the split quadrature's 0.9 / 0.1 weight sums
# exo_k fixture: band edges (cm^-1, absolute) and interpolated k (relative)
EXOK_TOL = 1e-9


def _write(
    path,
    *,
    nb=NB,
    nu0=100.0,
    zeros=False,
    g_shift=0.0,
    kunits="cm^2/molecule",
    punits="bar",
    method="petit_samples",
    ngauss=None,
):
    """A miniature file in ExoMolOP's petitRADTRANS layout, with the real files' unit attributes and string datasets."""
    import h5py

    edges = nu0 * np.exp(np.arange(nb + 1) / 1000.0)
    t = np.linspace(200.0, 3000.0, NT)
    p = np.logspace(-5, 2, NP)
    g = np.linspace(0.02, 0.98, NG) + g_shift
    w = np.full(NG, 1.0 / NG)
    rng = np.random.default_rng(0)
    # ExoMolOP order: (n_P, n_T, n_band, n_g), k ascending in g
    k = np.sort(rng.lognormal(-50.0, 1.0, size=(NP, NT, nb, NG)), axis=-1)
    if zeros:
        k[0, 0, 0, :] = 0.0
    with h5py.File(path, "w") as f:
        f["kcoeff"] = k
        f["kcoeff"].attrs["units"] = kunits
        f["bin_edges"] = edges
        f["bin_centers"] = np.sqrt(edges[:-1] * edges[1:])
        f["t"] = t
        f["p"] = p
        f["p"].attrs["units"] = punits
        f["samples"] = g
        f["weights"] = w
        f["mol_mass"] = np.array([18])
        f["ngauss"] = NG if ngauss is None else ngauss
        f.create_dataset("method", data=np.array([method], dtype=h5py.string_dtype()))
        f.create_dataset("DOI", data=np.array(["x.xxxx/yyyyy"], dtype=h5py.string_dtype()))
    return k, edges, t, p, g, w


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    (tmp_path / "exomolop").mkdir()
    return tmp_path


@pytest.mark.usefixtures("data_root")
def test_missing_table_raises_with_the_fetch_command():
    """A missing table raises with the fetch command; nothing is downloaded at run time."""
    with pytest.raises(FileNotFoundError) as e:
        exomolop.load_tables(["H2O"], 100.0, 200.0)
    msg = str(e.value)
    assert "vulcan_forward.fetch_exomolop" in msg
    assert "--molecules H2O" in msg


@pytest.mark.parametrize(("mols", "err"), [(["NOT_A_MOLECULE"], KeyError), (["H2O", "H2O"], ValueError)])
@pytest.mark.usefixtures("data_root")
def test_unknown_or_repeated_molecules_raise_before_touching_the_disk(mols, err):
    """A repeated name would fold its opacity in twice."""
    with pytest.raises(err):
        exomolop.load_tables(mols, 100.0, 200.0)


@pytest.mark.usefixtures("data_root")
def test_an_empty_molecule_list_is_refused():
    with pytest.raises(ValueError, match="at least one molecule"):
        exomolop.load_tables([], 2000.0, 3000.0)


@pytest.mark.usefixtures("data_root")
def test_layout_is_transposed_to_the_engine_order():
    """ExoMolOP stores (n_P, n_T, n_band, n_g); everything downstream indexes (n_T, n_P, n_g, n_band).

    A wrong transpose still returns finite numbers of the right total size, so it has to be pinned elementwise.
    """
    k, edges, t, p, g, w = _write(exomolop.table_path("H2O"))
    pack = exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)
    got = np.asarray(pack.logk["H2O"])
    assert got.shape == (NT, NP, NG, NB)
    want = np.log(k.transpose(1, 0, 3, 2))
    assert np.allclose(got, want, rtol=ROUND_TOL, atol=0.0)
    assert np.allclose(np.asarray(pack.t_grid), t)
    assert np.allclose(np.asarray(pack.p_grid), p)
    assert np.allclose(np.asarray(pack.gg), g)
    assert np.allclose(np.asarray(pack.gw), w)
    assert pack.ng == NG


@pytest.mark.usefixtures("data_root")
def test_only_bands_fully_inside_the_request_are_kept():
    _, edges, *_ = _write(exomolop.table_path("H2O"))
    lo, hi = edges[5], edges[12]
    pack = exomolop.load_tables(["H2O"], lo, hi, verbose=False)
    assert pack.band_edges[0] >= lo
    assert pack.band_edges[-1] <= hi
    assert pack.nu_bands.size == 7
    # centres are the geometric means of the kept edges
    assert np.allclose(pack.nu_bands, np.sqrt(pack.band_edges[:-1] * pack.band_edges[1:]))
    # a request wholly outside the table keeps nothing, and says so
    with pytest.raises(ValueError, match="no band inside"):
        exomolop.load_tables(["H2O"], edges[-1] * 10, edges[-1] * 20, verbose=False)


@pytest.mark.usefixtures("data_root")
def test_zero_cross_sections_floor_instead_of_going_to_minus_infinity():
    """ExoMolOP tables carry exact zeros where a species has no lines in a band.

    log(0) would poison the bilinear (T, P) interpolation for every layer, not just that band, so it must floor.
    """
    _, edges, *_ = _write(exomolop.table_path("H2O"), zeros=True)
    pack = exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)
    lk = np.asarray(pack.logk["H2O"])
    assert np.all(np.isfinite(lk))
    assert lk.min() == pytest.approx(np.log(constants.K_FLOOR))


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"g_shift": 0.01}, "disagree on the g ordinates"),
        ({"nb": NB - 1}, "disagree on the band grid"),
        ({"nu0": 101.0}, "disagree on the band grid"),
        ({"kunits": "cm^2/g"}, "kcoeff units"),
        ({"punits": "Pa"}, "p units"),
        ({"method": "gauss_legendre"}, "method"),
        ({"ngauss": NG + 1}, "ngauss"),
    ],
)
@pytest.mark.usefixtures("data_root")
def test_tables_that_disagree_or_carry_a_wrong_header_are_refused(kw, match):
    """Mismatched tables and wrong headers are refused, naming the offender.

    The mixing is ordinate by ordinate, so tables from different releases cannot be combined, and a mislabelled header
    (per-gram units, a different quadrature, an ngauss that does not match the samples) would change every cross
    section. Both failures would be silent, not a crash.
    """
    _write(exomolop.table_path("H2O"))
    _write(exomolop.table_path("CO2"), **kw)
    with pytest.raises(ValueError, match=match) as e:
        exomolop.load_tables(["H2O", "CO2"], 1.0, 1.0e6, verbose=False)
    assert "CO2" in str(e.value)


def test_table_info_and_provenance_report_what_is_on_disk(data_root):
    """The metadata API reports the file verbatim and JSON-serializable, and raises with the fetch command when empty.

    Placeholder DOIs are reported as they are and absent fields as None, never substituted.
    """
    import json

    with pytest.raises(FileNotFoundError, match="fetch_exomolop"):
        exomolop.table_info("H2O")
    with pytest.raises(FileNotFoundError, match="fetch_exomolop"):
        exomolop.provenance()
    _, edges, t, p, _, _ = _write(exomolop.table_path("H2O"))
    info = exomolop.table_info("H2O")
    assert info["doi"] == "x.xxxx/yyyyy"
    assert info["mol_name"] is None
    assert info["date_id"] is None
    assert info["method"] == "petit_samples"
    assert info["ngauss"] == NG
    assert info["kcoeff_units"] == exomolop.KCOEFF_UNITS
    assert info["p_units"] == exomolop.P_UNITS
    assert info["t_range_k"] == [t[0], t[-1]]
    assert info["p_range_bar"] == [p[0], p[-1]]
    assert info["wl_range_um"] == pytest.approx([1e4 / edges[-1], 1e4 / edges[0]])
    assert info["n_bands"] == NB
    assert info["file"] == "H2O.ktable.h5"
    json.dumps(info)
    rec = {
        "H2O": {
            "dataset": "POKAZATEL",
            "iso": "1H2-16O",
            "file": "x.h5",
            "natural_abundance": False,
            "url": "https://example",
        }
    }
    (data_root / "exomolop" / "provenance.json").write_text(json.dumps(rec))
    assert exomolop.provenance() == rec


@pytest.mark.parametrize(
    ("dataset", "index", "value"),
    [
        ("p", ..., [1e-5, 1e-5, 1e2]),  # duplicate node
        ("t", ..., [300.0, np.nan, 3000.0, 3400.0]),  # non-finite
        ("samples", ..., [0.02, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]),  # g = 1
        ("weights", ..., [0.2] * NG),  # sum != 1
        ("kcoeff", (0, 0, 0, 0), -1.0),  # negative
        ("kcoeff", (0, 0, 0, 0), np.nan),  # non-finite
        ("kcoeff", (0, 0, 0, 0), 1.0),  # above every later g-node (~1e-22): unsorted
    ],
)
@pytest.mark.usefixtures("data_root")
def test_malformed_tables_are_refused(dataset, index, value):
    import h5py

    path = exomolop.table_path("H2O")
    _, edges, *_ = _write(path)
    with h5py.File(path, "r+") as f:
        f[dataset][index] = value
    with pytest.raises(ValueError):
        exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)


# Against a real table, when one is installed.


def _need_real(*mols):
    """Skip unless the tables exist, resolved at call time.

    test_contract.py clears the data root to prove the library imports with no data installed, so a module-level
    skipif would be evaluated against whichever environment existed when this module was collected.
    """
    mols = mols or ("H2O",)
    try:
        ps = [exomolop.table_path(m) for m in mols]
    except RuntimeError:  # no data root configured
        pytest.skip("no $VULCAN_FORWARD_DATA configured")
    missing = [str(p) for p in ps if not p.exists()]
    if missing:
        pytest.skip(f"no ExoMolOP table at {missing} (python -m vulcan_forward.fetch_exomolop)")


def test_real_tables_share_one_grid_signature():
    """Every installed table carries the same band grid and quadrature, so one grid signature.

    load_tables refuses to mix them, and a consumer keys its caches on the signature.
    """
    _need_real()
    signatures = {m: exomolop.table_info(m)["grid_sha256"] for m in exomolop.available()}
    assert len(set(signatures.values())) == 1, signatures


def test_real_table_uses_the_petitradtrans_split_quadrature():
    """The quadrature is petitRADTRANS' split rule, 8 points on [0, 0.9] and 8 on [0.9, 1]: weights split 0.9/0.1."""
    _need_real()
    pack = exomolop.load_tables(["H2O"], 2000.0, 2100.0, verbose=False)
    w = np.asarray(pack.gw)
    g = np.asarray(pack.gg)
    assert pack.ng == 16
    assert w.sum() == pytest.approx(1.0, abs=ROUND_TOL)
    assert w[:8].sum() == pytest.approx(0.9, abs=SPLIT_TOL)
    assert w[8:].sum() == pytest.approx(0.1, abs=SPLIT_TOL)
    assert np.all((g > 0.0) & (g < 1.0))
    assert np.all(np.diff(g) > 0.0)


def test_real_table_is_a_cross_section_per_molecule_not_per_gram():
    """Kcoeff is cm^2 per molecule: ``opacity_profile_xs_ckd`` multiplies by mmr and divides by the molecular mass.

    A cm^2/g table would come out ~3.3e22 times too large. An H2O band-mean cross section at 1200 K and 0.1 bar is of
    order 1e-23 to 1e-19 cm^2/molecule; per gram it would be 1e-1 to 1e3.
    """
    _need_real()
    pack = exomolop.load_tables(["H2O"], 2400.0, 2600.0, verbose=False)
    lk = np.asarray(pack.logk["H2O"])
    t = np.asarray(pack.t_grid)
    p = np.asarray(pack.p_grid)
    it = int(np.argmin(np.abs(t - 1200.0)))
    ip = int(np.argmin(np.abs(p - 0.1)))
    kbar = np.exp(lk[it, ip]).T @ np.asarray(pack.gw)  # (nband,)
    med = float(np.median(kbar))
    assert 1e-26 < med < 1e-17, med
    # the same claim from the file header, checked on every load
    assert exomolop.table_info("H2O")["kcoeff_units"] == exomolop.KCOEFF_UNITS


def test_real_tables_reproduce_the_exo_k_reference():
    """Oracle: exo_k 1.3.1 (Leconte 2021) with ``Ktable.RandOverlap`` on the same three tables.

    The fixture is tests/data/exok_ref_overlap.npz; its meta carries the generating script and the conventions. The
    reader is bitwise; the interpolation within 1e-9 (measured 2e-14); the overlap at the convention gap (exo_k rebins
    on the upper-edge cumulative g, ours on the midpoint): measured per-ordinate median 1.4e-3, p90 1.9e-2, band-mean
    median 2.2e-3, each asserted at 3x.
    """
    import json
    import jax.numpy as jnp
    from vulcan_forward import ckd

    with np.load(DATA / "exok_ref_overlap.npz") as f:
        z = dict(f)
    meta = json.loads(bytes(np.asarray(z["meta"])))
    mols = list(meta["molecules"])
    _need_real(*mols)
    lo, hi = meta["wn_range_cm1"]
    pack = exomolop.load_tables(mols, lo, hi, verbose=False)
    edges = np.asarray(pack.band_edges)
    ib = np.searchsorted(edges, z["wn_edges_lo"])
    assert np.allclose(edges[ib], z["wn_edges_lo"], rtol=0.0, atol=EXOK_TOL)
    for ours, theirs in (
        (pack.gg, z["ggrid"]),
        (pack.gw, z["weights"]),
        (pack.t_grid, z["tgrid"]),
        (pack.p_grid, z["pgrid"]),
    ):
        assert np.allclose(np.asarray(ours), theirs, rtol=1e-15, atol=0.0)
    it, ip = z["on_it"], z["on_ip"]
    # reader: bitwise
    k_on = {}
    for j, m in enumerate(mols):
        lk = np.asarray(pack.logk[m])[it, ip][:, :, ib]  # (node, g, band)
        assert np.array_equal(lk, np.log(z["k_on"][j])), m
        k_on[m] = np.exp(lk)
    # (T, P) interpolation at the off-grid nodes
    T, P = jnp.asarray(z["off_T"]), jnp.asarray(z["off_P"])
    for j, m in enumerate(mols):
        got = np.exp(np.asarray(ckd._interp_logk(pack.logk[m], pack.t_grid, pack.p_grid, T, P)))[:, :, ib]
        assert np.allclose(got, z["k_off"][j], rtol=EXOK_TOL, atol=0.0), m
    # random overlap: the fixture's left fold, on the same 58 bands (band-local)
    x = meta["vmr"]
    tot = None
    for m in mols:
        dt = jnp.asarray(k_on[m] * x[m])  # (node as layer, g, band)
        tot = dt if tot is None else ckd.overlap(tot, dt, pack.gg, pack.gw)
    tot = np.asarray(tot)
    rel = np.abs(tot / z["k_mix_on"] - 1.0)
    assert np.median(rel) < 5e-3, np.median(rel)
    assert np.percentile(rel, 90) < 6e-2, np.percentile(rel, 90)
    w = np.asarray(pack.gw)
    band_rel = np.abs(np.einsum("g,lgb->lb", w, tot) / np.einsum("g,lgb->lb", w, z["k_mix_on"]) - 1.0)
    assert np.median(band_rel) < 7e-3, np.median(band_rel)


@pytest.mark.parametrize("molecule", ["CO2", "CO", "CH4", "NH3", "HCN", "H2S", "SO2", "C2H2", "OCS", "SO"])
def test_core_molecule_ranges_are_where_the_table_absorbs(molecule):
    """``constants.CORE_MOLECULE_BANDS_UM`` against its rule: at 1000 K and 1 mbar, the R = 100 band-mean cross-section
    is at least 1 % of its 1-15 um peak. Every such band lies in a listed range, and every listed range holds one.
    """
    _need_real(molecule)
    k, grid = exomolop._read_bands(molecule, constants.WIDE_BAND_NU_MIN, constants.WIDE_BAND_NU_MAX)
    i_t = int(np.argmin(np.abs(grid["t"] - 1000.0)))
    i_p = int(np.argmin(np.abs(np.log(grid["p"] / 1.0e-3))))
    sigma = np.convolve(k[i_p, i_t] @ grid["weights"], np.ones(11) / 11.0, mode="same")  # 11 R = 1000 bands
    wl = 1.0e4 / (0.5 * (grid["bin_edges"][:-1] + grid["bin_edges"][1:]))
    strong = wl[sigma >= 0.01 * sigma.max()]
    ranges = constants.CORE_MOLECULE_BANDS_UM[molecule]
    outside = [float(x) for x in strong if not any(lo <= x <= hi for lo, hi in ranges)]
    assert not outside, f"{molecule} absorbs at {outside[:5]} um, outside its listed ranges {ranges}"
    empty = [r for r in ranges if not np.any((strong >= r[0]) & (strong <= r[1]))]
    assert not empty, f"{molecule}: no strong band inside the listed ranges {empty}"


@pytest.mark.usefixtures("data_root")
def test_tables_must_cover_the_temperature_window_and_an_empty_molecule_table_is_not_the_default(monkeypatch):
    """Tables that stop inside the engine's temperature window would clamp a layer at their edge, so the load refuses
    them. An empty molecule table is the caller's table, with no entry: it is not swapped for the engine's default.
    """
    _, edges, *_ = _write(exomolop.table_path("H2O"))
    exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)
    with pytest.raises(KeyError, match="no molecule spec for"):
        exomolop.load_tables(["H2O"], edges[0], edges[-1], molecule_table={}, verbose=False)
    monkeypatch.setattr(constants, "T_OPA_MIN_K", 150.0)
    with pytest.raises(ValueError, match=r"tables cover \[200, 3000\] K but the engine accepts columns from 150"):
        exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)


@pytest.mark.usefixtures("data_root")
def test_load_refuses_without_x64(monkeypatch):
    """Without x64 the load is refused: float32 tables shift every cross section by ~1e-5 relative, without a crash."""
    _, edges, *_ = _write(exomolop.table_path("H2O"))

    class _Off:
        jax_enable_x64 = False

    # _require_x64 does `from jax import config`, which is getattr(jax, "config"), so swapping the attribute is enough;
    # jax's Config has no setter, and flipping x64 globally mid-session would change every later test.
    monkeypatch.setattr(jax, "config", _Off())
    with pytest.raises(RuntimeError, match="x64"):
        exomolop.load_tables(["H2O"], edges[0], edges[-1], verbose=False)


# The fetcher, on recorded miniature pages and files: nothing here reaches the network.
GRID_FILE = "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"


def _pages(**over):
    """ExoMolOP's three levels for H2O: isotopologues, line lists, files."""
    root = fetch_exomolop.ROOT
    pages = {
        f"{root}/H2O/": 'x <a href="1H2-16O" class="i"> <a href="1H-2H-16O" class="i"> <a href="1H2-18O" class="i">',
        f"{root}/H2O/1H2-16O/": (
            '<a href="BT2" class="list-group-item">BT2</a>'
            '<a href="POKAZATEL" class="list-group-item">POKAZATEL (recommended)</a>'
            '<a href="HotWat78" class="list-group-item">HotWat78</a>'
        ),
        f"{root}/H2O/1H2-16O/POKAZATEL/": (
            f'<a href="/db/H2O/1H2-16O/POKAZATEL/{GRID_FILE}">k</a>'
            '<a href="/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.R15000_0.2-30mu.ktable.petitRADTRANS.h5">k</a>'
            '<a href="/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.R1000_0.3-50mu.xsec.TauREx.h5">x</a>'
        ),
    }
    pages.update(over)
    return pages


def test_resolve_takes_the_principal_isotopologue_the_recommended_list_and_the_engine_s_grid(monkeypatch):
    pages = _pages()
    monkeypatch.setattr(fetch_exomolop, "_get", pages.__getitem__)
    url, dataset, iso, natural = fetch_exomolop.resolve("H2O")
    assert url == f"{fetch_exomolop.BASE}/db/H2O/1H2-16O/POKAZATEL/{GRID_FILE}"
    assert (dataset, iso, natural) == ("POKAZATEL", "1H2-16O", False)


def test_resolve_prefers_the_natural_abundance_file(monkeypatch):
    root = fetch_exomolop.ROOT
    nat = "1H2-16O-all__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    files = f'<a href="/db/H2O/1H2-16O/POKAZATEL/{GRID_FILE}">k</a><a href="/db/H2O/1H2-16O/POKAZATEL/{nat}">k</a>'
    pages = _pages(**{f"{root}/H2O/1H2-16O/POKAZATEL/": files})
    monkeypatch.setattr(fetch_exomolop, "_get", pages.__getitem__)
    url, *_, natural = fetch_exomolop.resolve("H2O")
    assert url.endswith(nat)
    assert natural is True


@pytest.mark.parametrize(
    ("level", "html", "match"),
    [
        ("{root}/H2O/", "nothing here", "no isotopologue links"),
        ("{root}/H2O/", '<a href="13C-16O2" class="i">', "Refusing to guess"),
        (
            "{root}/H2O/1H2-16O/POKAZATEL/",
            '<a href="/db/x/1H2-16O__P.R15000_0.2-30mu.ktable.petitRADTRANS.h5">k</a>',
            "none on the R1000_0.3-50mu grid",
        ),
    ],
)
def test_resolve_refuses_what_it_cannot_identify(monkeypatch, level, html, match):
    pages = _pages(**{level.format(root=fetch_exomolop.ROOT): html})
    monkeypatch.setattr(fetch_exomolop, "_get", pages.__getitem__)
    with pytest.raises(RuntimeError, match=match):
        fetch_exomolop.resolve("H2O")


def test_resolve_says_when_exomolop_has_no_k_table(monkeypatch):
    root = fetch_exomolop.ROOT
    pages = _pages(**{f"{root}/H2O/1H2-16O/POKAZATEL/": '<a href="/db/x/y.xsec.TauREx.h5">x</a>'})
    monkeypatch.setattr(fetch_exomolop, "_get", pages.__getitem__)
    assert fetch_exomolop.resolve("H2O") is None


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def download(data_root, monkeypatch, tmp_path):
    """Serve ``payload`` as any table's download, and resolve every molecule to one dataset."""
    monkeypatch.setattr(paths, "_configured_root", lambda: data_root, raising=False)
    monkeypatch.setattr(paths, "ensure_layout", lambda: data_root)
    monkeypatch.setattr(fetch_exomolop, "resolve", lambda _m: ("https://x/y.h5", "POKAZATEL", "1H2-16O", False))
    source = tmp_path / "served.h5"

    def serve(**kw):
        _write(source, **kw)
        monkeypatch.setattr(fetch_exomolop.urllib.request, "urlopen", lambda *_a, **_k: _Response(source.read_bytes()))

    return serve, source


def test_fetch_installs_a_valid_download_and_records_it(download):
    serve, _ = download
    serve()
    fetch_exomolop.fetch(["H2O"])
    assert exomolop.table_path("H2O").is_file()
    assert not exomolop.table_path("H2O").with_name("H2O.ktable.h5.part").exists()
    assert exomolop.provenance()["H2O"]["dataset"] == "POKAZATEL"


def test_fetch_leaves_a_present_attributed_table_alone(download, monkeypatch):
    serve, _ = download
    serve()
    fetch_exomolop.fetch(["H2O"])

    def no_network(*_a, **_k):
        raise AssertionError("a present, attributed table needs no network")

    monkeypatch.setattr(fetch_exomolop.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(fetch_exomolop, "resolve", no_network)
    fetch_exomolop.fetch(["H2O"])


def test_a_corrupt_forced_download_cannot_replace_the_only_table(download, monkeypatch):
    serve, source = download
    serve()
    fetch_exomolop.fetch(["H2O"])
    dest = exomolop.table_path("H2O")
    before = dest.read_bytes()
    monkeypatch.setattr(
        fetch_exomolop.urllib.request, "urlopen", lambda *_a, **_k: _Response(b"<html>not a table</html>")
    )
    with pytest.raises(RuntimeError, match="not a valid k-table"):
        fetch_exomolop.fetch(["H2O"], force=True)
    assert dest.read_bytes() == before
    assert not dest.with_name("H2O.ktable.h5.part").exists()
    assert source.exists()


def test_a_download_on_another_grid_is_refused_and_deleted(download):
    serve, _ = download
    serve()
    fetch_exomolop.fetch(["H2O"])
    serve(nu0=101.0)
    with pytest.raises(RuntimeError, match=r"disagrees with H2O\.ktable\.h5 on the band grid"):
        fetch_exomolop.fetch(["CO2"])
    assert not exomolop.table_path("CO2").exists()
    assert not exomolop.table_path("CO2").with_name("CO2.ktable.h5.part").exists()


def test_fetch_refuses_to_overwrite_an_unreadable_provenance_record(download):
    serve, _ = download
    serve()
    record = paths.exomolop_dir() / "provenance.json"
    record.write_text("{not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        fetch_exomolop.fetch(["H2O"])
    assert record.read_text() == "{not json"


@pytest.mark.parametrize(
    ("iso", "principal"),
    [
        ("1H2-16O", True),
        ("12C-16O2", True),
        ("1H2-32S", True),
        ("13C-16O2", False),  # a minor isotope of carbon
        ("1H-2H-16O", False),  # deuterium
        ("1H2-99Xx", False),  # an element the table does not know
        ("H2O", False),  # not ExoMol's isotopologue form: the caller raises rather than accept it
    ],
)
def test_the_isotopologue_token_names_the_principal_form(iso, principal):
    assert fetch_exomolop._is_principal(iso) is principal


def test_a_forced_download_with_a_wrong_header_cannot_replace_the_table(download):
    """Per-gram units pass the layout check and would change every cross section."""
    serve, _ = download
    serve()
    fetch_exomolop.fetch(["H2O"])
    dest = exomolop.table_path("H2O")
    before = dest.read_bytes()
    serve(kunits="cm^2/g")
    with pytest.raises(RuntimeError, match="not a valid k-table"):
        fetch_exomolop.fetch(["H2O"], force=True)
    assert dest.read_bytes() == before


def test_a_page_that_fails_is_a_refusal_at_the_first_attempt(monkeypatch):
    """A retry can only hide a flaky network from the person watching; a re-run skips what already landed."""
    calls = []

    def down(_req, timeout):
        calls.append(timeout)
        raise OSError("unreachable")

    monkeypatch.setattr(fetch_exomolop.urllib.request, "urlopen", down)
    with pytest.raises(RuntimeError, match="unreachable"):
        fetch_exomolop._get("https://example.invalid/page")
    assert calls == [fetch_exomolop.PAGE_TIMEOUT_S]
