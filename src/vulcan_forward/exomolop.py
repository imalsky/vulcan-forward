"""ExoMolOP k-tables: the published high-temperature opacities, ingested.

WHY THIS EXISTS
---------------
``ckd.py`` holds HOW the opacity is integrated; this module supplies WHAT is
integrated. Building tables locally from HITRAN had three measured defects,
all pushing the same way (too little opacity in the WINDOWS, which inflates
spectral contrast): a 296 K database applied at 1200 K, terrestrial-air
pressure broadening in a hydrogen atmosphere (unfixable inside HITRAN -- H2O
has no H2/He columns there at all), and thin species coverage. The measured
numbers live in notes.md ("Opacity data: ExoMolOP, not HITRAN").

ExoMolOP (Chubb et al. 2021, A&A 646, A21) closes all three at once: it
publishes PRE-COMPUTED opacities for ~80 species, built from the ExoMol and
HITEMP high-temperature line lists with H2/He broadening already applied,
free and with no account, in each radiative-transfer code's native format.
Its petitRADTRANS-format k-tables are ~389 MB per species and land almost
exactly on the layout ``ckd`` uses, so this is an ingestion adapter rather
than a second opacity implementation: everything downstream -- the
random-overlap mixing, the (T, P) interpolation, the transmission and
emission solvers -- is shared code.

WHAT IS DIFFERENT ABOUT THEIR TABLES, and both matter
-----------------------------------------------------
* The QUADRATURE is not plain Gauss-Legendre: petitRADTRANS' split scheme,
  8 Gauss-Legendre points on [0, 0.9] plus 8 on [0.9, 1.0]. ``ckd.overlap``
  takes the nodes and weights as arguments, so they carry through from the
  file -- but never assume 16-point Gauss-Legendre downstream.
* Their PRESSURE grid stops at 1e-5 bar, while the RT column runs to about
  5e-9 bar. ``ckd._interp_logk`` clamps rather than extrapolating, so the
  layers above 1e-5 bar all use the 1e-5 bar table entry. That is defensible
  physics -- up there the lines are Doppler-dominated and k stops depending on
  pressure -- and it is what petitRADTRANS itself does, but it is an
  ASSUMPTION applied to real layers, so ``load_tables`` prints it rather than
  letting it pass silently.

UNITS: verified empirically AND checked on every load. Their ``kcoeff`` is in
cm^2 per MOLECULE, the same convention ``opacity_profile_xs_ckd`` consumes --
order unity against a HITRAN-built H2O table, not the 3.34e22 a per-gram
convention would give (notes.md). ``_header`` refuses a file whose
``kcoeff``/``p`` unit attributes, ``method`` or ``ngauss`` are not what this
reader assumes; a mislabelled table must never load as if it were right.

TABLES ARE NEVER DOWNLOADED AT RUN TIME (standing fail-loud rule): a missing
table raises with the fetch command.

This module is importable WITHOUT the RT stack: h5py and jax are imported
inside the functions that need them, so the path helpers (``table_dir``,
``table_path``, ``available``) and ``provenance`` serve stdlib-only consumers
such as jwst_tool.datacheck.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vulcan_forward import constants, paths

# log k floor. Their tables contain exact zeros where a species has no lines
# in a band; log(0) would poison the interpolation with -inf, and a zero cross
# section is physically "no absorption", so it floors to a value far below any
# optical depth that matters (1e-60 cm^2 over a full column is ~1e-40 in tau).
K_FLOOR = 1.0e-60

# The header values this reader is built for. Anything else is a different
# product (or a mislabelled file) and is refused, never reinterpreted.
KCOEFF_UNITS = "cm^2/molecule"
P_UNITS = "bar"
METHOD = "petit_samples"


def table_dir() -> Path:
    return paths.exomolop_dir()


def table_path(molecule: str) -> Path:
    return table_dir() / f"{molecule}.ktable.h5"


def _fetch_hint(missing) -> str:
    return ("ExoMolOP tables are never downloaded at run time. Fetch them "
            "with:\n  python -m vulcan_forward.fetch_exomolop --molecules "
            + ",".join(missing)
            + "\nIf that reports SKIP for a species, ExoMolOP does not publish "
              "one for it (CS2 and C2H6 are the two this engine's molecule "
              "table names); drop the species.")


def available() -> list:
    """Molecules with an ExoMolOP table present, sorted."""
    d = table_dir()
    if not d.is_dir():
        return []
    return sorted(p.name.split(".")[0] for p in d.glob("*.ktable.h5"))


def provenance() -> dict:
    """What each installed table is, as recorded by ``fetch_exomolop``.

    ``{molecule: {dataset, iso, file, natural_abundance, url}}`` from the
    tree's ``provenance.json``. stdlib only, so a data-status check can read it
    without jax or h5py. Raises with the fetch command when the record is
    absent (the fetcher writes it, and backfills tables already on disk
    without downloading them again).
    """
    import json
    path = table_dir() / "provenance.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no ExoMolOP provenance record at {path}. fetch_exomolop writes "
            "it and backfills tables already on disk without re-downloading:\n"
            "  python -m vulcan_forward.fetch_exomolop --molecules "
            + ",".join(available() or ["H2O"]))
    return json.loads(path.read_text())


def _scalar_str(f, key):
    """One-element string dataset -> str; None when the file has no such key
    (the H2O table lacks mol_name and Date_ID)."""
    if key not in f:
        return None
    v = np.asarray(f[key][()]).ravel()[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _validated_layout(f, path) -> dict:
    """Data-integrity checks on the table layout: ordered t/p/bin_edges, finite
    positive coordinates, quadrature weights summing to 1, kcoeff dims. The
    kcoeff values themselves (finite, non-negative) are checked by ``load_tables``.
    """
    required = ("kcoeff", "bin_edges", "t", "p", "samples", "weights")
    missing = [name for name in required if name not in f]
    if missing:
        raise ValueError(f"{path}: missing required datasets {missing}")

    arrays = {name: np.asarray(f[name], dtype=np.float64)
              for name in ("bin_edges", "t", "p", "samples", "weights")}
    for name, values in arrays.items():
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"{path}: {name} must be a 1-D array with at least two values")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {name} contains non-finite values")

    for name in ("bin_edges", "t", "p", "samples"):
        values = arrays[name]
        if np.any(values <= 0.0) or not np.all(np.diff(values) > 0.0):
            raise ValueError(f"{path}: {name} must be finite, positive, and strictly increasing")
    if np.any(arrays["samples"] >= 1.0):
        raise ValueError(f"{path}: samples must lie strictly inside (0, 1)")
    weights = arrays["weights"]
    if np.any(weights <= 0.0) or weights.shape != arrays["samples"].shape:
        raise ValueError(f"{path}: weights must be positive and match samples")
    if not np.isclose(np.sum(weights), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{path}: quadrature weights sum to {np.sum(weights):.17g}, not 1")

    expected = (arrays["p"].size, arrays["t"].size,
                arrays["bin_edges"].size - 1, arrays["samples"].size)
    if f["kcoeff"].shape != expected:
        raise ValueError(f"{path}: kcoeff shape {f['kcoeff'].shape} != expected {expected}")
    return arrays


def _header(f, path) -> dict:
    """Verify the load-bearing header of an open k-table and return it.

    Load-bearing (a wrong value silently changes every cross section, so it
    RAISES with the file and the offending value): the ``kcoeff`` unit
    attribute, the ``p`` unit attribute, ``method`` (the split quadrature) and
    ``ngauss`` against the number of g-samples. Informational (DOI, mol_name,
    Date_ID) is reported verbatim -- placeholder DOIs included -- or None when
    absent; it is never substituted.
    """
    def _attr(ds, name):
        v = f[ds].attrs.get(name)
        return v.decode() if isinstance(v, bytes) else (None if v is None else str(v))

    rec = dict(
        kcoeff_units=_attr("kcoeff", "units"), p_units=_attr("p", "units"),
        method=_scalar_str(f, "method"),
        ngauss=(int(np.asarray(f["ngauss"][()]).ravel()[0]) if "ngauss" in f else None),
        doi=_scalar_str(f, "DOI"), mol_name=_scalar_str(f, "mol_name"),
        date_id=_scalar_str(f, "Date_ID"))
    n_samples = int(f["samples"].shape[0])
    bad = []
    if rec["kcoeff_units"] != KCOEFF_UNITS:
        bad.append(f"kcoeff units {rec['kcoeff_units']!r}, need {KCOEFF_UNITS!r} "
                   "(a per-gram table would be ~3.3e22 too large)")
    if rec["p_units"] != P_UNITS:
        bad.append(f"p units {rec['p_units']!r}, need {P_UNITS!r}")
    if rec["method"] != METHOD:
        bad.append(f"method {rec['method']!r}, need {METHOD!r} (the split 8+8 "
                   "quadrature)")
    if rec["ngauss"] != n_samples:
        bad.append(f"ngauss {rec['ngauss']!r} != len(samples) {n_samples}")
    if bad:
        raise ValueError(
            f"{path}: " + "; ".join(bad) + ". Not a petitRADTRANS-format "
            "ExoMolOP k-table this engine can read; re-fetch it with "
            "python -m vulcan_forward.fetch_exomolop --force --molecules "
            f"{Path(path).name.split('.')[0]}")
    return rec


def table_info(molecule: str) -> dict:
    """Header of one installed k-table as plain JSON-serializable values.

    Keys: molecule, file, doi, mol_name, date_id, method, ngauss,
    kcoeff_units, p_units, n_bands, t_range_k, p_range_bar, wl_range_um.
    Runs the same header check as ``load_tables``, so it raises on the same
    files. Placeholder DOIs are reported verbatim; absent DOI / mol_name /
    Date_ID are None.
    """
    import h5py                       # function-local, as in load_tables
    path = table_path(molecule)
    if not path.exists():
        raise FileNotFoundError(
            f"ExoMolOP k-table missing for {molecule}: looked for {path}\n"
            + _fetch_hint([molecule]))
    with h5py.File(path, "r") as f:
        arrays = _validated_layout(f, path)
        rec = _header(f, path)
        t, p, e = (arrays[k] for k in ("t", "p", "bin_edges"))
        # Signature of the grid as the loader's 1e-12 agreement rule sees it:
        # round before hashing so tables that load_tables accepts together
        # share one key; each array is prefixed by its shape.
        digest = hashlib.sha256()
        for key in ("t", "p", "bin_edges", "samples", "weights"):
            values = arrays[key]
            rounded = (np.round(np.log10(values), 10) if key in ("t", "p", "bin_edges")
                       else np.round(values, 12))
            rounded = np.ascontiguousarray(rounded, dtype="<f8")
            digest.update(np.asarray(rounded.shape, dtype="<i8").tobytes())
            digest.update(rounded.tobytes())
        centers = np.sqrt(e[:-1] * e[1:])
        resolving_power = centers / np.diff(e)
    rec.update(molecule=molecule, file=path.name, n_bands=int(e.size - 1),
               t_range_k=[float(t[0]), float(t[-1])],
               p_range_bar=[float(p[0]), float(p[-1])],
               wl_range_um=[float(1e4 / e[-1]), float(1e4 / e[0])],
               grid_sha256=digest.hexdigest(),
               band_resolution=float(np.median(resolving_power)))
    return rec


def load_tables(molecules, nu_min, nu_max, *, molecule_table=None,
                verbose=True) -> SimpleNamespace:
    """Load one ExoMolOP k-table per molecule, restricted to [nu_min, nu_max].

    Returns the namespace the correlated-k core consumes (``logk`` per
    molecule + band edges, (T, P) grids, and the quadrature).

    Raises rather than downloading, raises on a header this reader is not
    built for (``_header``), and raises rather than mixing tables that do not
    share a band grid or a quadrature -- ``ckd.overlap`` combines species
    ordinate by ordinate, so two tables on different g-nodes would be silently
    added as if they were the same distribution.
    """
    import h5py                       # only needed for ingestion
    import jax.numpy as jnp           # keeps the module importable bare
    from jax import config as _jax_config

    if not _jax_config.jax_enable_x64:
        # Without x64, jnp.asarray silently returns float32 tables. log k spans
        # roughly -140 to -40, so float32 costs ~1e-5 relative on every cross
        # section and breaks this engine's float64 contract -- and it would
        # show up as a small unexplained spectrum shift, not as a crash. The
        # engine enables x64 at import (vulcan_chem, exojax_rt); a caller who
        # reached here without it has a broken import order.
        raise RuntimeError(
            "jax x64 is not enabled, so the k-tables would be silently "
            "downcast to float32. Import vulcan_forward.vulcan_chem (or "
            "vulcan_forward.exojax_rt) before loading opacities, or call "
            "jax.config.update('jax_enable_x64', True).")

    tbl = molecule_table or constants.MOLECULES
    mols = list(molecules)
    unknown = [m for m in mols if m not in tbl]
    if unknown:
        raise KeyError(f"no molecule spec for {unknown} in the molecule table")
    missing = [m for m in mols if not table_path(m).exists()]
    if missing:
        raise FileNotFoundError(
            f"ExoMolOP k-table missing for {missing}: looked in {table_dir()}\n"
            + _fetch_hint(missing))

    out, ref = {}, None
    for m in mols:
        with h5py.File(table_path(m), "r") as f:
            arrays = _validated_layout(f, table_path(m))
            _header(f, table_path(m))
            edges, gg, gw = (arrays[k] for k in ("bin_edges", "samples", "weights"))
            t_grid, p_grid = (arrays[k] for k in ("t", "p"))
            # bands fully inside the requested span; edges has n_band+1 entries
            keep = np.where((edges[:-1] >= nu_min) & (edges[1:] <= nu_max))[0]
            if keep.size == 0:
                raise ValueError(
                    f"ExoMolOP table for {m} covers "
                    f"[{edges[0]:.1f}, {edges[-1]:.1f}] cm^-1 and has no band "
                    f"inside the requested [{nu_min:g}, {nu_max:g}]")
            b0, b1 = int(keep[0]), int(keep[-1]) + 1
            # hyperslab read: (n_P, n_T, n_band, n_g) -> only the bands wanted
            k = np.asarray(f["kcoeff"][:, :, b0:b1, :], dtype=np.float64)
            sub_edges = edges[b0:b1 + 1]

        if not np.all(np.isfinite(k)) or np.any(k < 0.0):
            raise ValueError(f"{table_path(m)}: kcoeff must be finite and non-negative")
        if np.any(np.diff(k, axis=-1) < 0.0):
            raise ValueError(
                f"{table_path(m)}: kcoeff must be non-decreasing along the g ordinate")

        if ref is None:
            ref = (sub_edges, t_grid, p_grid, gg, gw, m)
        else:
            e0, t0, p0, g0, w0, m0 = ref
            for name, a, b in (("band grid", sub_edges, e0),
                               ("temperature nodes", t_grid, t0),
                               ("pressure nodes", p_grid, p0),
                               ("g ordinates", gg, g0),
                               ("g weights", gw, w0)):
                if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12,
                                                         atol=0.0):
                    raise ValueError(
                        f"ExoMolOP tables for {m} and {m0} disagree on the "
                        f"{name}. They are mixed ordinate by ordinate, so "
                        "tables from different releases cannot be combined; "
                        "re-fetch both from the same ExoMolOP release.")
        # (n_P, n_T, n_band, n_g) -> (n_T, n_P, n_g, n_band)
        out[m] = jnp.asarray(np.log(np.maximum(k, K_FLOOR)).transpose(1, 0, 3, 2))

    sub_edges, t_grid, p_grid, gg, gw, _ = ref
    if verbose:
        print(f"[ckd] ExoMolOP: {len(mols)} species, {sub_edges.size - 1} bands "
              f"over [{sub_edges[0]:.1f},{sub_edges[-1]:.1f}] cm^-1, ng={gg.size}, "
              f"T {t_grid[0]:.0f}-{t_grid[-1]:.0f} K, "
              f"P {p_grid[0]:.1e}-{p_grid[-1]:.1e} bar", flush=True)
        print(f"[ckd]   NOTE: the table's pressure floor is "
              f"{p_grid[0]:.1e} bar; RT layers above it reuse that entry "
              "(k is Doppler-dominated and pressure-independent there). "
              "Same treatment as petitRADTRANS.", flush=True)

    return SimpleNamespace(
        logk=out, band_edges=sub_edges,
        nu_bands=np.sqrt(sub_edges[:-1] * sub_edges[1:]),
        t_grid=jnp.asarray(t_grid), p_grid=jnp.asarray(p_grid),
        gg=jnp.asarray(gg), gw=jnp.asarray(gw), ng=int(gg.size))
