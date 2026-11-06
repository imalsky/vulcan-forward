"""ExoMolOP k-tables: load them, check their headers and report what each one is.

Tables are never downloaded at run time: a missing table raises with the fetch command. h5py and jax are imported inside
the functions that need them, so ``table_path``, ``available`` and ``provenance`` import without either. docs/method.md
describes the tables, their quadrature and the pressure floor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from vulcan_forward import constants, paths

if TYPE_CHECKING:
    from collections.abc import Iterable

    import h5py

__all__ = ["K_FLOOR", "available", "load_tables", "provenance", "table_info", "table_path"]

logger = logging.getLogger(__name__)

# log k floor, cm^2 per molecule. The tables hold exact zeros where a species has no lines in a band, and log(0) would
# poison the interpolation; 1e-60 cm^2 over a full column is ~1e-32 in tau.
K_FLOOR = 1.0e-60

# The header values this reader is built for; anything else is refused, never reinterpreted.
KCOEFF_UNITS = "cm^2/molecule"
P_UNITS = "bar"
METHOD = "petit_samples"

# Two tables mix only if their grids agree to this relative tolerance (float64 rounding of one published grid).
_GRID_RTOL = 1.0e-12
# Decimals table_info rounds to before hashing: the g-samples and weights at
# _GRID_RTOL, the log10 of the t / p / band-edge grids at 1e-10.
_HASH_DECIMALS = round(-math.log10(_GRID_RTOL))
_HASH_LOG_DECIMALS = 10
# The quadrature weights must sum to 1 to this tolerance (float64 rounding).
_WEIGHT_SUM_RTOL, _WEIGHT_SUM_ATOL = 1.0e-10, 1.0e-12


def table_path(molecule: str) -> Path:
    """Return the path of one molecule's k-table, which need not exist."""
    return paths.exomolop_dir() / f"{molecule}.ktable.h5"


def _fetch_hint(missing: Iterable[str]) -> str:
    """Return the message that says how to fetch the ``missing`` tables."""
    return (
        "ExoMolOP tables are never downloaded at run time. Fetch them "
        "with:\n  python -m vulcan_forward.fetch_exomolop --molecules "
        + ",".join(missing)
        + "\nIf that reports SKIP for a species, ExoMolOP does not publish "
        "one for it (CS2 and C2H6 are the two this engine's molecule "
        "table names); drop the species."
    )


def available() -> list[str]:
    """Return the molecules with a table present, sorted."""
    d = paths.exomolop_dir()
    if not d.is_dir():
        return []
    return sorted(p.name.split(".")[0] for p in d.glob("*.ktable.h5"))


def provenance() -> dict[str, dict[str, Any]]:
    """Return what each installed table is, as recorded by ``fetch_exomolop``.

    Returns
    -------
    dict
        ``{molecule: {dataset, iso, file, natural_abundance, url}}``, plus ``doi`` for datasets whose header DOI is a
        placeholder.

    Raises
    ------
    FileNotFoundError
        If the record is absent, with the fetch command; the fetcher backfills tables already on disk.
    """
    path = paths.exomolop_dir() / "provenance.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no ExoMolOP provenance record at {path}. fetch_exomolop writes "
            "it and backfills tables already on disk without re-downloading:\n"
            "  python -m vulcan_forward.fetch_exomolop --molecules " + ",".join(available() or ["H2O"])
        )
    record: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return record


def _scalar_str(f: h5py.File, key: str) -> str | None:
    """Read a one-element string dataset, or None when the file has no such key (H2O lacks mol_name)."""
    if key not in f:
        return None
    v = np.asarray(f[key][()]).ravel()[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _validated_layout(f: h5py.File, path: Path) -> dict[str, np.ndarray]:
    """Check the table layout and return its coordinate arrays.

    Checks ordered, finite, positive t, p and bin_edges, weights summing to 1 and the kcoeff shape. ``load_tables``
    checks the kcoeff values.
    """
    required = ("kcoeff", "bin_edges", "t", "p", "samples", "weights")
    missing = [name for name in required if name not in f]
    if missing:
        raise ValueError(f"{path}: missing required datasets {missing}")

    arrays = {name: np.asarray(f[name], dtype=np.float64) for name in ("bin_edges", "t", "p", "samples", "weights")}
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
    if not np.isclose(np.sum(weights), 1.0, rtol=_WEIGHT_SUM_RTOL, atol=_WEIGHT_SUM_ATOL):
        raise ValueError(f"{path}: quadrature weights sum to {np.sum(weights):.17g}, not 1")

    expected = (arrays["p"].size, arrays["t"].size, arrays["bin_edges"].size - 1, arrays["samples"].size)
    if f["kcoeff"].shape != expected:
        raise ValueError(f"{path}: kcoeff shape {f['kcoeff'].shape} != expected {expected}")
    return arrays


# The arrays two tables must share to be mixed ordinate by ordinate, and the
# name each is reported under.
_GRID = (
    ("bin_edges", "band grid"),
    ("t", "temperature nodes"),
    ("p", "pressure nodes"),
    ("samples", "g ordinates"),
    ("weights", "g weights"),
)


def _grid_mismatch(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> str | None:
    """Name the first ``_GRID`` array on which two layouts disagree at ``_GRID_RTOL``, or return None."""
    for key, name in _GRID:
        if a[key].shape != b[key].shape or not np.allclose(a[key], b[key], rtol=_GRID_RTOL, atol=0.0):
            return name
    return None


def _header(f: h5py.File, path: Path) -> dict[str, Any]:
    """Check the header fields that change every cross section and return the header.

    Raises on the ``kcoeff`` and ``p`` units, ``method`` and ``ngauss``. DOI, mol_name and Date_ID are reported
    verbatim, placeholder DOIs included, or None when absent.
    """

    def _attr(ds: str, name: str) -> str | None:
        """Read one attribute of dataset ``ds`` as a string, or None."""
        v = f[ds].attrs.get(name)
        return v.decode() if isinstance(v, bytes) else (None if v is None else str(v))

    rec = {
        "kcoeff_units": _attr("kcoeff", "units"),
        "p_units": _attr("p", "units"),
        "method": _scalar_str(f, "method"),
        "ngauss": (int(np.asarray(f["ngauss"][()]).ravel()[0]) if "ngauss" in f else None),
        "doi": _scalar_str(f, "DOI"),
        "mol_name": _scalar_str(f, "mol_name"),
        "date_id": _scalar_str(f, "Date_ID"),
    }
    n_samples = int(f["samples"].shape[0])
    bad = []
    if rec["kcoeff_units"] != KCOEFF_UNITS:
        bad.append(
            f"kcoeff units {rec['kcoeff_units']!r}, need {KCOEFF_UNITS!r} (a per-gram table would be ~3.3e22 too large)"
        )
    if rec["p_units"] != P_UNITS:
        bad.append(f"p units {rec['p_units']!r}, need {P_UNITS!r}")
    if rec["method"] != METHOD:
        bad.append(f"method {rec['method']!r}, need {METHOD!r} (the split 8+8 quadrature)")
    if rec["ngauss"] != n_samples:
        bad.append(f"ngauss {rec['ngauss']!r} != len(samples) {n_samples}")
    if bad:
        raise ValueError(
            f"{path}: " + "; ".join(bad) + ". Not a petitRADTRANS-format "
            "ExoMolOP k-table this engine can read; re-fetch it with "
            "python -m vulcan_forward.fetch_exomolop --force --molecules "
            f"{Path(path).name.split('.')[0]}"
        )
    return rec


def table_info(molecule: str) -> dict[str, Any]:
    """Return the header of one installed k-table as JSON-serializable values.

    Runs the layout and header checks of ``load_tables`` but not its kcoeff value checks, so ``load_tables`` can still
    refuse a file this accepts.

    Parameters
    ----------
    molecule : str
        The table's molecule.

    Returns
    -------
    dict
        Keys: molecule, file, doi, mol_name, date_id, method, ngauss, kcoeff_units, p_units, n_bands, t_range_k,
        p_range_bar, wl_range_um, grid_sha256, band_resolution.

    Raises
    ------
    FileNotFoundError
        If the table is not installed.
    """
    import h5py

    path = table_path(molecule)
    if not path.exists():
        raise FileNotFoundError(
            f"ExoMolOP k-table missing for {molecule}: looked for {path}\n" + _fetch_hint([molecule])
        )
    with h5py.File(path, "r") as f:
        arrays = _validated_layout(f, path)
        rec = _header(f, path)
        t, p, e = (arrays[k] for k in ("t", "p", "bin_edges"))
        # Round before hashing to absorb float noise; values within _GRID_RTOL can still round apart. The band grid is
        # hashed whole, not the slice load_tables compares.
        digest = hashlib.sha256()
        for key in ("t", "p", "bin_edges", "samples", "weights"):
            values = arrays[key]
            rounded = (
                np.round(np.log10(values), _HASH_LOG_DECIMALS)
                if key in ("t", "p", "bin_edges")
                else np.round(values, _HASH_DECIMALS)
            )
            rounded = np.ascontiguousarray(rounded, dtype="<f8")
            digest.update(np.asarray(rounded.shape, dtype="<i8").tobytes())
            digest.update(rounded.tobytes())
        centers = np.sqrt(e[:-1] * e[1:])
        resolving_power = centers / np.diff(e)
    rec.update(
        molecule=molecule,
        file=path.name,
        n_bands=int(e.size - 1),
        t_range_k=[float(t[0]), float(t[-1])],
        p_range_bar=[float(p[0]), float(p[-1])],
        wl_range_um=[float(constants.UM_PER_CM / e[-1]), float(constants.UM_PER_CM / e[0])],
        grid_sha256=digest.hexdigest(),
        band_resolution=float(np.median(resolving_power)),
    )
    return rec


# size-limit-ok: the refusals in order, one table at a time
def load_tables(
    molecules: Iterable[str],
    nu_min: float,
    nu_max: float,
    *,
    molecule_table: dict[str, dict[str, Any]] | None = None,
    verbose: bool = True,
) -> SimpleNamespace:
    """Load one k-table per molecule, restricted to the bands inside [nu_min, nu_max].

    Parameters
    ----------
    molecules : sequence of str
        The species, each once.
    nu_min : float
        The lower band limit, cm^-1.
    nu_max : float
        The upper band limit, cm^-1.
    molecule_table : dict, optional
        Replaces ``constants.MOLECULES``.
    verbose : bool, optional
        Log the grid and the pressure-floor assumption.

    Returns
    -------
    types.SimpleNamespace
        ``logk`` per molecule [nT, nP, ng, nband], ``band_edges``, ``nu_bands``, ``t_grid``, ``p_grid``, ``gg``, ``gw``,
        ``ng``.

    Raises
    ------
    RuntimeError
        If jax x64 is off: float32 tables would shift every cross section by ~1e-5.
    KeyError
        If a molecule has no entry in the molecule table.
    ValueError
        On a repeated species, a band or header mismatch, or an invalid kcoeff.
    FileNotFoundError
        If a table is missing, with the fetch command.
    """
    import h5py
    import jax.numpy as jnp
    from jax import config as _jax_config

    if not _jax_config.jax_enable_x64:  # type: ignore[attr-defined]  # jax declares its options at run time
        # float32 tables would shift every cross section by ~1e-5 relative
        # without an error.
        raise RuntimeError(
            "jax x64 is not enabled, so the k-tables would be silently "
            "downcast to float32. Import vulcan_forward.vulcan_chem (or "
            "vulcan_forward.exojax_rt) before loading opacities, or call "
            "jax.config.update('jax_enable_x64', True)."
        )

    tbl = molecule_table or constants.MOLECULES
    mols = list(molecules)
    unknown = [m for m in mols if m not in tbl]
    if unknown:
        raise KeyError(
            f"no molecule spec for {unknown} in the molecule table "
            f"(have: {sorted(tbl)}). Pass profile['molecule_table'] with one "
            "entry per molecule: vulcan, molmass."
        )
    if len(set(mols)) != len(mols):
        # the fold adds one opacity per list entry, so a repeat counts twice
        raise ValueError(f"molecules {mols} name a species more than once; each would add its opacity again")
    missing = [m for m in mols if not table_path(m).exists()]
    if missing:
        raise FileNotFoundError(
            f"ExoMolOP k-table missing for {missing}: looked in {paths.exomolop_dir()}\n" + _fetch_hint(missing)
        )

    out, ref, m0 = {}, None, None
    for m in mols:
        with h5py.File(table_path(m), "r") as f:
            arrays = _validated_layout(f, table_path(m))
            _header(f, table_path(m))
            edges = arrays["bin_edges"]
            # bands fully inside the requested span; edges has n_band+1 entries
            keep = np.where((edges[:-1] >= nu_min) & (edges[1:] <= nu_max))[0]
            if keep.size == 0:
                raise ValueError(
                    f"ExoMolOP table for {m} covers "
                    f"[{edges[0]:.1f}, {edges[-1]:.1f}] cm^-1 and has no band "
                    f"inside the requested [{nu_min:g}, {nu_max:g}]"
                )
            b0, b1 = int(keep[0]), int(keep[-1]) + 1
            # hyperslab read: (n_P, n_T, n_band, n_g) -> only the bands wanted
            k = np.asarray(f["kcoeff"][:, :, b0:b1, :], dtype=np.float64)
            grid = dict(arrays, bin_edges=edges[b0 : b1 + 1])

        if not np.all(np.isfinite(k)) or np.any(k < 0.0):
            raise ValueError(f"{table_path(m)}: kcoeff must be finite and non-negative")
        if np.any(np.diff(k, axis=-1) < 0.0):
            raise ValueError(f"{table_path(m)}: kcoeff must be non-decreasing along the g ordinate")

        if ref is None:
            ref, m0 = grid, m
        elif (bad := _grid_mismatch(grid, ref)) is not None:
            raise ValueError(
                f"ExoMolOP tables for {m} and {m0} disagree on the {bad}. "
                "They are mixed ordinate by ordinate, so tables from different "
                "releases cannot be combined; re-fetch both from the same "
                "ExoMolOP release."
            )
        # (n_P, n_T, n_band, n_g) -> (n_T, n_P, n_g, n_band)
        out[m] = jnp.asarray(np.log(np.maximum(k, K_FLOOR)).transpose(1, 0, 3, 2))

    if ref is None:
        raise ValueError("load_tables needs at least one molecule; got an empty list")
    sub_edges, t_grid, p_grid, gg, gw = (ref[k] for k in ("bin_edges", "t", "p", "samples", "weights"))
    if verbose:
        logger.info(
            "[ckd] ExoMolOP: %d species, %d bands over [%.1f,%.1f] cm^-1, ng=%d, T %.0f-%.0f K, P %.1e-%.1e bar",
            len(mols),
            sub_edges.size - 1,
            sub_edges[0],
            sub_edges[-1],
            gg.size,
            t_grid[0],
            t_grid[-1],
            p_grid[0],
            p_grid[-1],
        )
        logger.info(
            "[ckd]   the table's pressure floor is %.1e bar; RT layers above it reuse that entry "
            "(k is Doppler-dominated and pressure-independent there).",
            p_grid[0],
        )

    return SimpleNamespace(
        logk=out,
        band_edges=sub_edges,
        nu_bands=np.sqrt(sub_edges[:-1] * sub_edges[1:]),
        t_grid=jnp.asarray(t_grid),
        p_grid=jnp.asarray(p_grid),
        gg=jnp.asarray(gg),
        gw=jnp.asarray(gw),
        ng=int(gg.size),
    )
