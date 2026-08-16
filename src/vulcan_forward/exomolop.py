"""ExoMolOP k-tables: the published high-temperature opacities, ingested.

WHY THIS EXISTS
---------------
``ckd.py`` holds HOW the opacity is integrated; this module supplies WHAT is
integrated. Building tables locally from HITRAN had three measured defects,
all pushing the same way (too little opacity in the WINDOWS, which inflates
spectral contrast): a 296 K database applied at 1200 K, terrestrial-air
pressure broadening in a hydrogen atmosphere (unfixable inside HITRAN -- H2O
has no H2/He columns there at all), and thin species coverage. The measured
numbers live in the README ("Opacity data: ExoMolOP, not HITRAN").

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

UNITS: verified empirically, not assumed. Their ``kcoeff`` is in cm^2 per
MOLECULE, the same convention ``opacity_profile_xs_ckd`` consumes -- order
unity against our own HITRAN H2O table, not the 3.34e22 a per-gram convention
would give (the cross-check numbers are in the README).

TABLES ARE NEVER DOWNLOADED AT RUN TIME (standing fail-loud rule): a missing
table raises with the fetch command.

This module is importable WITHOUT the RT stack: h5py and jax are imported
inside ``load_tables`` only, so path helpers (``table_dir``, ``table_path``,
``available``) serve stdlib-only consumers such as jwst_tool.datacheck.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vulcan_forward import constants, paths

# Their pressure grid floor. Layers above this share its k; see the module
# docstring. Kept as a named constant so the disclosure and the check cannot
# drift apart.
P_TABLE_MIN_BAR = 1.0e-5

# log k floor. Their tables contain exact zeros where a species has no lines
# in a band; log(0) would poison the interpolation with -inf, and a zero cross
# section is physically "no absorption", so it floors to a value far below any
# optical depth that matters (1e-60 cm^2 over a full column is ~1e-40 in tau).
K_FLOOR = 1.0e-60


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
              "table names). Drop the species, or run the whole model on "
              "opacity_mode='lbl' knowingly -- sampled line-by-line on this "
              "engine's affordable grids is measurably biased (README, "
              "'Opacity: correlated-k').")


def available() -> list:
    """Molecules with an ExoMolOP table present, sorted."""
    d = table_dir()
    if not d.is_dir():
        return []
    return sorted(p.name.split(".")[0] for p in d.glob("*.ktable.h5"))


def load_tables(molecules, nu_min, nu_max, *, molecule_table=None,
                verbose=True) -> SimpleNamespace:
    """Load one ExoMolOP k-table per molecule, restricted to [nu_min, nu_max].

    Returns the namespace the correlated-k core consumes (``logk`` per
    molecule + band edges, (T, P) grids, and the quadrature).

    Raises rather than downloading, and raises rather than mixing tables that
    do not share a band grid or a quadrature -- ``ckd.overlap`` combines
    species ordinate by ordinate, so two tables on different g-nodes would be
    silently added as if they were the same distribution.
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
        raise KeyError(f"no opacity spec for {unknown} in the molecule table")
    missing = [m for m in mols if not table_path(m).exists()]
    if missing:
        raise FileNotFoundError(
            f"ExoMolOP k-table missing for {missing}: looked in {table_dir()}\n"
            + _fetch_hint(missing))

    out, ref = {}, None
    for m in mols:
        with h5py.File(table_path(m), "r") as f:
            edges = np.asarray(f["bin_edges"], dtype=np.float64)
            gg = np.asarray(f["samples"], dtype=np.float64)
            gw = np.asarray(f["weights"], dtype=np.float64)
            t_grid = np.asarray(f["t"], dtype=np.float64)
            p_grid = np.asarray(f["p"], dtype=np.float64)
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
