"""Fetch ExoMolOP k-tables (offline step; never runs from the RT path).

    python -m vulcan_forward.fetch_exomolop --molecules H2O,CO2,CO,CH4,SO2

Downloads into ``$VULCAN_FORWARD_DATA/exomolop/`` as ``<MOL>.ktable.h5``,
about 389 MB each, and writes a ``provenance.json`` recording which ExoMol
dataset every file came from. Resumable: an existing file is left alone.

The download URLs are RESOLVED, not guessed. ExoMolOP's pages are three levels
deep and the filenames do not follow one pattern (H2O is
``1H2-16O__POKAZATEL__R1000_...`` with a double underscore, everything else is
``<iso>__<dataset>.R1000_...`` with a dot), so guessing 404s. The walk is:

    /data/data-types/opacity/<MOL>/              -> isotopologue pages
    /data/data-types/opacity/<MOL>/<ISO>/        -> dataset pages
    /data/data-types/opacity/<MOL>/<ISO>/<SET>/  -> the /db/... file links

Three selection rules, all deliberate:
  * the ISOTOPOLOGUE must be the PRINCIPAL one (most abundant isotope of every
    element), and an unrecognised naming form RAISES rather than guessing.
    Taking the first sorted entry instead picked N2O's ~0.4%-abundance
    ``14N-15N-16O``, understating its opacity ~1e2.
  * the dataset ExoMol marks "recommended" wins when there is one;
  * within it, the NATURAL-ABUNDANCE file ("<MOL>-all__", "*-NatAbund__")
    wins over the principal isotopologue, because VULCAN tracks a total
    molecular VMR and the opacity must therefore include the minor
    isotopologues that VMR stands for.

exomol.com returns 403 to a default urllib User-Agent, so a browser one is
sent. Downloads run one at a time on purpose: they are I/O bound, and this
machine has been killed by memory exhaustion from parallel jobs before.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

from vulcan_forward import exomolop

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
BASE = "https://www.exomol.com"
ROOT = f"{BASE}/data/data-types/opacity"
# The ONE product this engine can mix: R=1000 bands over 0.3-50 um. Every
# table in a correlated-k mixture must share a band grid, so a file at another
# resolution or span is not a substitute -- see resolve().
GRID_TOKEN = "R1000_0.3-50mu"


def _get(url, retries=3):
    """Fetch a page or RAISE. Swallowing a network failure into "" made an
    offline run print "ExoMolOP publishes no k-table for this species" --
    indistinguishable from genuinely-unpublished (standing loud-errors rule).
    A SKIP must mean a real page with no k-table link, nothing else."""
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                                # noqa: BLE001
            last = e
            if k < retries - 1:
                time.sleep(2.0)
    raise RuntimeError(
        f"failed to fetch {url} after {retries} attempts: {last}. "
        "Cannot tell whether ExoMolOP publishes this species; fix the "
        "network and re-run rather than skipping.") from last


# Most abundant isotope of each element, by mass number. Used to identify the
# PRINCIPAL isotopologue among ExoMol's listings; see the module docstring for
# the N2O case that made this necessary.
_PRINCIPAL_ISOTOPE = {
    "H": 1, "He": 4, "Li": 7, "C": 12, "N": 14, "O": 16, "F": 19, "Na": 23,
    "Mg": 24, "Al": 27, "Si": 28, "P": 31, "S": 32, "Cl": 35, "K": 39,
    "Ca": 40, "Ti": 48, "V": 51, "Cr": 52, "Fe": 56,
}
_ISO_TOKEN = re.compile(r"^(\d+)([A-Z][a-z]?)(\d*)$")


def _is_principal(iso: str) -> bool:
    """True if every element in ``iso`` carries its most abundant isotope.

    ``iso`` is ExoMol's isotopologue token, e.g. ``1H2-16O``, ``14N2-16O``,
    ``12C-32S``. An unparseable token returns False, which makes the caller
    raise rather than silently accept an unknown naming form.
    """
    parts = iso.split("-")
    if not parts:
        return False
    for tok in parts:
        m = _ISO_TOKEN.match(tok)
        if not m:
            return False
        mass, elem = int(m.group(1)), m.group(2)
        if _PRINCIPAL_ISOTOPE.get(elem) != mass:
            return False
    return True


def resolve(mol: str):
    """(url, dataset, iso, natural_abundance) for ``mol``, or None."""
    html = _get(f"{ROOT}/{mol}/")
    isos = [i for i in sorted(set(re.findall(r'href="([0-9A-Za-z-]+)(?:#[^"]*)?"\s',
                                             html))) if re.match(r"^\d", i)]
    principal = [i for i in isos if _is_principal(i)]
    if isos and not principal:
        raise RuntimeError(
            f"{mol}: none of the isotopologues ExoMolOP lists {isos} parses as "
            "the principal one (most abundant isotope of every element). "
            "Refusing to guess: a rare isotopologue's cross section paired "
            "with a total molecular VMR understates the opacity by orders of "
            "magnitude. Extend _PRINCIPAL_ISOTOPE or name the isotopologue "
            "explicitly.")
    if len(principal) > 1:
        raise RuntimeError(
            f"{mol}: {principal} all parse as principal isotopologues, which "
            "should be impossible. Refusing to pick one arbitrarily.")
    for iso in principal:                      # principal isotopologue page
        sets = []
        dhtml = _get(f"{ROOT}/{mol}/{iso}/")
        for m in re.finditer(r'href="([A-Za-z0-9_-]+)(?:#[^"]*)?"[^>]*>(.*?)</a>',
                             dhtml, re.S):
            if "list-group-item" in m.group(0):
                sets.append((m.group(1), "recommended" in m.group(2)))
        if not sets:
            return None
        rec = [s for s in sets if s[1]] or sets
        ds = rec[-1][0]
        fhtml = _get(f"{ROOT}/{mol}/{iso}/{ds}/")
        prt = [h for h in sorted(set(re.findall(r'href="(/db/[^"]+)"', fhtml)))
               if "petitRADTRANS" in h and h.endswith(".h5")]
        if not prt:
            return None
        # More than one petitRADTRANS product per species, not interchangeable:
        # O2's only k-table is R15000_0.2-30mu, 11.8 GB on a different grid,
        # which taking prt[0] would download.
        onthe = [h for h in prt if GRID_TOKEN in h]
        if not onthe:
            raise RuntimeError(
                f"{mol}: ExoMolOP has petitRADTRANS files for {iso}/{ds} but "
                f"none on the {GRID_TOKEN} grid this engine uses "
                f"(found {[h.rsplit('/', 1)[1] for h in prt]}). A different "
                "resolution or wavelength span is NOT a drop-in substitute -- "
                "every table in a mixture must share one band grid.")
        nat = [h for h in onthe if "-all__" in h or "NatAbund__" in h]
        return BASE + (nat[0] if nat else onthe[0]), ds, iso, bool(nat)
    return None


def _assert_grid_matches(mol, dest, dest_dir):
    """Verify the DOWNLOADED file shares the grid of the tables already here.

    ``resolve``'s filename filter is a convention check, which is exactly what
    a wrong guess also passes; this opens the file and compares the arrays
    ``load_tables`` will demand agree. Deletes the offender before raising --
    left on disk, the next run finds it and skips the download.
    """
    import h5py                                    # offline step only
    import numpy as np

    peers = sorted(p for p in dest_dir.glob("*.ktable.h5") if p != dest)
    if not peers:
        return                                     # first table: nothing to compare
    keys = ("bin_edges", "t", "p", "samples", "weights")
    with h5py.File(peers[0], "r") as f:
        ref = {k: np.asarray(f[k], dtype=np.float64) for k in keys}
    with h5py.File(dest, "r") as f:
        got = {k: np.asarray(f[k], dtype=np.float64) for k in keys}
    for k in keys:
        a, b = got[k], ref[k]
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=0.0):
            os.unlink(dest)
            raise RuntimeError(
                f"{mol}: the downloaded table disagrees with {peers[0].name} "
                f"on '{k}' (shape {a.shape} vs {b.shape}). Correlated-k tables "
                "are mixed ordinate by ordinate, so they must share one band "
                "grid, one (T, P) grid and one quadrature. The file has been "
                "deleted. Re-fetch every table from the same ExoMolOP release.")


def fetch(molecules, force=False):
    dest_dir = exomolop.table_dir()
    os.makedirs(dest_dir, exist_ok=True)
    prov_path = dest_dir / "provenance.json"
    prov = {}
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
        except ValueError:
            prov = {}
    for mol in molecules:
        dest = exomolop.table_path(mol)
        if dest.exists() and not force:
            if mol in prov:
                print(f"{mol:6s} have  {dest.stat().st_size/1e6:7.1f} MB")
                continue
            # Present but unattributed -- what an interrupted fetch leaves
            # (provenance is written once, at the end). Record it without
            # re-downloading, so provenance always describes what is on disk.
            got = resolve(mol)
            if got is None:
                print(f"{mol:6s} have  {dest.stat().st_size/1e6:7.1f} MB  "
                      "(provenance UNRESOLVED)")
                continue
            url, ds, iso, nat = got
            prov[mol] = {"url": url, "dataset": ds, "iso": iso,
                         "natural_abundance": nat,
                         "file": url.rsplit("/", 1)[1]}
            print(f"{mol:6s} have  {dest.stat().st_size/1e6:7.1f} MB  "
                  f"(provenance backfilled: {ds} {iso})")
            continue
        got = resolve(mol)
        if got is None:
            print(f"{mol:6s} SKIP  ExoMolOP publishes no petitRADTRANS "
                  f"k-table for this species")
            continue
        url, ds, iso, nat = got
        tmp = str(dest) + ".part"
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, \
                    open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(tmp, dest)
        except Exception as e:                                # noqa: BLE001
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise RuntimeError(f"failed to fetch {mol} from {url}: {e}") from e
        _assert_grid_matches(mol, dest, dest_dir)
        prov[mol] = {"url": url, "dataset": ds, "iso": iso,
                     "natural_abundance": nat,
                     "file": url.rsplit("/", 1)[1]}
        tag = "natural-abundance" if nat else f"principal ({iso})"
        print(f"{mol:6s} GET   {dest.stat().st_size/1e6:7.1f} MB  "
              f"{time.time()-t0:5.1f}s  {ds} {tag}")
    prov_path.write_text(json.dumps(prov, indent=1, sort_keys=True))
    print(f"\nprovenance -> {prov_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--molecules", required=True,
                    help="comma-separated, e.g. H2O,CO2,CO,CH4,SO2")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is present")
    a = ap.parse_args(argv)
    fetch([m.strip() for m in a.molecules.split(",") if m.strip()],
          force=a.force)
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
