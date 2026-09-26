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

Selection rules:
  * the ISOTOPOLOGUE must be the PRINCIPAL one (most abundant isotope of every
    element), and an unrecognised naming form RAISES rather than guessing.
  * the dataset ExoMol marks "recommended" wins when there is one;
  * within it, the NATURAL-ABUNDANCE file ("<MOL>-all__", "*-NatAbund__")
    wins over the principal isotopologue, because VULCAN tracks a total
    molecular VMR and the opacity must therefore include the minor
    isotopologues that VMR stands for.

exomol.com returns 403 to a default urllib User-Agent, so a browser one is
sent. Downloads run sequentially (I/O bound).
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request

from vulcan_forward import exomolop, paths

# What a failed page fetch or table download raises (URLError and socket
# timeouts are OSErrors too); anything else is a bug and propagates.
_NET_ERRORS = (urllib.error.URLError, http.client.HTTPException, TimeoutError,
               OSError)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
BASE = "https://www.exomol.com"
ROOT = f"{BASE}/data/data-types/opacity"
# The ONE product this engine can mix: R=1000 bands over 0.3-50 um. Every
# table in a correlated-k mixture must share a band grid, so a file at another
# resolution or span is not a substitute -- see resolve().
GRID_TOKEN = "R1000_0.3-50mu"

HTTP_RETRIES = 3           # attempts per page fetch
RETRY_WAIT_S = 2.0         # pause between attempts
PAGE_TIMEOUT_S = 90        # one HTML page
DOWNLOAD_TIMEOUT_S = 600   # socket timeout while streaming a ~389 MB table
CHUNK_BYTES = 1 << 20      # 1 MiB read size for the table stream

# Line-list DOIs for datasets whose k-table header carries a placeholder DOI
# (`x.xxxx/yyyyy`, `xxxxxxx/xxxxxxxxx/xxxxxx`); provenance.json supplies them.
# Other datasets keep their header DOI. Keyed by ExoMol dataset name.
_DATASET_DOI = {
    "MM": "10.1093/mnras/stae148",       # CH4  Yurchenko+ 2024, MNRAS 528, 3719
    "Dozen": "10.1093/mnras/staf2135",   # CO2  Yurchenko+ 2025, MNRAS 545
    "TYM": "10.1093/mnras/stae2201",     # N2O  Yurchenko+ 2024, MNRAS 534, 1364
    "OYT8": "10.1093/mnras/stae1110",    # OCS  Owens+ 2024, MNRAS 530, 4004
}


def _record(url, ds, iso, nat):
    """One provenance.json entry. `doi` is present only for the datasets in
    _DATASET_DOI; every other dataset is described by its header DOI."""
    rec = {"url": url, "dataset": ds, "iso": iso, "natural_abundance": nat,
           "file": url.rsplit("/", 1)[1]}
    doi = _DATASET_DOI.get(ds)
    if doi:
        rec["doi"] = doi
    return rec


def _get(url, retries=HTTP_RETRIES):
    """Fetch a page or raise after ``retries`` attempts, so a network failure
    is never reported as SKIP (a page with no k-table link)."""
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=PAGE_TIMEOUT_S) as r:
                return r.read().decode("utf-8", "replace")
        except _NET_ERRORS as e:
            last = e
            if k < retries - 1:
                time.sleep(RETRY_WAIT_S)
    raise RuntimeError(
        f"failed to fetch {url} after {retries} attempts: {last}. "
        "Cannot tell whether ExoMolOP publishes this species; fix the "
        "network and re-run rather than skipping.") from last


# Most abundant isotope of each element, by mass number. Used to identify the
# PRINCIPAL isotopologue among ExoMol's listings.
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
    if not isos:
        raise RuntimeError(
            f"{mol}: no isotopologue links parsed from {ROOT}/{mol}/ -- the page "
            "layout changed or the response was not the molecule page. Not a "
            "SKIP (that means a page with no k-table link); fix the parser.")
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


def _assert_grid_matches(mol, part, dest, dest_dir):
    """Verify the DOWNLOADED file (``part``, not yet installed) shares the grid
    of the tables already here.

    ``resolve`` only checks the filename; this opens the file and runs
    ``load_tables``' layout checks and grid comparison. Deletes the download before
    raising on a grid mismatch; ``dest`` (a table ``--force`` would replace)
    is left untouched.
    """
    import h5py                                    # offline step only

    peers = sorted(p for p in dest_dir.glob("*.ktable.h5") if p != dest)
    if not peers:
        return                                     # first table: nothing to compare
    with h5py.File(peers[0], "r") as f:
        ref = exomolop._validated_layout(f, peers[0])
    with h5py.File(part, "r") as f:
        got = exomolop._validated_layout(f, part)
    bad = exomolop._grid_mismatch(got, ref)
    if bad is not None:
        os.unlink(part)
        raise RuntimeError(
            f"{mol}: the downloaded table disagrees with {peers[0].name} on "
            f"the {bad}. Correlated-k tables are mixed ordinate by ordinate, "
            "so they must share one band grid, one (T, P) grid and one "
            "quadrature. The download has been deleted and any installed "
            "table left as it was. Re-fetch every table from the same "
            "ExoMolOP release.")


def fetch(molecules, force=False):
    paths.ensure_layout()          # a setup command creates the data root
    dest_dir = paths.exomolop_dir()
    prov_path = dest_dir / "provenance.json"
    prov = {}
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
        except ValueError as e:
            raise RuntimeError(
                f"{prov_path} is not valid JSON ({e}); refusing to overwrite the "
                "provenance record. Repair or move it, then rerun.") from e
    for mol in molecules:
        dest = exomolop.table_path(mol)
        if dest.exists() and not force:
            if mol in prov:
                # Already attributed: no network, but still re-stamp the
                # curated DOI so a change to _DATASET_DOI reaches records
                # written before it existed. Nothing else is re-derived.
                doi = _DATASET_DOI.get(prov[mol].get("dataset"))
                if doi:
                    prov[mol]["doi"] = doi
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
            prov[mol] = _record(url, ds, iso, nat)
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
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r, \
                    open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
        except _NET_ERRORS as e:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise RuntimeError(f"failed to fetch {mol} from {url}: {e}") from e
        _assert_grid_matches(mol, tmp, dest, dest_dir)
        os.replace(tmp, dest)
        prov[mol] = _record(url, ds, iso, nat)
        tag = "natural-abundance" if nat else f"principal ({iso})"
        print(f"{mol:6s} GET   {dest.stat().st_size/1e6:7.1f} MB  "
              f"{time.time()-t0:5.1f}s  {ds} {tag}")
    prov_path.write_text(json.dumps(prov, indent=1, sort_keys=True))
    print(f"\nprovenance -> {prov_path}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
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
