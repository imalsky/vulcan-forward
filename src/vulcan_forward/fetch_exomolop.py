"""Fetch ExoMolOP k-tables and record where they came from; an offline step, never run from the RT path.

    python -m vulcan_forward.fetch_exomolop --molecules H2O,CO2,CO,CH4,SO2

Downloads into ``$VULCAN_FORWARD_DATA/exomolop/`` as ``<MOL>.ktable.h5``, about 389 MB each, and writes
``provenance.json``. An existing file is left alone. URLs are resolved from ExoMolOP's pages; the selection rules are in
docs/method.md.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.request
from typing import TYPE_CHECKING, Any

from vulcan_forward import exomolop, paths

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = ["fetch", "main", "resolve"]

# What a failed page fetch or table download raises (URLError, socket timeouts and TLS errors are OSErrors).
_NET_ERRORS = (http.client.HTTPException, OSError)

# exomol.com returns 403 to a default urllib User-Agent.
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
BASE = "https://www.exomol.com"
ROOT = f"{BASE}/data/data-types/opacity"
# The one product this engine can mix: every table in a correlated-k mixture must share a band grid.
GRID_TOKEN = "R1000_0.3-50mu"

PAGE_TIMEOUT_S = 90  # one HTML page
DOWNLOAD_TIMEOUT_S = 600  # socket timeout while streaming a ~389 MB table
CHUNK_BYTES = 1 << 20  # 1 MiB read size for the table stream

# Line-list DOIs, keyed by ExoMol dataset name, for the datasets whose k-table header carries a placeholder DOI
# (`x.xxxx/yyyyy`, `xxxxxxx/xxxxxxxxx/xxxxxx`); provenance.json supplies them. Other datasets keep their header DOI.
_DATASET_DOI = {
    "MM": "10.1093/mnras/stae148",  # CH4  Yurchenko+ 2024, MNRAS 528, 3719
    "Dozen": "10.1093/mnras/staf2135",  # CO2  Yurchenko+ 2025, MNRAS 545
    "TYM": "10.1093/mnras/stae2201",  # N2O  Yurchenko+ 2024, MNRAS 534, 1364
    "OYT8": "10.1093/mnras/stae1110",  # OCS  Owens+ 2024, MNRAS 530, 4004
}


def _record(url: str, ds: str, iso: str, nat: bool) -> dict[str, Any]:
    """Build one provenance.json entry; ``doi`` is present only for the datasets in ``_DATASET_DOI``."""
    rec = {"url": url, "dataset": ds, "iso": iso, "natural_abundance": nat, "file": url.rsplit("/", 1)[1]}
    doi = _DATASET_DOI.get(ds)
    if doi:
        rec["doi"] = doi
    return rec


def _get(url: str) -> str:
    """Fetch a page or raise, so a network failure is never reported as SKIP."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=PAGE_TIMEOUT_S) as r:
            return r.read().decode("utf-8", "replace")
    except _NET_ERRORS as e:
        raise RuntimeError(
            f"failed to fetch {url}: {e}. Cannot tell whether ExoMolOP publishes this species; "
            "fix the network and re-run rather than skipping."
        ) from e


# Most abundant isotope of each element, by mass number, to identify the principal isotopologue among ExoMol's listings.
_PRINCIPAL_ISOTOPE = {
    "H": 1,
    "He": 4,
    "Li": 7,
    "C": 12,
    "N": 14,
    "O": 16,
    "F": 19,
    "Na": 23,
    "Mg": 24,
    "Al": 27,
    "Si": 28,
    "P": 31,
    "S": 32,
    "Cl": 35,
    "K": 39,
    "Ca": 40,
    "Ti": 48,
    "V": 51,
    "Cr": 52,
    "Fe": 56,
}
_ISO_TOKEN = re.compile(r"^(\d+)([A-Z][a-z]?)(\d*)$")


def _is_principal(iso: str) -> bool:
    """Say whether every element in ``iso`` carries its most abundant isotope.

    ``iso`` is ExoMol's isotopologue token, such as ``1H2-16O``. An unknown token returns False, so the caller raises
    rather than accept an unknown naming form.
    """
    for tok in iso.split("-"):
        m = _ISO_TOKEN.match(tok)
        if not m:
            return False
        mass, elem = int(m.group(1)), m.group(2)
        if _PRINCIPAL_ISOTOPE.get(elem) != mass:
            return False
    return True


def resolve(mol: str) -> tuple[str, str, str, bool] | None:
    """Find the table to fetch for one molecule by walking ExoMolOP's pages.

    Parameters
    ----------
    mol : str
        The molecule.

    Returns
    -------
    tuple or None
        ``(url, dataset, iso, natural_abundance)``, or None when ExoMolOP publishes no petitRADTRANS k-table for it.

    Raises
    ------
    RuntimeError
        If the pages cannot be parsed, no principal isotopologue is found, or no file is on the engine's band grid.
    """
    html = _get(f"{ROOT}/{mol}/")
    isos = [i for i in sorted(set(re.findall(r'href="([0-9A-Za-z-]+)(?:#[^"]*)?"\s', html))) if re.match(r"^\d", i)]
    principal = [i for i in isos if _is_principal(i)]
    if not isos:
        raise RuntimeError(
            f"{mol}: no isotopologue links parsed from {ROOT}/{mol}/ -- the page "
            "layout changed or the response was not the molecule page. Not a "
            "SKIP (that means a page with no k-table link); fix the parser."
        )
    if not principal:
        raise RuntimeError(
            f"{mol}: none of the isotopologues ExoMolOP lists {isos} parses as "
            "the principal one (most abundant isotope of every element). "
            "Refusing to guess: a rare isotopologue's cross section paired "
            "with a total molecular VMR understates the opacity by orders of "
            "magnitude. Extend _PRINCIPAL_ISOTOPE or name the isotopologue "
            "explicitly."
        )
    if len(principal) > 1:
        raise RuntimeError(
            f"{mol}: {principal} all parse as principal isotopologues, which "
            "should be impossible. Refusing to pick one arbitrarily."
        )
    iso = principal[0]
    sets = []
    dhtml = _get(f"{ROOT}/{mol}/{iso}/")
    for m in re.finditer(r'href="([A-Za-z0-9_-]+)(?:#[^"]*)?"[^>]*>(.*?)</a>', dhtml, re.DOTALL):
        if "list-group-item" in m.group(0):
            sets.append((m.group(1), "recommended" in m.group(2)))
    if not sets:
        return None
    rec = [s for s in sets if s[1]] or sets
    ds = rec[-1][0]
    fhtml = _get(f"{ROOT}/{mol}/{iso}/{ds}/")
    prt = [
        h for h in sorted(set(re.findall(r'href="(/db/[^"]+)"', fhtml))) if "petitRADTRANS" in h and h.endswith(".h5")
    ]
    if not prt:
        return None
    # A species can have other petitRADTRANS products: O2's only k-table is R15000_0.2-30mu, 11.8 GB on another grid.
    on_grid = [h for h in prt if GRID_TOKEN in h]
    if not on_grid:
        raise RuntimeError(
            f"{mol}: ExoMolOP has petitRADTRANS files for {iso}/{ds} but "
            f"none on the {GRID_TOKEN} grid this engine uses "
            f"(found {[h.rsplit('/', 1)[1] for h in prt]}). A different "
            "resolution or wavelength span is NOT a drop-in substitute -- "
            "every table in a mixture must share one band grid."
        )
    nat = [h for h in on_grid if "-all__" in h or "NatAbund__" in h]
    return BASE + (nat[0] if nat else on_grid[0]), ds, iso, bool(nat)


def _assert_grid_matches(mol: str, part: Path, dest: Path, dest_dir: Path) -> None:
    """Check that the downloaded file ``part`` is a valid table and shares the grid of the tables already installed.

    ``resolve`` checks only the filename. Deletes the download before raising; ``dest``, which ``--force`` would
    replace, stays as it was.
    """
    import h5py

    try:
        with h5py.File(part, "r") as f:
            got = exomolop._validated_layout(f, part)
            exomolop._header(f, part)
    except (OSError, ValueError) as e:
        part.unlink()
        raise RuntimeError(f"{mol}: the download is not a valid k-table ({e}). It has been deleted.") from e
    peers = sorted(p for p in dest_dir.glob("*.ktable.h5") if p != dest)
    if not peers:
        return
    with h5py.File(peers[0], "r") as f:
        ref = exomolop._validated_layout(f, peers[0])
    bad = exomolop._grid_mismatch(got, ref)
    if bad is not None:
        part.unlink()
        raise RuntimeError(
            f"{mol}: the downloaded table disagrees with {peers[0].name} on "
            f"the {bad}. Correlated-k tables are mixed ordinate by ordinate, "
            "so they must share one band grid, one (T, P) grid and one "
            "quadrature. The download has been deleted and any installed "
            "table left as it was. Re-fetch every table from the same "
            "ExoMolOP release."
        )


def fetch(molecules: Iterable[str], force: bool = False) -> None:
    """Download the tables of ``molecules`` and update provenance.json.

    Parameters
    ----------
    molecules : iterable of str
        The species.
    force : bool, optional
        Download a table even if it is present.

    Raises
    ------
    RuntimeError
        If provenance.json is not valid JSON, or a download fails.
    """
    paths.ensure_layout()
    dest_dir = paths.exomolop_dir()
    prov_path = dest_dir / "provenance.json"
    prov = {}
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
        except ValueError as e:
            raise RuntimeError(
                f"{prov_path} is not valid JSON ({e}); refusing to overwrite the "
                "provenance record. Repair or move it, then rerun."
            ) from e
    for mol in molecules:
        dest = exomolop.table_path(mol)
        if dest.exists() and not force:
            if mol in prov:
                # Already attributed: no network, but re-stamp the curated DOI so a change to _DATASET_DOI reaches it.
                doi = _DATASET_DOI.get(prov[mol].get("dataset"))
                if doi:
                    prov[mol]["doi"] = doi
                print(f"{mol:6s} have  {dest.stat().st_size / 1e6:7.1f} MB")
                continue
            # Present but unattributed, as an interrupted fetch leaves it: provenance is written once, at the end.
            got = resolve(mol)
            if got is None:
                print(f"{mol:6s} have  {dest.stat().st_size / 1e6:7.1f} MB  (provenance UNRESOLVED)")
                continue
            url, ds, iso, nat = got
            prov[mol] = _record(url, ds, iso, nat)
            print(f"{mol:6s} have  {dest.stat().st_size / 1e6:7.1f} MB  (provenance backfilled: {ds} {iso})")
            continue
        got = resolve(mol)
        if got is None:
            print(f"{mol:6s} SKIP  ExoMolOP publishes no petitRADTRANS k-table for this species")
            continue
        url, ds, iso, nat = got
        tmp = dest.with_name(dest.name + ".part")
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r, tmp.open("wb") as fh:
                while True:
                    chunk = r.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
        except _NET_ERRORS as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"failed to fetch {mol} from {url}: {e}") from e
        _assert_grid_matches(mol, tmp, dest, dest_dir)
        tmp.replace(dest)
        prov[mol] = _record(url, ds, iso, nat)
        tag = "natural-abundance" if nat else f"principal ({iso})"
        print(f"{mol:6s} GET   {dest.stat().st_size / 1e6:7.1f} MB  {time.time() - t0:5.1f}s  {ds} {tag}")
    prov_path.write_text(json.dumps(prov, indent=1, sort_keys=True))
    print(f"\nprovenance -> {prov_path}")


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        The arguments; default ``sys.argv``.

    Returns
    -------
    int
        The exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--molecules", required=True, help="comma-separated, e.g. H2O,CO2,CO,CH4,SO2")
    ap.add_argument("--force", action="store_true", help="re-download even if the file is present")
    a = ap.parse_args(argv)
    fetch([m.strip() for m in a.molecules.split(",") if m.strip()], force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
