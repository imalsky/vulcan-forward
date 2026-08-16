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

Two selection rules, both deliberate:
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


def resolve(mol: str):
    """(url, dataset, iso, natural_abundance) for ``mol``, or None."""
    html = _get(f"{ROOT}/{mol}/")
    isos = [i for i in sorted(set(re.findall(r'href="([0-9A-Za-z-]+)(?:#[^"]*)?"\s',
                                             html))) if re.match(r"^\d", i)]
    for iso in isos[:1]:                       # principal isotopologue page
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
        nat = [h for h in prt if "-all__" in h or "NatAbund__" in h]
        return BASE + (nat[0] if nat else prt[0]), ds, iso, bool(nat)
    return None


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
            print(f"{mol:6s} have  {dest.stat().st_size/1e6:7.1f} MB")
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
