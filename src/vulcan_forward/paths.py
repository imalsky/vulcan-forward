"""Data-root contract for the shared forward engine.

The engine needs two external data trees at RUN time: the ExoMolOP k-tables
(~389 MB per species) and the offline opacity cache holding the two H2-H2 /
H2-He CIA tables. Together these are ~10 GB, so they are never vendored -- the
consumer says where they live.

CONTRACT (deliberately explicit -- no inference, no silent fallback):

    $VULCAN_FORWARD_DATA   the data root. Expected layout:
                             <root>/opacity_cache/       CIA tables
                             <root>/exomolop/            ExoMolOP k-tables
                                 (<MOL>.ktable.h5 + provenance.json;
                                 python -m vulcan_forward.fetch_exomolop)

    The cache tree can be overridden independently:
      $VULCAN_FORWARD_OPACITY_CACHE  overrides <root>/opacity_cache

A consumer may also set these programmatically with ``set_data_root()`` before
building a model, which is what an application with its own configuration
surface should do.

Why this shape. These paths must never be resolved at import time, from this
package's ``__file__``, or against a marker directory a consumer happens to
own: the engine has to be importable on a machine with no data installed.
Nothing touches the filesystem until a path is actually needed, and then it
fails loudly with the offending value and the remedy (the standing fail-fast
rule in every sibling repo).
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_ROOT = "VULCAN_FORWARD_DATA"
ENV_OPACITY_CACHE = "VULCAN_FORWARD_OPACITY_CACHE"

# Programmatic override, for a consumer that owns its own config surface.
_root_override: Path | None = None


def set_data_root(root: str | os.PathLike) -> None:
    """Point the engine at a data root for this process.

    Takes precedence over $VULCAN_FORWARD_DATA. The per-tree env var still
    wins, so a caller can relocate the cache tree without moving the rest.
    """
    global _root_override
    _root_override = Path(root).expanduser()


def data_root() -> Path:
    """The configured data root, or raise with the remedy.

    Existence is checked here (a typo'd root is a configuration error worth
    reporting immediately), but nothing below it is required -- individual
    trees are validated by the accessors that need them.
    """
    if _root_override is not None:
        root = _root_override
    else:
        env = os.environ.get(ENV_ROOT, "").strip()
        if not env:
            raise RuntimeError(
                f"vulcan_forward needs a data root: set ${ENV_ROOT} to the "
                "directory holding opacity_cache/ and exomolop/, or call "
                "vulcan_forward.paths.set_data_root(...) before building a "
                "model. (The k-tables and CIA tables are ~10 GB, so they are "
                "never bundled with the package.)")
        root = Path(env).expanduser()
    if not root.is_dir():
        raise RuntimeError(
            f"vulcan_forward data root does not exist: {root}. Set ${ENV_ROOT} "
            "to an existing directory holding opacity_cache/ and exomolop/.")
    return root


def ensure_layout() -> Path:
    """Create the data layout under the configured root and return the root.

    ``data_root`` is deliberately strict, because a missing directory during a
    run is a configuration error. A SETUP tool is the opposite case: it knows the
    directories should exist and its job is to make them. Use this from a fetch
    or bootstrap command, never from a model build.

    The root itself still has to be configured -- this function will not guess a
    location -- but it is created if it does not exist, together with the two
    subdirectories the engine reads.
    """
    if _root_override is not None:
        root = _root_override
    else:
        env = os.environ.get(ENV_ROOT, "").strip()
        if not env:
            raise RuntimeError(
                f"vulcan_forward needs a data root: set ${ENV_ROOT} to the "
                "directory that should hold opacity_cache/ and exomolop/, "
                "then run the setup command again.")
        root = Path(env).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    override = os.environ.get(ENV_OPACITY_CACHE, "").strip()
    cache = Path(override).expanduser() if override else root / "opacity_cache"
    cache.mkdir(parents=True, exist_ok=True)
    (root / "exomolop").mkdir(parents=True, exist_ok=True)
    return root


def _tree(env_var: str, subdir: str, *, what: str) -> Path:
    """Resolve one data tree: per-tree env var wins, else <root>/<subdir>."""
    env = os.environ.get(env_var, "").strip()
    path = Path(env).expanduser() if env else data_root() / subdir
    if not path.is_dir():
        raise RuntimeError(
            f"vulcan_forward {what} directory not found: {path}. Set "
            f"${env_var} to override it, or create it under the data root "
            f"(${ENV_ROOT}).")
    return path


def opacity_cache_dir() -> Path:
    """Offline opacity cache: the two CIA tables."""
    return _tree(ENV_OPACITY_CACHE, "opacity_cache", what="opacity-cache")


def exomolop_dir() -> Path:
    """ExoMolOP k-table tree (<MOL>.ktable.h5 + provenance.json).

    No existence check HERE, unlike the ``_tree`` accessor: the loud
    FileNotFoundError with the exact fetch command lives in
    ``exomolop.load_tables``, and datacheck wants the path even when the
    tree is absent so it can report per-molecule MISSING items.
    """
    return data_root() / "exomolop"


def cia_h2h2_file() -> Path:
    """H2-H2 CIA table. Missing is NOT fatal here: exojax auto-fetches it
    (~24 MB from hitran.org) and the caller announces that before it happens
    (its downloader swallows failures, so an offline failure would otherwise be
    unattributable)."""
    return opacity_cache_dir() / "H2-H2_2011.cia"


def cia_h2he_file() -> Path:
    """H2-He CIA table (He is ~14% by number; a real continuum contribution).

    Download once:
    https://hitran.org/data/CIA/main/H2-He_2011.cia  (~147 MB; the /main/
    segment is required -- the bare /data/CIA/ URL 404s). The RT builder
    REFUSES to build without it rather than silently dropping the He
    continuum.
    """
    return opacity_cache_dir() / "H2-He_2011.cia"
