"""Data-root contract: where the k-tables and the CIA tables live.

The layout under ``$VULCAN_FORWARD_DATA`` (or ``set_data_root``) is ``opacity_cache/`` for the two CIA tables and
``exomolop/`` for ``<MOL>.ktable.h5`` plus ``provenance.json``. ``$VULCAN_FORWARD_OPACITY_CACHE`` relocates the cache
alone. Paths resolve on use, never at import, so the engine imports with no data installed.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ENV_OPACITY_CACHE",
    "ENV_ROOT",
    "cia_h2h2_file",
    "cia_h2he_file",
    "data_root",
    "ensure_layout",
    "exomolop_dir",
    "opacity_cache_dir",
    "set_data_root",
]

ENV_ROOT = "VULCAN_FORWARD_DATA"
ENV_OPACITY_CACHE = "VULCAN_FORWARD_OPACITY_CACHE"

# Set by set_data_root, for a consumer that owns its own configuration.
_root_override: Path | None = None


def set_data_root(root: str | os.PathLike[str]) -> None:
    """Point the engine at a data root for this process.

    Takes precedence over ``$VULCAN_FORWARD_DATA``; ``$VULCAN_FORWARD_OPACITY_CACHE`` still relocates the cache.

    Parameters
    ----------
    root : str or os.PathLike
        The directory holding ``opacity_cache/`` and ``exomolop/``.
    """
    global _root_override  # noqa: PLW0603  # the one process-wide data root
    _root_override = Path(root).expanduser()


def _configured_root() -> Path:
    """Return ``set_data_root`` or ``$VULCAN_FORWARD_DATA``, not checked for existence; raise when neither is set."""
    if _root_override is not None:
        return _root_override
    env = os.environ.get(ENV_ROOT, "").strip()
    if not env:
        raise RuntimeError(
            f"vulcan_forward needs a data root: set ${ENV_ROOT} to the "
            "directory holding opacity_cache/ and exomolop/, or call "
            "vulcan_forward.paths.set_data_root(...) before building a "
            "model. (The k-tables and CIA tables are ~10 GB, so they are "
            "never bundled with the package.)"
        )
    return Path(env).expanduser()


def _cache_override() -> Path | None:
    """Return ``$VULCAN_FORWARD_OPACITY_CACHE``, or None when unset."""
    env = os.environ.get(ENV_OPACITY_CACHE, "").strip()
    return Path(env).expanduser() if env else None


def data_root() -> Path:
    """Return the configured data root, which must exist; the trees below it are checked by their accessors."""
    root = _configured_root()
    if not root.is_dir():
        raise RuntimeError(
            f"vulcan_forward data root does not exist: {root}. Set ${ENV_ROOT} "
            "to an existing directory holding opacity_cache/ and exomolop/."
        )
    return root


def ensure_layout() -> Path:
    """Create the configured root and its two subdirectories, for a fetch command, and return the root."""
    root = _configured_root()
    root.mkdir(parents=True, exist_ok=True)
    (_cache_override() or root / "opacity_cache").mkdir(parents=True, exist_ok=True)
    (root / "exomolop").mkdir(parents=True, exist_ok=True)
    return root


def opacity_cache_dir() -> Path:
    """Return the cache of the two CIA tables: ``$VULCAN_FORWARD_OPACITY_CACHE``, else ``<root>/opacity_cache``."""
    path = _cache_override() or data_root() / "opacity_cache"
    if not path.is_dir():
        raise RuntimeError(
            f"vulcan_forward opacity-cache directory not found: {path}. Set "
            f"${ENV_OPACITY_CACHE} to override it, or create it under the data "
            f"root (${ENV_ROOT})."
        )
    return path


def exomolop_dir() -> Path:
    """Return the k-table tree, unchecked: ``exomolop.load_tables`` raises the fetch command."""
    return data_root() / "exomolop"


def cia_h2h2_file() -> Path:
    """Return the H2-H2 CIA table path, which ``build_rt_model`` requires (hitran.org/data/CIA/main/, 24 MB)."""
    return opacity_cache_dir() / "H2-H2_2011.cia"


def cia_h2he_file() -> Path:
    """Return the H2-He CIA table path, which ``build_rt_model`` requires (hitran.org/data/CIA/main/, 147 MB)."""
    return opacity_cache_dir() / "H2-He_2011.cia"
