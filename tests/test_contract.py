"""Contract tests for the shared engine.

These are the properties that made extraction worth doing, so they are pinned
here rather than left as prose: the package imports with no data installed, its
data-root contract fails loudly instead of guessing, the molecule table is
injectable, and planet geometry is required rather than defaulted to WASP-39 b.

Deliberately dependency-light -- no jax, no exojax, no vulcan_jax -- so this
file runs anywhere. The physics itself is validated by the consumers' suites
against real spectra.
"""
from __future__ import annotations

import os
import sys

import pytest


def test_constants_import_without_data_or_heavy_deps():
    """constants must be importable with no data root set and no jax present.

    Before extraction this was impossible: the config module resolved a repo
    root from __file__ at import time and RAISED unless a WASP-39 b
    observed-spectra directory existed as a sibling.
    """
    from vulcan_forward import constants

    assert constants.MOLECULES["CO"]["source"] == "exomol_cached"
    assert constants.ART_PTOP_BAR < constants.ART_PBTM_BAR
    assert set(constants.ATOM_COLS) >= {"H", "O", "C", "He", "N", "S"}
    # the table must stay location-independent: no absolute paths in it
    for name, spec in constants.MOLECULES.items():
        assert not os.path.isabs(spec["db"]), (name, spec["db"])


def test_paths_module_imports_clean_and_fails_loudly(monkeypatch):
    """paths must import with nothing configured, then raise with a remedy."""
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_LINELISTS, raising=False)
    monkeypatch.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    monkeypatch.setattr(paths, "_root_override", None, raising=False)

    with pytest.raises(RuntimeError, match=paths.ENV_ROOT):
        paths.data_root()

    monkeypatch.setenv(paths.ENV_ROOT, "/nonexistent/vulcan-forward-data")
    with pytest.raises(RuntimeError, match="does not exist"):
        paths.data_root()


def test_data_root_and_per_tree_overrides(tmp_path, monkeypatch):
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_LINELISTS, raising=False)
    monkeypatch.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    root = tmp_path / "data"
    (root / "exojax_linelists").mkdir(parents=True)
    (root / "opacity_cache").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_ROOT, str(root))
    monkeypatch.setattr(paths, "_root_override", None, raising=False)

    assert paths.linelist_dir() == root / "exojax_linelists"
    assert paths.opacity_cache_dir() == root / "opacity_cache"
    assert paths.cia_h2he_file().name == "H2-He_2011.cia"
    # db suffixes resolve against the right tree, so the table needs no paths
    assert paths.resolve_db("CO/12C-16O/Li2015", "exomol_cached").startswith(
        str(root / "opacity_cache"))
    assert paths.resolve_db("H2O", "hitran").startswith(
        str(root / "exojax_linelists"))

    # a per-tree override relocates one tree without moving the rest
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(paths.ENV_LINELISTS, str(other))
    assert paths.linelist_dir() == other
    assert paths.opacity_cache_dir() == root / "opacity_cache"


def test_set_data_root_is_honored(tmp_path, monkeypatch):
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    root = tmp_path / "prog"
    (root / "opacity_cache").mkdir(parents=True)
    try:
        paths.set_data_root(root)
        assert paths.data_root() == root
    finally:
        monkeypatch.setattr(paths, "_root_override", None, raising=False)


def test_geometry_is_required_not_wasp39b():
    """The RT builder must refuse a profile with no planet geometry.

    It used to fall back to WASP-39 b constants, so a caller who forgot rp_cm
    silently modeled a different planet. Checked on the helper so the test
    needs neither exojax nor a data tree.
    """
    pytest.importorskip("exojax")
    from vulcan_forward import exojax_rt

    with pytest.raises(ValueError, match="required planet geometry"):
        exojax_rt._require_geometry({"gs_cgs": 422.0}, "rp_cm", "gs_cgs",
                                    "rstar_cm")
    # a complete profile passes
    exojax_rt._require_geometry(
        {"rp_cm": 9.1e9, "gs_cgs": 422.0, "rstar_cm": 6.5e10},
        "rp_cm", "gs_cgs", "rstar_cm")


def test_import_order_guard_is_documented_and_live():
    """vulcan_chem must refuse to be imported after exojax.

    Run in a subprocess so the guard sees a clean interpreter: importing exojax
    first must produce the actionable RuntimeError, not a subtly wrong model.
    """
    pytest.importorskip("exojax")
    import subprocess

    code = ("import exojax\n"
            "import vulcan_forward.vulcan_chem\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode != 0, "importing after exojax must fail"
    assert "must be imported BEFORE exojax" in proc.stderr, proc.stderr
