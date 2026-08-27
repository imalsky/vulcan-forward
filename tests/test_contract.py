"""Contract tests for the shared engine.

These are the engine's library properties, pinned here rather than left as
prose: the package imports with no data installed, its
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
    """constants must be importable with no data root set and no jax present."""
    from vulcan_forward import constants

    assert constants.MOLECULES["CO"]["molmass"] == pytest.approx(28.010)
    assert constants.ART_PTOP_BAR < constants.ART_PBTM_BAR
    assert set(constants.ATOM_COLS) >= {"H", "O", "C", "He", "N", "S"}
    # exactly the two fields the correlated-k path reads; no paths, no
    # line-list names (those live in the k-table tree's provenance.json)
    for name, spec in constants.MOLECULES.items():
        assert set(spec) == {"vulcan", "molmass"}, name


def test_every_molmass_matches_its_own_formula():
    """molmass is the only mass the engine uses; it must equal the formula.

    Seven entries once carried an isotopologue or a pre-2009 sulfur weight.
    Harmless while the mass cancels out of the optical depth -- which is
    exactly why nothing else would catch it.
    """
    import re

    from vulcan_forward import constants

    weights = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}
    for name, entry in constants.MOLECULES.items():
        want = sum(
            weights[el] * (int(n) if n else 1)
            for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", name)
            if el
        )
        assert entry["molmass"] == pytest.approx(want, abs=5e-4), name


def test_paths_module_imports_clean_and_fails_loudly(monkeypatch):
    """paths must import with nothing configured, then raise with a remedy."""
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    monkeypatch.setattr(paths, "_root_override", None, raising=False)

    with pytest.raises(RuntimeError, match=paths.ENV_ROOT):
        paths.data_root()

    monkeypatch.setenv(paths.ENV_ROOT, "/nonexistent/vulcan-forward-data")
    with pytest.raises(RuntimeError, match="does not exist"):
        paths.data_root()


def test_data_root_and_per_tree_overrides(tmp_path, monkeypatch):
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    root = tmp_path / "data"
    (root / "opacity_cache").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_ROOT, str(root))
    monkeypatch.setattr(paths, "_root_override", None, raising=False)

    assert paths.opacity_cache_dir() == root / "opacity_cache"
    assert paths.cia_h2he_file().name == "H2-He_2011.cia"
    # no existence check on the k-table tree: datacheck wants the path even
    # when nothing is installed, to report per-molecule MISSING items
    assert paths.exomolop_dir() == root / "exomolop"

    # the per-tree override relocates the cache without moving the rest
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(paths.ENV_OPACITY_CACHE, str(other))
    assert paths.opacity_cache_dir() == other
    assert paths.exomolop_dir() == root / "exomolop"


def test_set_data_root_is_honored(tmp_path, monkeypatch):
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    root = tmp_path / "prog"
    (root / "opacity_cache").mkdir(parents=True)
    try:
        paths.set_data_root(root)
        assert paths.data_root() == root
    finally:
        # PLAIN assignment, not monkeypatch.setattr: monkeypatch records the
        # value it finds (already set to `root` by the call above) as the
        # original and restores it at teardown, so cleaning up through
        # monkeypatch here actually leaked this now-deleted tmp_path into
        # paths._root_override for every later test in the session.
        paths._root_override = None


def test_geometry_is_required_not_wasp39b():
    """The RT builder must refuse a profile with no planet geometry.

    A defaulted rp_cm would silently model a different planet. Checked on the
    helper so the test needs neither exojax nor a data tree.
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


def test_chem_params_named_api_matches_the_vector_form():
    """The engine's primitive is named ChemParams; the positional vector stays
    supported as the adapter a sampler or a jvp needs.

    Runs in a SUBPROCESS on purpose. Importing vulcan_chem needs a clean
    interpreter: an earlier test in this file imports exojax_rt, and after that
    the import-order guard correctly refuses vulcan_chem.
    """
    pytest.importorskip("jax")
    import subprocess

    code = r"""
import numpy as np, jax
from vulcan_forward.vulcan_chem import ChemParams, params_from_vector

# with a tp_eval hook the tail is that hook's parameter block
theta = [0.5, -0.25, 1.5, 1200.0, -1.0]
p = params_from_vector(theta, n_tp_params=2, has_tp_eval=True)
assert (float(p.lnZ), float(p.c_o), float(p.lnKzz)) == (0.5, -0.25, 1.5)
assert np.allclose(np.asarray(p.tp), [1200.0, -1.0])
assert np.allclose(np.asarray(p.to_vector()), theta)

# without one, the tail is a single uniform temperature offset in K
q = params_from_vector([0.0, 0.0, 0.0, 25.0], n_tp_params=0, has_tp_eval=False)
assert np.asarray(q.tp).shape == (1,)
assert float(q.tp[0]) == 25.0

# a NamedTuple, so it is a JAX pytree and round-trips through to_vector
assert len(jax.tree_util.tree_leaves(ChemParams(1.0, 2.0, 3.0, (4.0,)))) == 4
assert np.allclose(
    np.asarray(ChemParams(1.0, 2.0, 3.0, (4.0,)).to_vector()),
    [1.0, 2.0, 3.0, 4.0])
print("OK")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


def test_ensure_layout_creates_the_trees(tmp_path, monkeypatch):
    """A setup tool must be able to CREATE the layout.

    data_root() is strict on purpose, but a fetch command knows the directories
    should exist.
    """
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    monkeypatch.setattr(paths, "_root_override", None, raising=False)
    root = tmp_path / "made-by-setup"
    monkeypatch.setenv(paths.ENV_ROOT, str(root))

    assert not root.exists()
    assert paths.ensure_layout() == root
    assert (root / "opacity_cache").is_dir()
    assert (root / "exomolop").is_dir()
    # and the strict reader is satisfied afterwards
    assert paths.data_root() == root
    assert paths.opacity_cache_dir() == root / "opacity_cache"

    # it still refuses to guess a location
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    with pytest.raises(RuntimeError, match=paths.ENV_ROOT):
        paths.ensure_layout()


def _import_in_clean_cwd(modules, tmp_path, extra_env=None):
    """Import `modules` in a fresh process whose CWD is an EMPTY directory.

    Returns whatever that directory contains afterwards. Run out-of-process
    because import side effects are one-shot: a module already imported by
    this pytest session would appear clean no matter what it does.
    """
    import subprocess
    work = tmp_path / "cleanroom"
    work.mkdir()
    env = dict(os.environ)
    env.pop("VULCAN_FORWARD_DATA", None)
    env.pop("VULCAN_FORWARD_OPACITY_CACHE", None)
    env.update(extra_env or {})
    code = "import " + ", ".join(modules)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(work),
                       capture_output=True, text=True, env=env, timeout=600)
    return work, r


def test_importing_the_engine_writes_nothing_to_the_callers_cwd(tmp_path):
    """A LIBRARY must not litter the directory its caller happens to be in.

    The light modules are checked first because they are the ones a consumer
    imports merely to read a constant or resolve a path.
    """
    work, r = _import_in_clean_cwd(
        ["vulcan_forward.constants", "vulcan_forward.paths",
         "vulcan_forward.interp_map"], tmp_path)
    assert r.returncode == 0, (
        f"importing the light engine modules failed:\n{r.stderr}")
    leftovers = sorted(p.name for p in work.iterdir())
    assert leftovers == [], (
        f"importing vulcan_forward wrote {leftovers} into the caller's "
        f"working directory. The engine is a library: it must write only "
        f"where the caller points it.")


def test_exomolop_helpers_and_provenance_import_without_jax_or_h5py(tmp_path):
    """jwst_tool.datacheck reads the table paths and the provenance record
    stdlib-side, so neither jax nor h5py may be needed to reach them. Run
    out-of-process with both blocked: this session has already imported jax."""
    import json
    import subprocess
    root = tmp_path / "data"
    (root / "exomolop").mkdir(parents=True)
    (root / "exomolop" / "provenance.json").write_text(json.dumps(
        {"H2O": {"dataset": "POKAZATEL", "iso": "1H2-16O", "file": "f.h5",
                 "natural_abundance": False, "url": "https://example"}}))
    code = (
        "import sys; sys.modules['jax'] = None; sys.modules['h5py'] = None\n"
        "from vulcan_forward import exomolop\n"
        "assert exomolop.table_path('H2O').name == 'H2O.ktable.h5'\n"
        "assert exomolop.available() == []\n"
        "assert exomolop.provenance()['H2O']['dataset'] == 'POKAZATEL'\n"
        "print('ok')")
    env = dict(os.environ, VULCAN_FORWARD_DATA=str(root))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_the_engine_never_resolves_its_data_root_from___file__():
    """Data paths come from the environment / set_data_root, never __file__.

    Resolving the root from this package's own `__file__` would pin the engine
    to one repo's directory nesting. `vulcan_chem` may still read
    `vulcan_jax.__file__` -- that locates the SIBLING package's vendored
    thermo data, which is a different thing and is allowed.
    """
    import ast
    from pathlib import Path as _P
    import vulcan_forward
    pkg = _P(vulcan_forward.__file__).parent
    offenders = []
    for src in sorted(pkg.glob("*.py")):
        tree = ast.parse(src.read_text(), str(src))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and node.id == "__file__"):
                continue
            offenders.append(f"{src.name}:{node.lineno}")
    # vulcan_chem's vulcan_jax.__file__ read is an Attribute, not a bare Name,
    # so it does not appear above; a bare __file__ would.
    assert offenders == [], (
        f"bare __file__ used in {offenders}: the engine must not resolve its "
        f"own location. Use $VULCAN_FORWARD_DATA or paths.set_data_root.")
