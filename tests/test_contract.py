"""The engine's library contract.

The package imports with no data installed, its data-root contract fails loudly instead of guessing, the molecule
table is injectable, and planet geometry is required rather than defaulted to WASP-39 b. The file needs no jax, exojax
or vulcan3 beyond the tests that skip without them.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def test_every_molmass_matches_its_own_formula():
    """Molmass is the only mass the engine uses; it must equal the formula mass.

    It cancels out of the optical depth, so no other test would catch a wrong one.
    """
    import re

    from vulcan_forward import constants

    weights = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}
    for name, entry in constants.MOLECULES.items():
        # the two fields the correlated-k path reads; what each table is lives in the k-table tree's provenance.json
        assert set(entry) == {"vulcan", "molmass"}, name
        want = sum(weights[el] * (int(n) if n else 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", name) if el)
        assert entry["molmass"] == pytest.approx(want, abs=5e-4), name


@pytest.fixture
def paths(monkeypatch):
    """vulcan_forward.paths with no data root, cache override or set_data_root in effect, restored afterwards."""
    from vulcan_forward import paths

    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    monkeypatch.setattr(paths, "_root_override", None, raising=False)
    return paths


def test_paths_module_imports_clean_and_fails_loudly(paths, monkeypatch):
    """Paths must import with nothing configured, then raise with a remedy."""
    with pytest.raises(RuntimeError, match=paths.ENV_ROOT):
        paths.data_root()

    monkeypatch.setenv(paths.ENV_ROOT, "/nonexistent/vulcan-forward-data")
    with pytest.raises(RuntimeError, match="does not exist"):
        paths.data_root()


def test_data_root_and_per_tree_overrides(paths, tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "opacity_cache").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_ROOT, str(root))

    assert paths.opacity_cache_dir() == root / "opacity_cache"
    assert paths.cia_h2he_file().name == "H2-He_2011.cia"
    # no existence check on the k-table tree: a consumer's data check wants the path even when nothing is installed, to
    # report each missing molecule
    assert paths.exomolop_dir() == root / "exomolop"

    # the per-tree override relocates the cache without moving the rest
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(paths.ENV_OPACITY_CACHE, str(other))
    assert paths.opacity_cache_dir() == other
    assert paths.exomolop_dir() == root / "exomolop"


def test_set_data_root_is_honored(paths, tmp_path):
    root = tmp_path / "prog"
    (root / "opacity_cache").mkdir(parents=True)
    paths.set_data_root(root)
    assert paths.data_root() == root


def test_geometry_is_required_not_wasp39b(monkeypatch):
    """The RT builder refuses a profile with no planet geometry, column bottom or reference pressure, before it reads
    a table.

    A defaulted rp_cm would silently model a different planet.
    """
    if importlib.util.find_spec("exojax") is None:
        pytest.skip("exojax not installed (light-CI environment)")
    from vulcan_forward import exojax_rt, exomolop

    def no_table(*_a, **_k):
        raise AssertionError("the geometry refusal comes before any table is read")

    monkeypatch.setattr(exomolop, "load_tables", no_table)
    profile = {"molecules": ["H2O"], "nu_min": 2000.0, "nu_max": 3000.0, "art_nlayer": 20, "gs_cgs": 422.0}
    with pytest.raises(ValueError, match=r"missing \['rp_cm', 'rstar_cm', 'p_ref_bar', 'art_pbtm_bar'\], which have"):
        exojax_rt.build_rt_model(profile)
    # a complete profile passes
    full = {"rp_cm": 9.1e9, "gs_cgs": 422.0, "rstar_cm": 6.5e10, "p_ref_bar": 1.0e-3, "art_pbtm_bar": 10.0}
    exojax_rt._require_positive(full, *exojax_rt._RT_KEYS)


def test_import_order_guard_is_documented_and_live():
    """vulcan_chem must refuse to be imported after exojax.

    Run in a subprocess so the guard sees a clean interpreter: importing exojax
    first must produce the actionable RuntimeError, not a subtly wrong model.
    """
    if importlib.util.find_spec("exojax") is None:
        pytest.skip("exojax not installed (light-CI environment)")
    import subprocess

    code = "import exojax\nimport vulcan_forward.vulcan_chem\n"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode != 0, "importing after exojax must fail"
    assert "must be imported BEFORE exojax" in proc.stderr, proc.stderr


def test_the_network_is_selected_before_vulcan3_is_imported_and_only_once():
    """``select_network`` is how a network reaches vulcan3: importing the engine selects nothing.

    In a fresh interpreter it writes the pair vulcan3 reads at its first import, accepts the same pair again, and
    refuses another network or atom list once vulcan3 is imported. An environment that already names another network
    is refused too: the caller's settings decide, never a stray variable.
    """
    if importlib.util.find_spec("vulcan3") is None:
        pytest.skip("vulcan3 not installed (light-CI environment)")
    import subprocess

    net, atoms = "thermo/NCHO_photo_network.txt", "H,O,C,N"
    code = (
        "import os\n"
        "from vulcan_forward import vulcan_chem\n"
        "assert 'VULCAN3_NETWORK' not in os.environ and 'VULCAN3_ATOM_LIST' not in os.environ\n"
        f"net, atoms = {net!r}, {atoms!r}\n"
        "vulcan_chem.select_network(net, atoms.split(','))\n"
        "assert (os.environ['VULCAN3_NETWORK'], os.environ['VULCAN3_ATOM_LIST']) == (net, atoms)\n"
        "import vulcan3.config\n"
        "assert vulcan3.config.default_config().network == net\n"
        "vulcan_chem.select_network(net, atoms)\n"
        "for other in (('thermo/SNCHO_photo_network_2025.txt', 'H,O,C,N,S'), (net, 'H,O,C')):\n"
        "    try:\n"
        "        vulcan_chem.select_network(*other)\n"
        "    except RuntimeError as exc:\n"
        "        assert 'already imported' in str(exc), exc\n"
        "    else:\n"
        "        raise SystemExit('a second network was accepted')\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("VULCAN3_")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("ok")

    code = f"from vulcan_forward import vulcan_chem\nvulcan_chem.select_network({net!r}, {atoms!r})\n"
    env["VULCAN3_NETWORK"] = "thermo/SNCHO_photo_network_2025.txt"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600, check=False)
    assert r.returncode != 0
    assert "the environment disagrees with the network this call selects" in r.stderr, r.stderr


def test_ensure_layout_creates_the_trees(paths, tmp_path, monkeypatch):
    """A fetch command creates the layout that the strict data_root() refuses to guess."""
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


def _import_in_clean_cwd(modules, tmp_path):
    """Import ``modules`` in a fresh process whose working directory is empty; return it and the CompletedProcess.

    Out of process because import side effects are one-shot: a module this session already imported would look clean.
    """
    import subprocess

    work = tmp_path / "cleanroom"
    work.mkdir()
    env = dict(os.environ)
    env.pop("VULCAN_FORWARD_DATA", None)
    env.pop("VULCAN_FORWARD_OPACITY_CACHE", None)
    code = "import " + ", ".join(modules)
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=str(work), capture_output=True, text=True, env=env, timeout=600, check=False
    )
    return work, r


def test_importing_the_engine_writes_nothing_to_the_callers_cwd(tmp_path):
    """A library must not write into the directory its caller happens to be in.

    These are the light modules a consumer imports merely to read a constant or resolve a path.
    """
    work, r = _import_in_clean_cwd(
        ["vulcan_forward.constants", "vulcan_forward.paths", "vulcan_forward.interp_map"], tmp_path
    )
    assert r.returncode == 0, f"importing the light engine modules failed:\n{r.stderr}"
    leftovers = sorted(p.name for p in work.iterdir())
    assert leftovers == [], (
        f"importing vulcan_forward wrote {leftovers} into the caller's "
        f"working directory. The engine is a library: it must write only "
        f"where the caller points it."
    )


def test_exomolop_helpers_and_provenance_import_without_jax_or_h5py(tmp_path):
    """A consumer's data check reads the table paths and the provenance record with neither jax nor h5py installed.

    Run out of process with both blocked: this session has already imported jax.
    """
    import json
    import subprocess

    root = tmp_path / "data"
    (root / "exomolop").mkdir(parents=True)
    (root / "exomolop" / "provenance.json").write_text(
        json.dumps(
            {
                "H2O": {
                    "dataset": "POKAZATEL",
                    "iso": "1H2-16O",
                    "file": "f.h5",
                    "natural_abundance": False,
                    "url": "https://example",
                }
            }
        )
    )
    code = (
        "import sys; sys.modules['jax'] = None; sys.modules['h5py'] = None\n"
        "from vulcan_forward import exomolop\n"
        "assert exomolop.table_path('H2O').name == 'H2O.ktable.h5'\n"
        "assert exomolop.available() == []\n"
        "assert exomolop.provenance()['H2O']['dataset'] == 'POKAZATEL'\n"
        "print('ok')"
    )
    env = dict(os.environ, VULCAN_FORWARD_DATA=str(root))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_the_data_root_follows_the_environment_from_a_relocated_install(tmp_path):
    """A copy of the package elsewhere, run from an unrelated directory: paths come from the configured root alone.

    With no root set it refuses, rather than finding one relative to its own location. ``paths`` imports without jax.
    """
    import shutil
    import subprocess
    from pathlib import Path

    import vulcan_forward

    site = tmp_path / "site"
    shutil.copytree(Path(vulcan_forward.__file__).parent, site / "vulcan_forward")
    root, cwd = tmp_path / "somewhere" / "data", tmp_path / "unrelated"
    (root / "opacity_cache").mkdir(parents=True)
    cwd.mkdir()
    # a tree beside the package, which a root resolved from the package's own location would pick up
    (site / "vulcan_forward" / "data" / "exomolop").mkdir(parents=True)
    code = (
        "import vulcan_forward\n"
        "from vulcan_forward import exomolop, paths\n"
        "print(vulcan_forward.__file__)\n"
        "print(paths.data_root())\n"
        "print(exomolop.table_path('H2O'))\n"
        "print(paths.cia_h2he_file())\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("VULCAN_FORWARD")}
    env.update(PYTHONPATH=str(site), PYTHONSAFEPATH="1")

    def run(**extra):
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env={**env, **extra}, cwd=cwd, check=False
        )

    r = run(VULCAN_FORWARD_DATA=str(root))
    assert r.returncode == 0, r.stderr
    pkg, data, table, cia = r.stdout.split()
    assert Path(pkg).parent == site / "vulcan_forward"
    assert Path(data) == root
    assert Path(table).is_relative_to(root / "exomolop")
    assert Path(cia).is_relative_to(root / "opacity_cache")
    r = run()
    assert r.returncode != 0
    assert "VULCAN_FORWARD_DATA" in r.stderr


def test_a_required_absorber_is_refused_and_the_others_are_reported():
    """Over 2.9-5.2 um a list without NH3 is refused; the other core absorbers are reported unless the caller names
    them.
    """
    from vulcan_forward import constants

    band = (1923.0, 3450.0)
    with pytest.raises(ValueError, match=r"needs core absorbers.*NH3 \(absorbs at 2.7-3.3, 5.1-15 um\)\. Add"):
        constants.check_core_molecules(["H2O", "CO2", "CO", "CH4"], [], *band)
    five = ["H2O", "CO2", "CO", "CH4", "NH3"]
    rest = ["HCN", "H2S", "SO2", "C2H2", "OCS", "SO"]
    assert constants.check_core_molecules(five, [], *band) == rest
    assert constants.check_core_molecules(five, ["HCN"], *band) == rest[1:]
    assert constants.check_core_molecules([*five, *rest], [], *band) == []
    assert constants.check_core_molecules(["H2O", "CH4", "NH3"], [], 4280.0, 4360.0) == ["H2S"]  # 2.29-2.34 um


def test_the_fixed_values_can_be_stored_in_a_run_record():
    """``constants.fixed_values`` is every fixed number by name, in a form a consumer can write as JSON."""
    import json

    from vulcan_forward import constants

    values = constants.fixed_values()
    assert values["T_OPA_MIN_K"] == constants.T_OPA_MIN_K
    assert values["REQUIRED_MOLECULES"] == sorted(constants.REQUIRED_MOLECULES)
    assert "MOLECULES" not in values
    assert not any(callable(v) for v in values.values())
    assert json.loads(json.dumps(values))["EMISSION_NSTREAM"] == constants.EMISSION_NSTREAM
