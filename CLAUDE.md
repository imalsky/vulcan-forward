# CLAUDE.md — vulcan-forward

The shared VULCAN-JAX -> ExoJAX forward engine. Extracted from vulcan-retrieval
on 2026-07-29 so that vulcan-retrieval and vulcan-jwst-tool are siblings over one
engine instead of the planner depending on another application. The workspace
dependency graph is a DAG: `vulcan-jax <- vulcan-forward <- {vulcan-retrieval,
vulcan-jwst-tool}`. The architecture rationale lives in the ROOT CLAUDE.md.

- **Fail fast and loud (standing rule, all sibling repos):** no behavior-changing
  fallbacks. Missing data raises with the offending path and the remedy; a check
  that cannot run must SAY so; unknown values raise instead of defaulting.
- **This is a LIBRARY.** It must be importable with no data installed, must not
  write to the caller's working directory, and must not resolve paths from its
  own `__file__`. All three were violated before extraction and each has a
  regression test in `tests/test_contract.py`. Keep them.
- **Import order is load-bearing:** `vulcan_chem` sets the VULCAN_JAX_*
  import-frozen env vars and jax x64, so it must precede any exojax import. It
  raises on a late arrival, and also on a prior `vulcan_jax` import whose frozen
  network conflicts. The env vars use `setdefault` — never assignment, or a
  caller driving a custom network gets silently overridden.
- **Data contract:** `$VULCAN_FORWARD_DATA` (or `paths.set_data_root`), with
  `$VULCAN_FORWARD_LINELISTS` / `$VULCAN_FORWARD_OPACITY_CACHE` per-tree
  overrides. Resolved on USE, never at import. Molecule-table `db` values are
  path SUFFIXES — keep them relative so the table stays location-independent.
- **`exojax==2.2.3` is pinned and the pin is load-bearing in four places** (see
  README "The exojax version"). exojax caps `numpy<2`; containing that here is a main
  reason this package exists, so do not relax the pin without re-verifying the
  gravity-profile workaround, the `dit_grid_resolution` routing, the two
  compat-shim imports, and the `mdb.gamma_air` monkey-patch.
- **Physics changes need the bit-identical gate.** The extraction was validated by
  forcing a fresh solve of vulcan-jwst-tool's default WASP-39 b case and diffing
  against a pre-split cached spectrum (`wl_um`, `depth`, `depth_wo`, `T`,
  `p_bar`, all max|diff| = 0). Any refactor claiming to be behavior-neutral runs
  that same check. This repo's own suite covers packaging contracts only —
  physics validation lives in the consumers.
- **Named parameters (0.2.0):** `ChemParams(lnZ, c_o, lnKzz, tp)` is the
  primitive; `params_from_vector` / `ChemParams.to_vector` convert to and from
  the positional `[lnZ, c_o, lnKzz, *tp]` layout, which every entry point still
  accepts. KEEP the vector path: the retrieval's sampler and the planner's
  `jax.jvp` differentiate with respect to that array, and a tangent has to match
  its shape — that is an adapter, not debt. `_as_params` is the single
  normalization point; `_prep` reads fields by name. When `tp_eval` is None the
  tail is `theta[3:4]` (a uniform T offset), not `theta[3:3]`.
- Suite: `python -m pytest tests -q` (dependency-light; the geometry and
  import-order tests skip without exojax). Install editable:
  `pip install --no-deps -e .`.
