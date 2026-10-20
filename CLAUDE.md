# CLAUDE.md — vulcan-forward (vulcan-forward / vulcan_forward)

The shared forward engine: it solves vulcan3 chemistry for a parameter vector and turns
the converged column into a transmission or emission spectrum with ExoJAX correlated-k
radiative transfer, differentiably. vulcan-retrieval and jwst-transit-authority both
build on it, in the chain `vulcan3 <- vulcan-forward <- {vulcan-retrieval,
jwst-transit-authority}`. A forward-model change usually belongs here, not in a consumer
(notes "Library and data contract").

## Docs

- `README.md` is hand-written by the maintainer: never edit it unless asked for that edit.
- `notes.md`: what is true and why, by section; the evidence behind each rule below.
- `todo.md`: open, owed and undecided work. Read it first; open or close items in the
  same pass.
- `CLAUDE.md`: directions only, no results.
- The repo is public: write `notes.md`, `todo.md` and `CLAUDE.md` for public reading. No
  personal names except cited authors, no emails, IDs, accounts or keys, no chat quotes. No other `.md` file. Cite a notes
  section by its heading name, never a number. Put every rejected approach in notes
  "Do not retry" in the same pass.
- Format and where each fact goes: the `doc-upkeep` skill. Policy (no bloat, the todo
  bar) is the project-root `CLAUDE.md`.

## Setup and commands

- Install editable: as a root uv workspace member (`uv sync` in `VULCAN_Project/`) or
  standalone with `pip install --no-deps -e .`.
- Data: set `$VULCAN_FORWARD_DATA` (or call `paths.set_data_root`). Fetch k-tables once,
  offline: `python -m vulcan_forward.fetch_exomolop --molecules H2O,CO2,...`.
- Use: pass one profile dict to `build_chem_model`, `build_rt_model` and
  `build_emis_model` (import order under Rules).
- Normal tier (everyday checks and CI; keep it under 10 minutes on daw), inside the CPU
  cap:
  `JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 taskset -c <CPUs> uv run python -m pytest tests -q -n 6`
- Full tier: add `VULCAN_FORWARD_RUN_FULL=1` and `--dist worksteal`. Run it before a push
  that touches the solve or its batched routes. It is a long, memory-heavy run: ask
  first (cost in notes "Test tiers and CI").
- Both tiers need single-threaded XLA as above: multi-threaded Eigen breaks the bitwise
  lane checks. Run them on the CPU (`tests/conftest.py` defaults `JAX_PLATFORMS=cpu`):
  on a GPU in use the suite runs out of memory.
- Lint: `uv run ruff check src tests`.
- CI (`.github/workflows/tests.yml`) runs the normal tier serially
  (`python -m pytest tests -q -ra`), without exojax (the RT tests skip), with vulcan3
  from `imalsky/VULCAN3@main`: push VULCAN3 `main` first when a change here needs a
  newer vulcan3. After a push, check
  `gh run list --repo imalsky/vulcan-forward` before reporting done.
- Releases: the `vulcan-release` skill; the version is in `src/vulcan_forward/_version.py`.

## Rules

### Library and data contract

- Fail fast and loud: no behavior-changing fallbacks. Missing data raises with the
  offending path and the remedy; a check that cannot run says so; an unknown value
  raises instead of defaulting.
- Stay a library: importable with no data installed, never writing to the caller's cwd,
  never resolving a path from its own `__file__`. `exomolop`'s path helpers and
  `provenance()` import without jax or h5py. Each property has a test in
  `tests/test_contract.py`; keep them.
- Data root: `$VULCAN_FORWARD_DATA` or `paths.set_data_root`, resolved on use, never at
  import. `<root>/exomolop/` holds the k-tables and `provenance.json`;
  `<root>/opacity_cache` holds the CIA tables (`$VULCAN_FORWARD_OPACITY_CACHE` overrides
  it). Never fetch tables at run time.
- `constants.MOLECULES` carries only `vulcan` and `molmass`. What a table is comes from
  `exomolop.provenance()` (written by the fetcher), its header from
  `exomolop.table_info()`: provenance is recorded, never asked of the caller.
- The molecule table is injectable (`profile['molecule_table']`): never tell a consumer
  to edit a constant inside the engine.
- Planet geometry is required: `build_rt_model` needs `rp_cm`, `gs_cgs`, `rstar_cm`.
  Never add a default planet.
- Every builder calls `constants.check_profile_keys`, which refuses any key outside
  `constants.PROFILE_KEYS`. A new profile reader adds its key there; deleting a reader
  removes it. `cfg_overrides` keys pass `vulcan3.validate_overrides` before they apply
  (notes "Import order, profile keys and refusals").
- Import order is load-bearing: `vulcan_chem` sets the import-frozen `VULCAN3_*` env vars
  (with `setdefault`, never assignment) and jax x64, so it precedes any exojax import.
  Keep its raises on a late arrival and on a prior `vulcan3` import whose frozen
  network conflicts.
- `exojax` is pinned exactly in `pyproject.toml`. Moving it needs the RT and derivative
  tests plus a spectrum diff against the old version on fixed inputs
  (notes "RT validation and the exojax pin").

### Chemistry solve and derivatives

- `ChemParams(lnZ, c_o, lnKzz, tp)` is the primitive; `params_from_vector` /
  `ChemParams.to_vector` convert to and from the positional `[lnZ, c_o, lnKzz, *tp]`
  layout every entry point accepts. Keep the vector path: consumers pass and
  differentiate that array. `_as_params` is the single normalization point; `_prep`
  reads fields by name; without a `tp_eval` hook the tail is `theta[3:4]` (a uniform T
  offset).
- Every cold solve starts from vulcan3's in-graph equilibrium seed at the proposal's own
  T-P and column ratios. To start from the build's column, pass it as a continuation:
  `warm_y=chem.y0, lnZ_ref=0, c_o_ref=0` (`run_diag` takes the same arguments) (notes
  "Cold start, abundance map, carbon repair").
- `ConvDiag.conv_normal` is `vulcan3.conv_normal(final, cfg)`, ANDed with the tangent
  certificate on the jvp path. Never re-implement the runner's certificate terms here
  (notes "Convergence certificate").
- A forward-mode derivative of the solve goes through `converged_y_jvp`, never
  `jax.jvp(converged_y)` (it stops when the column certifies and leaves a warm start's
  tangent unrelaxed) and never a step floor. Pass the direction in finite-difference-step
  units and divide `dy` by the step. Several directions go in as one stacked
  `(D, n_theta)` tangent (`dy` comes back `(D, nz, ni)`), never a `jax.vmap` of the call
  (notes "Convergence certificate").
- The batched primal is `converged_y_batch` or `converged_y_queue` (lanes refilled from a
  theta queue; `active=` masks padding jobs). Both agree with the solo solve at the
  convergence scale only, and the queue is not bitwise the batch: compare certificates
  (`conv_normal`), never bytes, and compare accept counts exactly only when both routes
  start from one column. A plain `jax.jvp` through either is supported; a
  tangent-certified derivative stays on `converged_y_jvp` (notes "Batched routes").
- Set no photolysis refresh cadence in the engine: the config YAML decides, and a
  consumer overrides per column through `cfg_overrides`
  (notes "Photolysis refresh cadence").

### Geometry and anchors

- `rp_cm`/`gs_cgs` apply at `p_ref_bar` (default `constants.P_REF_BAR`), not at the RT
  grid bottom. exojax's `radius_btm` is the bottom layer's lower boundary, so
  `_radius_at` carries an extra half-layer node: keep it, or the depth depends on
  `art_nlayer` (notes "Geometry and anchors").
- Emission anchors radius and gravity at the constant `constants.P_REF_EMISSION_BAR` (not
  a profile key); `emission_radius()` gives the consumer's prefactor radius. The eclipse
  prefactor is the tau = 2/3 radius per k-ordinate (`eclipse_flux_tau`).
- lnR0 is a radius change at fixed GM in both observables: `transmission_depth_r` and
  every emission function take `lnR0` and re-anchor. A consumer passes it through and
  never multiplies a depth by e^(2 lnR0).
- Validate every anchor pressure against the grid span (`jnp.interp` clamps silently). A
  clamped chemistry top or bottom raises. The chemistry `P_t` follows `art_ptop_bar`
  (default `constants.ART_PTOP_BAR`) unless `cfg_overrides` sets `P_t`.
  `_check_cia_span` validates the CIA tables.

### Opacity and radiative transfer

- Opacity is correlated-k over the published ExoMolOP tables, the only path.
  `build_rt_model` refuses `opacity_mode != "exomolop"`, and the removed line-by-line and
  Mie keys raise (notes "Opacity: correlated-k on ExoMolOP").
- exojax and exo_k are oracles only (`tests/test_ckd.py`, `tests/test_exomolop.py`); run
  exo_k in a throwaway venv, never the jax env.
- The fetcher takes the principal isotopologue, its natural-abundance file where one
  exists, on the engine's band grid only, and checks the `.part` grid before it replaces
  a table. Curated DOIs live in the fetcher record (`fetch_exomolop._DATASET_DOI`), never
  in header echoes.
- Quadrature nodes and weights come from the file. Keep the refusals in `load_tables`,
  `build_rt_model` (`art_pbtm_bar` deeper than the table's pressure ceiling) and
  `_validated_layout`.
- Mixtures use random-overlap resort-rebin in `ckd.overlap`: keep it despite its jumps
  where pairwise sums swap order (notes "Do not retry"). It is non-associative, so
  `_fold_wo` keeps the zero folded in the exact naive order.
- Emission runs `_run_emis_ckd_linsap`, never `ArtEmisPure.run_ckd`, and `ibased_linsap`,
  never `ibased`. The g-major/band-minor flatten and flux-side g-weighting are pinned
  (`tests/test_emission_solver.py`); the linsap source lives at the layer boundaries via
  `_boundary_temperature` (notes "Emission solver and gravity").
- Consumers check `tau_bottom` and refuse a see-through column; the fix is a deeper
  `art_pbtm_bar`, never a wider tolerance. `emission_flux_tau` builds dtau once for the
  flux and `tau_bottom`.
- Gravity is the inverse-square profile at layer midpoints in transmission and emission
  (`_gravity_profile_invsq`, the engine's own copy; `tests/test_gravity_profile.py`).
- The RT is verified against petitRADTRANS, transmission and emission, both codes reading
  the same k-table file (the pRT version is pinned in `tests/test_e2e_fixtures.py`). Do
  not re-open "is our RT wrong" without new evidence. `tests/data/prt_ref_*.npz` and
  `tests/test_e2e_rt_reference.py` gate the RT physics; pRT is never a test dependency.
  Before comparing with external spectra, read the pRT flux-unit and gCMCRT-decode
  entries in notes "RT validation and the exojax pin".

### Changes and tests

- Bit-identical gate: a refactor claiming to be behavior-neutral must reproduce
  jwst-transit-authority's default WASP-39 b case (forced fresh solve) as solved on the
  code before the change: `wl_um`, `depth`, `depth_wo`, `T`, `p_bar`, max|diff| = 0.
  jwst's e2e reference test is a tolerance band, not this gate. Deliberate physics
  changes carry new fixtures (notes "Library and data contract").
- A test that compiles a tangent runner or takes minutes goes in the full tier (`skipif`
  on `VULCAN_FORWARD_RUN_FULL`).
- Never `pytest.importorskip("exojax")` in a test module (it breaks the import-order
  contract for every later module): use `importlib.util.find_spec` and import
  `vulcan_chem` first. The repeated test-module preamble is deliberate: never move the
  `vulcan_chem` import into a conftest (it would reorder the contract test).
- Keep ruff's pinned rule set (`pyproject.toml` `[tool.ruff.lint]`): import sorting would
  reorder those preambles.

## Where things live

- `vulcan_chem.py`: `build_chem_model`, `ChemParams`, `ConvDiag`, every solve entry point.
- `exojax_rt.py`: `build_rt_model`, `build_emis_model`, geometry anchors, emission solver.
- `ckd.py`: correlated-k interpolation and the overlap fold. `exomolop.py`: table loading
  and `provenance()`. `fetch_exomolop.py`: the offline fetcher.
- `interp_map.py`: chemistry grid to RT grid. `paths.py`: the data root. `constants.py`:
  `PROFILE_KEYS`, anchor and grid constants, `MOLECULES`.
- `tests/test_contract.py`: the library contract. `tests/test_e2e_rt_reference.py`: the
  pRT gate.
