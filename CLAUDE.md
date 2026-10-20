# CLAUDE.md — vulcan-forward

## Docs

Four root docs: `README.md` (hand-written by the maintainer; never edit it unless asked
for that edit), `notes.md` (what is true and why), `todo.md` (open, owed and undecided
work) and this file. `notes.md`, `todo.md` and `CLAUDE.md` are tracked and public: write
them for public reading. No other `.md` file in this tree. Policy (no bloat, doc upkeep)
follows the project-root `CLAUDE.md`.

Read `todo.md` first; open or close items in the same pass. Add every rejected approach
to notes.md "Do not retry" in the same pass. This file is the contract; notes.md holds
the evidence, cited below by section name.

## Rules

The shared vulcan3 -> ExoJAX forward engine. DAG: `vulcan3 <- vulcan-forward <-
{vulcan-retrieval, jwst-transit-authority}` (notes "Library and data contract").

- **Fail fast and loud:** no behavior-changing fallbacks. Missing data raises with the
  offending path and the remedy; a check that cannot run says so; unknown values raise
  instead of defaulting.
- **Profile keys are checked:** every builder calls `constants.check_profile_keys`,
  which refuses any key outside `constants.PROFILE_KEYS`. A new profile reader adds its
  key there; deleting a reader removes it. `cfg_overrides` keys pass
  `vulcan3.validate_overrides` before they apply (notes "Import order, profile keys and
  refusals").
- **Every cold solve starts from the equilibrium seed** (vulcan3's in-graph Gibbs column
  at the proposal's own T-P and column ratios). To start from the build's column, pass it
  as a continuation: `warm_y=chem.y0, lnZ_ref=0, c_o_ref=0` (`run_diag` takes the same
  arguments) (notes "Cold start, abundance map, carbon repair").
- **`ConvDiag.conv_normal` is `vulcan3.conv_normal(final, cfg)`**, ANDed with the
  tangent certificate on the jvp path. Never re-implement the runner's certificate terms
  here (notes "Convergence certificate").
- **This is a library:** importable with no data installed, never writes to the
  caller's cwd, never resolves paths from its own `__file__`; each has a regression test
  in `tests/test_contract.py`. `exomolop`'s path helpers and `provenance()` stay
  importable without jax or h5py (notes "Library and data contract").
- **Import order is load-bearing:** `vulcan_chem` sets the import-frozen `VULCAN3_*` env
  vars and jax x64, so it precedes any exojax import. It raises on a late arrival and on
  a prior `vulcan3` import whose frozen network conflicts. The env vars use
  `setdefault`, never assignment.
- **Data contract:** `$VULCAN_FORWARD_DATA` (or `paths.set_data_root`), with
  `$VULCAN_FORWARD_OPACITY_CACHE` overriding `<root>/opacity_cache` (CIA tables), plus
  `<root>/exomolop/` (k-tables + `provenance.json`); resolved on use, never at import.
  `constants.MOLECULES` carries only `vulcan` + `molmass`; what each table is lives in
  `exomolop.provenance()`, the header in `exomolop.table_info()`. Provenance is
  recorded, not requested.
- **Planet geometry is required:** `build_rt_model` needs `rp_cm`, `gs_cgs`, `rstar_cm`;
  there is no default planet.
- **The molecule table is injectable** (`profile['molecule_table']`); never tell a
  consumer to edit a constant inside the engine.
- **`exojax==2.6.0` is pinned exactly.** Moving it needs the RT and derivative tests plus
  a spectrum diff against the old version on fixed inputs (notes "RT validation and the
  exojax pin").
- **Bit-identical gate:** a refactor claiming to be behavior-neutral must reproduce
  jwst-transit-authority's default WASP-39 b case (forced fresh solve) as solved on the
  code before the change: `wl_um`, `depth`, `depth_wo`, `T`, `p_bar`, max|diff| = 0.
  jwst's e2e reference test is a tolerance band, not this gate. Deliberate physics
  changes carry new fixtures instead.
- **Named parameters:** `ChemParams(lnZ, c_o, lnKzz, tp)` is the primitive;
  `params_from_vector` / `ChemParams.to_vector` convert to and from the positional
  `[lnZ, c_o, lnKzz, *tp]` layout every entry point accepts. Keep the vector path:
  consumers pass and differentiate that array. `_as_params` is the single normalization
  point; `_prep` reads fields by name; without a `tp_eval` hook the tail is `theta[3:4]`
  (a uniform T offset).
- **A forward-mode derivative of the solve goes through `converged_y_jvp`**, not
  `jax.jvp(converged_y)`, which stops when the column certifies and leaves a warm
  start's tangent unrelaxed. Pass the direction in finite-difference-step units and
  divide `dy` by the step. Several directions go in as one stacked `(D, n_theta)`
  tangent (`dy` comes back `(D, nz, ni)`), never a `jax.vmap` of the call. Never a step
  floor instead (notes "Convergence certificate").
- **The batched primal is `converged_y_batch` or `converged_y_queue`** (lanes refilled
  from a theta queue; `active=` masks padding jobs). Both agree with the solo solve at
  the convergence scale only, and the queue is not bitwise the batch: compare
  certificates, never bytes, and compare accept counts exactly only when both routes
  start from one column. A plain `jax.jvp` through either is supported; a
  tangent-certified derivative stays on `converged_y_jvp`. The warm cap rides the carry
  (`count_max_dyn`), solo path included: no second runner (notes "Batched routes").
- **Geometry** (notes "Geometry and anchors"):
  - `rp_cm`/`gs_cgs` apply at `p_ref_bar` (default 1e-3 bar), not at the RT grid
    bottom. exojax's `radius_btm` is the bottom layer's lower boundary, so `_radius_at`
    carries an extra half-layer node; dropping it makes the depth depend on
    `art_nlayer`.
  - Emission anchors radius and gravity at `p_ref_emission_bar` (0.1 bar);
    `emission_radius()` gives the consumer's prefactor radius. lnR0 is a radius change
    at fixed GM in both observables: `transmission_depth_r` and every emission function
    take `lnR0` and re-anchor, so a consumer passes it through and never multiplies a
    depth by e^(2 lnR0). The eclipse prefactor uses the tau = 2/3 radius per k-ordinate
    (`eclipse_flux_tau`).
  - Every anchor pressure is validated against the grid span (`jnp.interp` clamps
    silently). A clamped chemistry top or bottom raises. `ART_PTOP_BAR = 1e-9`, and the
    chemistry `P_t` follows `art_ptop_bar` unless `cfg_overrides` sets it.
    `_check_cia_span` validates the CIA tables.
- **Opacity is correlated-k over the published ExoMolOP tables, the only path.**
  `build_rt_model` refuses `opacity_mode != "exomolop"`; the removed line-by-line and Mie
  keys raise (notes "Opacity: correlated-k on ExoMolOP"). exojax and exo_k are oracles
  only (`tests/test_ckd.py`, `tests/test_exomolop.py`; exo_k in a throwaway venv, never
  the jax env).
  - Tables are fetched once, offline (`python -m vulcan_forward.fetch_exomolop`), never
    at run time. The fetcher takes the natural-abundance file and the principal
    isotopologue and checks the `.part` grid before it replaces a table; curated DOIs
    live in the fetcher record, never in header echoes.
  - The quadrature is pRT's split 8+8, nodes and weights from the file. Keep the
    refusals in `load_tables`, `build_rt_model` (`art_pbtm_bar` deeper than the 100 bar
    table ceiling) and `_validated_layout`.
  - Mixtures use random-overlap resort-rebin in `ckd.overlap`: keep it despite its jumps
    where pairwise sums swap order. It is non-associative, so `_fold_wo` keeps the zero
    folded in the exact naive order.
- **Emission runs `_run_emis_ckd_linsap`, never `ArtEmisPure.run_ckd`, and
  `ibased_linsap`, never `ibased`.** g-major/band-minor flatten and flux-side
  g-weighting are pinned (`tests/test_emission_solver.py`); the linsap source lives at
  the layer boundaries via `_boundary_temperature`. Consumers check `tau_bottom` and
  refuse a see-through column (fix: a deeper `art_pbtm_bar`, not a wider tolerance).
  `emission_flux_tau` builds dtau once for flux and `tau_bottom`.
- **Gravity is the inverse-square profile at layer midpoints** in transmission and
  emission (`_gravity_profile_invsq`, the engine's own copy;
  `tests/test_gravity_profile.py`).
- **The RT is verified against petitRADTRANS 3.4.0**, transmission and emission, both
  codes reading the same k-table file. Do not re-open "is our RT wrong" without new
  evidence. `tests/data/prt_ref_*.npz` + `tests/test_e2e_rt_reference.py` gate the RT
  physics; pRT is never a test dependency. Before comparing with external spectra, read
  the pRT flux-unit and gCMCRT-decode entries in notes "RT validation and the exojax
  pin".

## Tests

- **Normal tier** is what CI runs: `python -m pytest tests -q -ra`, serial, Python
  3.10-3.12, no exojax (the RT tests skip). It stays under 10 minutes on daw. Locally,
  inside the CPU cap:
  `JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 taskset -c <CPUs> uv run python -m pytest tests -q -n 6`
- Both tiers need single-threaded XLA, as CI sets it: multi-threaded Eigen breaks the
  bitwise lane checks. On daw also set `JAX_PLATFORMS=cpu`: the workspace env carries
  CUDA jax, and on a GPU in use the suite runs out of memory (notes "Test tiers and CI").
- **Full tier:** add `VULCAN_FORWARD_RUN_FULL=1` and `--dist worksteal` (about 20 min,
  about 24 GB peak). It adds the gradient tests, the `converged_y_jvp` tests, the
  8-species pRT case and the route comparisons whose bars hold on daw's CPU but not the
  GitHub runner's. Run it before a push that touches the solve or its batched routes. A
  test that compiles a tangent runner or takes minutes goes in the full tier (`skipif`
  on `VULCAN_FORWARD_RUN_FULL`).
- Never `pytest.importorskip("exojax")` in a test module (it breaks the import-order
  contract for every later module): use `importlib.util.find_spec` and import
  `vulcan_chem` first. The repeated test-module preamble is deliberate: never move the
  `vulcan_chem` import into a conftest (it would reorder the contract test). ruff stays pinned to `E4`/`E7`/`E9`/`F` (import sorting would reorder
  those preambles).
- Install editable: root uv workspace member (`uv sync` in `VULCAN_Project/`) or
  standalone `pip install --no-deps -e .`.
- CI installs vulcan3 from `imalsky/VULCAN3@main` (no exojax): push VULCAN3 `main`
  before a push here that needs a newer vulcan3. After a push, check
  `gh run list --repo imalsky/vulcan-forward` before reporting done.
