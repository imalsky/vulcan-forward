# vulcan-forward notes

What is true about the forward engine and why: the evidence behind each rule in
`CLAUDE.md`, the design records that still govern the code, retired designs and the
do-not-retry list. Open, owed and undecided work is in `todo.md`. Retrieval-side studies
are in vulcan-retrieval's notes; solver findings in VULCAN3's notes.
Tags: [M] measured, [E] estimated, [I] inferred.

## Current state

- Version 0.29.0 (`801f7e7`, tag `v0.29.0`); last code commit `39c2c45`. The
  2026-10-20 cleanup on `main`: no build-time warm-up solve, `co_mode` default `fixed_O`,
  the emission anchor a constant, merged docs. One branch, `main`.
- Floors (`pyproject.toml`): `exojax==2.6.0`, `radis<0.17`, `vulcan3>=0.18.0` (first
  release under that name; carries `conv_normal` and `validate_overrides`),
  `jax>=0.6.2`, `numpy>=1.26`, `h5py>=3.9`.
- Consumers: vulcan-retrieval requires `>=0.28.0`; jwst-transit-authority (the planner)
  requires `>=0.29.0` and pins `801f7e7` in `deploy/pins.env`.
- Production: the WASP-39b baseline v2 nautilus run used `d26e9d3` (0.28.0, tag
  `w39b-baseline-v2`). All three W39b production runs carry the 0.27.2 carbon repair.
- Normal tier: plain `pytest tests`. CI runs it serially on Python 3.10, 3.11 and 3.12,
  single-threaded XLA, numpy<2, vulcan3 from `imalsky/VULCAN3@main`, no exojax (the RT
  tests skip). Last reading on file: 95 passed / 8 skipped in 189 s at `-n 6` on daw, code
  of `d26e9d3` [M]. `39c2c45` added one normal-tier test (`test_masked_jobs_take_no_step`);
  not re-run at HEAD.
- Full tier (`VULCAN_FORWARD_RUN_FULL=1`): the `converged_y_jvp` tests, the five `@FULL`
  queue/gradient tests and the 8-species pRT case. At `a6a1094`: 100 passed / 3 failed
  in 20.0 min at `-n 6 --dist worksteal` (the three knife-edge tests in `todo.md`); at
  `4ca5fa6`: 103 passed in 20.7 min, 23.9 GB peak [M]. After the 2026-10-20 cleanup (`co_mode`
  default `fixed_O`, no warm-up solve): 95 passed / 2 failed in 31.4 min at `-n 6 --dist
  worksteal` (after the further cuts; was 102 / 2) [M]; the two failures are the queue-vs-batch knife edge at exactly
  the recorded 6.874e-2, and the warm jvp case now certifies.
- Known defect, no shipped number moves: the `ckd.overlap` sort-crossing jumps (see
  "Opacity").

## RT speed (speed2_rt, 2026-10-17)

- XLA's argsort along the middle axis of `ckd.overlap` was 82% of the RT stage on a GH200,
  107.4 ms per spectrum [M].
- The "kv" rewrite (sums on the last axis, one stable two-operand `lax.sort` with the
  weights as payload) runs at 25.3 ms per spectrum (4.25x); a 2376-row RT pass 256.0 ->
  60.2 s. Depths bitwise on all 3,920,400 values, log L bitwise on 2376/2376, jvp
  bitwise, vjp within 1.6e-14 relative [M]. Adopted: `5e8d43a` (`ckd.overlap`).
- After it the sort is still 51% of the stage (11.7 ms per spectrum). Not adopted: an
  unrolled searchsorted (24.4 ms, bitwise), index gather (26.9-36.8 ms), compare_all
  (not bitwise), rt_chunk 8-64 (no gain), a CUDA fold prototype (8.2 ms for the fold) [M].
- `39c2c45`: `converged_y_queue(active=)` lets the retrieval mask padding rows. A masked
  job's lane starts done and takes no step; the active jobs keep the unmasked schedule
  (bitwise on the GPU per the retrieval's check; the CPU test bar is 1e-9).

## Warm starts

- Every warm start begins from the build's pre-loop photo carry, not the warm column's
  own field: `state0` is packed from the build's RunState and `_prep`'s
  `state0._replace` does not touch the photo flux fields [I: code]. Whether that
  pre-loop `dflux_u` is zero was not verified [I]. ad_advice sample: for 27 of 98
  certified lanes the photo carry was the last gate to clear [M].
- Weighted vs unweighted inventory: the projection fixes each layer's total and the
  unweighted (layer-summed) column ratios to H; the element budget tracks the
  dz-weighted column (`budget_ref`, "Convergence certificate"). So two starts at one
  theta can carry different weighted inventories: algebraically real. On the shipped
  HD209 column the weighted and unweighted column ratios agree to 1e-11..2e-10,
  because the column's atoms sit in the bottom scale heights [M, one column, second
  referee]; W39b is unchecked. vulcan-retrieval's 97 of 1320 warm-certify, cold-fail
  rows (baseline_v2) are fixed-point path dependence, a different effect.

## Provenance audit

151 commits are dated before 2026-10-13 [M: `git log`]. They were made on other machines
and their provenance is not recorded on NAS. The items below are live at HEAD and on by
default unless noted. The maintainer confirmed the first (2026-10-20: the terms and the
per-theta re-anchoring stay); the rest stay live.

- `b8cae5c`, `b72b0d2`, `3bb7cab`, `aa79f25`, `dfa2a0f`: `conv_normal` adds geometry and
  element-budget certificate terms and per-theta re-anchoring. This decides which draws
  are rejected and goes beyond VULCAN's original criteria. The terms live in vulcan3
  (VULCAN3 notes "Decided by the maintainer, not to be re-opened").
- `0ee65a4` + `fb244c7`: chemistry top `P_t` forced to the RT top, now 1e-9 bar
  (`build_chem_model` sets `cfg.P_t` from `art_ptop_bar`; `constants.ART_PTOP_BAR`). A
  model choice, not a parity item (evidence: "Geometry and anchors").
- `3b7654b`, `2838121`, `5018e4d`, `d40211d`: cold start is in-graph equilibrium at the
  proposal's T-P; the 'baseline' start was deleted.
- `a6a1094`: carbon repaired through CO+CH4 (used in all three W39b production runs).
- `2807b9a`, `0250ce0`: elemental abundance map only.
- `4446e6e`, `e2aacba`: transit radius anchored at 1 mbar (`constants.P_REF_BAR`, a model
  choice: "Geometry and anchors"); bottom boundary; emission solver linsap.
- `d26e9d3`, `801f7e7`: fixed planet mass under lnR0. The daw record calls the emission
  half (0.29.0) the maintainer's choice.
- `773837e`: hand-written correlated-k random overlap, T/P clamps, `K_FLOOR` 1e-60.
- `db94d6e`: line-by-line and Mie paths removed (also removed a runtime overwrite of
  MdbHitran gamma_air/n_air; ExoJAX source itself was never modified).
- `aedbcbc`: 12 of 27 molecular masses changed. The mass cancels in the line-opacity
  conversion ("Library and data contract"), so the edit moves no spectrum beyond
  rounding.
- `6aa0cb8`: `warm_count_max` defaults to `count_max`. `c63815c`: emission uses g(r).
  `ccdf92f`: exojax 2.6.0. `879a99c`: the jvp path also needs a tangent certificate.
  `55613bc`: `eclipse_flux_tau` (opt-in).
- `5e8d43a` and `39c2c45` were made on NAS on the maintainer's 2026-10-17 instructions;
  both are bitwise.

## Library and data contract

- No data at import, no writes to the caller's cwd, no paths from `__file__`: each has a
  test in `tests/test_contract.py`. The old `__file__`-relative path (`parents[3]` plus a
  `data/cm24_wasp39b` marker) made the package un-importable elsewhere. A build is
  cwd-clean because vulcan3 dropped the pre-loop `Output()` (VULCAN-JAX `d8856c6`, in the
  0.18.0 floor): a build plus one solve in an empty directory left it empty, so the
  `output_dir` redirect was removed [M]. The test checks imports only.
- `exomolop`'s path helpers and `provenance()` import without jax or h5py (a subprocess
  test). The planner's `datacheck.exomolop_table_path` re-derives the path; a data-gated
  planner test pins that the two agree.
- `$VULCAN_FORWARD_DATA` / `paths.set_data_root`, resolved on use: no measurement. A
  library cannot know a data path, and a guessed one fails late or reads other data [I].
- Planet geometry is required: it used to fall back to WASP-39 b, so a forgotten key
  silently modeled another planet. `build_rt_model` requires `rp_cm`, `gs_cgs`,
  `rstar_cm`; `build_emis_model` requires `rp_cm`, `gs_cgs` (`_require_geometry`).
- Injectable molecule table (`profile['molecule_table']`, passed to
  `exomolop.load_tables`): no measurement; consumers choose their own absorber menus
  (the planner's `EXTRA_MOLECULES_DEFAULT`). `constants.MOLECULES` carries only
  `vulcan` and `molmass`; the mass cancels out of dtau (a 10x-wrong H2S mass left the pRT
  fixtures bit-unchanged [M]).
- Bit-identical gate: the extraction of `vulcan_chem`, `exojax_rt` and `interp_map` from
  vulcan-retrieval reproduced the planner's default WASP-39 b case at max|diff| = 0 [M].
  Deliberate physics changes since (model top, eclipse radius, emission g(r), lnR0)
  carried new fixtures.
- Why a separate distribution: the exact exojax pin is load-bearing, and VULCAN3 is
  standalone with vendored data; an RT stack and an external data path there would widen
  its blast radius. DAG: `vulcan3 <- vulcan-forward <- {vulcan-retrieval,
  jwst-transit-authority}`.

## Import order, profile keys and refusals

- `vulcan_chem` sets `VULCAN3_NETWORK` / `VULCAN3_ATOM_LIST` (vulcan3 freezes them at
  import) with `setdefault`, and jax x64, before any exojax import. A late arrival raises;
  so does a prior `vulcan3` import whose frozen network (`chem_funs._NETWORK.network_path`,
  not an env var) conflicts. No measurement: the values freeze at import, so a wrong
  order fails much later [I]. `test_import_order_guard_is_documented_and_live` pins it.
- `pytest.importorskip("exojax")` imports exojax ahead of `vulcan_chem` for every later
  test module, so test modules use `importlib.util.find_spec`. A `conftest.py` moved
  `vulcan_chem` ahead of the jax-free contract test, so the preamble repeats per module.
- ruff stays on `E4`/`E7`/`E9`/`F`: ruff 0.16's defaults found 96 items, including import
  sorting that would reorder those preambles, and dropped `E402`, which the `noqa`s rely
  on [M].
- `PROFILE_KEYS` is the union of the keys the three builders read (consumers pass one
  dict to all three); `check_profile_keys` refuses anything else (`affaecf`, 0.26.0).
  Before it, deleting a reader turned a set key into a silent no-op. `cfg_overrides` pass
  `vulcan3.validate_overrides` first (0.27.0).
- Every build-time refusal names the remedy: missing table (fetch command), missing
  geometry, out-of-grid anchor, clamped chemistry top or bottom, CIA span, `art_pbtm_bar`
  deeper than the table's 100 bar ceiling, unknown profile key, unknown `co_mode`,
  repeated molecule, a non-empty `T_cross_sp`. The last because `_prep` leaves the
  T-dependent cross sections at the baseline; no shipped config sets it.
  (`use_fix_all_bot`, dead in upstream master, is no longer a vulcan3 key.)

## Opacity: correlated-k on ExoMolOP

- Sampled line-by-line is wrong below R = 700,000 (exojax evaluates cross sections on the
  grid it is given; `warn_resolution` critical R 700,000). W39b vs PRISM: an R = 1477
  sample gave 2.80x the data's 448 ppm amplitude; R = 700,000 is 5.1 M points over 1-15 um,
  about 20 min per spectrum. Emission is hit harder: at R = 1477, 45% of the 3-5 um band
  flux and 58.7% rms per bin; correlated-k gives 100.5% and 1.9% [M].
- HITRAN lowers window opacity in a hot hydrogen atmosphere three ways: 296 K tabulation
  (H2O band mean 0.01-0.85 dex low vs POKAZATEL at 1200 K), air broadening (H2O has 0%
  H2/He coverage), 5 species. W39b amplitude vs the data's 448 ppm: R = 700,000 HITRAN 870
  (1.94x), ExoMolOP 5 species 695 (1.55x), + H2S 663 (1.48x), published gCMCRT 622
  (1.39x) [M].
- ExoMolOP (Chubb et al. 2021): natural-abundance files (VULCAN tracks a total VMR); split
  8+8 quadrature, nodes and weights from the file; cm^2 per molecule; pressure grid 1e-5
  to 100 bar. Layers above 1e-5 bar reuse the top entry, as pRT does (logged at load). A
  deeper grid bottom would reuse an under-broadened entry, so it is refused. 25 usable
  tables, ngauss 16.
- Header metadata is not trusted: the CH4, N2O, OCS and CO2 header DOIs are placeholders
  (the real ones are in `fetch_exomolop._DATASET_DOI` and go to `provenance.json` as
  `doi`); NO's header says XABC, the table is HITEMP, and its `mol_mass = 46` is inert (NO
  identity checked from the data: peak at 1924 cm^-1). The engine never reads `mol_mass`.
- Fetcher: the first sorted isotopologue gave a rare N2O isotopologue (~1e2 low) and the
  first `.h5` gave a wrong O2 product (another band grid, 11.8 GB). Hence `_is_principal()`
  (unknown naming raises), the grid-token filter, and `_assert_grid_matches()` on the
  `.part` download before `os.replace`, so `--force` cannot destroy a good table.
- The species menu is complete: every IR-active SNCHO species was swept against ExoMolOP.
  Cannot be added: O2 (only R15000, another band grid); CH3OH, CH3CN, HC3N, NO2, C6H6,
  CH3CHO, HO2 (no petitRADTRANS file); CS2, C2H6 (no pRT-format table); ~19 radicals and
  nitriles (absent). S2 is homonuclear; HSO has no published list.
- Oracles (neither replaces the JAX code): `ckd._interp_logk` vs exojax
  `interpolate_log_k_2d` max |dlogk| 2.8e-14 (asserted 1e-12); exo_k 1.3.1 fixture
  `exok_ref_overlap.npz`: reader bitwise, interpolation 2.1e-14, overlap per ordinate
  median 1.4e-3, max 0.22 (exo_k's upper-edge vs our midpoint cumulative g) [M].
- `ckd.overlap` is continuous only piecewise: the midpoint coordinate moves when two
  pairwise sums with unequal weights swap sort order, so k and its derivative jump.
  Largest jump measured (W39b column, one VMR scanned through a crossing): 0.56 ppm
  native, 0.056 ppm at R = 100, rms 0.011 ppm; a 25-molecule fold on the same column is
  no worse (0.35 ppm native) [M]. The maintainer decided to keep the rebin (see "Do not
  retry").
- Leave-one-out: resort-rebin is not associative, so `ckd._fold_wo` gives each wo output
  the exact op sequence of a naive refold with the zero operand; consumers gate
  max|diff| = 0.
- `ckd.fold` (0.25.0) runs the molecule fold under one `lax.scan`, bitwise the Python loop
  on the CPU (rtol 1e-12 elsewhere: a GH200 differed in the last bits). RT vjp per lane,
  12 absorbers, 1650 bands, 67 layers, CPU: loop 13.49 GiB / 11.9 s, scan 4.56 GiB /
  9.8 s [M].

## Geometry and anchors

- `p_ref_bar = 1e-3`: `rp_cm`/`gs_cgs` apply there and convert hydrostatically to
  exojax's `radius_btm`, the bottom layer's lower boundary (`p[-1] k^-0.5`), which is
  why `_radius_at` carries a half-layer node. Re-anchoring at 1 mbar reproduced the JWST
  ERS WASP-39 b 3.0-5.5 um median depth to 0.4% (21,299 vs 21,381 ppm). Depth moves about
  4.6 H/Rp per decade of `p_ref_bar` [M].
- lnR0 at fixed GM: transmission (0.28.0) and emission (0.29.0) scale (rp_cm, gs_cgs) at
  p_ref to (rp_cm e^lnR0, gs_cgs e^(-2 lnR0)). Before 0.28.0 transmission scaled the
  bottom radius at fixed g, so the planet was e^(2 lnR0) too light (0.83 in
  vulcan-retrieval run 90302) [M]. At lnR0 = 0 the outputs are bitwise the old ones;
  `test_e2e_rt_reference.py` pins lnR0 against a model rebuilt at the scaled radius and
  gravity to 1e-12.
- Emission anchors radius and gravity at the constant `constants.P_REF_EMISSION_BAR`
  (0.1 bar), not a profile key; the emission model still carries the value as
  `p_ref_emission_bar`. A transit radius with emission gravity puts GM 8.9% off.
  `emission_radius()` gives the prefactor radius.
- Eclipse prefactor: the tau = 2/3 radius per k-ordinate (`eclipse_flux_tau`; Fortney et
  al. 2019). A single radius reads low on W39b's dayside by 7% median, 13% in the CO2
  core; under 2.5% on HD 189733 b [M]. Pinned against a numpy loop to 9e-16 and a
  gray-cloud analytic photosphere.
- Model top: clamping chemistry VMRs above a 1e-7 bar chemistry top cost 73.47 ppm at
  3.0 um; truncating the RT grid instead cost 80.82 ppm [M]. So the chemistry `P_t`
  follows `art_ptop_bar` and a clamped top raises. Top 1e-8 vs 1e-9 bar moved 14.65 ppm
  (fails the 5 ppm gate); 1e-9 vs 1e-10 moved 1.14 ppm, hence `ART_PTOP_BAR = 1e-9` [M].
- `_check_cia_span`: production caps `nu_max` at the H2-H2 edge (10000 cm^-1); worst case
  +3.30 ppm, unreachable in shipped consumers [M].

## Emission solver and gravity

- `ArtEmisPure.run_ckd` hard-codes `ibased` (no interior source): 22-99% less flux on a
  thin column. `ibased` loses 12-52% of the flux at admitted optical depths; at 60
  layers, bottom tau 10, `ibased_linsap` sits 0.13-0.40% from its converged answer vs
  2.9-5.3% [M]. The deep `_boundary_temperature` entry is the interior source. Pinned:
  degenerate-g identity 3.3e-16, blackbody limit 6.7e-16.
- Gravity, isothermal gray W39b-like column at 60 layers vs an independent chord
  quadrature: constant g -102 ppm, exojax's `gravity_profile` (linear in 1/r) -51 ppm,
  inverse-square at layer midpoints +1.5 ppm [M]. The engine keeps its own
  `_gravity_profile_invsq` so an exojax change cannot move it.
- The emission optical depth uses the same g(r) (`c63815c`). The old single gravity at
  the 0.1 bar emission anchor put the eclipse flux 0.5-1% high on a W39b-like column, 0.9-2% on a
  WASP-107b-like one [M]. The pRT emission gate did not move (its large rp holds g
  constant).

## RT validation and the exojax pin

- Committed fixtures `tests/data/prt_ref_*.npz` gate the RT in
  `tests/test_e2e_rt_reference.py`. Meta holds the pRT version (3.4.0, pinned in
  `tests/test_e2e_fixtures.py`), opacity files, constants,
  generating script and achieved agreement; tolerances are 3x achieved; pRT ran in a
  throwaway venv. Both codes read the same k-table file [M]:
  - transmission, isothermal H2O: mean ratio 1.00003, rms 0.0013%; 6 species 1.0010 /
    0.021%; W39b 8 species (Tsai et al. 2023 inputs) 0.986 / 24 ppm on 23,000 ppm.
  - emission, H2O, 900-2200 K, 3-5 um: bias +0.034%, rms 0.020%, worst band 0.091%.
  - transmission vs the analytic grey solution: 0.012-0.016 scale heights over 5-7
    decades.
- Unit trap: exojax flux is per cm^-1; pRT's wavelength-mode flux is per cm of
  wavelength, so `F_lambda = F_nu_tilde * nu_tilde^2`.
- Published gCMCRT spectra decode as `depth = (H(1)^2 + 2*col2)/Rstar^2` with each file's
  own header H(1) (`exp_3D_sph_atm_trans.f90`).
- exojax 2.2.3 -> 2.6.0 (0.27.1, daw, same inputs): every optical depth bit-identical;
  transmission depth <= 1.5e-10 absolute, emission flux <= 4.9e-10 relative, RT jvps
  <= 4.7e-8; the 8-species pRT case passes (mean ratio 0.998288) [M]. Causes, from the
  source diff: `m_u` CODATA 2014 -> 2022 and `piBarr` rewritten with `expm1`. Floor
  reason: forward-mode AD through exojax was NaN-poisoned before `safe_sqrt` (2.1). The
  chemistry never imports exojax.

## Convergence certificate (`ConvDiag`)

- `conv_normal` is `vulcan3.conv_normal(final, cfg)` (0.27.0, `dfa2a0f`): tight or loose
  branch, the flux gate, the geometry term and the element-budget term, ANDed with
  `tangent_ok` on the `converged_y_jvp` path only (`_conv_diag`). The switch from the
  local copy was exact: 175 arrays (every ConvDiag field, y, dy) array_equal on the
  batch, queue and jvp test profiles [M]. `conv_branch` stays the column's, so an
  unsettled tangent is distinguishable from an uncertified column.
- Per-theta init: `_prep` seeds the element-budget reference on this theta's column and
  grid (`budget_ref`, zeroed `budget_err`/`budget_drift`, `y_prev = y0p`), so the
  certificate charges the solver, not the geometry refresh. Retrieval smoke (nz 30, 6
  draws): `budget_drift_max` 0.13-12.9 (6/6 uncertified) without the `y_prev` seed,
  2.5e-8 to 1.3e-3 with it [M]. Kept by the maintainer (2026-10-20).
- On the planner's W39b default (C/O 0.55) the loose-branch exit is set by N2O at
  3.5 mbar, VMR 6e-18: one ultratrace cell decides when the solve stops [M].
- `converged_y_jvp` runs `OuterLoop.run_jvp` and stops only when the column and the
  carried tangent both certify, per cell in the tangent's units. A plain
  `jax.jvp(converged_y)` stops when the column certifies, which from a converged warm
  start is `count_min` with the tangent unrelaxed. A stacked `(D, n_theta)` tangent rides
  one primal: stacked vs single 6.2e-9; a vmap of the call 2.8e-3 (the two stopping
  rules) and D primal integrations; 3 directions 2.4 s stacked vs 3.4 s vmap [M]. The
  planner's AD rows call it.
- A warm tangent blowing up after the primal converged, dt uncapped, is the settled
  uncapped-dt case in VULCAN3's notes, not a new defect.

## Batched routes (`converged_y_batch`, `converged_y_queue`)

- Both agree with the solo solve at the convergence scale only (batch vs per-lane vmap max
  rel 7.2e-3) [M]. A batch lane is bitwise a batch of one since VULCAN-JAX `dd2c7a0`
  (before, a lane's eq seed moved ~4e-15 with the batch width).
- The cold queue is not bitwise the batch: `run_queue` builds the seed inside its jitted
  loop, and the two compilations of `_prep` differ by 1.1e-15 in `y_ini`, amplified until
  accept counts split (queue [121,122,121,121] vs batch [121,121,121,166]) [M]. Started
  both from `y0` they still split (queue [121,127,122,148] vs batch [123,...]) since the
  carbon repair, and a 1e-12 lnZ nudge moves the queue-vs-batch max rel from 0.4e-2 to
  7e-2 on every job [M]: tests compare `conv_normal` (not `conv_branch`: an S4 cell at
  1e-20 certifies tight in one compilation, loose in another) and the convergence scale
  (bar 0.1, `yconv_min`), and a plain-jvp tangent only between routes that stop on the
  same accept count (two that stop a step apart differ by 9.9e-1).
- Warm-started stage 2: every theta certifies in both arms at different relaxation
  depths; the worst cells are floor sulfur; H2O, CO, H2, He <= 2.8e-4 rel, CO2 and CH4 up
  to 2.0e-2 [M]. The transit-depth effect is unmeasured.
- The queue test profile runs the production cap `dt_max` 1e11 s. Uncapped, queue-vs-batch
  CO read 1.06e-3 (over the 1e-3 bar); capped, 1.26e-4. A tight capped reference puts
  every route 1.2e-3 to 8.4e-3 off in CO: the routes differ by less than their
  convergence error [M].
- A plain `jax.jvp` runs through both (batch jvp vs per-theta `jax.jvp(converged_y)`
  1.0e-4). The gradient tests assert column max rel 4e-1, median 1e-3, tangent
  stack-relative 2e-4; measured cold 1.70e-1 / 2.08e-4 / 1.73e-5, warm 1.16e-1 /
  5.74e-5 / 2.56e-5 [M].
- Per-lane `lnZ_ref` / `c_o_ref` are bitwise the scalar path [M].
- Test bars, calibrated on daw (Xeon 8462Y+): queue-vs-batch primal 0.05 (`REL_MAX`);
  batch-vs-solo 0.1 (6e-2 and 8e-2 measured across Python 3.10-3.12 and numpy
  1.26-2.x); queued tangent 1e-2 on the settled thetas (<= 4.6e-4 measured) and a 1.0
  wiring bound on theta 2, whose queued tangent is undetermined (a 1e-12 lnZ nudge moves
  it 6.8e-2) [M]. GitHub runner run 36254404095 (`bdbf2d1`): queue-vs-batch primal
  6.87e-2, queued tangent 3.47e-2, and the eq-seed warm jvp ran to its 3001 cap with the
  tangent diverged; those three tests are in the full tier. GH200 is unverified; the
  maintainer decided no run is planned (nautilus is primal-only).

## Cold start, abundance map, carbon repair

- The only cold start is vulcan3's in-graph Gibbs `eq_seed` at the proposal's own T-P and
  column ratios (0.27.0). The build column as a continuation (`warm_y=chem.y0,
  lnZ_ref=0, c_o_ref=0`) is bitwise the retired `cold_seed="baseline"` on
  vulcan-retrieval's condensation recipe [M]. Baseline vs eq, jwst-like W39b column: at
  the file T-P identical step counts and 6.9e-7 ppm in depth; on a Guillot profile 1617
  vs 1497 steps and 0.034 ppm at R = 100 [M].
- One abundance map: every initial column is renormalized to sum n_i = M, repaired onto
  the exact theta column ratios, and `atom_ini` is rebuilt from it. The retired masks map
  was not an exact elemental direction (sum n != M at init).
- Repair groups (0.27.2, `_ELEMENTAL_REPAIR`): each element is repaired by one common
  factor on a group of carriers, carbon on (CO, CH4). CO alone needed factors down to -21
  on a cool CH4-dominated column and the solve died at step 1 (NaN). On a 7x7 (lnZ, c_o)
  box over six W39b profiles: CO-only minimum factor <= 0 at 10-26 of 49 points; with the
  group 0/49, smallest factor 0.94, ratios exact to 3.0e-15 at 3 passes (4-6 passes no
  better) [M]. Every negative point sat where stage 1 already exhausted `count_max`, so
  the fix rescued no draw there.
- `co_bz_margin(y)` is the fixed-O seed map's positivity margin; the planner gates its AD
  dlnCO on it.

## Photolysis refresh cadence

- Config policy, never engine policy: `vulcan_chem` sets neither `ini_update_photo_frq`
  nor `final_update_photo_frq`. The YAML decides (W39b 5/5, other configs 100/5, as
  upstream VULCAN 2); a caller overrides per column through `cfg_overrides`.
- Cadence 1 fixes the planner's TOI-7169 b stall (at cadence 5 one step is rejected 23
  times every cycle against a stale water column; cadence 1 certified at 3422 steps) and
  moves the planner's six presets by <= 3.8 ppm native [M]. It was tried engine-wide and
  reverted: the planner's transmission FD/FD and emission AD/FD closure tests pass at
  cadence 5 and fail at cadence 1 (emission warm AD row corr 0.9976 -> 0.558;
  transmission Tirr row corr 0.998 -> 0.981 against a 0.99 gate) [M]. The consumer
  escalates per column via `cfg_overrides` and records it; vulcan-retrieval sets neither
  key. So "all derivatives are validated" is not a claim this repo can make.

## Test tiers and CI

- XLA compile, not the solve, sets the chemistry tests' cost: the tangent runner (nz 20,
  photo off) compiles in 112-148 s, the primal in ~18 s, a 41-step run takes ~1.5 s. One
  photo-on route of the cold gradient map is ~325 s compile + ~117 s run at ~16 GB [M].
- A model keeps every program compiled on it: the single cold three-route test took
  1362 s and peaked at 35.4 GB; split by route with a per-test model, each peaks ~21 GB
  [M]. That is above a hosted runner's 16 GB, so CI runs the normal tier only.
- Walls at `2ab2049` (12 CPUs, `-n 6`): normal 3.3-3.6 min, 4.7 GB peak; full 22.5 min
  with `--dist worksteal`, 29.7 min with the default `load` scheduler, which queued the
  heavy tests on one worker [M]. A CI job takes ~6-8 min with its install [E].
- Without single-threaded XLA, `test_converged_y_batch` fails its bitwise lane check on
  daw [M]. With the workspace env's CUDA jax and a busy GPU the suite runs out of memory
  [M].

## Superseded designs

- Sampled line-by-line opacity and the Mie cloud deck: wrong contrast below
  R = 700,000 (see "Opacity"). Removed in 0.11.0, `db94d6e`.
- Interim HITRAN-built correlated-k mode (`opacity_mode="ckd"`, 0.6.0-0.7.x): right
  machinery (32 ppm rms vs R = 700,000), wrong line data. Removed in 0.8.0, `773837e`.
- Radius anchored at the 7 bar RT grid bottom: depth 26,754 vs 21,381 ppm on W39b.
  Replaced by `p_ref_bar`, `4446e6e` / `e2aacba`.
- `cold_seed` (baseline / eq) and the masks abundance map with `reanchor_atom_ini`: one
  cold start and one exact elemental map replaced them (see "Cold start"). Removed in
  0.27.0, `5018e4d`, `d40211d`, `2807b9a`.
- A local copy of the runner's certificate terms in `ConvDiag`: replaced by
  `vulcan3.conv_normal` with an exact switch. `dfa2a0f`.
- `warm_dt`, an exit-dt step seed for warm solves: a wash (stage-2 steps 756 -> 750,
  977 -> 1316, 876 -> 653) [M]. Added `3d43367`, reverted exactly in `7e5ea47`.
- Band tiling of the correlated-k fold (`rt_band_tiles`): 2.5% of the per-lane gradient
  peak; the unrolled molecule loop held the memory. Added `faa580a`, removed in 0.25.0
  `fd88866` for `ckd.fold`'s scan; `integ_warm` went in the same commit.
- `converged_ymix`: no caller. Deleted `a5d2190`.
- `warm_cap` / `warm_count_max`, a step cap for the SMC sampler's mutation solves (it
  rode the carry's `count_max_dyn`): no caller after the SMC removal, both retrieval call
  sites passed `warm_cap=False`. Removed with the unread `ConvDiag` fields
  (`count_since_new_min`, `budget_drift_max`, `budget_drift_atom`) and `atom_order`.
- One gravity in the emission optical depth: replaced by g(r), `c63815c`.
- Consumer-side e^(2 lnR0) eclipse prefactor at fixed g: replaced by lnR0 at fixed GM in
  the engine, `d26e9d3` (transmission) and `801f7e7` (emission).
- CO-only carbon repair: replaced by the (CO, CH4) group, `a6a1094`.
- Middle-axis argsort in `ckd.overlap`: replaced by the last-axis two-operand sort,
  `5e8d43a`.
- The `speed` branch (0.21.0-0.27.0, from `43e895d`): merged; `main` is the only line.
- The build-time warm-up solve (`baseline_conv_normal`, `skip_warmup`): a full solve at
  build whose column was never the returned seed and whose bit nothing gated on (every
  solve certifies itself). Removed in the 2026-10-20 cleanup; the first real solve now
  compiles the runner. `p_ref_emission_bar` as a profile key went in the same pass.

## Do not retry

- Sampled line-by-line below R = 700,000: wrong contrast, non-convergent with more points
  (see "Opacity"); its keys (`nu_pts`, `broadening`, `dit_grid_resolution`, `mie_*`)
  raise.
- HITRAN line data in any mode; `broadening="h2he"` cannot fix H2O (no H2/He widths).
- Guessing ExoMolOP URLs: 404s; walk the page levels (the site also 403s a default urllib
  user agent).
- First sorted isotopologue or first `.h5` on a page: wrong products (see "Opacity").
- Trusting header DOIs or `mol_mass`: placeholders and an inert mass (see "Opacity").
- A network error printed as SKIP: SKIP means a page with no k-table link, nothing else.
- `ArtEmisPure.run_ckd` or `ibased`: they lose the interior source or flux at admitted
  depths (see "Emission solver and gravity").
- exojax `gravity_profile` or constant g: biased depths (see "Emission solver and
  gravity").
- Anchoring at the bottom layer centre: depth depends on `art_nlayer` (+1.02% at 20
  layers).
- A transit radius with emission gravity, or a single eclipse radius for the whole band
  (see "Geometry and anchors").
- The `R_ref^2 + col2` gCMCRT decode: halves the amplitude, a fake 2.2x discrepancy.
- Dividing pRT flux by c: wrong by ~1e18.
- Separate `emission_flux` / `tau_bottom` calls: dtau built twice; `emission_flux_tau`
  builds it once.
- `jit(overlap)`: 2-7%, below the 10% adoption bar. (The last-axis sort layout read ~10%
  on the CPU and was set aside, then adopted after the GH200 measurement: 4.25x.)
- Skipping the dropped molecule instead of folding its zero: not guaranteed bitwise.
- A conservative rebin in `ckd.overlap` to remove the sort-crossing jumps: <= 0.06 ppm at
  R = 100 against data sigmas of tens of ppm, and it would move every spectrum, fixture
  and cache. Maintainer decision: keep the rebin.
- Truncating the RT grid at the chemistry top: worse than the clamp it replaced (see
  "Geometry and anchors").
- `art.change_temperature_range`: dead in both modes (exojax reads Tlow/Thigh only in its
  own T-P constructors).
- A `conftest.py` for the test preamble: reorders `vulcan_chem` ahead of the jax-free
  contract test.
- Comparing the ERS CO2 model-grid case without the bundle's exact inputs (T-P source,
  radius anchor, v30 grid): rms 155/199 vs 106/149, not reproducible; the inputs are not
  on daw.
- `np.trapezoid` on numpy 1.x or `np.trapz` on 2.x: use
  `getattr(np, "trapezoid", None) or np.trapz`.
- Photolysis cadence 1 as the engine default: the planner's closure tests fail (see
  "Photolysis refresh cadence"). Reconsider only with both closures and the
  reference-accuracy gate satisfied.
- A checkpointed body for `ckd.fold`'s scan: 3.48 vs 4.56 GiB per lane but 14.4 vs 9.8 s
  per RT vjp. Take it only if GPU memory, not time, binds.
- Sharing the opacity tables across CPU lanes: the W39b tables are 1.4 GiB against
  ~17.4 GiB per gradient lane, so ~10%; production runs on the GPU.
- Queue tests uncapped (see "Batched routes"), or an uncapped tight reference: the
  40000-step one never certified (element-budget drift 13-55%).
- An exact accept-count check between the cold queue and the cold batch: each builds its
  own seed (see "Batched routes"); start both from one column.
- Shrinking the photo-on gradient tests by nz, directions or thetas: still 362-406 s and
  15.9-18 GB, since compile is ~3/4 of it; split by route instead.
- A tangent-tail mode for `converged_y_jvp`: VULCAN3 rejected the tail; the derivative
  stays `run_jvp`.

## Known and deliberately untracked

- `test_batched_lanes_are_their_own_solves_and_agree_with_the_vmap` (bitwise lane vs
  batch-of-one) passes on daw and CI; an outside x86 container reported lane 1 differing
  (124 vs 121 steps), so bitwise lane independence may depend on the CPU [I].
- The photolysis-cadence switch threshold stays at VULCAN3's derived 1.0 when a profile
  overrides `yconv_min`: only the refresh timing moves, the certificate does not.
- `co_mode` defaults to `fixed_O`, which both consumers pass; `proxy` is kept and no test
  in any repo sets it. The modes differ only in a warm continuation's guess, which the
  elemental projection puts on the exact targets.
- `transmission_depth`, `emission_flux` and `tau_bottom` have no production caller (grep
  of vulcan-retrieval/src and vulcan-jwst-tool/src).
- Importing `vulcan_chem` sets `OMP_NUM_THREADS=1` (`setdefault`, undocumented).
- `build_chem_model` is one ~800-line function. `ChemParams.tp = ()` defaults to an
  IndexError [I].
- A non-default `VULCAN3_NETWORK` is kept when `vulcan_chem` is imported first, refused
  when `vulcan3` was.
- No engine test sets `cfg_overrides`, `run_diag`, `use_rayleigh`,
  `rt_integration` or `molecule_table` (consumers use all but the last).
  `test_contract.py`'s docstring says the planner's datacheck reads the path helpers; it
  re-derives the path.
- The eq seed reads vulcan3's process-default config at trace time
  (`ini_abun.element_vector`; solver controls from `ini_abun._CFG`), not the model's.
  Harmless while the elemental ratios set every seed element but H and the seed controls
  agree [I].
- `constants.ATOMIC_MASSES` / `ATOM_COLS` duplicate vulcan3's composition data (checked
  at build by `_assert_composition_tables`); they stay because the planner's
  `tests/unit/test_engine_api_contract.py` pins the names.
- The observed-cell comparison (mix floor, max rel) is written out in each converged_y
  test module; one shared copy needs a shared test file.
