# vulcan-forward notes

What is true about the forward engine and why: the evidence behind each rule in
`CLAUDE.md`, the design records that still govern the code, retired designs and the
do-not-retry register. Open, owed and undecided work is in `todo.md`. Retrieval-side
studies are in vulcan-retrieval's notes; solver findings in VULCAN3's notes.
Tags: [M] measured, [E] estimated, [I] inferred.

## Current state

- Version 0.31.0 in the working tree: 0.30.0 plus a breaking interface change. The profile names the chemistry
  (`network`, `atom_list`, `vulcan_cfg_name`) and the RT column (`art_pbtm_bar`, `p_ref_bar`): the engine fills none
  in, and `opacity_mode` is gone ("Import order, profile keys and refusals"). 0.30.0 on `main`, the only branch, is
  0.29.0 plus the WASP-39b v5 engine options,
  a sulfur knob (`s_o`) and a Kzz power law as optional entries of the chemistry layout,
  `lnM` and a grey deck in the RT ("Cold start, abundance map, carbon repair", "Kzz
  shape", "Geometry and anchors", "Opacity: correlated-k on ExoMolOP"). 0.29.0
  (`801f7e7`, tag `v0.29.0`): no build-time warm-up solve, `co_mode` default `fixed_O`,
  the emission anchor a constant.
- Floors (`pyproject.toml`): `exojax==2.6.0`, `radis<0.17`, `vulcan3>=0.18.0` (first
  release under that name; carries `conv_normal` and `validate_overrides`),
  `jax>=0.6.2`, `numpy>=1.26`, `h5py>=3.9`.
- Consumers: vulcan-retrieval floors `>=0.30.0` and its `requirements-hpc.txt` pins
  `50bef74`; jwst-transit-authority (the planner) floors `>=0.29.0` and pins `0accd36`
  in `deploy/pins.env`.
- Production: the WASP-39b baseline v2 nautilus run used `d26e9d3` (0.28.0, tag
  `w39b-baseline-v2`). All three W39b production runs carry the 0.27.2 carbon repair
  (`a6a1094`).
- The engine has no default network (0.31.0): a profile names it. The consumers' cases name
  `thermo/SNCHO_photo_network_2025.txt` (93 species), upstream's WASP-39b network. Every run before 0.30.0, v5
  included, solved the 2021 `SNCHO_photo_network.txt` (89 species).
- Normal tier: plain `pytest tests`. CI runs it serially on Python 3.10, 3.11 and 3.12,
  single-threaded XLA, numpy<2, vulcan3 from `imalsky/VULCAN3@main`, no exojax (the RT
  tests skip). Readings [M]: daw, `-n 2`, single-threaded XLA: 135 passed / 8
  skipped, line coverage 90.6 %. CI's `coverage` job (exojax, no data tree, `fail_under` 80), read here: 125 passed / 12
  skipped, 90.5 %.
- Full tier (`VULCAN_FORWARD_RUN_FULL=1`): the `converged_y_jvp` tests, the five `@FULL`
  queue/gradient tests and the 8-species pRT case. Readings at `-n 6 --dist worksteal`
  [M]: `4ca5fa6` 103 passed in 20.7 min, 23.9 GB peak; the 0.29.0 tree 95 passed /
  2 failed in 31.4 min, the failures the queue-vs-batch knife edge at the recorded
  6.874e-2 ("Batched routes") while the eq-seed warm jvp case certifies; the
  2025-network tree 98 passed in 35.1 min on daw.
- Known defect, no shipped number moves: the `ckd.overlap` sort-crossing jumps
  ("Opacity: correlated-k on ExoMolOP").

## Library and data contract

- No data at import, no writes to the caller's cwd, no paths from `__file__`: each has a
  test in `tests/test_contract.py` (imports only). A `__file__`-relative path
  (`parents[3]` plus a `data/cm24_wasp39b` marker) made the package un-importable
  elsewhere. A build is cwd-clean because vulcan3 dropped the pre-loop `Output()`
  (VULCAN-JAX `d8856c6`, in the 0.18.0 floor): a build plus one solve in an empty
  directory left it empty, so the engine carries no `output_dir` redirect [M].
- `exomolop`'s path helpers and `provenance()` import without jax or h5py (a subprocess
  test). The planner's `datacheck.exomolop_table_path` re-derives the path; a data-gated
  planner test pins that the two agree.
- `$VULCAN_FORWARD_DATA` / `paths.set_data_root`, resolved on use: no measurement. A
  library cannot know a data path, and a guessed one fails late or reads other data [I].
- Planet geometry is required: `build_rt_model` needs `rp_cm`, `gs_cgs`, `rstar_cm`;
  `build_emis_model` needs `rp_cm`, `gs_cgs` (`_require_positive`). The retired WASP-39 b
  fallback let a forgotten key silently model another planet.
- Injectable molecule table (`profile['molecule_table']`, passed to
  `exomolop.load_tables`): no measurement; consumers choose their own absorber menus
  (the planner's `EXTRA_MOLECULES_DEFAULT`). `constants.MOLECULES` carries only
  `vulcan` and `molmass`; the mass cancels out of dtau (a 10x-wrong H2S mass left the pRT
  fixtures bit-unchanged [M]).
- Bit-identical gate: the extraction of `vulcan_chem`, `exojax_rt` and `interp_map` from
  vulcan-retrieval reproduced the planner's default WASP-39 b case at max|diff| = 0 [M].
  The deliberate physics changes since (model top, eclipse radius, emission g(r), lnR0)
  carried new fixtures.
- The three builders return a `SimpleNamespace` whose functions are bound methods of
  frozen, `eq=False` classes (`_ChemSetup`, `_ChemSolves`, `_Transmission`, `_Emission`;
  the last two share `_Opacities`): jwst-transit-authority's
  contract test reads the record's keywords off the source, and each attribute stays one
  function object for a `jax.jit` of it. `_ChemSetup` must not reach the runner:
  `OuterLoop.run_queue` compiles its methods into a program the runner holds, and JAX's
  trace caches keep them until that program dies; with `_conv_diag` on the runner's side a
  dropped model stayed alive through `gc.collect()`, where the base freed it [M].
- Why a separate distribution: the exact exojax pin is load-bearing, and VULCAN3 is
  standalone with vendored data; an RT stack and an external data path there would widen
  its blast radius. DAG: `vulcan3 <- vulcan-forward <- {vulcan-retrieval,
  jwst-transit-authority}`.

## Import order, profile keys and refusals

- The network is the caller's: `vulcan_chem.select_network(network, atom_list)` writes `VULCAN3_NETWORK` /
  `VULCAN3_ATOM_LIST`, which vulcan3 freezes at its first import, and `build_chem_model` calls it with the profile's
  pair. Importing the engine selects nothing. A disagreeing environment is refused (the caller's settings decide), and
  once vulcan3 is imported another network or atom list is refused against what vulcan3 built at import
  (`chem_funs.NETWORK.network_path`, `jax_step.IMPORT_ATOM_LIST`): one process, one network. The build also sets the
  loaded config's network and atoms to the profile's, so vulcan3's own pre-loop check compares the profile's pair. Before 0.31.0 the engine set the 2025 sulfur network with `setdefault` at
  import and the environment silently beat the network a config named.
  `test_the_network_is_selected_before_vulcan3_is_imported_and_only_once` pins it in a fresh interpreter. The suite's
  own conftest names its one network, because a few tests reach vulcan3 through a helper before any model is built.
- `vulcan_chem` sets jax x64 before any exojax import, and a late arrival raises
  (`test_import_order_guard_is_documented_and_live`).
- No default stands in for a case (0.31.0): `build_chem_model` requires `network`, `atom_list` and `vulcan_cfg_name`
  (it loaded the WASP-39 b config when none was named); `build_rt_model` requires `art_pbtm_bar` and `p_ref_bar` beside
  the planet geometry (7 bar and 1 mbar were WASP-39 b's). In-range outputs are bitwise unchanged with the keys
  stated: 29 arrays (four chemistry columns with and without photochemistry, transmission with and without Rayleigh,
  cloud and deck, leave-one-out, a T jvp, emission and eclipse flux) on the real tables [M, daw CPU].
- A setting stated twice or where it cannot act is refused: a top-level key repeated in `cfg_overrides` (the override
  won silently), `network` or `atom_list` in `cfg_overrides` (frozen at import), `runtime` there without `dt_max`
  (vulcan3 derives `dt_max` from the file's `runtime` at load, so it did not follow), `kzz_p_ref_bar` outside the power
  law. A `yconv_min` override still leaves vulcan3's derived photolysis-switch threshold unchanged ("Known and
  deliberately untracked"): jwst-transit-authority relies on it.
- One planet: `G Mp` of the vulcan3 config (after overrides) and `gs_cgs rp_cm^2` of the profile must agree to
  `constants.GM_MATCH_RTOL`, 2 % (the maintainer decided). The shipped WASP-39 b pair agrees to 3e-6 [M]. A retrieval
  that samples the RT mass is checked at its base values only.
- The start column is checked after the elemental repair (`_chem_model._physical_start`): a repair factor at or below
  zero, or a negative or non-finite density, is an error when eager and a NaN column under jit, which comes back
  uncertified. Without the check a warm start with one negative density froze vulcan3 at `dt_min` in 132 s
  (`DtMinFrozenError`) [M].
- The Gibbs data comes from the network's own tree: a network with no `NASA9` folder beside it is refused before the
  pre-loop setup. It fell back to vulcan3's packaged data without a message.
- `build_emis_model` refuses a profile whose molecules, band, grid, anchor, geometry or integration rule differ from
  its transmission model's. It ignored most of those keys.
- `pytest.importorskip("exojax")` imports exojax ahead of `vulcan_chem` for every later
  test module, so test modules use `importlib.util.find_spec`. A `conftest.py` moved
  `vulcan_chem` ahead of the jax-free contract test, so the preamble repeats per module.
- Ruff's import sorting would reorder those preambles, so `tests/**` ignores `I` and `E402`.
- `PROFILE_KEYS` is the union of the keys the three builders read (consumers pass one
  dict to all three); `check_profile_keys` refuses anything else (`affaecf`, 0.26.0).
  Before it, deleting a reader turned a set key into a silent no-op. `cfg_overrides` pass
  `vulcan3.validate_overrides` first (0.27.0).
- Every build-time refusal names the remedy: missing table (fetch command), missing
  geometry, out-of-grid anchor, clamped chemistry top or bottom, an absent CIA table (exojax would fetch it behind a
  downloader that swallows failures), CIA span, `art_pbtm_bar`
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
  no worse (0.35 ppm native) [M]. The rebin stays by decision ("Do not retry").
- Leave-one-out: resort-rebin is not associative, so `ckd._fold_wo` gives each wo output
  the exact op sequence of a naive refold with the zero operand; consumers gate
  max|diff| = 0.
- `ckd.fold` (0.25.0) runs the molecule fold under one `lax.scan`, bitwise the Python loop
  on the CPU (rtol 1e-12 elsewhere: a GH200 differed in the last bits). RT vjp per lane,
  12 absorbers, 1650 bands, 67 layers, CPU: loop 13.49 GiB / 11.9 s, scan 4.56 GiB /
  9.8 s [M].
- H2 and He Rayleigh is on unless the profile sets `use_rayleigh` false (it was off by default before; both
  consumers always passed true). Callers that did not name the key, so ran without it and now change: jax_paper's
  `scripts/retrieval/paper_profiles.py` profiles (used by `run_figs.py`, `sensitivity.py`,
  `zco_information/build_zco_jacobians.py`, `build_zco_walk.py`) and jwst-transit-authority's
  `validation/scripts/rtcase.py` (a grep of every `build_rt_model(` caller in the workspace). On minus off on a W39b-like test column (40 layers, 700-1600 K, H2O 1e-3, CO 5e-4,
  CO2 1e-5, SO2 3e-6, CH4 1e-7, no cloud) [M, daw CPU]: mean 10.6 ppm (max 29.4) at 1.0-1.2 um, 1.1 (6.0) at
  1.2-2.0 um, 0.07 (0.27) at 2-3 um, 0.01 (0.03) at 3-5.3 um.
- Bounds refused in the engine: a band outside 1-15 um (`constants.check_band`; below 1 um Na, K and hazes set the spectrum),
  and a column the model does not cover (`_rt_opacity._column_ok`: a temperature outside 200-3000 K, a mean
  molecular weight below atomic hydrogen's, a mixing ratio outside [0, 1]; an error when eager, a NaN spectrum under
  jit). Planet radius, gravity, stellar radius, the column bottom and the reference pressure must be finite and
  positive (`_require_positive`). The window is the opacity data's: both CIA files start at 200 K, the H2-H2 file ends
  at 3000 K, and the k-tables span 100-3400 K; the build refuses installed tables that do not cover it. It was
  300-3000 K with no reason for the lower edge. It bounds the RT only: vulcan3's equilibrium start breaks down below
  about 225 K (VULCAN-JAX notes), where an eager solve raises and a jitted one comes back uncertified. exojax's CIA interpolation is `jnp.interp` in T, so it clamps like the k-tables; the H2-H2 file spans
  200-3000 K and the H2-He file 200-9900 K. In-window outputs are bitwise unchanged by the guard: 12 arrays
  (transmission with and without Rayleigh, leave-one-out, a T jvp, emission flux, tau and eclipse flux) on the real
  tables [M].
- Core absorbers (`constants.CORE_MOLECULE_BANDS_UM`, checked by `constants.check_core_molecules`): H2O, CO2, CO,
  CH4, NH3, HCN, H2S, SO2, C2H2, OCS, SO. Left out of a band it absorbs in and not named in `omit_core_molecules`, one
  of the first five is refused (`REQUIRED_MOLECULES`) and one of the others is warned about and echoed as
  `unlisted_core_molecules` (the maintainer decided: HCN and the sulfur gases depend on the planet, though on WASP-39 b
  SO2 is the third largest effect, 1030 ppm, and HCN the fourth, 742). The first eight are the largest leave-one-out
  effects in jwst-transit-authority's record (its notes "Opacity and radiative transfer": on WASP-39 b each is 134 ppm
  or more over 1-15 um), and the carriers Moses et al. 2013 name. The last three are warned about because they reach
  tens to hundreds of ppm in the same record: OCS 58.5 and SO 47.7 ppm on WASP-39 b, C2H2 1434 ppm on its carbon-rich
  column. C2H4 (949 ppm there) matters only at C/O above 1 and is left to a case's own absorber screen. The ranges follow one rule (R = 100 band-mean cross-section at 1000 K, 1 mbar, at
  least 1 % of the 1-15 um peak), checked against the installed tables by `tests/test_exomolop.py`; H2O is the whole
  band, because its own peak rule would drop the 1.0-1.3 um bands where it is the main absorber. Against the callers
  [M]: both retrieval base specs pass (12 molecules); jwst-transit-authority's GUI default passes, its API default
  (no `extra_mols`: H2O, CO2, CO, CH4, SO2) is refused for NH3, and its sulfur-free network passes with a warning for
  H2S and SO2; the jax_paper profiles (`paper_profiles.py`) name what they leave out.
- A bare `converged_y` or `converged_y_batch` returns a column only if it certified (`_chem_model._certified`: an
  error when eager, NaN under jit); `return_conv_diag=True` hands the caller the certificate instead, and
  `allow_uncertified=True` is the opt-out vulcan-retrieval's ungated stage 1 passes. Before this a caller that did
  not ask for the diagnostics got a column that may have stopped on its step cap.
- `build_chem_model` refuses a stellar spectrum that does not cover every photolysis threshold of the network
  (`vulcan_chem._check_stellar_span`): vulcan3 cuts its photolysis grid to the file's span, as VULCAN 2 does. For the
  2025 SNCHO network the thresholds reach 655.5 nm (HCO). Of vulcan3's packaged files and the survey's, only
  `sflux-epseri.txt` (115-283 nm) is refused; the others start at 2 nm or below and reach 700 nm or more [M].
- Grey deck (`deck`, log10 P_deck in bar, independent of the power-law `cloud`): every
  layer deeper than the deck gets dtau = `CLOUD_DECK_TAU` (1e3) through a logistic step in
  ln P of width `CLOUD_DECK_WIDTH_LNP` (0.02, a sixteenth of a production RT layer). Along
  the chord the deck is ~1e5 optical depths thick, so the effective boundary sits ~13
  widths (0.26 in ln P, 0.1 dex) above P_deck. `tests/test_cloud_deck.py`: in a gas-free
  isothermal column the transit radius is within 0.5 H of R(P_deck), band-independent to
  the last bit, and a deck three decades below the grid reproduces the deck-free spectrum
  bitwise.

## RT speed (speed2_rt)

- GH200, `ckd.overlap`: XLA's argsort along the middle axis was 82% of the RT stage,
  107.4 ms per spectrum [M].
- Adopted (`5e8d43a`): sums on the last axis, one stable two-operand `lax.sort` with the
  weights as payload. 25.3 ms per spectrum (4.25x); a 2376-row RT pass 256.0 -> 60.2 s.
  Depths bitwise on all 3,920,400 values, log L bitwise on 2376/2376, jvp bitwise, vjp
  within 1.6e-14 relative [M]. The sort is then still 51% of the stage (11.7 ms per
  spectrum); the alternatives measured against it are under "Do not retry".

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
- `lnM` (`transmission_depth_r`, default 0): the gravity at `p_ref_bar` scales e^lnM at
  the unchanged scaled radius, so GM scales e^lnM while lnR0 keeps its fixed-GM meaning.
  RT-only: the chemistry column and a consumer's T-P tau keep M0. Pinned against a model
  rebuilt at the scaled gravity (`tests/test_e2e_rt_reference.py`); lnM = 0 is the lnR0
  path bit for bit. Transmission only; the emission functions take lnR0 alone.

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
- Stream count: `constants.EMISSION_NSTREAM` is 8, exojax's default. Against 32 streams the eclipse flux over
  1-15 um differs by at most 1.8e-4 (rms 1.1e-4) on two rising profiles and 1.3e-3 (rms 3.3e-4) on a 2400-1200 K
  inversion; 16 streams differ by 9e-6 and 1.3e-4. An isothermal column agrees to 1e-14 [M, daw CPU, 80 layers,
  H2O, CO2, CO, CH4, SO2].
- A grey deck is available in emission (`deck=`, the transmission deck's meaning): the opaque layers emit at the
  temperature of the level they sit at. On a gas-free column the flux is pi B(T) at a level within 0.2 dex above the
  deck pressure and does not see the temperature below it (`tests/test_cloud_deck.py`).
- The emission optical depth uses the same g(r) (`c63815c`). A single gravity at the
  0.1 bar emission anchor put the eclipse flux 0.5-1% high on a W39b-like column, 0.9-2%
  on a WASP-107b-like one [M]. The pRT emission gate did not move (its large rp holds g
  constant).

## RT validation and the exojax pin

- Committed fixtures `tests/data/prt_ref_*.npz` gate the RT in
  `tests/test_e2e_rt_reference.py`. Meta holds the pRT version (3.4.0, pinned in
  `tests/test_e2e_fixtures.py`), opacity files, constants, generating script and achieved
  agreement; tolerances are 3x achieved; pRT ran in a throwaway venv. Both codes read the
  same k-table file [M]:
  - transmission, isothermal H2O: mean ratio 1.00003, rms 0.0013%; 6 species 1.0010 /
    0.021%; W39b 8 species (Tsai et al. 2023 inputs) 0.986 / 24 ppm on 23,000 ppm.
  - emission, H2O, 900-2200 K, 3-5 um: bias +0.034%, rms 0.020%, worst band 0.091%.
  - transmission vs the analytic grey solution: 0.012-0.016 scale heights over 5-7
    decades.
- The continuum fixtures `tests/data/prt_ref_cont_*.npz` (`tools/prt_continuum_reference.py`, pRT 3.4.0, normal
  tier) cover CIA and Rayleigh, on H2O-poor columns where the continuum sets the windows: 1-5 um, H2O 1e-5,
  800-1800 K, W39b geometry, in transmission; H2O 1e-4, 900-2200 K, in emission; He/H2 0.19384. Both codes read the
  engine's ExoMolOP H2O table and its HITRAN 2011 CIA files (`H2-H2_2011.cia`, `H2-He_2011.cia`, Richard et al.
  2012), which the script converts into pRT's CIA format in a private input tree; the CIA tables pRT ships (its
  `BoRi` files) are not installed on daw, so that data difference is unmeasured. Rayleigh is each code's own: pRT's
  H2 cross-section is Dalgarno & Williams 1962, and exojax's polarizability form with King factor 1.0 is 2.1-3.6 %
  above it for H2 and 5.6 % for He. Measured, engine over pRT over 1.01-4.95 um [M, daw CPU]:
  - CIA, transmission: mean ratio 1.000164, rms 0.0086 %, max 0.024 %; depth difference rms 1.8 ppm, max 7.3 ppm.
    Lines-only on the same column: 1.000166, rms 0.0103 %, so the residual is the lines'. The CIA term itself
    (with minus without) is 297.2 ppm mean in the engine against 297.1 in pRT, differing by rms 1.0 ppm, max 4.5.
  - CIA and Rayleigh, transmission: mean ratio 1.000188, rms 0.0126 %, max 0.036 %; depth difference rms 2.6 ppm,
    max 11.3 ppm, on a Rayleigh term of 161 ppm mean at 1.0-1.2 um. With pRT's cross-sections swapped into the
    engine: rms 0.0081 %, max 0.024 %, the CIA case's numbers, so the extra residual is the cross-section formula,
    not the method.
  - CIA, emission: mean ratio 0.99947, rms 0.052 %, max 0.178 %. Lines-only on the same column: 0.999465, rms
    0.056 %, larger at 1-2 um (0.05-0.07 %) than at 3-5 um (0.014 %). The CIA term (flux change) is -1.829 % mean in
    the engine against -1.830 % in pRT, differing by rms 0.010 %, max 0.066 %.
  - exojax takes the CIA table node at `digitize(nu)` where pRT interpolates linearly; at the band centres the two
    differ by rms 0.11-0.13 %, max 0.57 %, in the CIA coefficient, inside the term agreement above.
  - With the term off in the engine call, each case fails its bars: CIA off gives max 4.4 % (transmission) and
    10.2 % (emission) against bars of 0.073 % and 0.53 %; the H2-He pair alone off gives max 0.175 % and 0.88 %;
    Rayleigh off gives max 1.36 % against 0.109 %. He is about 1 % of the Rayleigh term, too small to gate alone.
- The W39b 8-species case builds its grid to 20 bar, not the fixture's 50 bar: the Tsai et al. 2023 profile reaches
  3507 K at 50 bar (3000 K at 23 bar, 3400 K at 43 bar), outside the opacity window, so the engine refuses it. Those
  layers are opaque in transit: at 20 bar the mean ratio is 0.998289, rms 0.0960 %, max 0.342 % against the 50 bar
  run's 0.998288, 0.0958 %, 0.341 % [M, daw CPU], inside the unchanged bars.
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
  2.5e-8 to 1.3e-3 with it [M]. The maintainer decided the seed stays.
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
  uncapped-dt case in VULCAN3's notes, not a new defect. The free-endpoint jvp test hit it
  once VULCAN3's collision limit counted M: at `dt_max` 1e15 the warm jvp ran to its
  3000-step cap, `tangent_longdy` 1.2e154 (81 steps, 0.070 before that change); at the
  production cap 1e11 it settles in 252 steps, 0.038 (81, 0.070 before) [M, daw CPU]. So
  that test's profile runs 1e11, as the queue profile does.

## Batched routes (`converged_y_batch`, `converged_y_queue`)

- Both agree with the solo solve at the convergence scale only (batch vs per-lane vmap max
  rel 7.2e-3) [M]. A lane is bitwise independent of its neighbours and its position:
  theta 1 of `test_converged_y_batch` at widths 2, 3, 4 and 8, beside itself or theta 0,
  2025 network, arm64 [M] (before VULCAN-JAX `dd2c7a0` a lane's eq seed moved ~4e-15 with
  the width). A batch of ONE is a different XLA program: on arm64 it splits from width 2
  by one ulp in three H2S cells at accept 115 and ends 1758 cells apart, at the
  convergence scale; with the geometry refresh off (`update_frq` 1e6) the two stay bitwise
  for 400 steps [M]. The op is not identified (not the refresh's mu sum, not the
  stage-repair defect sum, each tried as a fixed-order sum). Which CPUs show the split
  varies (arm64 and one CI x86 runner do, daw does not): tests compare a lane against the
  same theta at the batch's width, never a batch of one.
- The cold queue is not bitwise the batch: `run_queue` builds the seed inside its jitted
  loop, and the two compilations of `_prep` differ by 1.1e-15 in `y_ini`, amplified until
  accept counts split (queue [121,122,121,121] vs batch [121,121,121,166]) [M]. Started
  both from `y0` they still split (queue [121,127,122,148] vs batch [123,...]) since the
  carbon repair, and a 1e-12 lnZ nudge moves the queue-vs-batch max rel from 0.4e-2 to
  7e-2 on every job [M]: tests compare `conv_normal` (not `conv_branch`: an S4 cell at
  1e-20 certifies tight in one compilation, loose in another) and the convergence scale
  (bar 0.1, `yconv_min`), and a plain-jvp tangent only at `dt_max` 1e7 (below).
- `converged_y_queue(active=)` (`39c2c45`) lets the retrieval mask padding rows. A masked
  job's lane starts done and takes no step; the active jobs keep the unmasked schedule
  (bitwise on the GPU per the retrieval's check; the CPU test bar is 1e-9).
- Warm-started stage 2: every theta certifies in both arms at different relaxation
  depths; the worst cells are floor sulfur; H2O, CO, H2, He <= 2.8e-4 rel, CO2 and CH4 up
  to 2.0e-2 [M]. The transit-depth effect is unmeasured.
- The queue test profile runs the production cap `dt_max` 1e11 s. Uncapped, queue-vs-batch
  CO read 1.06e-3 (over the 1e-3 bar); capped, 1.26e-4. A tight capped reference puts
  every route 1.2e-3 to 8.4e-3 off in CO: the routes differ by less than their
  convergence error [M].
- A plain tangent at the production cap has no stable value [M, 41 draws, lnZ nudged by
  1e-15 to 7e-12, daw, `run_queue` deterministic on repeats]: queue-vs-batch median 6.7e-5 /
  2.1e-3 / 2.5e-4 / 2.0e-2 on jobs 0-3, max 3.3e-1 / 4.0 / 1.0 / 5.4; the batch moves
  against itself as much (job 1 5.1e-1 under 1e-14), stop counts scatter (121 to 245 on a
  job), and vulcan3 `e91db66`, before the data fixes, shows the same tail, so its earlier
  pass was its bit pattern. CPU count picks the pattern: job 0 reads 8.1e-2 on three CPUs, and
  job 3 reads 2.1e-2 with job 0 at 5.7e-5 on two. At that 8.1e-2 draw the certified tangent
  (1.4e-5) and central FD (2.6e-4 at h 1e-2) side with the batch; the gap is layer 0 (H, H2,
  H2S, SH). `dt_max` 1e9 still reads up to 1.4e-1; `yconv_min` 1e-2 does not help (up to
  3e7). At `dt_max` 1e7, 19 draws read <= 1.5e-5, stops fixed at 146, 127, 155, 124 (the
  queue's refilled jobs 156, 125) [M]. The AD path is capped lower (the planner's
  `AD_BUILD_OVERRIDES`, 1e5 s).
- A plain `jax.jvp` runs through both (batch jvp vs per-theta `jax.jvp(converged_y)`
  1.0e-4). The gradient tests assert column max rel 4e-1, median 1e-3, tangent
  stack-relative 2e-4; measured cold 1.70e-1 / 2.08e-4 / 1.73e-5, warm 1.16e-1 /
  5.74e-5 / 2.56e-5 [M].
- Per-lane `lnZ_ref` / `c_o_ref` are bitwise the scalar path [M].
- Test bars, calibrated on daw (Xeon 8462Y+): queue-vs-batch primal 0.05 (`REL_MAX`);
  batch-vs-solo 0.1 (6e-2 and 8e-2 measured across Python 3.10-3.12 and numpy
  1.26-2.x); queued tangent 1e-2 on every job at `dt_max` 1e7 (<= 1.5e-5 measured; a refill
  that drops a job's tangent reads 1.0, one that takes the next job's 3.9e-1) [M]. GitHub
  runner run 36254404095 (`bdbf2d1`): queue-vs-batch primal 6.87e-2, and the eq-seed warm jvp
  ran to its 3001 cap with the tangent diverged; those tests are in the full tier. GH200 is unverified; the
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
- `s_o`, ln of the S/O multiple, is an optional entry of the positional layout
  (`CHEM_EXTRAS`; `build_chem_model(chem_extras=...)`, `params_from_vector(extras=...)`):
  sulfur's target ratio scales by e^(lnZ + s_o) while O and N keep e^lnZ, the warm guess
  scales the S-bearing species by the s_o increment and the projection repairs the oxygen
  they carry (SO, SO2, OCS, HSO) through H2O; a continuation carries `s_o_ref` beside
  `lnZ_ref` / `c_o_ref`. Without the entry every layout and consumer is unchanged (the
  planner's `ChemParams` has s_o = 0). Gate: `tests/test_converged_y_batch.py`, the
  built column's ratios exact to 1e-9 with positive repair factors from the seed and from a
  warm guess at another sulfur reference. How solves behave at |s_o| = ln 10 is
  unmeasured; the retrieval's v5 smoke is the first measurement [I].

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
  because the column's atoms sit in the bottom scale heights [M, one column, external
  review]; W39b is unchecked. vulcan-retrieval's 97 of 1320 warm-certify, cold-fail
  rows (baseline_v2) are fixed-point path dependence, a different effect.

## Kzz shape

- `kzz_mode` "scale" (the default) multiplies the config's Kzz profile by e^lnKzz;
  "powerlaw" anchors Kzz0 at `kzz_p_ref_bar` (1 mbar by default, log-log interpolated
  on the config profile at the interior interfaces pico[1:-1], where the runner reads
  Kzz) and tilts it by (P / P_ref)^-kzz_slope, the slope an optional layout entry
  (`CHEM_EXTRAS`). Slope 0 is a constant Kzz equal to the profile's value at the
  reference level, so lnKzz keeps its meaning; a slope entry with the scale mode is
  refused at build (a silent no-op otherwise). Pinned by `tests/test_converged_y_batch.py`
  on the prepped `pv.Kzz` (no solve).

## Photolysis refresh cadence

- Config policy, never engine policy: `vulcan_chem` sets neither `ini_update_photo_frq`
  nor `final_update_photo_frq`. The YAML decides (W39b 5/5, other configs 100/5, as
  upstream VULCAN 2); a caller overrides per column through `cfg_overrides`.
- Cadence 1 fixes the planner's TOI-7169 b stall (at cadence 5 one step is rejected 23
  times every cycle against a stale water column; cadence 1 certified at 3422 steps) and
  moves the planner's six presets by <= 3.8 ppm native [M]. Engine-wide cadence 1 was
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

## Provenance audit

The commits that moved the engine away from VULCAN 2's defaults, each live at HEAD and
on by default unless noted. The maintainer decided the certificate terms and the
per-theta re-anchoring stay; the other items stay live as model choices.

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
- `d26e9d3`, `801f7e7`: fixed planet mass under lnR0; the emission half (0.29.0) is the
  maintainer's choice.
- `773837e`: hand-written correlated-k random overlap, T/P clamps, `K_FLOOR` 1e-60.
- `db94d6e`: line-by-line and Mie paths removed (also removed a runtime overwrite of
  MdbHitran gamma_air/n_air; ExoJAX source itself was never modified).
- `aedbcbc`: 12 of 27 molecular masses changed. The mass cancels in the line-opacity
  conversion ("Library and data contract"), so the edit moves no spectrum beyond
  rounding.
- `6aa0cb8`: `warm_count_max` defaults to `count_max` (the cap is since removed,
  "Superseded designs"). `c63815c`: emission uses g(r). `ccdf92f`: exojax 2.6.0.
  `879a99c`: the jvp path also needs a tangent certificate. `55613bc`: `eclipse_flux_tau`
  (opt-in).
- `5e8d43a`, `39c2c45`: the last-axis overlap sort and the queue `active=` mask, both
  bitwise ("RT speed (speed2_rt)", "Batched routes").

## Known and deliberately untracked

- 0.30.0 is on `main` with no TestPyPI release and no tag (not tracked, the maintainer decided): the consumers
  install the commit they pin.
- Where the vertical optical depth never reaches 2/3, `eclipse_flux_tau` takes the grid-bottom radius for that
  ordinate. The engine does not refuse it: a consumer checks `tau_bottom` and refuses a see-through column. On four
  W39b-like columns over 1-15 um with the bottom at 7 bar the least bottom optical depth is 3.4, so no ordinate is
  clamped [M].
- The Planck source, the CIA, Rayleigh and the power-law cloud are evaluated at band centres, not band-averaged: at
  R = 1000 each is smooth across a band [E].
- The 151 commits dated before 2026-10-13 were made on other machines; NAS holds no
  provenance for them beyond "Provenance audit" [M: `git log`].
- The photolysis-cadence switch threshold stays at VULCAN3's derived 1.0 when a profile
  overrides `yconv_min`: only the refresh timing moves, the certificate does not.
- `co_mode` defaults to `fixed_O`, which both consumers pass; `proxy` is kept and no test
  in any repo sets it. The modes differ only in a warm continuation's guess, which the
  elemental projection puts on the exact targets.
- `transmission_depth`, `emission_flux` and `tau_bottom` have no production caller (grep
  of vulcan-retrieval/src and vulcan-jwst-tool/src).
- Importing `vulcan_chem` sets `OMP_NUM_THREADS=1` (`setdefault`, undocumented).
- `ChemParams.tp = ()` defaults to an IndexError [I].
- No engine test sets `run_diag` or `rt_integration` (consumers use both).
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

## Superseded designs

- Sampled line-by-line opacity and the Mie cloud deck: wrong contrast below
  R = 700,000 ("Opacity: correlated-k on ExoMolOP"). Removed in 0.11.0, `db94d6e`.
- Interim HITRAN-built correlated-k mode (`opacity_mode="ckd"`, 0.6.0-0.7.x): right
  machinery (32 ppm rms vs R = 700,000), wrong line data. Removed in 0.8.0, `773837e`.
- Radius anchored at the 7 bar RT grid bottom: depth 26,754 vs 21,381 ppm on W39b.
  Replaced by `p_ref_bar`, `4446e6e` / `e2aacba`.
- `cold_seed` (baseline / eq) and the masks abundance map with `reanchor_atom_ini`: one
  cold start and one exact elemental map replaced them ("Cold start, abundance map,
  carbon repair"). Removed in 0.27.0, `5018e4d`, `d40211d`, `2807b9a`.
- A local copy of the runner's certificate terms in `ConvDiag`: replaced by
  `vulcan3.conv_normal` with an exact switch, `dfa2a0f`.
- `warm_dt`, an exit-dt step seed for warm solves: a wash (stage-2 steps 756 -> 750,
  977 -> 1316, 876 -> 653) [M]. Added `3d43367`, reverted exactly in `7e5ea47`.
- Band tiling of the correlated-k fold (`rt_band_tiles`): 2.5% of the per-lane gradient
  peak; the unrolled molecule loop held the memory. Added `faa580a`, removed in 0.25.0
  `fd88866` for `ckd.fold`'s scan; `integ_warm` went in the same commit.
- `converged_ymix`: no caller. Deleted `a5d2190`.
- `warm_cap` / `warm_count_max`, a step cap for the SMC sampler's mutation solves (it
  rode the carry's `count_max_dyn`): no caller after the SMC removal, both retrieval call
  sites passed `warm_cap=False`. Removed in `19fa8a9` with the unread `ConvDiag` fields
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
  solve certifies itself). Removed in 0.29.0 (`1c68022`); the first real solve compiles
  the runner. `p_ref_emission_bar` as a profile key went in the same commit.

## Do not retry

- RT-stage sort alternatives on the GH200, against the adopted 25.3 ms per spectrum: an
  unrolled searchsorted (24.4 ms, bitwise), index gather (26.9-36.8 ms), compare_all (not
  bitwise), rt_chunk 8-64 (no gain), a CUDA fold prototype (8.2 ms for the fold) [M].
- `jit(overlap)`: 2-7%, below the 10% adoption bar. The last-axis sort layout reads ~10%
  on the CPU and was set aside on that reading; its GH200 gain (4.25x) adopted it.
- A checkpointed body for `ckd.fold`'s scan: 3.48 vs 4.56 GiB per lane but 14.4 vs 9.8 s
  per RT vjp. Take it only if GPU memory, not time, binds.
- Sharing the opacity tables across CPU lanes: the W39b tables are 1.4 GiB against
  ~17.4 GiB per gradient lane, so ~10%; production runs on the GPU.

- Sampled line-by-line below R = 700,000: wrong contrast, non-convergent with more points
  ("Opacity: correlated-k on ExoMolOP"); its keys (`nu_pts`, `broadening`,
  `dit_grid_resolution`, `mie_*`) raise.
- HITRAN line data in any mode; `broadening="h2he"` cannot fix H2O (no H2/He widths).
- Guessing ExoMolOP URLs: 404s; walk the page levels (the site also 403s a default urllib
  user agent).
- First sorted isotopologue or first `.h5` on a page: wrong products.
- Trusting header DOIs or `mol_mass`: placeholders and an inert mass.
- A network error printed as SKIP: SKIP means a page with no k-table link, nothing else.
- `ArtEmisPure.run_ckd` or `ibased`: they lose the interior source or flux at admitted
  depths ("Emission solver and gravity").
- exojax `gravity_profile` or constant g: biased depths ("Emission solver and gravity").
- Anchoring at the bottom layer centre: depth depends on `art_nlayer` (+1.02% at 20
  layers).
- A transit radius with emission gravity, or a single eclipse radius for the whole band
  ("Geometry and anchors").
- The `R_ref^2 + col2` gCMCRT decode: halves the amplitude, a fake 2.2x discrepancy.
- Dividing pRT flux by c: wrong by ~1e18.
- Separate `emission_flux` / `tau_bottom` calls: dtau built twice; `emission_flux_tau`
  builds it once.
- Skipping the dropped molecule instead of folding its zero: not guaranteed bitwise.
- A conservative rebin in `ckd.overlap` to remove the sort-crossing jumps: <= 0.06 ppm at
  R = 100 against data sigmas of tens of ppm, and it would move every spectrum, fixture
  and cache. The maintainer decided to keep the rebin.
- Truncating the RT grid at the chemistry top: worse than the clamp it replaced
  ("Geometry and anchors").
- `art.change_temperature_range`: dead in both modes (exojax reads Tlow/Thigh only in its
  own T-P constructors).
- A `conftest.py` for the test preamble: reorders `vulcan_chem` ahead of the jax-free
  contract test.
- Comparing the ERS CO2 model-grid case without the bundle's exact inputs (T-P source,
  radius anchor, v30 grid): rms 155/199 vs 106/149, not reproducible; the inputs are not
  on daw.
- `np.trapezoid` on numpy 1.x or `np.trapz` on 2.x: use
  `getattr(np, "trapezoid", None) or np.trapz`.
- Photolysis cadence 1 as the engine default: the planner's closure tests fail
  ("Photolysis refresh cadence"). Reconsider only with both closures and the
  reference-accuracy gate satisfied.
- Comparing plain tangents at `dt_max` 1e11, on the same accept count or not: bit-level
  noise reaches O(1) ("Batched routes").
- Queue tests uncapped ("Batched routes"), or an uncapped tight reference: the
  40000-step one never certified (element-budget drift 13-55%).
- An exact accept-count check between the cold queue and the cold batch: each builds its
  own seed ("Batched routes"); start both from one column.
- Shrinking the photo-on gradient tests by nz, directions or thetas: still 362-406 s and
  15.9-18 GB, since compile is ~3/4 of it; split by route instead.
- A tangent-tail mode for `converged_y_jvp`: VULCAN3 rejected the tail; the derivative
  stays `run_jvp`.
