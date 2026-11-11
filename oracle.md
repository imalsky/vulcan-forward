# oracle.md — what vulcan-forward must achieve

This file states the goal of vulcan-forward (this repo) and what counts as success or failure. It
is the yardstick for judging whether the repo is ready to ship. It holds targets and decisions, not
measurements: the evidence lives in notes.md, named here by section heading. Only the maintainer
changes this file. The goals of the consumers are in the project-root `oracle.md`; vulcan3's are in
its own `oracle.md`.

## Goal: one forward model, shared

vulcan-forward exists so that vulcan-retrieval and jwst-transit-authority do not each carry their
own forward model. It turns a parameter vector into a converged vulcan3 chemistry column and then
into a transmission or thermal-emission spectrum with ExoJAX, differentiably. It has no science
goal of its own: it serves the two consumers, and both get the same physics from the same code.

"Correct" radiative transfer means, for now, agreement with petitRADTRANS on the same opacity
files, within the tolerances its tests set.

### Success

- **One copy.** Chemistry driving and radiative transfer live here and nowhere else. A consumer
  holds settings, not a second implementation; a forward-model change lands here.
- **Certified columns.** Every column carries vulcan3's own certificate (`conv_normal`). A column
  that did not certify, or froze at the smallest step, is reported as such, never returned as
  converged.
- **The RT matches petitRADTRANS.** Transmission and emission agree with petitRADTRANS, both codes
  reading the same k-table files, within the bars of `tests/test_e2e_rt_reference.py`, and
  transmission agrees with the analytic grey solution (notes "RT validation and the exojax pin").
  The comparison covers every opacity term a default run uses: lines, collision-induced absorption
  and Rayleigh scattering.
- **A realistic default atmosphere.** Molecular lines are correlated-k on the published ExoMolOP
  tables. H2-H2 and H2-He collision-induced absorption is always on and cannot be switched off.
  H2 and He Rayleigh scattering is on by default in transmission. A cloud is in the spectrum when
  the caller asks for one, and only then. The consumer names its molecules; the engine has no
  default set and drops none silently.
- **The core absorbers are there.** H2O, CO2, CO, CH4 and NH3 are in the radiative transfer
  wherever the band overlaps their absorption, or the caller names the one it leaves out. HCN, H2S
  and SO2 depend on the planet: leaving one out is reported, not refused. A minor species may be
  left out freely.
- **The star covers the photochemistry.** A stellar spectrum that does not reach every photolysis
  threshold of the network is refused.
- **Certified derivatives.** Forward-mode derivatives of the solve pass the certified-tangent
  tests (`tests/test_converged_y_jvp.py`).
- **Batched solves.** A batched or queued solve certifies as a single solve does and agrees with
  it at the convergence scale (notes "Batched routes").
- **Neutral changes are exact.** A change that claims to alter no behaviour reproduces the
  reference with zero difference: jwst-transit-authority's default WASP-39 b case, solved fresh on
  the code before the change, in wavelength, depth, the depth with each leave-one-out molecule
  removed, temperature and pressure (notes "Library and data contract"). A deliberate physics
  change carries new reference files.
- **The ExoJAX pin is deliberate.** The version moves only with the RT and derivative tests and a
  spectrum comparison against the old version on fixed inputs.
- **A well-behaved library.** No data read at import, nothing written to the caller's folder, no
  path taken from its own install folder, and no default planet or column (`tests/test_contract.py`).
  No default network: what belongs to a case, the case states. No table is fetched at run time.
  Unknown settings, missing opacity tables and pressure anchors outside the grid are refused.
- **Settings are visible.** A number that changes a result is a named setting the user can read and
  change in the configuration, with its unit and its reason. It is not a value buried in the code.
  A physical constant or a numerical guard that is fixed on purpose is a named constant with its
  reason beside it.
- **Inside its bounds, or an error.** A band outside 1-15 um, a temperature outside the opacity
  window, or a grid bottom below the opacity tables is refused. Where the code cannot raise, the
  spectrum is NaN, never a clamped value.

### Not modelled, by scope

The model is a hydrogen-dominated atmosphere with C-H-N-O-S gas chemistry, from 1 to 15 um, inside
the temperature window of the opacity tables. It has no alkali metals (Na, K), no H- continuum, no
metal oxides or atoms, no collision-induced absorption other than H2-H2 and H2-He, no Rayleigh
scattering by gases other than H2 and He, and no scattering in emission. The cloud terms are
nuisance opacities, not cloud physics. Where the code can tell that one of these matters, it
refuses; where it cannot, the consumer says so beside its result.

### Failure

- A spectrum from a column that did not certify.
- An absorber, or a continuum term, dropped without an error; a required absorber left out without
  being named.
- Photolysis switched off over part of the spectrum because the stellar file is short.
- Rayleigh scattering off, or a cloud on, when the caller did not ask for it.
- A clamped pressure grid, a clamped temperature, or a see-through column, accepted.
- A default planet, network or column.
- A number that changes a result, hidden in the code where the user cannot see or set it.
- A change that claims to be neutral and moves the reference spectrum.
- An ExoJAX version change without the checks.
- A consumer carrying its own copy of the chemistry driver or the RT.
- A silent fallback or default where the code should refuse (project-root `CLAUDE.md`, "Error
  fast").

## Ready to ship

The repo is ready when, at the commit to be released:

- the normal tier passes locally and in CI, and the full tier passes when the change touches the
  solve or its batched routes;
- the petitRADTRANS gate passes, not skips, on a machine that holds the opacity data;
- the consumers' version floors and pins name released, tested commits;
- no `[owed]` or `[unverified]` item in todo.md backs a claim the release makes, and nothing above
  under Failure applies.
