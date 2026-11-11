# Method

How the engine goes from chemistry parameters to a spectrum, with the source of each method. Results and the evidence
behind the rules are in `notes.md`.

## Modules

| Module | Job |
|---|---|
| `constants` | Every fixed number, under four headings (physical constants, fixed model choices, numerical tolerances, defaults a caller may override), the default molecule table, the profile-key check |
| `paths` | The data root: k-tables and CIA tables |
| `exomolop` | ExoMolOP k-table ingestion, header checks, provenance |
| `fetch_exomolop` | The offline fetcher |
| `ckd` | (T, P) interpolation of log k and the random-overlap mixture |
| `vulcan_chem` | The import-order preamble, `select_network` and `build_chem_model`, the chemistry driver's setup |
| `_chem_model` | Theta to the converged column, forward-mode differentiable: `_ChemSetup` (what a build fixes, a proposal's initial state and exit report), `_ChemSolves` (every solve entry point) |
| `_chem_layout` | `ChemParams`, the positional layout, `ConvDiag` |
| `_elemental` | The composition masks, the elemental repair, the C/O margin |
| `interp_map` | Chemistry grid to RT grid |
| `exojax_rt` | The transmission and emission models |
| `_rt_geometry` | Radius and gravity anchors, layer boundaries, the tau = 2/3 level |
| `_rt_opacity` | The optical depth: lines, CIA, Rayleigh, clouds |

## Chemistry

`build_chem_model` runs VULCAN 3 (a JAX port of VULCAN; Tsai et al. 2021) once for its pre-loop, then re-converges the
column per proposal. Rate constants and the T- and composition-dependent structure ($D_{zz}$, $v_m$, $v_s$, $K_{zz}$, the
initial carry geometry) are rebuilt on-graph. T-dependent photolysis cross sections are refused at build, because the
rebuild does not follow them.

### Parameter vector

`ChemParams(lnZ, c_o, lnKzz, tp, s_o, kzz_slope)` is the named form. `params_from_vector` and `ChemParams.to_vector` convert
to and from the positional layout `[lnZ, c_o, lnKzz, *extras, *tp]`, where the extras are the `chem_extras` the model was
built with (a subset of `CHEM_EXTRAS`, in its order; none by default). Without a `tp_eval` hook the tail is one entry, a
uniform temperature offset in K. A warm continuation carries three references: `lnZ_ref`, `c_o_ref`, `s_o_ref`.

### Composition map

The theta targets are column ratios to elemental H: He fixed, O and N times $Z$, C times $Z e^{c_o}$, S times
$Z e^{s_o}$. A cold solve starts from the network's Gibbs equilibrium at the proposal's T-P and ratios; a warm solve from
a converged column with only the composition increments applied through masks. Either way the guess goes through the
repair: renormalize to $\sum_i n_i = M$ per layer, solve the linear system for one common factor per element's adjuster
species (He; H2O; CO and CH4; N2; H2S), and repeat. The residual is about $10^{-15}$ relative; `audit_init` measures it.

### Fixed-O C/O guess

In `co_mode="fixed_O"` a warm guess scales every C-bearing species by $e^c$ and the O-only carriers by
$b_z = 1 + (1 - e^c) O^C_z / O^O_z$, keeping each layer's O total. $b_z > 0$ only while
$c < \ln(1 + \min_z O^O_z / O^C_z)$, which `co_bz_bound` and `co_bz_margin` report.

### Kzz

`kzz_mode="scale"` multiplies the config's profile by $e^{\ln K_{zz}}$. `"powerlaw"` reads $K_{zz,0}$ off the profile at
`kzz_p_ref_bar` and uses $K_{zz,0} e^{\ln K_{zz}} (P / P_{\rm ref})^{-\alpha}$.

### Condensation

Condensation follows the live T(P): `_prep` rebuilds every T-dependent condensation array per proposal. The cold-trap
index and the active-layer set are discrete, so a jvp through a condensing state is valid only away from those switches.
A condensing model does not reduce to a condensation-free one when nothing condenses: the `fix_species` pin freezes the
reservoirs at their `stop_conden_time` state. Enable it only where the species condenses.

### Derivatives

The runner's `lax.while_loop` supports jvp, not vjp, so forward mode is the end-to-end route. `converged_y_jvp` holds the
tangent to the same change-over-lookback tolerance as the column.

## Opacity

ExoMolOP (Chubb et al. 2021) publishes k-tables built from the ExoMol and HITEMP high-temperature line lists with H2/He
broadening applied; HITRAN's 296 K tabulation and air broadening read window opacity low in a hot hydrogen atmosphere. Its
petitRADTRANS-format tables (Mollière et al. 2019), about 389 MB per species, are read as they are:

- The quadrature is petitRADTRANS' split scheme, 8 Gauss-Legendre points on [0, 0.9] plus 8 on [0.9, 1]. Nodes and weights
  come from the file; nothing downstream assumes 16-point Gauss-Legendre.
- The pressure grid stops at $10^{-5}$ bar while the RT column runs to $10^{-9}$ bar. Layers above reuse the $10^{-5}$ bar
  entry (lines are Doppler-dominated there; petitRADTRANS does the same). `load_tables` logs this.
- `kcoeff` is cm$^2$ per molecule, the convention `opacity_profile_xs_ckd` consumes. `_header` refuses other units.

exojax (Kawahara et al. 2022) computes cross sections directly on the output grid and warns below $R = 700\,000$, so a
JWST-band model at $R$ near 1500 is biased; correlated-k integrates once, offline.

### Mixtures

exojax's `opacity_profile_xs_ckd` takes one species. Tables are per species because the composition changes every run, so
they are combined by random-overlap resort-rebin (Amundsen et al. 2017): every pairwise sum with weight $w_i w_j$, sorted,
the weight accumulated into a cumulative $g$, re-interpolated onto the $n_g$ grid. It assumes the two absorbers' line
positions are uncorrelated within a band. It is written in JAX, so tangents pass through.

### Continua

CIA (H2-H2, H2-He, always on), H2 and He Rayleigh in transmission only (on unless `use_rayleigh` is false; it is
scattering, and the emission solver is pure absorption), exojax's `powerlaw_clouds`, $\kappa(\nu) = \kappa_0 (\nu / \nu_0)^\alpha$ per gram of atmosphere, and a grey deck below a free
pressure, in transmission and in emission (in emission the opaque layers radiate at the temperature of the level they
sit at). The power law is a uniformly mixed nuisance, not a physical cloud. All are smooth across a band, so their band
value adds identically to every g-ordinate.

### Bounds

The band must lie inside 1-15 um (`constants.WIDE_BAND_NU_MIN`, `WIDE_BAND_NU_MAX`): shortward of 1 um the Na and K
resonance lines and hazes set the spectrum, and the model has neither. Every RT temperature must lie inside
`constants.T_OPA_MIN_K` to `T_OPA_MAX_K` (200-3000 K, the span both CIA tables cover; the build refuses installed
tables that do not), because the k-table and CIA interpolations clamp at their edges. This window bounds the opacity
only: the chemistry's equilibrium start is known to break down below about 225 K, where a solve raises or comes back
uncertified. A column
outside the window raises when the call is eager; under `jax.jit` nothing can raise, so the spectrum comes back NaN and
the caller checks it. The same holds for a mean molecular weight below that of atomic hydrogen and for a mixing ratio
outside [0, 1]. Planet radius, gravity, stellar radius, the grid bottom `art_pbtm_bar` and the reference pressure
`p_ref_bar` have no default and must be finite and positive. The grid bottom may not lie
below the k-table pressure ceiling; layers above the tables' lowest pressure reuse its entry, as petitRADTRANS does.

The core absorbers are H2O, CO2, CO, CH4, NH3, HCN, H2S and SO2, the main carriers of oxygen, carbon, nitrogen and
sulfur in a hydrogen-dominated atmosphere (Moses et al. 2013), seven of them measured in transmission with JWST
(Rustamkulov et al. 2023; Bell et al. 2023; Welbanks et al. 2024; Fu et al. 2024), and C2H2, OCS and SO, which reach
tens to hundreds of ppm in this model's own columns. `constants.CORE_MOLECULE_BANDS_UM`
lists where each absorbs: H2O everywhere, the others where the table's R = 100 band-mean cross-section at 1000 K and
1 mbar is at least 1 % of its peak. Where the band overlaps a molecule's range and the molecule is not in the RT, the
build refuses H2O, CO2, CO, CH4 and NH3 (`constants.REQUIRED_MOLECULES`) and warns for the other six, which depend
on the planet; the warned names come back as `unlisted_core_molecules` for the consumer to report. A caller that
leaves one out on purpose names it in `omit_core_molecules`. A molecule outside the core set may be left out freely.

### Fetching tables

`fetch_exomolop` resolves URLs by walking ExoMolOP's three levels of pages, because filenames follow no one pattern. The
isotopologue must be the principal one, and an unknown naming form raises. The dataset ExoMol marks recommended wins;
with none marked, or several, the last one listed wins. Within it the natural-abundance file wins over the principal
isotopologue, because VULCAN tracks a total molecular VMR. Only the R1000 0.3-50 um product is accepted, since all
tables in a mixture must share a band grid.

## Grid map

`interp_map` interpolates linearly in log pressure with `jnp.interp`, so tangents pass. It is not column- or
mass-conservative and can smear photochemical transitions sharper than the ART layer spacing; the consumers' vertical-grid
ladders check that neither matters at the quoted precision.

## Geometry

exojax defines `radius_btm` and `gravity_btm` at the lower boundary of the bottom layer, half a log-layer below the deepest
centre. A published planet radius is the transit radius, near the millibar photosphere. Handing exojax the literature pair
stacks the whole column on top of a radius that already is the photospheric one.

Hydrostatic equilibrium at fixed GM, with $g(r) = g_{\rm ref} (r_{\rm ref} / r)^2$, is

$$\frac{dr}{d\ln P} = -C(P) r^2, \qquad C = \frac{k T}{\mu m_u g_{\rm ref} r_{\rm ref}^2},$$

and $u = 1/r$ linearizes it: $u(P) = u_{\rm ref} + \int_{\ln P_{\rm ref}}^{\ln P} C \, d\ln P$. No ODE solve is needed and the
result stays differentiable. The integration nodes extend half a layer past both end centres, holding the end layer's T
and mean molecular weight.

Gravity inside the column is inverse-square at layer midpoints, the law exojax's height integrator uses, so opacity columns
and heights share one gravity.

### Emission anchor

The slant path through a hot Jupiter's limb is 35-90 times the vertical one (Fortney 2005), so transmission sees near a
millibar while the dayside photosphere sits near 0.1 bar. Using the transit radius and gravity for emission biases
planet-to-star flux ratios by 5% typically and 10-25% for low-gravity hot Jupiters (Fortney et al. 2019). 0.1 bar is HyDRA's
convention (Gandhi & Madhusudhan 2018). `eclipse_flux_tau` goes further and applies the radius at vertical
$\tau = 2/3$ per k-ordinate.

## Emission solver

`ArtEmisPure.run_ckd` hard-codes `rtrun_emis_pureabs_ibased`, which has no bottom-boundary term, so every photon entering
the grid from below is lost. `_run_emis_ckd_linsap` is upstream's flatten-solve-reweight with `ibased_linsap` in its place.
The flatten is g-major and band-minor, matching `jnp.tile`'s tiling of the source: element $(g, b)$ lands at flat index
$g n_b + b$ on both sides. The g-weights multiply the flux, because the k-distribution reorders wavenumber within a band.

## Glossary

| Symbol | Meaning |
|---|---|
| `lnZ`, `c_o`, `s_o` | ln of the metallicity, C/O and S/O multiples of the baseline |
| `lnKzz`, `kzz_slope`, `Kzz0` | ln of the eddy-diffusion multiple, the power-law slope, the baseline $K_{zz}$ |
| `Dzz` | Molecular diffusion coefficient |
| `T`, `T_art`, `Ti` | Temperature on the chemistry grid, on the ART grid, at interfaces |
| `P`, `dP`, `logP_*`, `lnp_*` | Pressure, a layer's pressure thickness, log10 pressure, natural-log pressure |
| `M`, `Mn` | Total number density per layer |
| `Hp_i`, `Hpi_i` | Scale height at cells and at interfaces |
| `A0`, `A`, `B`, `Bs`, `E_np`, `Msys` | Column element totals and the matrices of the elemental repair |
| `R0_*` | Baseline column ratios to H |
| `OC_z`, `OO_z` | Per-layer O in C-bearing species and in O-only species |
| `lnR0`, `lnM` | ln of the radius and mass multiples, RT-only |
| `Rp*` | Planet radius |

## References

- Amundsen, D. S., et al. 2017, A&A, 598, A97, doi:10.1051/0004-6361/201629322
- Chubb, K. L., et al. 2021, A&A, 646, A21, doi:10.1051/0004-6361/202038350
- Fortney, J. J. 2005, MNRAS, 364, 649, doi:10.1111/j.1365-2966.2005.09587.x
- Fortney, J. J., Lupu, R. E., Morley, C. V., Freedman, R. S., & Hood, C. 2019, ApJL, 880, L16, arXiv:1904.00025
- Gandhi, S., & Madhusudhan, N. 2018, MNRAS, 474, 271, arXiv:1710.06433
- Heng, K., & Kitzmann, D. 2017, MNRAS, 470, 2972, doi:10.1093/mnras/stx1453 (the test of the cloud deck)
- Kawahara, H., et al. 2022, ApJS, 258, 31
- Mollière, P., et al. 2019, A&A, 627, A67
- Tsai, S.-M., et al. 2021, ApJ, 923, 264
