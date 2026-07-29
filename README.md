# vulcan-forward

The shared forward-model engine: live [VULCAN-JAX](https://github.com/imalsky/jax-vulcan)
photochemical kinetics chained into an [ExoJAX](https://github.com/HajimeKawahara/exojax)
transmission or emission spectrum, differentiably.

This package exists so that the two applications built on it are siblings rather
than a chain. Both depend on this engine and on VULCAN-JAX; neither depends on
the other:

```
vulcan-jax          chemistry kernels + steady-state solver
      |
vulcan-forward      this package: chemistry driver + radiative transfer
      |
      +-- vulcan-retrieval     SMC / forward-mode-MALA atmospheric retrieval
      +-- vulcan-jwst-tool     JWST observation planner
```

It imports as `vulcan_forward`.

## Modules

| Module | Contents |
|---|---|
| `constants` | shared physics constants and the default molecule/opacity table |
| `paths` | the external data-root contract (line lists, opacity cache, CIA tables) |
| `vulcan_chem` | chemistry driver: parameters to converged volume mixing ratios, forward-mode differentiable |
| `interp_map` | chemistry grid to RT grid log-pressure interpolation |
| `exojax_rt` | opacities, CIA, `ArtTransPure` / `ArtEmisPure`, transit depth or emergent flux |

## Install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vulcan-forward
```

From a checkout, alongside its siblings:

```bash
pip install --no-deps -e .
```

`--no-deps` because vulcan-jax lives on TestPyPI rather than PyPI.

## Import order is load-bearing

`vulcan_chem` must be imported before anything from exojax. It sets the
`VULCAN_JAX_*` import-frozen environment variables and enables jax float64, all
of which are read once at first import. Arriving late cannot be worked around
silently, so the module raises instead:

```python
from vulcan_forward import constants, vulcan_chem, interp_map, exojax_rt
```

The same guard also fires when `vulcan_jax` was imported first with a
conflicting network, which used to fail much later with a message blaming the
environment variable rather than the import order.

## Data

Line lists, the offline opacity cache, and the two CIA tables are tens of
gigabytes and are never bundled. Point the engine at them:

```bash
export VULCAN_FORWARD_DATA="$HOME/vulcan/forward-data"     # holds:
#   exojax_linelists/     HITRAN + ExoMol caches (downloaded on first use)
#   opacity_cache/        cached CO ExoMol tree, H2-H2 and H2-He CIA tables
```

Individual trees can be relocated with `VULCAN_FORWARD_LINELISTS` and
`VULCAN_FORWARD_OPACITY_CACHE`, or set programmatically with
`vulcan_forward.paths.set_data_root(...)` by an application that owns its own
configuration surface.

Nothing touches the filesystem until a path is actually needed, and then it
fails with the offending value and the remedy. The H2-He CIA table is required
physics (helium is about 14% by number): a missing file raises rather than
silently dropping a real continuum term.

## Planet geometry is required

`build_rt_model` requires `rp_cm`, `gs_cgs`, and `rstar_cm` in its profile.
These set the transit-depth normalization and the hydrostatic scale, so there is
no safe default. They previously fell back to WASP-39 b constants, which meant a
caller who forgot one silently modeled a different planet.

## The exojax pin

`exojax==2.2.3` is pinned, and the pin is load-bearing in four independent
places:

- **Gravity profile.** `exojax_rt._gravity_profile_invsq` computes
  `g(r) = g_btm (R_btm/r)^2` locally, because exojax's own `gravity_profile`
  returns a value linear in `1/r` while its height integrator documents a
  `1/r^2` dependence. On WASP-39 b the correction is measured at +1.5 ppm
  against a reference, versus -101.8 ppm for constant gravity and -50.8 ppm for
  exojax's profile.
- **Deprecated keyword.** `dit_grid_resolution` is deprecated in favor of
  `broadening_resolution`, whose accepted dict shape is 2.x-only.
- **Compatibility shims.** Two imports go through `exojax.database.api` and
  `exojax.database.contdb`, both of which announce removal in a future major
  release.
- **Broadening.** H2/He widths are applied by writing `mdb.gamma_air` and
  `mdb.n_air` directly, which depends on the snapshot attribute surface.

exojax 2.2.3 itself caps `numpy<2`. Isolating that constraint here, rather than
in the chemistry core, is a main reason this package is separate: VULCAN-JAX
stays lean (jax, numpy, scipy, PyYAML) for the users who only want chemistry.

## Scope and known work

The numerical core is theta-free and geometry-parameterized, but one piece of
its calling convention is inherited rather than designed: `vulcan_chem`'s public
entry points take a **positional parameter vector**, unpacking
`theta[0:3]` as `[lnZ, c_o, lnKzz]` and `theta[3:3+n_tp]` as the temperature
block. That is the retrieval's contract, and a library should instead offer
keyword or dataclass parameters with the vector form as a consumer-side adapter.
Changing it is additive and planned; it was kept out of the extraction commit so
that the move could be proven not to change any spectrum.

Physics validation lives with the consumers, which exercise this engine against
real spectra and against PandExo; this package's own suite covers the packaging
contracts (importable with no data, loud path failures, injectable molecule
table, required geometry, import-order guard).

## License

GPLv3, inherited from VULCAN.
