# vulcan-forward

vulcan-forward computes exoplanet atmosphere spectra. It runs
[VULCAN-JAX](https://github.com/imalsky/jax-vulcan) photochemical kinetics to get
the chemical abundances. It then runs
[ExoJAX](https://github.com/HajimeKawahara/exojax) radiative transfer to get a
transmission or emission spectrum. The whole chain is differentiable, so you can
compute derivatives of the spectrum with respect to the model parameters.

Two applications use this engine. Each one depends on this package. Neither one
depends on the other:

```
vulcan-jax          chemistry kernels and the steady-state solver
      |
vulcan-forward      this package: chemistry driver and radiative transfer
      |
      +-- vulcan-retrieval     atmospheric retrieval (SMC sampler)
      +-- vulcan-jwst-tool     JWST observation planner
```

The package name is `vulcan-forward`. The import name is `vulcan_forward`.

## Modules

| Module | Contents |
|---|---|
| `constants` | Physics constants and the default molecule table |
| `paths` | The location of the external data files |
| `vulcan_chem` | The chemistry driver. It computes converged volume mixing ratios from the model parameters |
| `interp_map` | Interpolation from the chemistry pressure grid to the radiative-transfer pressure grid |
| `exojax_rt` | Opacities, collision-induced absorption, and the radiative transfer. It returns a transit depth or an emergent flux |

## Install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vulcan-forward
```

To install from a local copy of the repository:

```bash
pip install --no-deps -e .
```

Use `--no-deps` because vulcan-jax is on TestPyPI and not on PyPI.

## Import order

Import `vulcan_chem` before you import anything from exojax. This order is
necessary. `vulcan_chem` does two things when Python imports it. It sets the
`VULCAN_JAX_*` environment variables that select the reaction network. It also
enables 64-bit floating point in jax. VULCAN-JAX and exojax read both of these
settings only once, at their own first import. A later change has no effect.

Use this order:

```python
from vulcan_forward import constants, vulcan_chem, interp_map, exojax_rt
```

`vulcan_chem` stops with an error if exojax is already imported. It also stops
with an error if `vulcan_jax` is already imported with a different reaction
network. In that condition the package cannot apply the network selection. The run
would then fail at a later step. Its message would name the environment
variable, not the import order.

## Data files

This package needs three sets of data files at run time:

- the HITRAN and ExoMol line lists,
- the offline opacity cache, which holds the cached carbon-monoxide line list,
- the two collision-induced absorption (CIA) tables, for H2-H2 and H2-He.

These files are tens of gigabytes, so the package does not include them. Tell the
package where they are:

```bash
export VULCAN_FORWARD_DATA="$HOME/vulcan/forward-data"
```

The package expects two directories below that root:

```
exojax_linelists/     HITRAN and ExoMol caches (downloaded on first use)
opacity_cache/        cached carbon-monoxide line list, H2-H2 and H2-He CIA tables
```

You can move one directory without moving the other. Set
`VULCAN_FORWARD_LINELISTS` or `VULCAN_FORWARD_OPACITY_CACHE` to do this. An
application with its own configuration can instead call
`vulcan_forward.paths.set_data_root(...)`.

The root does not have to exist yet. A setup command creates the layout with
`vulcan_forward.paths.ensure_layout()`. A model build never creates directories:
it reports a missing one as an error.

The package reads no file until it needs one. If a file or directory is missing,
the package stops and shows the path and the remedy. The H2-He CIA table is
required physics, because helium is about 14% of the atmosphere by number. A
missing H2-He table stops the run. The package never continues without that
absorption.

## Planet geometry is required

`build_rt_model` needs three values in its profile: `rp_cm` (the planet radius in
centimeters), `gs_cgs` (the surface gravity in cm/s^2), and `rstar_cm` (the
stellar radius in centimeters). These values set the transit-depth normalization
and the atmospheric scale height. There is no safe default value, so the function
stops with an error if one is missing.

Earlier versions of this code used WASP-39 b values when a key was absent. A
caller who forgot one key then modeled a different planet. No message reported
the substitution.

## The exojax version

This package requires exactly `exojax==2.2.3`. Four parts of `exojax_rt` depend on
the behavior of that version:

1. **Gravity profile.** `exojax_rt` computes its own gravity profile,
   `g(r) = g_btm (R_btm/r)^2`. The exojax function `gravity_profile` returns a
   value proportional to `1/r` instead, but the exojax height integrator states a
   dependence proportional to `1/r^2`. For WASP-39 b the measured difference in
   transit depth is +1.5 parts per million with the local profile, -101.8 parts
   per million with constant gravity, and -50.8 parts per million with the exojax
   profile.
2. **Deprecated argument.** exojax deprecated the `dit_grid_resolution` argument
   and replaced it with `broadening_resolution`. The accepted format of the new
   argument applies to exojax 2.x only.
3. **Compatibility modules.** Two imports use `exojax.database.api` and
   `exojax.database.contdb`. Both modules state that a future major release will
   remove them.
4. **Broadening data.** The code writes `mdb.gamma_air` and `mdb.n_air` directly
   to apply H2/He line widths. This depends on the attribute names of the exojax
   database object.

exojax 2.2.3 also requires `numpy<2`. This package holds that limit, so
vulcan-jax does not have to. vulcan-jax needs only jax, numpy, scipy and PyYAML,
and users who want chemistry alone install nothing more.

## Scope and known work

The numerical core does not depend on any application. It takes the planet
geometry as arguments, and it contains no sampler concepts.

### Chemistry parameters

`vulcan_chem` has a named parameter type. Use it in new code:

```python
from vulcan_forward.vulcan_chem import ChemParams

params = ChemParams(lnZ=0.0, c_o=0.0, lnKzz=0.7, tp=(1200.0, -1.0))
y = chem.converged_y(params)
```

`lnZ` scales the metals. `c_o` moves carbon at fixed oxygen. `lnKzz` scales the
eddy-diffusion profile. `tp` holds the temperature parameters. The meaning of
`tp` is fixed when you build the model. With a `tp_eval` hook it is that hook's
parameter block. Without one it is a single uniform temperature offset in
kelvin.

Every function also accepts a positional vector, `[lnZ, c_o, lnKzz, *tp]`. Keep
that form for a sampler or for forward-mode automatic differentiation. In those
cases the parameters are a vector, and the tangent must have the same shape.
`params_from_vector` reads a vector into a `ChemParams`, and
`ChemParams.to_vector` writes one back.

The tests in this repository check the packaging rules only. Each test confirms
one rule:

- the package imports with no data installed,
- a missing data path stops the run and shows the remedy,
- an application can replace the molecule table,
- the planet geometry is required,
- the import-order check works.

The two applications validate the physics against real spectra.

## License

GPLv3, inherited from VULCAN.
