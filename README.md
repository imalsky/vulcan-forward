# vulcan-forward

vulcan-forward computes exoplanet atmosphere spectra: VULCAN-JAX
photochemical kinetics for the abundances, then ExoJAX radiative transfer for
a transmission or emission spectrum. The whole chain is differentiable. Two
applications build on it; neither depends on the other:

```
vulcan-jax          chemistry kernels and the steady-state solver
      |
vulcan-forward      this package: chemistry driver and radiative transfer
      |
      +-- vulcan-retrieval     atmospheric retrieval (SMC sampler)
      +-- vulcan-jwst-tool     JWST observation planner
```

Modules: `constants` (physics constants + molecule table), `paths` (data
locations), `vulcan_chem` (chemistry driver), `interp_map` (chemistry-to-RT
grid map), `exojax_rt` (opacities, CIA, radiative transfer), `ckd`
(correlated-k core), `exomolop` (published k-table adapter), and
`fetch_exomolop` (offline k-table download).

## Install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vulcan-forward
```

Import `vulcan_chem` before anything from exojax; it sets the import-frozen
network selection and jax float64, and it stops with an error if the order is
wrong. The exojax pin (2.2.3) has both a ceiling (four load-bearing
workarounds, numpy<2) and a floor: forward-mode AD through exojax was
NaN-poisoned before safe_sqrt landed in 2.1.

## Data

Point `VULCAN_FORWARD_DATA` at a directory holding `exojax_linelists/`,
`opacity_cache/`, and `exomolop/` (tens of GB; fetched, never bundled).
Missing files stop the run with the path and the remedy.

```bash
export VULCAN_FORWARD_DATA="$HOME/vulcan/forward-data"
python -m vulcan_forward.fetch_exomolop --molecules H2O,CO2,CO,CH4,SO2
```

## Use

`vulcan_chem.ChemParams(lnZ, c_o, lnKzz, tp)` is the parameter type; every
entry point also accepts the positional vector `[lnZ, c_o, lnKzz, *tp]` for
samplers and forward-mode AD. Planet geometry (`rp_cm`, `gs_cgs`,
`rstar_cm`) is required; there is no default planet.

The radiative transfer is verified against petitRADTRANS 3.4.0 in both
observables, pinned by committed fixtures in `tests/`; the chemistry port is
verified against VULCAN 2.0. Figures: `validation/figures/`.

```bash
python -m pytest tests -q
```

## License

GPLv3, inherited from VULCAN.
