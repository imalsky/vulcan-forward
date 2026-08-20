# vulcan-forward

`vulcan-forward` is the shared forward model that I'm using for different VULCAN work. It runs
[VULCAN-JAX](https://github.com/imalsky/jax-vulcan) chemistry and then uses
[ExoJAX](https://github.com/HajimeKawahara/exojax) radiative transfer to make
a transmission or thermal-emission spectrum. I separated it here because I have
several projects that need this forward model (a retrieval and a JWST
observation planning tool).

Two applications use this package:

```text
vulcan-jax          chemical kinetics
    |
vulcan-forward      chemistry driver and radiative transfer
    |
    +-- vulcan-retrieval    atmospheric retrieval
    +-- vulcan-jwst-tool    JWST observation planning
```

The model supports line-by-line opacity and ExoMolOP correlated-k tables. It
includes molecular absorption, H2-H2 and H2-He collision-induced absorption,
Rayleigh scattering, and optional cloud opacity. The main model parameters are
metallicity, C/O, eddy diffusion, and temperature-profile parameters.

## Install

```bash
python -m pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  vulcan-forward
```

The package pins ExoJAX 2.2.3. Do not replace this version without rerunning
the radiative-transfer and derivative tests.

Import `vulcan_chem` before ExoJAX. This sets the chemistry network and enables
64-bit JAX calculations before those settings become fixed.

```python
from vulcan_forward import constants, vulcan_chem, interp_map, exojax_rt
```

## Data

Opacity files are not included in the Python package. Set the data directory:

```bash
export VULCAN_FORWARD_DATA="$HOME/vulcan/forward-data"
python -m vulcan_forward.fetch_exomolop \
  --molecules H2O,CO2,CO,CH4,SO2
```

The directory can contain `exomolop/`, `exojax_linelists/`, and
`opacity_cache/`. Missing data cause a clear error; the model does not silently
omit an absorber.

The model also needs planet geometry. Supply the planet radius and reference
gravity, both quoted at the reference pressure, and the stellar radius. There
is no default planet.

## Validation and limits

The tests compare transmission and emission calculations with
[petitRADTRANS](https://doi.org/10.1051/0004-6361/201935470), and compare the
chemistry with VULCAN. The committed figures are in
[`validation/figures/`](validation/figures/).

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

The result is only as complete as the selected reaction network, opacity
tables, pressure grid, and cloud model. Check wavelength and temperature
coverage before using a new molecule or atmosphere. Condensation is not
validated for gradient inference.

## Papers and citation

Published work should cite the components used by the run:

- VULCAN: [Tsai et al. (2017)](https://doi.org/10.3847/1538-4365/228/2/20)
  and [Tsai et al. (2021)](https://doi.org/10.3847/1538-4357/ac29bc)
- ExoJAX: [Kawahara et al. (2022)](https://arxiv.org/abs/2105.14782) and
  [Kawahara et al. (2025)](https://arxiv.org/abs/2410.06900)
- ExoMolOP tables: [Chubb et al. (2021)](https://doi.org/10.1051/0004-6361/202038350)
- FastChem initialization: [Stock et al. (2018)](https://doi.org/10.1093/mnras/sty1531)

Also record the package versions, opacity sources, reaction network, and model
configuration.

## License

`vulcan-forward` is released under GPLv3.
