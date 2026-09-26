"""Shared physics constants for the VULCAN-JAX -> ExoJAX forward model.

Pure constants (plus the profile-key check): NO heavy imports (no jax, no
vulcan_jax, no exojax) and NO filesystem access, so this module is safe to
import before the env-order-sensitive VULCAN-JAX setup runs and safe to import
on a machine with no data installed. Data locations live in ``paths.py``;
application-specific settings (planet geometry, run profiles,
parameter-vector layouts) belong to the consumer, not here.
"""
from __future__ import annotations

# VULCAN-JAX network selection
# These must be in the environment BEFORE the first ``import vulcan_jax``
# (VULCAN-JAX freezes network/atom_list at first import). ``vulcan_chem``
# applies them; see its module docstring for the import-order contract.
DEFAULT_NETWORK = "thermo/SNCHO_photo_network.txt"
DEFAULT_ATOM_LIST = "H,O,C,N,S"
# Default ``vulcan_jax.load_config`` name. A caller normally passes
# ``profile["vulcan_cfg_name"]`` instead; this is only the fallback.
DEFAULT_CFG_NAME = "W39b"

# Composition tables: mirrors of vulcan_jax.composition (this module stays
# import-light); vulcan_chem checks them at build time and raises on drift.
#
# Index of each element column we touch when building the Z / C/O knobs.
ATOM_COLS = {"H": 0, "O": 1, "C": 2, "He": 3, "N": 4, "S": 5}

# Molar masses (g/mol) for every compo_array column, in column order. Used to
# turn a VMR profile into a mean-molecular-weight profile
# (compo_array @ this vector).
# atom_list = (H,O,C,He,N,S,P,Na,K,Si,Fe,Ar,Ti,V,Mg,Ca,e)
ATOMIC_MASSES = [
    1.008, 15.999, 12.011, 4.0026, 14.007, 32.06, 30.974, 22.990,
    39.098, 28.085, 55.845, 39.948, 47.867, 50.942, 24.305, 40.078, 5.4858e-4,
]

# Opacity / radiative-transfer defaults
# Reference wavenumber (cm^-1) for the ExoJAX powerlaw_clouds retrieval cloud:
# kappa(nu) = kappac0 * (nu/CLOUD_NUC0)^alphac, kappac0 in cm^2 per gram of
# atmosphere (pRT convention; alphac = 0 is a gray cloud). 2857 cm^-1 = 3.5 um.
# This is a uniformly mixed power-law opacity ("haze slope + gray deck"
# nuisance), NOT a physical cloud: no cloud-top pressure, condensation,
# sedimentation, or patchiness.
CLOUD_NUC0 = 2857.0

# ART pressure bounds (bar). Chemistry and RT share the model top (vulcan_chem
# sets P_t from art_ptop_bar; interp_map refuses an uncovered ART grid). At
# 1e-9 bar the top is converged: one decade higher moves the R=100 depth by
# ~1 ppm (vulcan-retrieval validation/top_pressure_ladder).
ART_PTOP_BAR = 1.0e-9   # model top
ART_PBTM_BAR = 7.0      # grid bottom

# Pressure (bar) at which rp_cm / gs_cgs apply: a published transit radius
# belongs near the transmission photosphere, not the RT grid bottom where
# exojax defines radius_btm. On WASP-39 b, re-anchoring at 1 mbar reproduces the
# JWST ERS 3.0-5.5 um median depth to 0.4%. Override with profile["p_ref_bar"].
P_REF_BAR = 1.0e-3

# The emission column probes MUCH deeper than the limb: the slant path is ~35-90x
# the vertical one (Fortney 2005), so transmission sees ~mbar while the dayside
# photosphere sits near 0.1 bar. Using the transit radius/gravity for emission
# biases planet-to-star flux ratios by ~5% typically and 10-25% for low-gravity
# hot Jupiters (Fortney, Lupu, Morley, Freedman & Hood 2019, ApJL 880, L16).
# 0.1 bar is HyDRA's stated convention, "the mean pressure of the tau=1 surface"
# (Gandhi & Madhusudhan 2018). This only re-anchors the column GRAVITY; the
# fully correct treatment computes a wavelength-dependent radius at vertical
# tau = 2/3, as POSEIDON and PLATON II do: exojax_rt's eclipse_flux_tau.
P_REF_EMISSION_BAR = 1.0e-1

# Physical constants and unit factors, cgs, local so this module stays
# stdlib-only. K_B_CGS is CODATA 2018 (1.380649e-16). The chemistry uses
# vulcan_jax.phy_const.kb = 1.38064852e-16 (CODATA 2014), upstream VULCAN's
# value, so its rates match the reference code; the two differ by 3.5e-7
# relative.
K_B_CGS = 1.380649e-16      # Boltzmann constant, erg/K
M_U_CGS = 1.66053906660e-24  # atomic mass unit, g (mmw is in amu)
BAR_CGS = 1.0e6              # dyn/cm^2 per bar
UM_PER_CM = 1.0e4            # micron per cm: wavelength (um) = UM_PER_CM / wavenumber (cm^-1)

# Temperature window the consumers treat as the valid RT range: the retrieval
# derives its T-P prior window from these (draws outside are rejected, never
# clipped). The ExoMolOP k-tables themselves span 100-3400 K and are clamped
# at their edges by ckd._interp_logk.
T_OPA_MIN_K = 300.0
T_OPA_MAX_K = 3000.0

# Supported wide band, 1-15 um, in wavenumber (cm^-1). The short edge is set by
# the H2-H2 CIA table (stops at 1 um / 10000 cm^-1); the k-tables reach 50 um.
# Consumers that want the full supported window start from these.
WIDE_BAND_NU_MIN = 667.0     # 15 um
WIDE_BAND_NU_MAX = 10000.0   # 1 um

# Molecule table: VULCAN species name and molar mass (g/mol). The opacity is
# the ExoMolOP k-table <MOL>.ktable.h5 (provenance: exomolop.provenance()).
# molmass is the only mass the engine uses (the tables' mol_mass header is not
# read; ExoMolOP's NO file carries 46) and each entry equals its formula mass
# (tests/test_contract.py). Tables are the principal isotopologue except CO and
# CO2 (natural abundance); paired with the total VMR that is a <= ~2% opacity
# deficit on the multi-carbon species, below the tables' accuracy. Callers may
# pass their own table via profile["molecule_table"].
MOLECULES = {
    "CO":  {"vulcan": "CO",  "molmass": 28.010},
    "H2O": {"vulcan": "H2O", "molmass": 18.015},
    "CO2": {"vulcan": "CO2", "molmass": 44.009},
    "CH4": {"vulcan": "CH4", "molmass": 16.043},
    "SO2": {"vulcan": "SO2", "molmass": 64.058},
    # High-C/O + sulfur discriminators: C2H2/HCN carry the signal near C/O ~ 1,
    # H2S is the reduced-S reservoir.
    "HCN":  {"vulcan": "HCN",  "molmass": 27.026},
    "C2H2": {"vulcan": "C2H2", "molmass": 26.038},
    "H2S":  {"vulcan": "H2S",  "molmass": 34.076},
    # Cool-planet nitrogen carrier (e.g. WASP-107b-class).
    "NH3":  {"vulcan": "NH3",  "molmass": 17.031},
    # Second equilibrium sulfur carrier (nu3 band ~4.85 um, inside G395H and
    # PRISM). The SNCHO network names the species COS (consumers alias the
    # token).
    "OCS":  {"vulcan": "COS",  "molmass": 60.070},
    # Photochemical sulfur carrier and a CH4-destruction product. ExoMolOP
    # publishes NO k-table for either (CS2: no ExoMol line list; C2H6: page but
    # no petitRADTRANS file), so they are listed for completeness and refused
    # at load with the fetch hint.
    "CS2":  {"vulcan": "CS2",  "molmass": 76.131},
    "C2H6": {"vulcan": "C2H6", "molmass": 30.070},
    # Simple hydrocarbon: photochemical CH4-destruction product.
    "C2H4": {"vulcan": "C2H4", "molmass": 28.054},
    # Radicals in VULCAN's SNCHO network; the published Tsai et al. 2023
    # WASP-39 b output carries SH and SO.
    "OH": {"vulcan": "OH", "molmass": 17.007},
    "SH": {"vulcan": "SH", "molmass": 33.068},
    "SO": {"vulcan": "SO", "molmass": 48.059},
    # ExoMolOP's NO table is built from HITEMP (14N-16O__HITEMP.R1000_0.3-50mu).
    "NO": {"vulcan": "NO", "molmass": 30.006},
    # Second-tier species: listed so the menu is complete;
    # jwst-transit-authority's EXTRA_MOLECULES_DEFAULT picks the defaults.
    # Absent because ExoMolOP cannot supply them: O2 (another band grid), CH3OH
    # (no petitRADTRANS file), HSO (no published list), S2 (no IR dipole), ...
    "NS":   {"vulcan": "NS",   "molmass": 46.067},
    "CH3":  {"vulcan": "CH3",  "molmass": 15.035},
    "NH":   {"vulcan": "NH",   "molmass": 15.015},
    "CN":   {"vulcan": "CN",   "molmass": 26.018},
    "H2CO": {"vulcan": "H2CO", "molmass": 30.026},
    "CS":   {"vulcan": "CS",   "molmass": 44.071},
    "N2O":  {"vulcan": "N2O",  "molmass": 44.013},
    "CH":   {"vulcan": "CH",   "molmass": 13.019},
    "C2":   {"vulcan": "C2",   "molmass": 24.022},
    "H2O2": {"vulcan": "H2O2", "molmass": 34.014},
}

# Bulk gas used for CIA + the dominant background (H2).
BULK_H2_VULCAN = "H2"

# Every top-level profile key the engine reads. Consumers pass ONE dict to
# build_chem_model, build_rt_model and build_emis_model, so this is the union.
PROFILE_KEYS = frozenset({
    # build_chem_model
    "vulcan_cfg_name", "use_photo", "yconv_cri", "yconv_min", "nz",
    "count_min", "count_max", "warm_count_max", "dt_max", "cfg_overrides",
    "skip_warmup", "co_mode",
    # build_rt_model / build_emis_model (art_ptop_bar also sets the chemistry top)
    "molecules", "molecule_table", "nu_min", "nu_max", "opacity_mode",
    "art_nlayer", "art_ptop_bar", "art_pbtm_bar", "rt_integration",
    "use_rayleigh", "rp_cm", "gs_cgs", "rstar_cm", "p_ref_bar",
    "p_ref_emission_bar",
})


def check_profile_keys(profile) -> None:
    """Refuse profile keys the engine does not read: a misspelled or retired
    key would otherwise be a silent no-op. ``cfg_overrides`` contents are not
    checked here."""
    unknown = sorted(set(profile) - PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"profile keys {unknown} are not read by vulcan-forward (misspelled, "
            "or retired with a removed feature), so they would change nothing. "
            f"Drop or correct them. Known keys: {sorted(PROFILE_KEYS)}.")
