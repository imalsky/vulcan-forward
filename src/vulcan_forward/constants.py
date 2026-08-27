"""Shared physics constants for the VULCAN-JAX -> ExoJAX forward model.

Pure constants only: NO heavy imports (no jax, no vulcan_jax, no exojax) and NO
filesystem access, so this module is safe to import before the env-order-
sensitive VULCAN-JAX setup runs and safe to import on a machine with no data
installed. Data locations live in ``paths.py``; application-specific settings
(planet geometry, run profiles, parameter-vector layouts) belong to the
consumer, not here.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# VULCAN-JAX network selection
# ---------------------------------------------------------------------------
# These must be in the environment BEFORE the first ``import vulcan_jax``
# (VULCAN-JAX freezes network/atom_list at first import). ``vulcan_chem``
# applies them; see its module docstring for the import-order contract.
DEFAULT_NETWORK = "thermo/SNCHO_photo_network.txt"
DEFAULT_ATOM_LIST = "H,O,C,N,S"
# Default ``vulcan_jax.load_config`` name. A caller normally passes
# ``profile["vulcan_cfg_name"]`` instead; this is only the fallback.
DEFAULT_CFG_NAME = "W39b"

# ---------------------------------------------------------------------------
# Composition tables (hardcoded mirrors of vulcan_jax.composition)
# ---------------------------------------------------------------------------
# This module stays import-light (no vulcan_jax), so both positional tables are
# mirrors of the package's composition metadata. ``vulcan_chem`` verifies them
# against ``vulcan_jax.composition`` at build time and raises on any drift --
# that check is the guard against these two distributions diverging, so never
# weaken it.
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

# ---------------------------------------------------------------------------
# Opacity / radiative-transfer defaults
# ---------------------------------------------------------------------------
# Reference wavenumber (cm^-1) for the ExoJAX powerlaw_clouds retrieval cloud:
# kappa(nu) = kappac0 * (nu/CLOUD_NUC0)^alphac, kappac0 in cm^2 per gram of
# atmosphere (pRT convention; alphac = 0 is a gray cloud). 2857 cm^-1 = 3.5 um.
# NOTE this is a uniformly mixed power-law opacity ("haze slope + gray deck"
# nuisance), NOT a physical cloud: no cloud-top pressure, condensation,
# sedimentation, or patchiness.
CLOUD_NUC0 = 2857.0

# ART pressure bounds (bar). The bottom stays inside VULCAN's envelope; the TOP
# is set ABOVE VULCAN's 1e-7 bar chemistry top on purpose -- the log-P
# interpolation CLAMPS the topmost VULCAN value of whatever is mapped through it
# over the extra decade, i.e. a constant-abundance upper-atmosphere extension (T is
# evaluated analytically on the ART grid by both production forwards, so it is not
# clamped there; a common
# transmission-modeling convention, NOT chemistry: photochemical species can
# genuinely vary at sub-microbar pressures). Without it, strong bands (CO2 4.3,
# CO 4.7 um) go optically thick to the model top and the transit radius
# saturates into a flat "wall" at 4.2-5.2 um (saturated fraction 4.8% at 1e-6
# bar); extending to 1e-8 bar removes it (0.1%), letting the bands rise to real
# peaks. This is an EXPLICIT modeling choice, measured on WASP-39 b.
ART_PTOP_BAR = 1.0e-8
ART_PBTM_BAR = 7.0

# Pressure at which a consumer's rp_cm / gs_cgs are taken to be defined. A
# published planet radius is the TRANSIT radius, so it belongs at roughly the
# transmission photosphere, not at the bottom of the RT grid where exojax
# defines radius_btm. 1 mbar is the value validated against the JWST ERS
# WASP-39 b spectrum: re-anchoring there reproduced the published 3.0-5.5 um
# median depth to 0.4% (21,299 vs 21,381 ppm), where the un-converted literature
# radius gave 26,853. Override per planet with profile["p_ref_bar"].
P_REF_BAR = 1.0e-3

# The emission column probes MUCH deeper than the limb: the slant path is ~35-90x
# the vertical one (Fortney 2005), so transmission sees ~mbar while the dayside
# photosphere sits near 0.1 bar. Using the transit radius/gravity for emission
# biases planet-to-star flux ratios by ~5% typically and 10-25% for low-gravity
# hot Jupiters (Fortney, Lupu, Morley, Freedman & Hood 2019, ApJL 880, L16).
# 0.1 bar is HyDRA's stated convention, "the mean pressure of the tau=1 surface"
# (Gandhi & Madhusudhan 2018). NOTE this only re-anchors the column GRAVITY; the
# fully correct treatment computes a wavelength-dependent radius at vertical
# tau = 2/3, as POSEIDON and PLATON II do, and is not implemented here.
P_REF_EMISSION_BAR = 1.0e-1

# Physical constants, cgs (CODATA 2018). Local so this module stays stdlib-only;
# vulcan_jax.phy_const carries the same values for the chemistry side.
K_B_CGS = 1.380649e-16      # Boltzmann constant, erg/K
M_U_CGS = 1.66053906660e-24  # atomic mass unit, g (mmw is in amu)

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

# ---------------------------------------------------------------------------
# Molecule table
# ---------------------------------------------------------------------------
# Each molecule: VULCAN species name and molar mass (g/mol). The opacity itself
# is the published ExoMolOP k-table <MOL>.ktable.h5 (fetch_exomolop selects the
# principal isotopologue and prefers the natural-abundance file; the dataset,
# isotopologue and URL behind each table are recorded in the tree's
# provenance.json, exposed by ``exomolop.provenance()``). molmass is explicit
# and is the ONLY mass the engine uses: the k-tables are cm^2 per MOLECULE and
# their own mol_mass header is not read (ExoMolOP's NO file carries 46, an
# upstream metadata error; NO is 30). Do not "fix" molmass to match a file.
#
# Every table is the principal isotopologue except CO and CO2 (natural
# abundance), paired with the TOTAL molecular VMR -- a <= ~2% opacity deficit
# on the multi-carbon species, below the tables' own accuracy.
#
# Callers may pass their own table to build_rt_model via
# profile["molecule_table"]; this is the default, not a hardcoded lookup.
MOLECULES = {
    "CO":  {"vulcan": "CO",  "molmass": 28.010},
    "H2O": {"vulcan": "H2O", "molmass": 18.015},
    "CO2": {"vulcan": "CO2", "molmass": 43.990},
    "CH4": {"vulcan": "CH4", "molmass": 16.043},
    "SO2": {"vulcan": "SO2", "molmass": 64.066},
    # High-C/O + sulfur discriminators: C2H2/HCN carry the signal near C/O ~ 1,
    # H2S is the reduced-S reservoir.
    "HCN":  {"vulcan": "HCN",  "molmass": 27.025},
    "C2H2": {"vulcan": "C2H2", "molmass": 26.037},
    "H2S":  {"vulcan": "H2S",  "molmass": 34.081},
    # Cool-planet nitrogen carrier (e.g. WASP-107b-class).
    "NH3":  {"vulcan": "NH3",  "molmass": 17.031},
    # Second equilibrium sulfur carrier (nu3 band ~4.85 um, inside G395H and
    # PRISM). The SNCHO network names the species COS (consumers alias the
    # token).
    "OCS":  {"vulcan": "COS",  "molmass": 60.075},
    # Photochemical sulfur carrier and a CH4-destruction product. ExoMolOP
    # publishes NO k-table for either (CS2: no ExoMol line list; C2H6: page but
    # no petitRADTRANS file), so they are listed for completeness and refused
    # at load with the fetch hint.
    "CS2":  {"vulcan": "CS2",  "molmass": 76.131},
    "C2H6": {"vulcan": "C2H6", "molmass": 30.069},
    # Simple hydrocarbon: photochemical CH4-destruction product.
    "C2H4": {"vulcan": "C2H4", "molmass": 28.054},
    # RADICALS. All four are species in VULCAN's SNCHO network, and the
    # published Tsai et al. 2023 WASP-39 b output carries SH and SO, so they are
    # part of an apples-to-apples comparison with that model -- species coverage
    # was one of the three measured reasons this engine's spectra had too much
    # contrast.
    "OH": {"vulcan": "OH", "molmass": 17.007},
    "SH": {"vulcan": "SH", "molmass": 33.073},
    "SO": {"vulcan": "SO", "molmass": 48.064},
    # ExoMolOP's recommended NO opacity is built from HITEMP
    # (14N-16O__HITEMP.R1000_0.3-50mu). Identity confirmed from the data: peak
    # at 1924 cm^-1 (5.20 um) with an overtone at 2.66 um is the NO
    # fundamental, not NO2 or NS. NO peaks at 2.0e-08 (W39b).
    "NO": {"vulcan": "NO", "molmass": 30.006},
    # SECOND-TIER SPECIES, from sweeping every IR-active SNCHO species against
    # ExoMolOP. Present so the menu is COMPLETE, not because each is expected
    # to matter; which ones default ON is decided by measured ppm, in
    # vulcan-jwst-tool forward.EXTRA_MOLECULES_DEFAULT.
    #
    # CANNOT be added, do not re-sweep: O2 (published only at R15000_0.2-30mu
    # -- different band grid), CH3OH / CH3CN / HC3N / NO2 / C6H6 / CH3CHO / HO2
    # (page, no petitRADTRANS file), and ~19 radicals and nitriles absent
    # entirely. The painful two are HSO (1.2e-4) and S2 (9.7e-5), both MORE
    # abundant than SO2 on W39b: S2 is homonuclear so it has no IR dipole, HSO
    # has no published list. Full record: notes.md.
    "NS":   {"vulcan": "NS",   "molmass": 46.072},
    "CH3":  {"vulcan": "CH3",  "molmass": 15.035},
    "NH":   {"vulcan": "NH",   "molmass": 15.015},
    "CN":   {"vulcan": "CN",   "molmass": 26.018},
    "H2CO": {"vulcan": "H2CO", "molmass": 30.026},
    "CS":   {"vulcan": "CS",   "molmass": 44.076},
    "N2O":  {"vulcan": "N2O",  "molmass": 44.013},
    "CH":   {"vulcan": "CH",   "molmass": 13.019},
    "C2":   {"vulcan": "C2",   "molmass": 24.022},
    "H2O2": {"vulcan": "H2O2", "molmass": 34.015},
}

# Bulk gas used for CIA + the dominant background (H2).
BULK_H2_VULCAN = "H2"
