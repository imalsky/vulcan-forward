"""Shared physics constants for the VULCAN-JAX -> ExoJAX forward model.

Pure constants only: NO heavy imports (no jax, no vulcan_jax, no exojax) and NO
filesystem access, so this module is safe to import before the env-order-
sensitive VULCAN-JAX setup runs and safe to import on a machine with no data
installed. Data locations live in ``paths.py``; application-specific settings
(planet geometry, run profiles, parameter-vector layouts) belong to the
consumer, not here.

Provenance: these values were extracted from vulcan-retrieval's
``retrieval_framework/forward/config.py`` when the shared engine became its own
distribution (2026-07-29). That module mixed three unrelated things -- these
constants, the retrieval checkout's filesystem layout, and WASP-39b/demo
settings. Only the first kind belongs in a library.
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

# Pressure-broadening perturber for the HITRAN opacities. "air" = HITRAN's
# terrestrial gamma_air/n_air (a documented approximation for an H2/He
# envelope); "h2he" = HITRAN's planetary-broadener H2/He widths where the
# database provides them, blended with the number-fraction mix below (uncovered
# lines keep air widths for that partner, reported loudly per molecule at
# build). Callers may override per run via profile["broadening"].
BROADENING = "air"
H2HE_BROADENING_MIX = (0.85, 0.15)   # (f_H2, f_He) by number, ~solar envelope

# ART pressure bounds (bar). The bottom stays inside VULCAN's envelope; the TOP
# is set ABOVE VULCAN's 1e-7 bar chemistry top on purpose -- the log-P
# interpolation CLAMPS the topmost VULCAN VMR/T over the extra decade, i.e. a
# constant-abundance + isothermal upper-atmosphere extension (a common
# transmission-modeling convention, NOT chemistry: photochemical species can
# genuinely vary at sub-microbar pressures). Without it, strong bands (CO2 4.3,
# CO 4.7 um) go optically thick to the model top and the transit radius
# saturates into a flat "wall" at 4.2-5.2 um (saturated fraction 4.8% at 1e-6
# bar); extending to 1e-8 bar removes it (0.1%), letting the bands rise to real
# peaks. This is an EXPLICIT modeling choice, measured on WASP-39 b.
ART_PTOP_BAR = 1.0e-8
ART_PBTM_BAR = 7.0
T_OPA_MIN_K = 300.0
T_OPA_MAX_K = 3000.0

# Supported wide band, 1-15 um, in wavenumber (cm^-1). The short edge is set by
# the H2-H2 CIA table (stops at 1 um / 10000 cm^-1); line lists reach ~20 um.
# Consumers that want the full supported window start from these.
WIDE_BAND_NU_MIN = 667.0     # 15 um
WIDE_BAND_NU_MAX = 10000.0   # 1 um

# ---------------------------------------------------------------------------
# Molecule / opacity-source table
# ---------------------------------------------------------------------------
# Each molecule: VULCAN species name, molar mass (g/mol), and opacity source.
# CO is fully offline (cached ExoMol Li2015). The rest use HITRAN, downloaded on
# first run (small, public, no login -- only the main isotopologue, isotope=1;
# HITRAN intensities already carry the terrestrial isotopic abundance factor, so
# pairing them with the TOTAL molecular VMR is the standard slightly-
# conservative treatment).
#
# KNOWN LIMITS for real-data inference, not just methodology: (1) HITRAN's 296 K
# room-T lists under-represent hot bands at a ~770-1540 K limb relative to
# HITEMP/ExoMol -- absolute abundances lean optimistic where hot bands matter;
# swap "source" per molecule when the multi-GB ExoMol/HITEMP fetch is
# acceptable. (2) Default broadening is terrestrial air; set
# profile["broadening"]="h2he" for HITRAN's planetary H2/He widths where
# available.
#
# "source" is one of {"exomol_cached", "exomol", "hitran"}. "db" is a path
# suffix RESOLVED AGAINST THE DATA ROOTS in paths.py -- never an absolute path
# here, so this table stays location-independent (the pre-extraction version
# baked an absolute cache path into the CO entry). molmass is explicit (exojax
# isotope_molmass returns None for CH4).
#
# Callers may pass their own table to build_rt_model via
# profile["molecule_table"]; this is the default, not a hardcoded lookup.
MOLECULES = {
    "CO":  {"vulcan": "CO",  "molmass": 28.010, "source": "exomol_cached",
            "db": "CO/12C-16O/Li2015"},
    "H2O": {"vulcan": "H2O", "molmass": 18.015, "source": "hitran", "db": "H2O"},
    "CO2": {"vulcan": "CO2", "molmass": 43.990, "source": "hitran", "db": "CO2"},
    "CH4": {"vulcan": "CH4", "molmass": 16.043, "source": "hitran", "db": "CH4"},
    "SO2": {"vulcan": "SO2", "molmass": 64.066, "source": "hitran", "db": "SO2"},
    # High-C/O + sulfur discriminators: C2H2/HCN carry the signal near C/O ~ 1,
    # H2S is the reduced-S reservoir.
    "HCN":  {"vulcan": "HCN",  "molmass": 27.025, "source": "hitran", "db": "HCN"},
    "C2H2": {"vulcan": "C2H2", "molmass": 26.037, "source": "hitran", "db": "C2H2"},
    "H2S":  {"vulcan": "H2S",  "molmass": 34.081, "source": "hitran", "db": "H2S"},
    # Cool-planet nitrogen carrier (e.g. WASP-107b-class).
    "NH3":  {"vulcan": "NH3",  "molmass": 17.031, "source": "hitran", "db": "NH3"},
    # Second equilibrium sulfur carrier (nu3 band ~4.85 um, inside G395H and
    # PRISM). The SNCHO network names the species COS; the PICASO Visscher
    # tables call it OCS (consumers alias the token).
    "OCS":  {"vulcan": "COS",  "molmass": 60.075, "source": "hitran", "db": "OCS"},
    # Photochemical sulfur carrier (collaborator request 2026-07-31).
    "CS2":  {"vulcan": "CS2",  "molmass": 76.131, "source": "hitran", "db": "CS2"},
    # Simple hydrocarbons: photochemical CH4-destruction products.
    "C2H4": {"vulcan": "C2H4", "molmass": 28.054, "source": "hitran", "db": "C2H4"},
    "C2H6": {"vulcan": "C2H6", "molmass": 30.069, "source": "hitran", "db": "C2H6"},
}

# Bulk gas used for CIA + the dominant background (H2).
BULK_H2_VULCAN = "H2"
