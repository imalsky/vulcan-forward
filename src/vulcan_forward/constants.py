"""Shared physics constants, RT defaults, the default molecule table and the profile-key check.

Imports nothing heavy and touches no file, so it is safe before the environment-order-sensitive vulcan3 setup and on a
machine with no data. Data locations live in ``paths``; planet geometry and run settings belong to the consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ART_PTOP_BAR",
    "ATOMIC_MASSES",
    "ATOMIC_MASS_TOL_G_MOL",
    "ATOM_COLS",
    "BAR_CGS",
    "BULK_H2_VULCAN",
    "CIA_H2H2_FILE",
    "CIA_H2HE_FILE",
    "CLOUD_DECK_TAU",
    "CLOUD_DECK_WIDTH_LNP",
    "CLOUD_NUC0",
    "CORE_MOLECULE_BANDS_UM",
    "ELEMENTAL_REPAIR",
    "ELEMENTAL_REPAIR_ITERS",
    "EMISSION_NSTREAM",
    "EXOMOLOP_GRID_TOKEN",
    "GM_MATCH_RTOL",
    "GRID_RTOL",
    "KZZ_P_REF_BAR",
    "K_B_CGS",
    "K_FLOOR",
    "MOLECULES",
    "M_U_CGS",
    "PROFILE_KEYS",
    "P_REF_EMISSION_BAR",
    "RAYLEIGH_KING_FACTOR",
    "REQUIRED_MOLECULES",
    "TAU_PHOTOSPHERE",
    "TOP_MATCH_RTOL",
    "T_OPA_MAX_K",
    "T_OPA_MIN_K",
    "UM_PER_CM",
    "WEIGHT_SUM_ATOL",
    "WEIGHT_SUM_RTOL",
    "WIDE_BAND_NU_MAX",
    "WIDE_BAND_NU_MIN",
    "check_band",
    "check_core_molecules",
    "check_profile_keys",
    "fixed_values",
]

# ---- 1. Physical constants and unit factors ----

# Physical constants and unit factors, cgs, local so this module stays stdlib-only. K_B_CGS is CODATA 2018; the
# chemistry uses vulcan3.phy_const.kb = 1.38064852e-16 (CODATA 2014), upstream VULCAN's value, so its rates match the
# reference code. The two differ by 3.5e-7 relative.
K_B_CGS = 1.380649e-16  # Boltzmann constant, erg/K
M_U_CGS = 1.66053906660e-24  # atomic mass unit, g (mmw is in amu)
BAR_CGS = 1.0e6  # dyn/cm^2 per bar
UM_PER_CM = 1.0e4  # micron per cm: wavelength (um) = UM_PER_CM / wavenumber (cm^-1)

# Composition tables, mirrored from vulcan3.composition so this module stays import-light; vulcan_chem checks them at
# build time and raises on drift.
#
# Index of each element column the Z and C/O knobs use.
ATOM_COLS = {"H": 0, "O": 1, "C": 2, "He": 3, "N": 4, "S": 5}

# Molar mass, g/mol, of every compo_array column, in the column order H, O, C, He, N, S, P, Na, K, Si, Fe, Ar, Ti, V,
# Mg, Ca, e. compo_array @ ATOMIC_MASSES is each species' molar mass, from which the mean molecular weight follows.
ATOMIC_MASSES = [
    1.008,
    15.999,
    12.011,
    4.0026,
    14.007,
    32.06,
    30.974,
    22.990,
    39.098,
    28.085,
    55.845,
    39.948,
    47.867,
    50.942,
    24.305,
    40.078,
    5.4858e-4,
]

# ---- 2. Fixed model choices: not settable, each with its reason ----

# Supported wide band, 1-15 um, in wavenumber (cm^-1). The H2-H2 CIA table sets the short edge (it stops at 1 um,
# 10000 cm^-1); the k-tables reach 50 um.
WIDE_BAND_NU_MIN = 667.0  # 15 um
WIDE_BAND_NU_MAX = 10000.0  # 1 um

# Temperature window, K, of the opacity data: both CIA tables start at 200 K and the H2-H2 table ends at 3000 K; the
# ExoMolOP k-tables span 100-3400 K. A column outside it is refused, never clamped (_rt_opacity._column_ok), and the
# build checks that the installed tables cover it. It bounds the RT only: vulcan3's equilibrium start is known to break
# down below about 225 K, where a solve raises or comes back uncertified.
T_OPA_MIN_K = 200.0
T_OPA_MAX_K = 3000.0

# Pressure, bar, at which the emission column's radius and gravity are anchored: the dayside photosphere sits near
# 0.1 bar, two decades below the limb transmission sees, and 0.1 bar is HyDRA's convention (Gandhi & Madhusudhan 2018).
# eclipse_flux_tau refines it to the vertical tau = 2/3 radius per k-ordinate (docs/method.md "Emission anchor").
P_REF_EMISSION_BAR = 1.0e-1

# Vertical optical depth of the emitting photosphere (Fortney et al. 2019): the eclipse-depth prefactor is the radius
# where tau = 2/3 at each wavelength, not one radius for the band.
TAU_PHOTOSPHERE = 2.0 / 3.0

# Gauss-Legendre streams of the emission solve (exojax ArtEmisPure's default). Against 32 streams the eclipse flux at 8
# differs by at most 1.8e-4 on monotonic profiles and 1.3e-3 on a 1200 K inversion (notes "Emission solver and
# gravity"), far below the opacity data's own accuracy.
EMISSION_NSTREAM = 8

# King correction factor of the H2 and He Rayleigh cross-sections. 1.0 neglects depolarization: the correction is
# about 1 % for H2 and zero for He (an atom), below the cross-section data's own accuracy.
RAYLEIGH_KING_FACTOR = 1.0

# Reference wavenumber, cm^-1, of exojax's powerlaw_clouds: kappa(nu) = kappac0 (nu / CLOUD_NUC0)^alphac, with kappac0
# in cm^2 per gram of atmosphere (pRT convention; alphac = 0 is grey). 2857 cm^-1 is 3.5 um. A uniformly mixed nuisance
# opacity, not a physical cloud (docs/method.md "Continua").
CLOUD_NUC0 = 2857.0

# Grey cloud deck, the alternative to the power law: layers deeper than the deck pressure get the optical depth
# CLOUD_DECK_TAU, switched on by a logistic step in ln P of width CLOUD_DECK_WIDTH_LNP so the deck pressure stays
# differentiable. 0.02 in ln P is 0.009 dex, a sixteenth of a production 67-layer RT layer. Along the chord the deck is
# ~1e5 optical depths thick, so the effective boundary sits ~13 widths (0.1 dex) above the deck pressure
# (tests/test_cloud_deck.py).
CLOUD_DECK_TAU = 1.0e3
CLOUD_DECK_WIDTH_LNP = 0.02

# Floor on k before the log, cm^2 per molecule. The tables hold exact zeros where a species has no lines in a band, and
# log(0) would poison the interpolation; 1e-60 cm^2 over a full column is ~1e-32 in tau.
K_FLOOR = 1.0e-60

# The two collision-induced-absorption tables, HITRAN CIA 2011 (hitran.org/data/CIA/main/), in paths.opacity_cache_dir.
# H2-H2 covers 200-3000 K and 20-10000 cm^-1, H2-He 200-9900 K: they set T_OPA_MAX_K and WIDE_BAND_NU_MAX.
CIA_H2H2_FILE = "H2-H2_2011.cia"
CIA_H2HE_FILE = "H2-He_2011.cia"

# The ExoMolOP band grid this engine reads: R = 1000 over 0.3-50 um, 16 g-ordinates per band, never rebinned.
EXOMOLOP_GRID_TOKEN = "R1000_0.3-50mu"

# Element -> adjuster species, scaled by one common factor per element: its dominant carriers in H2-dominated gas, so
# the linear repair stays small and well-conditioned. Carbon needs CO and CH4 together: below 900 K CH4 holds the carbon
# and CO only 1e-3 of it, so a CO-only repair of a 1% carbon error drives CO negative. H is the reference element.
ELEMENTAL_REPAIR = (("He", ("He",)), ("O", ("H2O",)), ("C", ("CO", "CH4")), ("N", ("N2",)), ("S", ("H2S",)))
# Renormalize-and-repair passes: one takes a warm guess's 1e-1 residual to 1e-10, two reach machine precision.
ELEMENTAL_REPAIR_ITERS = 3

# ---- 3. Numerical tolerances ----

# Two k-tables mix only if their grids agree to this relative tolerance (float64 rounding of one published grid).
GRID_RTOL = 1.0e-12
# The quadrature weights of a k-table must sum to 1 to this tolerance (float64 rounding).
WEIGHT_SUM_RTOL, WEIGHT_SUM_ATOL = 1.0e-10, 1.0e-12
# Relative slack on the chemistry top: an ART top that equals it up to rounding is not a clamp.
TOP_MATCH_RTOL = 1.0e-9
# The chemistry takes its planet mass from the vulcan3 config and the RT its gravity and radius from the profile: G Mp
# and gs_cgs rp_cm^2 must agree to this relative tolerance, or the two halves model different planets. 2 % is above
# the rounding of a published mass and far below a wrong planet.
GM_MATCH_RTOL = 0.02
# g/mol. vulcan3's mass column carries mild rounding (O as 16.0, the electron as 1e-3); a swapped element differs by at
# least 1 g/mol.
ATOMIC_MASS_TOL_G_MOL = 0.01

# ---- 4. Defaults a caller may override through the profile ----
# The chemistry's network, atom list and config name, and the RT's planet geometry, bottom pressure and reference
# pressure have no default: they belong to the caller's case (vulcan_chem._CHEMISTRY_KEYS, exojax_rt._RT_KEYS).

# RT top pressure, bar. Chemistry and RT share the model top: vulcan_chem sets P_t from art_ptop_bar, and interp_map
# refuses an ART grid the chemistry does not cover. At 1e-9 bar the top is converged: one decade higher moves the
# R = 100 depth by ~1 ppm (vulcan-retrieval validation/top_pressure_ladder). The bottom, art_pbtm_bar, has no default:
# it must sit above the case's chemistry bottom.
ART_PTOP_BAR = 1.0e-9

# The pressure, bar, at which a Kzz profile's value is quoted and reported (1 mbar: where the WASP-39 b profile value
# is quoted). Not a default: under kzz_mode "powerlaw" the profile states its own kzz_p_ref_bar, where Kzz0 is read off
# the config's profile and tilted by (P / P_ref)^-slope.
KZZ_P_REF_BAR = 1.0e-3

# Molecule table: VULCAN species name and molar mass, g/mol. The opacity is the ExoMolOP k-table <MOL>.ktable.h5, and
# exomolop.provenance() says which one. molmass is the only mass the engine uses and equals the formula mass
# (tests/test_contract.py); the tables' mol_mass header is not read (ExoMolOP's NO file carries 46). The tables are the
# principal isotopologue except CO and CO2 (natural abundance); paired with the total VMR, that is an opacity deficit of
# ~2% on the multi-carbon species and ~5-6% on the sulfur species (32S is 95% of sulfur). A caller may pass its own
# table as profile["molecule_table"].
MOLECULES: dict[str, dict[str, Any]] = {
    "CO": {"vulcan": "CO", "molmass": 28.010},
    "H2O": {"vulcan": "H2O", "molmass": 18.015},
    "CO2": {"vulcan": "CO2", "molmass": 44.009},
    "CH4": {"vulcan": "CH4", "molmass": 16.043},
    "SO2": {"vulcan": "SO2", "molmass": 64.058},
    # High-C/O and sulfur discriminators: C2H2 and HCN carry the signal near C/O ~ 1, H2S is the reduced-S reservoir.
    "HCN": {"vulcan": "HCN", "molmass": 27.026},
    "C2H2": {"vulcan": "C2H2", "molmass": 26.038},
    "H2S": {"vulcan": "H2S", "molmass": 34.076},
    # Cool-planet nitrogen carrier (e.g. WASP-107b-class).
    "NH3": {"vulcan": "NH3", "molmass": 17.031},
    # Second equilibrium sulfur carrier (nu3 band ~4.85 um, inside G395H and PRISM). The SNCHO network names it COS.
    "OCS": {"vulcan": "COS", "molmass": 60.070},
    # Photochemical sulfur carrier and a CH4-destruction product. ExoMolOP publishes no k-table for either (CS2: no
    # ExoMol line list; C2H6: no petitRADTRANS file), so they are listed for completeness and refused at load.
    "CS2": {"vulcan": "CS2", "molmass": 76.131},
    "C2H6": {"vulcan": "C2H6", "molmass": 30.070},
    # Simple hydrocarbon: photochemical CH4-destruction product.
    "C2H4": {"vulcan": "C2H4", "molmass": 28.054},
    # Radicals in VULCAN's SNCHO network; the published Tsai et al. 2023 WASP-39 b output carries SH and SO.
    "OH": {"vulcan": "OH", "molmass": 17.007},
    "SH": {"vulcan": "SH", "molmass": 33.068},
    "SO": {"vulcan": "SO", "molmass": 48.059},
    # ExoMolOP's NO table is built from HITEMP (14N-16O__HITEMP.R1000_0.3-50mu).
    "NO": {"vulcan": "NO", "molmass": 30.006},
    # Second-tier species, listed so the menu is complete; jwst-transit-authority's EXTRA_MOLECULES_DEFAULT picks the
    # defaults. Absent because ExoMolOP cannot supply them: O2 (another band grid), CH3OH (no petitRADTRANS file), HSO
    # (no published list), S2 (no IR dipole) and others.
    "NS": {"vulcan": "NS", "molmass": 46.067},
    "CH3": {"vulcan": "CH3", "molmass": 15.035},
    "NH": {"vulcan": "NH", "molmass": 15.015},
    "CN": {"vulcan": "CN", "molmass": 26.018},
    "H2CO": {"vulcan": "H2CO", "molmass": 30.026},
    "CS": {"vulcan": "CS", "molmass": 44.071},
    "N2O": {"vulcan": "N2O", "molmass": 44.013},
    "CH": {"vulcan": "CH", "molmass": 13.019},
    "C2": {"vulcan": "C2", "molmass": 24.022},
    "H2O2": {"vulcan": "H2O2", "molmass": 34.014},
}

# Core absorbers: the gases that carry the oxygen, carbon, nitrogen and sulfur of a hydrogen-dominated atmosphere
# (Moses et al. 2013, ApJ 763, 25) and have been measured in transmission with JWST (H2O, CO2, CO, SO2: Rustamkulov et
# al. 2023; CH4: Bell et al. 2023; NH3: Welbanks et al. 2024; H2S: Fu et al. 2024; HCN is the predicted disequilibrium
# nitrogen carrier), plus three that reach tens to hundreds of ppm in this engine's own columns: C2H2 in carbon-rich
# gas, and the sulfur carriers OCS and SO (notes "Opacity: correlated-k on ExoMolOP"). Where a band overlaps one of a
# molecule's ranges and the molecule is neither in the RT nor named in profile["omit_core_molecules"], build_rt_model
# refuses a REQUIRED_MOLECULES member and warns for the others. Ranges, um: H2O absorbs across the whole band; for the
# others, where the R = 100 band-mean cross-section of the ExoMolOP table at 1000 K and 1 mbar is at least 1 % of its
# 1-15 um peak, rounded outward (tests/test_exomolop.py checks them against the installed tables).
CORE_MOLECULE_BANDS_UM: dict[str, tuple[tuple[float, float], ...]] = {
    "H2O": ((1.0, 15.0),),
    "CO2": ((2.6, 2.9), (4.1, 4.6), (13.0, 15.0)),
    "CO": ((4.3, 5.2),),
    "CH4": ((2.2, 2.5), (3.0, 3.8), (6.3, 9.1)),
    "NH3": ((1.9, 2.5), (2.7, 3.3), (5.1, 15.0)),
    "HCN": ((2.9, 3.3), (6.6, 7.9), (11.8, 15.0)),
    "H2S": ((1.5, 1.7), (1.8, 2.2), (2.3, 3.0), (3.3, 4.8), (5.5, 15.0)),
    "SO2": ((3.9, 4.2), (7.0, 9.8)),
    "C2H2": ((2.9, 3.2), (6.5, 8.2), (11.5, 15.0)),
    "OCS": ((3.4, 3.5), (4.7, 5.2), (5.3, 5.4), (9.2, 9.7), (11.4, 12.3)),
    "SO": ((4.3, 4.8), (8.2, 10.1)),
}
# The oxygen, carbon and nitrogen carriers every such atmosphere holds. HCN, C2H2 and the sulfur gases depend on the
# planet (a quench level, C/O, a sulfur-bearing network), so leaving one out is reported, not refused.
REQUIRED_MOLECULES = frozenset({"H2O", "CO2", "CO", "CH4", "NH3"})

# The bulk gas: the CIA partner and the dominant background.
BULK_H2_VULCAN = "H2"

# Every top-level profile key the engine reads. A consumer passes one dict to build_chem_model, build_rt_model and
# build_emis_model, so this is the union.
PROFILE_KEYS = frozenset(
    {
        # build_chem_model
        "network",
        "atom_list",
        "vulcan_cfg_name",
        "use_photo",
        "yconv_cri",
        "yconv_min",
        "nz",
        "count_min",
        "count_max",
        "dt_max",
        "cfg_overrides",
        "co_mode",
        "kzz_mode",
        "kzz_p_ref_bar",
        # build_rt_model / build_emis_model (art_ptop_bar also sets the chemistry top)
        "molecules",
        "molecule_table",
        "omit_core_molecules",
        "nu_min",
        "nu_max",
        "art_nlayer",
        "art_ptop_bar",
        "art_pbtm_bar",
        "rt_integration",
        "use_rayleigh",
        "rp_cm",
        "gs_cgs",
        "rstar_cm",
        "p_ref_bar",
    }
)


def check_profile_keys(profile: Mapping[str, object]) -> None:
    """Refuse profile keys the engine does not read: a misspelled or retired key would be a silent no-op.

    ``cfg_overrides`` contents are checked by ``vulcan3.validate_overrides``.

    Parameters
    ----------
    profile : mapping
        The settings dict passed to a builder.

    Raises
    ------
    ValueError
        Naming the unknown keys and the known ones.
    """
    unknown = sorted(set(profile) - PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"profile keys {unknown} are not read by vulcan-forward (misspelled, "
            "or retired with a removed feature), so they would change nothing. "
            f"Drop or correct them. Known keys: {sorted(PROFILE_KEYS)}."
        )


def check_band(nu_min: float, nu_max: float) -> None:
    """Refuse a band outside the supported 1-15 um: the model has no opacity for what absorbs beyond it.

    Parameters
    ----------
    nu_min : float
        Lower band edge, cm^-1.
    nu_max : float
        Upper band edge, cm^-1.

    Raises
    ------
    ValueError
        If the band is empty or reaches outside ``WIDE_BAND_NU_MIN`` to ``WIDE_BAND_NU_MAX``.
    """
    if not (WIDE_BAND_NU_MIN <= nu_min < nu_max <= WIDE_BAND_NU_MAX):
        raise ValueError(
            f"nu_min={nu_min:g}, nu_max={nu_max:g} cm^-1 ({UM_PER_CM / nu_max:.3g}-{UM_PER_CM / nu_min:.3g} um) is not "
            f"inside the supported band [{WIDE_BAND_NU_MIN:g}, {WIDE_BAND_NU_MAX:g}] cm^-1 (1-15 um). Shortward of "
            "1 um the spectrum is set by the Na and K resonance lines and by hazes, which this model does not have, "
            "and the H2-H2 CIA table ends there. Narrow nu_min/nu_max."
        )


def check_core_molecules(molecules: list[str], omitted: list[str], nu_min: float, nu_max: float) -> list[str]:
    """Refuse a band that needs a required absorber the RT lacks, and return the other core absorbers it lacks.

    Parameters
    ----------
    molecules : list of str
        The RT molecule set.
    omitted : list of str
        Core molecules left out on purpose (``profile["omit_core_molecules"]``).
    nu_min : float
        Lower band edge, cm^-1.
    nu_max : float
        Upper band edge, cm^-1.

    Returns
    -------
    list of str
        Core molecules outside ``REQUIRED_MOLECULES`` that absorb in the band and are in neither list, for the caller
        to report.

    Raises
    ------
    ValueError
        If ``omitted`` names a molecule that is not core or is in ``molecules``, or if the band overlaps a range of
        a ``REQUIRED_MOLECULES`` member that is in neither list.
    """
    bad = sorted(set(omitted) - set(CORE_MOLECULE_BANDS_UM)) + sorted(set(omitted) & set(molecules))
    if bad:
        raise ValueError(
            f"omit_core_molecules names {bad}: each entry must be a core molecule ({sorted(CORE_MOLECULE_BANDS_UM)}) "
            "that is not in 'molecules'. Correct the list."
        )
    lo_um, hi_um = UM_PER_CM / nu_max, UM_PER_CM / nu_min
    missing = {
        m: [r for r in ranges if r[0] < hi_um and r[1] > lo_um]
        for m, ranges in CORE_MOLECULE_BANDS_UM.items()
        if m not in molecules and m not in omitted
    }
    required = {m: r for m, r in missing.items() if r and m in REQUIRED_MOLECULES}
    if required:
        raise ValueError(
            f"the band {lo_um:.3g}-{hi_um:.3g} um needs core absorbers that 'molecules' {molecules} lacks: "
            + "; ".join(f"{m} (absorbs at {', '.join(f'{a:g}-{b:g}' for a, b in r)} um)" for m, r in required.items())
            + ". Add them to 'molecules', or name the ones left out on purpose in 'omit_core_molecules'."
        )
    return [m for m, r in missing.items() if r and m not in REQUIRED_MOLECULES]


def fixed_values() -> dict[str, object]:
    """Return the engine's fixed numbers and choices by name, for a consumer to store in a run's record.

    Returns
    -------
    dict
        Every constant of this module except the molecule table and the profile-key list; a set comes back sorted.
    """
    out: dict[str, object] = {}
    for name in __all__:
        value = globals()[name]
        if callable(value) or name in ("MOLECULES", "PROFILE_KEYS"):
            continue
        out[name] = sorted(value) if isinstance(value, frozenset) else value
    return out
