# todo

Open, owed or undecided work for vulcan-forward; the evidence is in `notes.md`.

1. [unverified] No petitRADTRANS reference covers collision-induced absorption or Rayleigh scattering.
2. [feature] Sodium and potassium: the model stops at 1 um and refuses a band below it, so their lines are never modelled.

## 1. petitRADTRANS reference with the continuum on

Evidence: notes "RT validation and the exojax pin". The three `tests/data/prt_ref_*.npz` cases are lines-only (their
meta says `continuum: none`), and `tests/test_e2e_rt_reference.py` passes zero H2 and He, so CIA and Rayleigh add
nothing there. CIA and Rayleigh are exercised only on miniature tables (`tests/test_rt_miniature.py`), as identities,
not values. `oracle.md` asks that the comparison cover every term a default run uses. The pRT script stored in the
W39b fixture's meta already has a `cont` variant (H2 and He Rayleigh, H2-H2 and H2-He CIA). Done when a committed
reference with the continuum on passes in `tests/test_e2e_rt_reference.py`, with its bars at 3x the achieved
agreement like the others.

## 2. Na and K

The Na (0.59 um) and K (0.77 um) resonance lines sit below the 1 um edge, where `constants.check_band` refuses a
band, so today the omission is an error, never a wrong spectrum. To model them: ExoMolOP publishes Na and K k-tables
in the format the engine reads (Chubb et al. 2021); the installed H2-H2 CIA file stops at 10000 cm^-1, so the band
needs a table that reaches shorter wavelengths; the C-H-N-O-S networks carry neither atom, so their abundance would
be a fixed solar-scaled value or a free parameter; and below 1 um hazes shape the spectrum too. The K line's far red
wing reaches past 1 um; its size inside the current band is not measured. Done when the maintainer asks for it and
the band, the tables, an abundance source and a petitRADTRANS reference with the alkalis are in.
