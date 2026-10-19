# vulcan-forward notes

Findings from the WASP-39b campaign (Oct 2026) that concern the forward model. The
retrieval-side studies, runs and open items are in vulcan-retrieval's `notes.md` and
`todo.md`; solver findings in VULCAN3's `notes.md`. [M] measured, [E] estimated.

## RT speed (speed2_rt, 2026-10-17)

- XLA's argsort along the middle axis of `ckd.overlap` was 82% of the RT stage, 107.4 ms
  per spectrum [M].
- The "kv" rewrite (sums on the last axis, one stable two-operand `lax.sort` with the
  weights as payload) runs at 25.3 ms per spectrum (4.25x); a 2376-row RT pass 256.0 ->
  60.2 s. Depths bitwise on all 3,920,400 values, log L bitwise on 2376/2376, jvp
  bitwise, vjp within 1.6e-14 relative. ADOPTED: 5e8d43a (ckd.py:76).
- After it the sort is still 51% of the stage (11.7 ms per spectrum). Not adopted: an
  unrolled searchsorted (24.4 ms, bitwise), index gather (26.9-36.8 ms), compare_all
  (not bitwise), rt_chunk 8-64 (no gain), a CUDA fold prototype (8.2 ms for the fold).
- 39c2c45: `converged_y_queue(active=)` (vulcan_chem.py:888) lets the retrieval mask
  padding rows; bitwise.

## Warm starts

- Every warm start zeroes dflux_u (vulcan_chem.py:783-788); for 27 of 98 certified lanes
  in the ad_advice sample that reset was the last gate to clear.

## Provenance audit (2026-10-19)

All commits before 2026-10-13 came from other machines and cannot be traced from NAS
(category D: provenance unknown; 154 commits). Live at HEAD and on by default unless
noted; each needs the user's confirmation:

- b8cae5c, b72b0d2, 3bb7cab, aa79f25, dfa2a0f: conv_normal adds geometry and
  element-budget certificate terms and per-theta re-anchoring. This decides which draws
  are rejected and goes beyond VULCAN's original criteria. The user's only reaction was
  "these all seem fine" to a one-line summary (10-17).
- 0ee65a4 + fb244c7: chemistry top P_t forced to the RT top, now 1e-9 bar
  (vulcan_chem.py:351-355, constants.py:50).
- 3b7654b, 2838121, 5018e4d, d40211d: cold start is in-graph equilibrium at the
  proposal's T-P; the 'baseline' start was deleted.
- a6a1094: carbon repaired through CO+CH4 (used in all three W39b production runs).
- 2807b9a, 0250ce0: elemental abundance map only.
- 4446e6e, e2aacba: transit radius anchored at 1 mbar (constants.py:58); bottom
  boundary; emission solver linsap.
- d26e9d3, 801f7e7: fixed planet mass under lnR0.
- 773837e: hand-written correlated-k random overlap, T/P clamps, K_FLOOR 1e-60.
- db94d6e: line-by-line and Mie paths removed (also removed a runtime overwrite of
  MdbHitran gamma_air/n_air; ExoJAX source itself was never modified).
- aedbcbc: 12 of 27 molecular masses changed.
- 6aa0cb8: warm_count_max defaults to count_max. c63815c: emission uses g(r).
  ccdf92f: exojax 2.6.0. 879a99c: the jvp path also needs a tangent certificate.
  55613bc: eclipse_flux_tau (opt-in).
- NAS commits 5e8d43a and 39c2c45 are bitwise and came from the user's 10-17
  instructions (5e8d43a was committed ~30 min before the specific OK).
