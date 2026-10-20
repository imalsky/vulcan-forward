# todo

Open, owed or undecided work for vulcan-forward; the evidence is in `notes.md`.

1. [decision] Confirm the per-theta budget re-anchoring in `_prep` (the certificate terms themselves are decided: they stay).
2. [stale] Two full-tier queue-vs-batch tests sit on knife edges that the 0.27.2 carbon repair tipped and fail on daw as on the GitHub runner; re-point or re-bar them.

## 1. conv_normal certificate terms

- The certificate terms are vulcan3's and stay (VULCAN-JAX notes "Decided by the
  maintainer, not to be re-opened"); this repo only ANDs `vulcan3.conv_normal` into
  `ConvDiag.conv_normal` (`_conv_diag` in `src/vulcan_forward/vulcan_chem.py`).
- This repo's part: the per-theta re-anchoring in `_prep` (`budget_ref`, zeroed
  `budget_err`/`budget_drift`, `y_prev = y0p`). Commits `b8cae5c`, `b72b0d2`, `3bb7cab`,
  `aa79f25`, `dfa2a0f`; evidence: notes.md "Provenance audit" (first bullet) and
  "Convergence certificate". The other provenance-unknown commits there can be confirmed
  in the same pass.
- Done means: the re-anchoring is confirmed or removed;
  the result goes to notes.md "Convergence certificate".

## 2. Two knife-edge full-tier tests

- Tests: `tests/test_converged_y_queue.py::test_queue_agrees_with_the_batch_with_and_without_refill`
  and `::test_queue_is_differentiable` (both stop at the queue-vs-batch primal check in
  `_report`, 6.87e-2 against `REL_MAX` 0.05), and
  `tests/test_converged_y_jvp.py::test_eq_seed_certifies_and_the_tangent_certificate_ends_the_run`
  (the warm jvp at `THETA_WARM` runs to its 3001 cap, `tangent_longdy` 2e159).
- Evidence: notes.md "Current state" and "Batched routes". `430732a` passes all three on
  daw; `a6a1094` fails them with the GitHub runner's values. The jvp case on the same
  profile at lnZ 0.29 / 0.30 / 0.31: `430732a` certifies / certifies / diverges,
  `a6a1094` certifies / diverges / certifies (the settled uncapped-dt edge). The test's
  own start barely moves (CH4 1.5e-4 of CO there; repair factor within 1e-9 of 1; the primal
  certifies in 77 vs 78 steps).
- Re-run after the 2026-10-20 cleanup (`co_mode` default `fixed_O`): both queue tests
  still fail at exactly 6.874e-2 (job 1 of 2; job 0 3.4e-3); the warm jvp test now
  certifies and passes [M]. Two tests remain red.
- Choice: move `THETA_WARM` and the queue thetas off the edge, or widen the bars.
- Command (daw, about 20 min, about 24 GB peak; keep the CPU cap):
  `JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 VULCAN_FORWARD_RUN_FULL=1 taskset -c <12 free CPUs> uv run python -m pytest tests -q -n 6 --dist worksteal`
- Done means: the full tier passes on daw.
