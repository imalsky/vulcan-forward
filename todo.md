# todo

Open, owed or undecided work for vulcan-forward; the evidence is in `notes.md`.

1. [decision] Confirm the extra `conv_normal` certificate terms (geometry, element budget) and the per-theta re-anchoring, whose provenance is not recorded.
2. [stale] Three full-tier tests sit on knife edges that the 0.27.2 carbon repair tipped and fail on daw as on the GitHub runner; re-point or re-bar them.

## 1. conv_normal certificate terms

- Evidence: notes.md "Provenance audit" (first bullet) and "Convergence certificate".
- Commits: `b8cae5c` (geometry term), `b72b0d2` (element-budget term), `3bb7cab`
  (per-theta budget reference), `aa79f25` (per-theta `y_prev` seed), `dfa2a0f`
  (`ConvDiag` delegates to `vulcan3.conv_normal`). The terms live in vulcan3; this repo
  ANDs them into `ConvDiag.conv_normal` (`_conv_diag` in
  `src/vulcan_forward/vulcan_chem.py`) and re-anchors per theta in `_prep`.
- Why it needs a decision: these terms decide which draws a retrieval rejects and go
  beyond VULCAN's original convergence criteria. vulcan-retrieval's todo lists the same
  question for the vulcan3 side.
- The other provenance-unknown commits in that notes section can be confirmed in the same
  pass.
- Done means: the maintainer keeps or removes each term; the decision is recorded in
  notes.md "Convergence certificate".

## 2. Three knife-edge full-tier tests

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
- Still open: commits since `a6a1094` (`d5d29b5` rename, `39c2c45` new normal-tier test)
  did not touch these tests' thetas or bars; not re-run at HEAD.
- Choice: move `THETA_WARM` and the queue thetas off the edge, or widen the bars.
- Command (daw, about 20 min, about 24 GB peak; keep the CPU cap):
  `JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 VULCAN_FORWARD_RUN_FULL=1 taskset -c <12 free CPUs> uv run python -m pytest tests -q -n 6 --dist worksteal`
- Done means: the full tier passes on daw.
