# PC-Slice scheduler diagnostic study (exploratory, November 2026)

This work is isolated on branch `experiments/scheduler-diagnostics-20261106`.
It DOES NOT modify the original recorded trial observations, figures or the
original `joint_plan` selection logic. Do not merge the branch into the
publication snapshot before evaluating it.

## Hypotheses

1. The 1 mm/s growing-clearance allowance may over-constrain longer horizons.
   Compare `JP`, `JP_G0` and `JP_G05` using unchanged physical constraints.
2. The bytes/useful-coverage objective may ignore waypoint progress.
   Compare exploratory `JP_P1` and `JP_P3` with the original `JP`.
3. Recheck the fixed-short (FS), event (EV), joint-fixed-short (JS) baselines.
   Baselines must retain their original transport and receiver checker.

## Exact matrix

Pilot: 1 development layout × 2 profiles × 1 seed × 4 variants = 8 trials.
Development: 2 development layouts × 4 profiles × 3 seeds × 8 variants = 192 trials.

Profiles: freefault1, delay2, packetloss3, communicationloss2.
Seeds: 31031, 31032, 31033. All are development; no held-out claims.

Variant code:
- FS / EV / JS: unchanged comparators.
- JP: unchanged joint-plan selection.
- JP_G0: same joint plan, but forecast-growth allowance = 0.
- JP_G05: same joint plan, but growth = 0.5 mm/s.
- JP_P1 / JP_P3: use additional waypoint-deficit weighting 1 or 3 for
  ranking feasible candidate horizons. This is a heuristic, not proven novel
  or clinically safe. Receiver check remains unchanged.

Comparison uses matched layouts/network/seed and reports safe completion,
breach trials, management bytes and capped task time. Planning wall time is
measured and injected into the simulated task clock; for this reason trials
run SEQUENTIALLY (not a 20-worker pool). CPU contention could change results,
so use consistent cluster load and document hardware. No significance claims
or new real-time guarantees are made.

## HPC

The existing offline Python venv and pinned physical/network assets are
required; these tests do not load the LLM weights or request GPU resources.

```bash
bash hpc/submit_scheduler_diagnostics.sh pilot
# Inspect logs and results, then:
bash hpc/submit_scheduler_diagnostics.sh development
```

Outputs: `results/scheduler_pilot_JOBID` or
`results/scheduler_development_JOBID`.
Each trial has `metrics.json`, `events.jsonl.gz`, `trajectory.jsonl.gz`,
and `completion.json`. Summary artifacts: `manifest.json`,
`observations.json` and `SUMMARY.json`. A failed task is a valid result.
An execution failure writes `ERROR.json` and exits nonzero.

The next step is to inspect and diagnose results, then freeze one method,
create previously unseen test geometries and seeds, and run an independent
held-out confirmatory study. This runner intentionally does NOT implement
that final test.
