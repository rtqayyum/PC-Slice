# Joint plan-length and transmission decisions: development protocol

## Progress/expiry repair validation — specified 5 November 2026

After downloaded development job620616, the user authorized repair and local
validation of missed task progress and expired-plan transmission. That run,
its scoring and prior studies remain frozen. The repair is DEVELOPMENT ONLY.

Task-level question: does executed reaching progress remain latched across
missed/reordered feedback, does the instrument hold after unchanged goal dwell,
and are unusable plans discarded causally before consuming link bytes?
No task thresholds, geometry, horizons, solver iterations or allocation
parameters are optimized in this repair.

Executed progress is sampled once per existing 60-Hz control tick. The receiver
latches ordered waypoint visits and the original goal dwell/completion, then
holds locally. The sender receives only sampled delayed/lost acknowledgments;
the planner may predict future phases but cannot assert executed completion.
P3P1 feedback has a 27-byte version/state/progress header, 156 state bytes for
the 12-joint instrument, 4 bytes per buffered plan ACK and 28 IP/UDP bytes:
211 bytes without plan ACKs, 215 with one. Completion-bearing feedback repeats
on the common sampling period through the trial; its bytes are charged after
completion as well. Sender planning stops only on received completion.

At measured planning availability, and during subsequent credit waiting,
discard a plan if current tick >= end or current tick + the existing causal
delay estimate >= end. Replan from the latest received state, subject to the
same byte credits and single-job compute availability. Preserve timestamps on
usable partial plans; do not peek at the next fault or shift old commands into
a new time window. All V3 controllers use the same rules, with discard reason,
planning sample/ack phase and estimated arrival recorded.

Before local execution, fix this functional validation matrix: development
layout dev_v3_yplus, seed31021, budget24 and unchanged16-second horizon.
Run fixed-short, event, joint and joint-fixed-short under pinned freefault1 and
delay2 (eight trials); both reduced MPC adaptations under freefault1 (two);
and joint under explicitly authored total outage (one). The eleven executions
are software/causality controls, not a comparative efficacy test or LLM run.
No parameters will be selected from their outcomes. The eight task outcomes
and MPC solver/timing failures are all retained. Record breaches, completion,
placement error, fallback and full bidirectional bytes, not only pass/fail.

Required invariants: progress recomputes from actual tip samples, sender phase
changes only through received feedback, completion produces next-tick hold,
no already-expired or predicted-arrival-expired plan is sent, actual serialized
byte totals agree and stay within budget. Four nominal non-MPC controls must
complete safely and the all-lost control must not complete or receive commands.
Delay/MPC outcomes are reported separately and do not release a final protocol.
Unit fixtures cover missing samples, out-of-order feedback, exact dwell,
completion acknowledgment, all nine transport methods and credit/compute waits.
They are labeled software fixtures, never model or clinical observations.

Basis: the existing pinned SurRoL task/dwell definitions and published NetFI
parameter emulations, plus actual620616 event/trajectory diagnosis in
analysis/surgical_v3_gate_620616/. New feedback metadata and deadline handling
are authored engineering protocol repairs, not external clinical requirements.
Entry point: bash hpc/validate_surgical_v3_repairs.sh; no HPC submission.

Authorized 4 November 2026. This revision keeps the same constrained SurRoL PSM
reaching task, geometry, goal/dwell criterion, speed and receiver checker. The
completed 604645/613555 runs and component manuscript remain frozen. Their
opened results are development evidence for this revision, never new test data.

Question: can choosing finite command-plan horizon AND refresh timing improve
the task/communication trade-off over the best fixed short-plan controller and
source-based predictive/event-triggered comparators, after accounting for
feedback traffic, feedback loss/delay and computation latency?

Primary outcomes: task completion without sampled constraint breaches, breach
count/duration, minimum clearance, placement error and completion time. Report
all timeouts and fallback/recovery plus control, telemetry and background bytes.
Improvement requires preserved task/constraint performance with fewer bytes or
better completion at matched bytes; packet savings alone are insufficient.

## Design basis and comparison fidelity

Pinned SurRoL and UVA NetFI source hashes and existing task specification remain
the source basis. All command plans, cases, task layouts and link scheduling
are authored engineering simulations/emulations, not clinical recordings.

- Nagahara/Quevedo, IFAC 2011, https://arxiv.org/abs/1308.1590:
  finite future-input packets and communication cost. We do not reproduce their
  sparse linear-input optimization or transfer their stability result to PSM.
- Puettschneider et al., ICCA 2025,
  https://arxiv.org/html/2409.18589v2, Algorithm 1 and OCP (2):
  error/coverage triggering, expected-arrival state prediction and optimization
  of a finite control trajectory. An implemented comparator uses a reduced
  shooting basis for the existing nonlinear articulated controller; its plant,
  cost and parameterization are documented adaptations, not a reproduction of
  the Furuta/O-RAN experiments or an assertion of their stability theorem.
- Wabersich/Zeilinger, https://arxiv.org/html/2105.10241v3:
  horizon constraint checking, backup and model-assumption limitations motivate
  candidate feasibility tests. The two-stage slack/terminal-barrier algorithm
  and its recovery proof are NOT reproduced by a shadow rollout.
- Akbarzadeh et al., https://arxiv.org/html/2606.24148v1:
  polynomial stochastic CBC/SOS synthesis is related work, not an interchangeable
  controller for the uncalibrated articulated PyBullet plant. Do not label a
  different controller as a reproduction of this paper.

## Candidate and shared controls, specified before development execution

Candidate `joint_plan`: enumerate 36/72/144 control-tick horizons at four-tick
knots; use byte cost and predicted useful coverage to choose horizon. Predict
the acknowledged actuator buffer from received feedback to estimated arrival;
reject candidate prefixes whose rollout clearance minus an explicitly declared
engineering uncertainty allowance falls below the unchanged checker margin.
Refresh on coverage/deadline or observed forecast error. All selected commands
remain subject to the common actual-state receiver check. No new formal safety
claim follows from finite model rollouts or an assumed uncertainty rate.
Joint decisions include waiting for byte credit when existing acknowledged
coverage can bridge the wait. All event-based comparators share suppression of
duplicate refreshes while waiting for a command acknowledgment, bounded by a
timeout based on measured return delay and the feedback sampling period.

Comparators: fixed short and long plans, a fixed 144-tick plan (so additional
coverage is not exclusive to the candidate), conventional event plans, an
event-triggered reduced-shooting MPC adaptation and its periodic counterpart;
joint fixed-horizon and no-checker ablations. Optimize comparator parameters
fairly on development data only, retaining their best valid settings. Reuse
the same source faults, state information and aggregate byte budgets.

New realism is shared by every v3 method: serialized state/buffer telemetry
charged to the same aggregate byte budget, delayed/lost return packets with
common seeded impairment draws; no perfect current sender observations.
Receiver safety checks still have current local state. Planning results become
transmittable only after measured compute latency on the task timeline;
checker deadline overruns trigger hold and are recorded. Deterministic software
fixtures may override timing only inside tests and must never become research
model results. Record all compute time and completion/fallback costs.
Controller/solver module initialization occurs before the task, like robot and
model loading; its cost is recorded separately. Per-plan optimization is timed
inside the task. This prevents one-time Python imports from masquerading as
repeated controller execution latency.

Telemetry, uncertainty, timing parameters are engineering sensitivity settings,
not source-measured clinical limits. Explicitly distinguish them from published
NetFI impairment parameters. Match byte costs and feedback/computation timing
across all methods. Keep the task unchanged and label any sensitivity strata.

## Development gate and release decision

Use only development layouts and fresh development seeds. Cover no-fault,
published delay/loss/outage profiles and the authored all-lost/bad-plan controls.
Before any new final sweep require software/byte/causality checks, preserved
constraint and task outcomes versus fixed-short, and a meaningful improvement
in communication or completion versus the strongest development comparators.
Retain development failures. Do not discard an adverse network stratum or choose
only favorable seeds. If this gate fails, do not advertise a superior method or
submit another long HPC job. Record what failed and conclude or redesign with
fresh development evidence. Development success is not final-test superiority.

A new final matrix, statistical margins and code/source/config freeze will be
declared only after development validation. No v3 HPC run is currently approved
as a completed submission, and no result is guaranteed. Fresh final task
configurations and independent workload sources are required for generalization;
new random seeds alone do not create independent surgical trials.

## Development findings recorded before any HPC submission

The first completed four-setting diagnostic (seed31001, two layouts, freefault1
and delay2, four workers) did not pass: joint control used 407.5 more management
bytes and took 0.542 s longer on average than event plans, despite improving
over fixed-plan settings. The single-worker six-setting diagnostic (seed31002,
two layouts, freefault1/delay1/delay2) passed its development gate: 6/6 safe
completions, mean management bytes 4,820.33 versus event 6,170.33 and fixed-short
9,230.67. An earlier dependency-failed run remains retained as well.

These are small, related DEVELOPMENT configurations with measured OS wall-time
and varying numerical trajectories. Do not select only the favorable run or
call them a held-out superiority result. Both completed diagnostics and all
failures stay under results/surgical_v3_dev_*. The 40-test software suite passes.
The next available entry point is a single HPC DEVELOPMENT allocation for
target-hardware timing and 300 actual integrated policy executions; it is not
a released final study. See hpc/SURGICAL_V3_DEVELOPMENT.md.

## Required LLM/RAG service evaluation

Keep all three installed checkpoints, deterministic interface, RAG/no-RAG/
missing-evidence and strict/constrained controls. Accepted service caps affect
the running scheduler after measured inference latency. Improve grounding and
approval instructions on labeled development requests only; freeze separate
authored evaluation cases and never feed scoring labels into prompts/guards.
Evaluate correct timely service changes, evidence support, authorization,
blocked valid requests, wrong activations, latency and resulting task outcomes.
No motion, geometry or safety limits are generated by an LLM. A deterministic
interface remains a first-class comparator even if it outperforms the models.
