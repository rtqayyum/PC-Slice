# Reported data and units

`study/observations.json` contains 588 derived per-execution records, preserving the measured fields from the audited archive. Duplicate prompt/response objects and machine-local paths are omitted; evaluated policy decisions are retained in `policy_detail`.

- `stage`: `systems` (288) or `policy` (300).
- `trial_id`: stable archived execution identifier.
- `safe_task_success`: actual ordered-waypoint and goal-dwell completion with no observed breach.
- `completion_time_s`: observed time, null if incomplete. Completion curves divide safe completions by all 32 system cases, retaining failures.
- `final_error_m`: final goal distance in meters, including incomplete trials.
- `minimum_clearance_m`: minimum across recorded physics substeps and declared sphere/workspace constraints. Negative means a declared breach, not necessarily tissue penetration.
- `management_bytes`: command + feedback wire bytes, including lost packets and repeated terminal feedback. Decimal kB in figures. Background traffic is separate but charged to the aggregate budget.
- `active_fallback_fraction`: fallback during the active task; excludes completed hold.
- `planning_seconds_p95`: per-trial p95 wall time per planning call, not cumulative planning time.
- `planning_available_after_start`: transmitted plans available after their original start; usable late prefixes are retained.
- `expired_unsent_plans`: already-expired plans discarded before transmission.
- `policy_detail`: evaluated action/schema/retrieval, measured request latency, and activation state.

Plots show medians, interquartile intervals, or observed ranges as labeled. These are descriptive summaries of opened cases, not confidence intervals or clinical population estimates. Policy groups have twelve authored requests, not twelve independent surgeries. The activation coverage denominator is five originally valid service-selection requests; the missing-evidence negative control can correctly abstain instead of fulfilling them.

`study/provenance.json` links the initial manuscript data to archived input SHA-256 hashes. `study/frozen_source_hashes.json` records original and published code hashes. `study/checkpoints.json` records the downloaded model revisions and original manifest identities. Instrument meshes and model weights are acquired from their upstream sources, not committed here.
