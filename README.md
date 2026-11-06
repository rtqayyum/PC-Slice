# PC-Slice

Communication-conditioned constrained reaching for a simulated surgical instrument, with finite command plans, receiver checking, and guarded LLM/RAG service management.

## Problem

A remote surgical instrument must make useful progress near a restricted region while its command and feedback links experience delay or loss. Saving packets is insufficient if the task cannot finish, and stopping motion is insufficient if it prevents useful progress. The study asks whether task/state-aware allocation improves constrained reaching relative to fixed and event-based controls with the same transport, checker, and fallback.

## Architecture

The sender constructs timestamped joint-velocity plans using received state and acknowledgments. Commands and return telemetry share a charged byte budget. The receiver checks candidate motion from actual state and otherwise holds. Executed waypoint visits and completion are returned through the same impaired feedback channel. An LLM optionally retrieves engineering service documents; an independent guard permits only registered service-cap changes after measured request latency.

## Main findings

This release reports an opened development workload, not a held-out clinical or robot-safety validation.

| Component | Recorded finding |
| --- | --- |
| Joint allocator | 25/32 safe completions; fixed-short, event-plan, and joint/fixed-short each achieve 26/32 with less management traffic |
| Receiver checker | No observed breach trials for the checked proposal; five for its unchecked counterpart |
| RAG | Qwen top-three retrieval improves contextual correctness from 8/12 to 10/12 requests |
| Task benefit of the language interface | Every interface group completes the selected task setting; no completion advantage is established |

The reduced MPC comparators are adaptations of published ideas, not faithful reproductions of full published controllers. Negative/null results are retained.

## Start here

- [Study and reproducibility guide](docs/REPRODUCIBILITY.md): environment, exact workloads, source acquisition, inference, Slurm, and figure generation.
- [Comparison methods](docs/COMPARATORS.md): literature relationship and implementation differences.
- [Result-data dictionary](docs/DATA.md): units, denominators, provenance, and limitations.
- `study/observations.json`: all 288 system and 300 integrated policy observations used in the reported comparisons.
- `study/checkpoints.json`: recorded model revisions and checkpoint-manifest hashes.
- `src/`: dependency-closed experimental implementation.
- `figures/`: TikZ/PGFPlots sources and observed plotting tables.

Regenerate plot data without a GPU, model, simulator, or hosted inference:

```bash
python3 tools/generate_plot_data.py --records study/observations.json --output figures/data
python3 tools/verify_release.py
```

No patient data or recorded surgeon requests are included. The reaching geometry and policy requests are authored engineering cases. Network conditions emulate published parameters; they are not measured cellular traces.

## Citation and license status

See [CITATION.cff](CITATION.cff) and [third-party notices](THIRD_PARTY_NOTICES.md). Manuscript author metadata and the original-code license remain to be supplied by the authors. This public source snapshot does not assign a new open-source license to third-party materials or model weights.
