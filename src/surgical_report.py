"""Task-first descriptive results; no automatic publication/safety declaration."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .policy_benchmark import digest


NUMERIC = ["safe_task_success", "task_success", "timeout", "minimum_clearance_m",
           "breach_duration_s", "maximum_penetration_m", "final_error_m",
           "control_packets", "background_packet_quanta", "control_bytes",
           "active_fallback_fraction", "fallback_breach_duration_s", "p95_active_age_s"]
OPTIONAL_NUMERIC = ['planning_seconds', 'checker_rejections', 'expired_plan_arrivals',
                    'skipped_plan_prefix_ticks', 'future_plans_received', 'buffer_empty_ticks',
                    'feedback_bytes','management_bytes','feedback_received','planning_jobs',
                    'planning_infeasible','checker_deadline_misses','solver_calls','solver_failures']


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    results = []
    for key, group in sorted(groups.items()):
        out = dict(zip(group_keys, key))
        out["trials"] = len(group)
        for metric in NUMERIC:
            values = [float(r[metric]) for r in group if r[metric] is not None]
            out[metric] = float(np.mean(values)) if values else None
        for metric in OPTIONAL_NUMERIC:
            values = [float(r[metric]) for r in group if r.get(metric) is not None]
            out[metric] = float(np.mean(values)) if values else None
        # Completion time conditional on success must never hide failures.
        values = [r["completion_time_s"] for r in group if r["completion_time_s"] is not None]
        out["successful_completion_time_s"] = float(np.mean(values)) if values else None
        out["any_breach_rate"] = sum(r["breach_count"] > 0 for r in group) / len(group)
        out["worst_clearance_m"] = min(r["minimum_clearance_m"] for r in group)
        out["worst_penetration_m"] = max(r["maximum_penetration_m"] for r in group)
        out["p95_breach_duration_s"] = float(np.quantile([r["breach_duration_s"] for r in group], .95))
        out["p95_control_packets"] = float(np.quantile([r["control_packets"] for r in group], .95))
        if group[0].get("policy"):
            for metric in ["contextual_correct", "original_request_correct", "wrong_service_activation",
                           "wrong_service_authorization", "valid_request_blocked",
                           "retrieval_recall", "retrieval_precision"]:
                out[metric] = float(np.mean([r["policy"]["scores"][metric] for r in group]))
            out["authorized_fraction"] = float(np.mean([r["policy"]["enforcement"]["authorized"] for r in group]))
            out["activation_fraction"] = float(np.mean([r["policy"]["activated"] for r in group]))
            out["schema_valid_fraction"] = float(np.mean([r["policy"]["enforcement"].get("schema_valid", False) for r in group]))
            out["p95_request_seconds"] = float(np.quantile([r["policy"]["measured_request_seconds"] for r in group], .95))
            out["mean_retrieval_seconds"] = float(np.mean([r["policy"]["retrieval_seconds"] for r in group]))
            out["mean_validation_seconds"] = float(np.mean([r["policy"]["validation_seconds"] for r in group]))
            out["response_within_trial_fraction"] = float(np.mean([r["policy"]["response_within_trial"] for r in group]))
            out['response_after_completion_fraction'] = float(np.mean([
                r['policy'].get('task_already_complete', False) for r in group]))
        results.append(out)
    return results


def policy_decisions(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"], r["decoding"])].append(r)
    result = []
    for key, group in sorted(groups.items()):
        metrics = {}
        f1s = []
        confusion = defaultdict(int)
        for r in group:
            scores = r["policy"]["scores"]
            confusion[(scores["contextual_oracle_action"], scores["predicted_action"])] += 1
        for action in ["accept", "reject", "abstain", "request_approval"]:
            tp = confusion[(action, action)]
            predicted = sum(n for (truth, pred), n in confusion.items() if pred == action)
            actual = sum(n for (truth, pred), n in confusion.items() if truth == action)
            precision, recall = tp / max(1, predicted), tp / max(1, actual)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            metrics.update({action + "_precision": precision, action + "_recall": recall, action + "_f1": f1})
            f1s.append(f1)
        result.append({"model": key[0], "condition": key[1], "decoding": key[2],
                       "executions": len(group), "authored_request_families": len({r["case_id"] for r in group}),
                       "contextual_macro_f1": float(np.mean(f1s)), **metrics,
                       "confusion": json.dumps({truth + "->" + pred: n for (truth, pred), n in confusion.items()})})
    return result


def comparisons(rows):
    """Paired layout-level descriptive intervals, not patient-level inference.

    Average paired network/seed/budget contrasts WITHIN authored task layouts,
    then resample layouts. Six engineered layouts are not independent surgeons.
    """
    paired = defaultdict(dict)
    for r in rows:
        if r["stress"] == "nominal":
            paired[(r["layout"], r["network"], r["seed"], r["budget_packets_s"])][r["method"]] = r
    rng = np.random.default_rng(8201)
    result = []
    packetized = any(r['method'] == 'adaptive_plan' for r in rows)
    proposed = 'adaptive_plan' if packetized else 'task_checked'
    baselines = ['instant_event_checked', 'fixed_plan_short', 'fixed_plan_long', 'event_plan', 'adaptive_no_checker'] if packetized else ['fixed', 'event', 'fixed_checked', 'event_checked']
    joint=any(r['method']=='joint_plan' for r in rows)
    if joint:
        proposed='joint_plan'
        baselines=sorted({r['method'] for r in rows}-{'joint_plan'})
    metrics=["safe_task_success", "task_success", "control_packets", "control_bytes", "final_error_m", "breach_duration_s", "active_fallback_fraction"]
    if joint:
        metrics+=['management_bytes','feedback_bytes','checker_deadline_misses']
    for baseline in baselines:
        for metric in metrics:
            layout_differences = defaultdict(list)
            for key, methods in paired.items():
                if baseline in methods and proposed in methods:
                    layout_differences[key[0]].append(float(methods[proposed][metric]) - float(methods[baseline][metric]))
            values = np.array([np.mean(v) for v in layout_differences.values()])
            if not len(values):
                continue
            boot = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
            result.append({"proposed": proposed, "baseline": baseline, "metric": metric,
                "paired_layouts": len(values), "mean_difference": float(values.mean()),
                "descriptive_95_interval": np.quantile(boot, [.025, .975]).tolist(),
                "unit": "authored simulation layout, not recording/operator/patient",
                "multiplicity": "unadjusted descriptive intervals; not confirmatory significance tests"})
    return result


def policy_task_contrasts(rows):
    """Matched task differences, retained even when LLM/RAG gives no benefit."""
    def key(r):
        return tuple(r[k] for k in ['layout', 'network', 'seed', 'budget_packets_s', 'case_id'])
    deterministic = {key(r): r for r in rows if r['model'] == 'deterministic'}
    no_rag = {(key(r), r['model'], r['decoding']): r for r in rows if r['condition'] == 'no_rag'}
    groups = defaultdict(list)
    for row in rows:
        if row['model'] == 'deterministic':
            continue
        comparators = [('deterministic', deterministic.get(key(row)))]
        if row['condition'] != 'no_rag':
            comparators.append(('same_model_no_rag', no_rag.get((key(row), row['model'], row['decoding']))))
        for baseline, reference in comparators:
            if reference is not None:
                groups[(row['model'], row['condition'], row['decoding'], baseline)].append((row, reference))
    results = []
    for (model, condition, decoding, baseline), pairs in sorted(groups.items()):
        result = dict(model=model, condition=condition, decoding=decoding, baseline=baseline,
                      matched_executions=len(pairs), authored_layouts=len({r['layout'] for r, _ in pairs}),
                      inference_scope='descriptive paired simulations; requests/layouts are correlated')
        for metric in ['safe_task_success', 'breach_duration_s', 'final_error_m', 'control_bytes', 'active_fallback_fraction']:
            result[metric + '_difference'] = float(np.mean([float(a[metric])-float(b[metric]) for a,b in pairs]))
        results.append(result)
    return results


def report(output):
    output = Path(output)
    paths = sorted(output.glob("system/*/metrics.json")) + sorted(output.glob("policy/*/metrics.json"))
    if not paths:
        raise ValueError("No completed research trials; validation fixtures are not research results")
    from .surgical_campaign import completed
    for path in paths:
        if not completed(path.parent):
            raise ValueError("Incomplete or unverified trial: " + str(path))
    rows = [json.loads(p.read_text()) for p in paths]
    if any("fixture" in r["model"].lower() or "oracle" in r["model"].lower() for r in rows):
        raise ValueError("Test fixtures must not be presented as research/model results")
    plan_path = output / "execution_plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"system": [], "policy": []}
    actual = {stage: {p.parent.name for p in paths if p.parent.parent.name == stage} for stage in ["system", "policy"]}
    missing = {stage: sorted(set(plan[stage]) - actual[stage]) for stage in actual}
    unexpected = {stage: sorted(actual[stage] - set(plan[stage])) for stage in actual}
    if plan_path.exists() and any(unexpected.values()):
        raise ValueError("Unexpected execution directories; do not mix different campaign versions")
    status = {"complete_matrix": plan_path.exists() and not any(missing.values()),
              "missing_executions": missing, "completed": {stage: len(ids) for stage, ids in actual.items()},
              "expected": {stage: len(ids) for stage, ids in plan.items()},
              "clinical_validation": False, "physical_safety_proof": False,
              "hypothesis_confirmed_automatically": False, "publication_guaranteed": False}
    (output / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n")
    system = [r for r in rows if r["model"] == "none"]
    policy = [r for r in rows if r["model"] != "none"]
    compact = [{"trial_id": p.parent.name, "artifact_directory": str(p.parent.relative_to(output)),
                **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}}
               for p, r in zip(paths, rows)]
    write_csv(output / "trial_metrics.csv", compact)
    main = summary(system, ["stress", "method", "budget_packets_s"])
    detailed = summary(system, ["stress", "method", "network", "budget_packets_s", "layout"])
    policies = summary(policy, ["model", "condition", "decoding", "case_id"])
    write_csv(output / "system_summary.csv", main)
    write_csv(output / "system_by_layout_network.csv", detailed)
    write_csv(output / "policy_task_summary.csv", policies)
    write_csv(output / 'policy_by_task.csv', summary(policy, ['model', 'condition', 'decoding', 'case_id', 'layout', 'network', 'budget_packets_s']))
    write_csv(output / "policy_decision_metrics.csv", policy_decisions(policy))
    (output / 'policy_task_comparisons.json').write_text(json.dumps(policy_task_contrasts(policy), indent=2) + '\n')
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons(system), indent=2) + "\n")
    lines = ["# PC-Slice surgical-instrument reaching experiment", "",
        "Matrix status: " + ("COMPLETE (not a scientific-success verdict)." if status["complete_matrix"] else "PARTIAL / no complete frozen matrix established."), "",
        f"Completed system/stress executions: {len(system)}. Integrated policy executions: {len(policy)}.", "",
        "The instrument is the pinned SurRoL articulated PSM in PyBullet. This is an authored constrained-reaching extension, not recorded surgeon behavior or a clinical trial.", "",
        "## Task outcomes at matched shared packet-service budgets", "",
        "| Method | Budget packet-equivalents/s | Trials | Safe task success | Any breach | Final error (mm) | Control packets | Control bytes | Active fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in main:
        if r["stress"] == "nominal":
            lines.append(f"| {r['method']} | {r['budget_packets_s']} | {r['trials']} | {r['safe_task_success']:.3f} | {r['any_breach_rate']:.3f} | {1000*r['final_error_m']:.2f} | {r['control_packets']:.1f} | {r['control_bytes']:.1f} | {r['active_fallback_fraction']:.3f} |")
    lines += ["", "## Interpretation and claim boundaries", "",
        "- Primary outcomes are task completion with constraint preservation, breaches, placement error, interruptions and resource trade-offs—not JSON accuracy or packet savings alone.",
        "- Failed tasks/timeouts remain in denominators. Completion time is separately reported conditional on success. Completed-task hold is not counted as fallback.",
        "- The shadow checker predicts finite-step command-and-brake execution in the same physics engine. It is not a formal instrument-safety proof; model mismatch and observation stresses are separately reported.",
        "- Constraint clearance covers a spherical tool-tip exclusion and workspace only. No tissue contact forces, whole-shaft collision safety or patient outcomes are established.",
        "- Restricted geometry, reference controller and service policies are authored engineering specifications, not clinical thresholds. Multiple layouts/seeds are not independent surgical recordings.",
        "- Shared service quanta and saturated background traffic are packet-service abstractions, not measured video quality, reserved radio bandwidth, PRBs or validated 5G/6G slicing.",
        "- Policy proposals are generated on demand for the executing task; measured request latency is injected into its discrete-event timeline. This is not real-time concurrent clinical deployment or an archived executor lookup.",
        "- Authorization does not prove semantic intent matching; wrong registered-service activations and valid-request rejection must be inspected.",
        "- No automatic novelty, publication-readiness or positive-result gate. Retain null and adverse outcomes.", "",
        "See system_by_layout_network.csv, policy_task_summary.csv, paired_comparisons.json, per-execution trajectories/events/policy outputs, and the frozen run/source/checkpoint manifests."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    # Modest static scientific figures; no generated illustrations.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(7, 4.5))
    joint=any(r['method']=='joint_plan' for r in rows)
    resource='management_bytes' if joint else 'control_bytes'
    for method in sorted({r["method"] for r in main}):
        selected = [r for r in main if r["method"] == method and r["stress"] == "nominal"]
        ax.plot([r[resource] for r in selected], [r["safe_task_success"] for r in selected], "o-", label=method)
    ax.set(xlabel="Mean control + feedback bytes per trial" if joint else "Mean control bytes per fixed-duration trial", ylabel="Task success with no sampled constraint breach", ylim=(-.03, 1.03))
    ax.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "task_resource_tradeoff.png", dpi=180)
    plt.close(figure)
    # Predeclared examples, not a search for favorable failures/successes.
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    example_count = 0
    packetized = any(r['method'] == 'adaptive_plan' for r in rows)
    for column, network in enumerate(['freefault1', 'delay2'] if packetized else ['freefault1', 'delay1']):
        projected_radius_mm = None
        packetized = any(r['method'] == 'adaptive_plan' for r in rows)
        example_methods = ['instant_event_checked', 'fixed_plan_long', 'event_plan', 'adaptive_plan'] if packetized else ['fixed', 'fixed_checked', 'task_checked']
        for method in example_methods:
            chosen = next((p for p, r in zip(paths, rows) if r["model"] == "none" and
                r["layout"] == ('v2_test_yplus_a' if packetized else 'test_yplus_low') and r["network"] == network and
                r["budget_packets_s"] == 24 and r["seed"] == (11001 if packetized else 5101) and
                r["stress"] == "nominal" and r["method"] == method), None)
            if chosen is None:
                continue
            with gzip.open(chosen.parent / "trajectory.jsonl.gz", "rt") as stream:
                trace = [json.loads(line) for line in stream]
            with gzip.open(chosen.parent / "events.jsonl.gz", "rt") as stream:
                initial = json.loads(next(stream))
            center = np.array(initial["restricted_center_m"])
            projected_radius_mm = 1000 * (initial["restricted_radius_m"] + initial["tip_radius_m"])
            xyz = (np.array([r["tip_m"] for r in trace]) - center) * 1000
            axes[0, column].plot(xyz[:, 0], xyz[:, 1], label=method)
            axes[1, column].plot([r["time_s"] for r in trace],
                                 [r["min_substep_clearance_m"] * 1000 for r in trace], label=method)
            example_count += 1
        axes[0, column].set(title=network + ": tool-tip path (XY projection)", xlabel="x relative to region centre (mm)", ylabel="y (mm)")
        axes[0, column].set_aspect("equal", adjustable="box")
        if projected_radius_mm is not None:
            axes[0, column].add_patch(plt.Circle((0, 0), projected_radius_mm, color="red", alpha=.12,
                                                label="Region + tip radius (projection)"))
        axes[1, column].axhline(0, color="red", linestyle="--")
        axes[1, column].set(xlabel="Simulation time (s)", ylabel="Minimum substep clearance (mm)")
        if example_count:
            axes[0, column].legend(fontsize=6)
            axes[1, column].legend(fontsize=6)
    if example_count:
        figure.tight_layout()
        figure.savefig(output / "predeclared_task_examples.png", dpi=180)
    plt.close(figure)
    hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    (output / "artifact_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Surgical task report:", output / "REPORT.md", flush=True)
    return status
