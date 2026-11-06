"""Exploratory V3 scheduler diagnostics; no model inference or held-out claims.

Sequential execution avoids CPU-worker contention that would directly alter
the measured planning latency injected into the simulation timeline.
Original V3 results and source files are not overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from .surgical_campaign import load_config, networks, run_trial, runtime
from .prepare_surgical_sources import verify

VARIANTS = {
    "FS": ("fixed_plan_short", {}),
    "EV": ("event_plan", {}),
    "JS": ("joint_fixed_short", {}),
    "JP": ("joint_plan", {}),
    "JP_G0": ("joint_plan", {"uncertainty_growth_m_s": 0.0}),
    "JP_G05": ("joint_plan", {"uncertainty_growth_m_s": 0.0005}),
    "JP_P1": ("joint_progress", {"progress_deficit_weight": 1.0}),
    "JP_P3": ("joint_progress", {"progress_deficit_weight": 3.0}),
}
PROFILES = ["freefault1", "delay2", "packetloss3", "communicationloss2"]


def run(output, mode):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    cfg = load_config("config/surgical_task_v3.json")
    if cfg["backend"] != "packetized-v3":
        raise ValueError("Expected V3 transport")
    verify(cfg["source_root"])
    available = networks(cfg)
    if any(name not in available for name in PROFILES):
        raise ValueError("Missing pinned network profile")
    layouts = [x for x in cfg["layouts"] if x["split"] == "development"]
    if not layouts:
        raise ValueError("No development layouts")
    if mode == "pilot":
        layouts, profiles, seeds, variants = layouts[:1], ["freefault1", "packetloss3"], [31031], ["FS", "EV", "JP", "JP_P1"]
    else:
        profiles, seeds, variants = PROFILES, [31031, 31032, 31033], list(VARIANTS)
    planned = [(lay, n, seed, variant) for lay in layouts for n in profiles for seed in seeds for variant in variants]
    here = Path(__file__).resolve()
    source_hash = hashlib.sha256(here.read_bytes()).hexdigest()
    transport_hash = hashlib.sha256((here.parent / "surgical_joint_transport.py").read_bytes()).hexdigest()
    manifest = {
        "scope": "EXPLORATORY DEVELOPMENT ONLY: no frozen held-out test",
        "mode": mode, "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "runtime": runtime(),
        "scorer_sha256": hashlib.sha256((here.parent/"surgical_scheduler_scoring.py").read_bytes()).hexdigest(),
        "runner_sha256": source_hash, "transport_sha256": transport_hash,
        "network_profiles": profiles, "seeds": seeds, "variants": variants,
        "layouts": layouts, "planned_trials": len(planned),
        "note": "Planning duration is measured and affects simulated readiness; execute sequentially and record hardware.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    for i, (layout, profile, seed, variant) in enumerate(planned, 1):
        method, overrides = VARIANTS[variant]
        config = {**cfg, **overrides}
        spec = {"layout": layout, "network": available[profile], "budget": 24,
                "seed": seed, "method": method, "stress": "nominal"}
        folder = root / "trials" / f"{layout['id']}_{profile}_{seed}_{variant}"
        try:
            run_trial(config, spec, folder)
            m = json.loads((folder / "metrics.json").read_text())
        except Exception as exc:
            (root / "ERROR.json").write_text(json.dumps({
                "index": i, "layout": layout["id"], "profile": profile,
                "seed": seed, "variant": variant, "error": repr(exc)}, indent=2)+"\n")
            raise
        row = {
            "layout": layout["id"], "network": profile, "seed": seed,
            "variant": variant, "method": method, "overrides": overrides,
            "safe": bool(m["safe_task_success"]), "task_success": bool(m["task_success"]),
            "breaches": int(m["breach_count"]),
            "bytes": int(m["management_bytes"]),
            "control_bytes": int(m["control_bytes"]),
            "feedback_bytes": int(m["feedback_bytes"]),
            "planning_seconds": float(m["planning_seconds"]),
            "planning_infeasible": int(m["planning_infeasible"]),
            "active_fallback": float(m["active_fallback_fraction"]),
            "capped_time_s": float(m["completion_time_s"]) if m["safe_task_success"] else cfg["duration_s"],
            "selected_horizons_ticks": m["selected_horizons_ticks"],
        }
        rows.append(row)
        (root / "observations.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(f"[{i}/{len(planned)}] {layout['id']} {profile} seed={seed} {variant} "
              f"safe={int(row['safe'])} breaches={row['breaches']} bytes={row['bytes']} "
              f"planning={row['planning_seconds']:.2f}s", flush=True)
    indexed = {(r["layout"],r["network"],r["seed"],r["variant"]):r for r in rows}
    matched = []
    for v in variants:
        if v == "JP":
            continue
        pairs = []
        for r in rows:
            if r["variant"] == "JP":
                b = indexed.get((r["layout"],r["network"],r["seed"],v))
                if b is not None:
                    pairs.append((r,b))
        if pairs:
            matched.append({"variant": v, "n": len(pairs),
                "jp_safe": sum(a["safe"] for a,b in pairs),
                "comparison_safe": sum(b["safe"] for a,b in pairs),
                "candidate_only_successes":sum(b["safe"] and not a["safe"] for a,b in pairs),
                "candidate_only_failures":sum(a["safe"] and not b["safe"] for a,b in pairs),
                "mean_comparison_minus_jp_bytes":sum(b["bytes"]-a["bytes"] for a,b in pairs)/len(pairs),
                "mean_comparison_minus_jp_capped_time_s":sum(b["capped_time_s"]-a["capped_time_s"] for a,b in pairs)/len(pairs),
                "comparison_breach_trials":sum(b["breaches"]>0 for a,b in pairs)})
    summary = {"development_only": True, "mode":mode, "complete":True,
               "n":len(rows), "comparisons_to_JP": matched,
               "interpretation":"Exploratory matched differences. No statistical or clinical efficacy conclusion. Hardware/planning-time effects may confound results."}
    (root/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("DONE", root/"SUMMARY.json",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["pilot","development"],required=True)
    p.add_argument("--output-dir",required=True)
    a=p.parse_args()
    run(a.output_dir,a.mode)


if __name__ == "__main__":
    main()
