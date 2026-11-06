"""Task-centered experiment orchestration. No old replay-to-policy joins.

Stages: validation (development layouts only), paired systems/stress study,
three-model on-demand LLM/RAG task executions, evidence-linked reporting.
"""
import argparse
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile

from .build_policy_cases import parse_profiles, SOURCE_PATH
from .policy_benchmark import digest
from .prepare_surgical_sources import verify


def code_hashes():
    paths = sorted(list(Path("src").glob("*.py")) + list(Path("config").glob("surgical*.json")) +
                   list(Path("tests").glob("test_surgical*.py")) +
                   [Path("requirements-surgical.txt"), Path("requirements-surgical-inference.txt"),
                    Path("hpc/submit_surgical.sh"), Path("hpc/surgical.slurm"),
                    Path("experiments/SURGICAL_TASK_EXPERIMENT.md"), Path("PROJECT_CHARTER.md"),
                    Path("EVALUATION_PROTOCOL.md"), Path("AGENTS.md")])
    paths += list(Path('hpc').glob('*surgical_v[23]*')) + list(Path('hpc').glob('*SURGICAL_V[23]*')) + list(Path('experiments').glob('SURGICAL_TASK_V[23].md'))
    return {str(p): digest(p) for p in paths if p.is_file()}


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError('Cyclic configuration inheritance')
    seen.add(path)
    config = json.loads(path.read_text())
    if 'extends' in config:
        parent = load_config(path.parent / config.pop('extends'), seen)
        def merge(base, changes):
            result = dict(base)
            for key, value in changes.items():
                result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
            return result
        config = merge(parent, config)
    if config["physics_hz"] % config["control_hz"]:
        raise ValueError("Physics frequency must be an integer multiple of control frequency")
    if config["duration_s"] <= 0 or config["goal_dwell_s"] <= 0:
        raise ValueError("Positive duration/dwell required")
    if len({l["id"] for l in config["layouts"]}) != len(config["layouts"]):
        raise ValueError("Duplicate task layout ID")
    if config.get('backend') in {'packetized-v2','packetized-v3'}:
        for key in ['short_plan_ticks', 'long_plan_ticks', 'plan_stride_ticks', 'max_lead_ticks',
                    'fixed_short_period_ticks', 'fixed_long_period_ticks', 'refresh_guard_ticks']:
            if not isinstance(config[key], int) or config[key] < 1:
                raise ValueError('Positive integer required: ' + key)
        if config['telemetry_delay_ticks'] < 1 or config['event_error_m'] <= 0 or config['adaptive_min_error_m'] <= 0:
            raise ValueError('Positive delayed telemetry/error thresholds required')
        if any(config[k] % config['plan_stride_ticks'] for k in ['short_plan_ticks', 'long_plan_ticks']):
            raise ValueError('Plan horizons must be multiples of stride')
    if config.get('backend') == 'packetized-v3':
        if not 0 < config['feedback_share'] < 1 or config['feedback_period_ticks'] < 1:
            raise ValueError('Invalid feedback allocation/sampling')
        if any(h < 1 or h % config['plan_stride_ticks'] for h in config['plan_horizons_ticks']):
            raise ValueError('Invalid candidate plan horizon')
    return config


def networks(config):
    root = Path(config["network_source_root"])
    manifest = json.loads((root / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise ValueError("Archived network source modified: " + entry["path"])
    return parse_profiles((root / SOURCE_PATH).read_text())


def validate(config, artifact_dir=None):
    """Engineering implementation gate, NOT a held-out efficacy experiment."""
    import numpy as np
    from .surgical_robot import Instrument, PredictiveChecker
    from .surgical_simulation import simulate
    verify(config["source_root"])
    profiles = networks(config)
    layout = next(l for l in config["layouts"] if l["split"] == "development")
    r = Instrument(config, layout)
    checker = PredictiveChecker(config, layout)
    try:
        error = float(np.linalg.norm(r.tip() - r.start))
        if error > 0.001 or r.clearance() <= 0:
            raise ValueError("Instrument initialization/geometry failed")
        before = r.tip().copy()
        q_before = r.state()[0].copy()
        checker.check(r, r.command(r.waypoints[0]))
        if not np.array_equal(q_before, r.state()[0]) or not np.array_equal(before, r.tip()):
            raise ValueError("Shadow checker mutated live robot")
        reference = r.waypoints[0]
        for _ in range(config["control_hz"]):
            r.set_velocity(r.command(reference))
            for _ in range(r.substeps):
                r.physics_step()
        progress = float(np.linalg.norm(before - reference) - np.linalg.norm(r.tip() - reference))
        if progress <= 0.005:
            raise ValueError("Instrument does not physically follow the reaching command")
    finally:
        r.close()
        checker.close()
    if artifact_dir is not None:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    manager = nullcontext(str(artifact_dir)) if artifact_dir is not None else tempfile.TemporaryDirectory(prefix="pcslice_surgical_gate_")
    extra = {}
    with manager as folder:
        method = config.get('proposed_method', 'fixed_checked')
        nominal = simulate(config, layout, profiles["freefault1"], 24, 9101, method, Path(folder) / "nominal")
        # Complete outage is an authored boundary test, not a measured trace.
        outage = simulate(config, layout, {"id": "validation_all_lost", "kind": "communication_loss",
            "parameters": [1, 100000, 100000, 0]}, 24, 9101, method, Path(folder) / "outage")
        if config.get('backend') == 'packetized-v2':
            delay = {'id': 'development_constant_220ms', 'kind': 'delay', 'parameters': [220, [1], [1e9]]}
            extra['delay_220ms'] = simulate(config, layout, delay, 24, 9101, method, Path(folder) / 'delay_220ms')
            short = {**config, 'duration_s': 6.}
            for name, comparator in [('fault_checked', 'adaptive_plan'), ('fault_unchecked', 'adaptive_no_checker')]:
                extra[name] = simulate(short, layout, profiles['freefault1'], 24, 9101,
                                       comparator, Path(folder) / name, stress='unsafe_plan')
            if (not extra['delay_220ms']['safe_task_success'] or
                extra['fault_checked']['checker_rejections'] == 0 or
                extra['fault_checked']['breach_count'] != 0 or
                extra['fault_unchecked']['breach_count'] == 0):
                raise ValueError('Finite-plan delay/fault development gate failed')
    if not nominal["safe_task_success"] or outage["delivered_commands"] != 0 or outage["task_success"]:
        raise ValueError("Nominal task or no-communication negative-control gate failed")
    return {"status": "PASS", "scope": "development implementation checks, not final research results",
            "initialization_error_m": error, "one_second_progress_m": progress,
            "nominal": nominal, "total_outage": outage, **extra}


def trial_specs(config, policy=False):
    profiles = networks(config)
    layouts = [l for l in config["layouts"] if l["split"] == "test"]
    if policy:
        settings = config["policy"]
        layouts = [l for l in layouts if l["id"] in settings["layouts"]]
        for l, profile, budget, seed in itertools.product(layouts, settings["network_profiles"],
                                                        settings["budgets_packets_s"], settings["seeds"]):
            yield {"layout": l, "network": profiles[profile], "budget": budget,
                   "seed": seed, "method": config.get('proposed_method', 'task_checked'), "stress": "nominal"}
    else:
        for l, profile, budget, seed, method in itertools.product(layouts, config["network_profiles"],
              config["budgets_packets_s"], config["seeds"], config["methods"]):
            yield {"layout": l, "network": profiles[profile], "budget": budget,
                   "seed": seed, "method": method, "stress": "nominal"}
        # Assumption-breaking stress is separately stratified, not pooled into
        # nominal efficacy or presented as covered by a safety guarantee.
        for l, profile, stress, method in itertools.product(layouts, config.get('stress_networks', ["freefault1", "delay2"]),
              [s for s in config["stress_conditions"] if s != "nominal"], config.get('stress_methods', ["fixed_checked", "task_checked"])):
            yield {"layout": l, "network": profiles[profile], "budget": 24,
                   "seed": 7101, "method": method, "stress": stress}


def trial_id(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


def execution_plan(config, policy):
    system = [trial_id(s) for s in trial_specs(config)]
    policy_ids = []
    for base, case in itertools.product(trial_specs(config, policy=True), policy["cases"]):
        policy_ids.append(trial_id({**base, "case": case, "condition": "no_rag", "model": "deterministic"}))
        for model, condition, decoding in itertools.product(config["policy"]["models"],
                config["policy"]["conditions"], config["policy"]["decoding"]):
            policy_ids.append(trial_id({**base, "case": case, "condition": condition,
                                       "model": model, "decoding": decoding}))
    if len(set(system)) != len(system) or len(set(policy_ids)) != len(policy_ids):
        raise ValueError("Duplicate planned execution IDs")
    return {"system": system, "policy": policy_ids}


def completed(folder):
    marker = folder / "completion.json"
    if not marker.exists():
        return False
    hashes = json.loads(marker.read_text())
    if not isinstance(hashes, dict) or not {"metrics.json", "events.jsonl.gz", "trajectory.jsonl.gz"} <= set(hashes):
        raise ValueError("Incomplete completion marker: " + str(marker))
    for name, sha in hashes.items():
        if digest(folder / name) != sha:
            raise ValueError("Completed trial artifact was modified: " + str(folder / name))
    return True


def run_trial(config, spec, folder, policy=None, engine=None):
    from .surgical_simulation import simulate
    folder = Path(folder)
    if completed(folder):
        return str(folder)
    public = {k: v for k, v in spec.items() if k not in {"case", "condition", "model", "decoding"}}
    simulate(config, **public, output=folder, policy_case=spec.get("case"), policy=policy,
             engine=engine, condition=spec.get("condition", "no_rag"), model_id=spec.get("model", "none"),
             decoding=spec.get("decoding", "strict"))
    hashes = {p.name: digest(p) for p in sorted(folder.iterdir())
              if p.is_file() and p.name not in {"completion.json", "completion.json.tmp"}}
    temporary = folder / "completion.json.tmp"
    temporary.write_text(json.dumps(hashes, indent=2) + "\n")
    temporary.replace(folder / "completion.json")
    return str(folder)


def _worker(args):
    return run_trial(*args)


def runtime():
    packages = {}
    for package in ["numpy", "scipy", "pybullet", "matplotlib", "torch", "transformers", "accelerate", "safetensors", "lm-format-enforcer"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "job_id": os.environ.get("SLURM_JOB_ID"), "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def preflight_models(config):
    from transformers import AutoConfig, AutoTokenizer
    policy = json.loads(Path(config.get('policy_path','config/surgical_policy.json')).read_text())
    from .surgical_policy import documents_for, messages_for
    manifests = {}
    for name in config["policy"]["models"]:
        print("Verifying checkpoint hashes and tokenizer budget:", name, flush=True)
        root = Path("models") / name
        marker = root / "pcslice_checkpoint.json"
        installed = json.loads(marker.read_text())
        if installed.get("status") != "COMPLETE":
            raise ValueError("Checkpoint incomplete: " + name)
        for file, sha in installed["files"].items():
            if digest(root / file) != sha:
                raise ValueError("Checkpoint modified: " + name + "/" + file)
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        model_config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        for case, condition in itertools.product(policy["cases"], config["policy"]["conditions"]):
            messages = messages_for(case, policy, documents_for(case, policy, condition), "approach")
            if name.startswith("Mistral-"):
                messages = [{"role": "user", "content": messages[0]["content"] + "\n" + messages[1]["content"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if count > config["policy"]["max_input_tokens"] or count + config["policy"]["max_new_tokens"] > model_config.max_position_embeddings:
                raise ValueError("Tokenizer budget exceeded: " + name)
        manifests[name] = digest(marker)
    return manifests


def run(config_path, output, workers, resume=False):
    config = load_config(config_path)
    if config.get('backend') == 'packetized-v3':
        raise ValueError('V3 is development only; no released final matrix. Run src.surgical_v3_development and inspect its gate.')
    policy = json.loads(Path(config.get('policy_path','config/surgical_policy.json')).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "code_hashes": code_hashes(),
        "source_manifest_sha256": digest(Path(config["source_root"]) / "pcslice_source_manifest.json"),
        "network_manifest_sha256": digest(Path(config["network_source_root"]) / "source_manifest.json"),
        "policy_sha256": digest("config/surgical_policy.json")}
    marker = output / "run_manifest.json"
    current_runtime = runtime()
    versions = current_runtime["packages"]
    if (not str(versions["torch"]).startswith("2.7.1") or versions["transformers"] != "4.51.3"
            or versions["pybullet"] != "3.2.7" or versions["lm-format-enforcer"] != "0.10.11"):
        raise ValueError("HPC inference/physics versions differ from the tested protocol; run submit_surgical.sh setup")
    if marker.exists():
        previous = json.loads(marker.read_text())
        if not resume or previous["identity"] != identity:
            raise ValueError("Existing run requires --resume and identical code/config/source hashes")
        if previous["runtime"]["packages"] != current_runtime["packages"]:
            raise ValueError("Resume dependency versions changed; preserve this run and create a new version")
    else:
        if any(output.iterdir()):
            raise ValueError("Use an empty run folder or a compatible --resume folder")
        marker.write_text(json.dumps({"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                                     "runtime": current_runtime}, indent=2) + "\n")
    with (output / "execution_sessions.jsonl").open("a") as stream:
        stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "runtime": current_runtime}) + "\n")
    (output / "execution_plan.json").write_text(json.dumps(execution_plan(config, policy), indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps(validate(config, output / "validation_trials"), indent=2) + "\n")
    checkpoint_hashes = preflight_models(config)
    model_manifest = output / "model_hashes.json"
    if model_manifest.exists() and json.loads(model_manifest.read_text()) != checkpoint_hashes:
        raise ValueError("Resume checkpoints changed")
    model_manifest.write_text(json.dumps(checkpoint_hashes, indent=2) + "\n")
    # Fail on actual CUDA/model generation before the long CPU stage, not after.
    import torch
    from .surgical_policy import LLMEngine
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible CUDA GPU is required")
    (output / "hardware.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_runtime": torch.version.cuda, "visible_gpus": torch.cuda.device_count(),
        "host_cpus": os.cpu_count(), "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK")}, indent=2) + "\n")
    for model in config["policy"]["models"]:
        engine = LLMEngine(model, config["policy"])
        try:
            response = engine([{"role": "system", "content": "Reply briefly."},
                               {"role": "user", "content": "Reply READY."}])
            if not response["raw"].strip():
                raise ValueError("Empty GPU generation: " + model)
            # Also exercise the structured decoder before the CPU campaign.
            case = policy["cases"][0]
            from .surgical_policy import documents_for, messages_for
            evidence = documents_for(case, policy, "rag_k3")
            structured = engine(messages_for(case, policy, evidence, "approach"), decoding="constrained")
            from .policy_benchmark import parse_output
            parsed = parse_output(structured["raw"])
            if set(parsed) != {"action", "service_id", "evidence_ids"}:
                raise ValueError("Structured generation gate failed: " + model)
            (output / ("gpu_check_" + model + ".json")).write_text(json.dumps(
                {"functional": response, "structured": structured}, indent=2) + "\n")
        finally:
            engine.close()
        print("GPU generation gate passed:", model, flush=True)
    specs = list(trial_specs(config))
    jobs = [(config, spec, output / "system" / trial_id(spec)) for spec in specs]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for index, _ in enumerate(pool.map(_worker, jobs, chunksize=1), 1):
            if index % 50 == 0 or index == len(jobs):
                print(f"System/stress trials: {index}/{len(jobs)}", flush=True)
    policy_specs = list(trial_specs(config, policy=True))
    # A conservative deterministic interface uses exactly the same task path.
    for base, case in itertools.product(policy_specs, policy["cases"]):
        spec = {**base, "case": case, "condition": "no_rag", "model": "deterministic"}
        run_trial(config, spec, output / "policy" / trial_id(spec), policy)
    for model in config["policy"]["models"]:
        engine = LLMEngine(model, config["policy"])
        try:
            combinations = list(itertools.product(policy_specs, policy["cases"], config["policy"]["conditions"], config["policy"]["decoding"]))
            for index, (base, case, condition, decoding) in enumerate(combinations, 1):
                spec = {**base, "case": case, "condition": condition, "model": model, "decoding": decoding}
                run_trial(config, spec, output / "policy" / trial_id(spec), policy, engine)
                if index % 20 == 0 or index == len(combinations):
                    print(f"Integrated policy tasks {model}: {index}/{len(combinations)}", flush=True)
        finally:
            engine.close()
    from .surgical_report import report
    status = report(output)
    if not status["complete_matrix"]:
        raise ValueError("Some planned executions are missing; preserve the run and resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/surgical_task.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["validate", "full", "report"], default="validate")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 20:
        parser.error("Use 1 to 20 workers within the allocated CPUs")
    if args.stage == "full":
        run(args.config, args.output_dir, args.workers, args.resume)
    elif args.stage == "report":
        from .surgical_report import report
        report(Path(args.output_dir))
    else:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        (out / "validation.json").write_text(json.dumps(validate(load_config(args.config), out / "trials"), indent=2) + "\n")
        print("pcslice surgical development validation: PASS", flush=True)


if __name__ == "__main__":
    main()
