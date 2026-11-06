"""Download the first checkpoint and record its immutable Hub revision."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .policy_benchmark import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    target = Path(args.model_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "pcslice_checkpoint.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["repo_id"] != args.repo:
            raise ValueError("Model directory belongs to another repository")
        revision = previous["revision"]
    else:
        if any(target.iterdir()):
            raise ValueError("Use an empty model directory or an existing pcslice checkpoint")
        revision = HfApi().model_info(args.repo).sha
        marker.write_text(json.dumps({"repo_id": args.repo, "revision": revision,
                                      "status": "DOWNLOADING"}, indent=2) + "\n")
    snapshot_download(
        repo_id=args.repo, revision=revision, local_dir=target, max_workers=4,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE*"],
    )
    index = target / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not all((target / name).is_file() for name in shards):
            raise ValueError("Incomplete checkpoint: weight shards missing")
    elif not (target / "model.safetensors").is_file():
        raise ValueError("Safetensors weights not found")
    manifest = {"repo_id": args.repo, "revision": revision, "status": "COMPLETE",
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "source_url": "https://huggingface.co/" + args.repo,
                "files": {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*"))
                          if p.is_file() and ".cache" not in p.relative_to(target).parts and p != marker}}
    temp = marker.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(marker)
    print(f"Checkpoint ready: {target}\nRevision: {revision}")


if __name__ == "__main__":
    main()
