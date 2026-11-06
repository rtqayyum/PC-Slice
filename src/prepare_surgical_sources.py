"""Acquire/verify pinned SurRoL assets. Never execute upstream setup code."""
import argparse
import json
from pathlib import Path
import subprocess

from .policy_benchmark import digest

COMMIT = "a903fa6de40b41e1cad12ddd2d4a6758ec78e817"
URL = "https://github.com/med-air/SurRoL.git"
ROOT = Path("data/sources/surrol-a903fa6de40b")


def verify(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "pcslice_source_manifest.json").read_text())
    if manifest["commit"] != COMMIT or not manifest["files"]:
        raise ValueError("Unapproved SurRoL source revision")
    for name, sha in manifest["files"].items():
        if digest(root / name) != sha:
            raise ValueError("Modified SurRoL source: " + name)
    return manifest


def prepare(root=ROOT):
    root = Path(root)
    if (root / "pcslice_source_manifest.json").exists():
        return verify(root)
    if not root.exists():
        root.mkdir(parents=True)
        for command in (["git", "init", str(root)],
                        ["git", "-C", str(root), "remote", "add", "origin", URL],
                        ["git", "-C", str(root), "fetch", "--depth", "1", "origin", COMMIT],
                        ["git", "-C", str(root), "checkout", "--detach", "FETCH_HEAD"]):
            subprocess.run(command, check=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError("Existing simulator checkout has wrong revision; preserve it and use a new folder")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--"], text=True):
        raise ValueError("Simulator tracked files were modified")
    files = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines()
    manifest = {"repository": URL, "commit": COMMIT,
                "paper": "https://arxiv.org/abs/2108.13035",
                "license": "LICENSE (MIT); retain asset provenance in surrol/robots/README.md",
                "limits": ["Upstream notes unresolved dynamics/limit validation.",
                           "No patient data or clinical safety limits provided."],
                "files": {name: digest(root / name) for name in files}}
    (root / "pcslice_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify_only else prepare()
    print("SurRoL source verified:", result["commit"], len(result["files"]), "files")
