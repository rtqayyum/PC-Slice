"""Read-only integrity/completeness audit of a downloaded surgical result root.

This verifies evidence integrity, never efficacy, novelty or clinical safety.
Works with the Python standard library; no GPU or model packages required.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path leaves artifact directory: ' + str(relative))
    return path


def audit(root):
    root = Path(root)
    run, bundle = root / 'run', root / 'bundle'
    hashes = json.loads((run / 'artifact_manifest.json').read_text())
    for relative, expected in hashes.items():
        if sha256(checked_path(run, relative)) != expected:
            raise ValueError('Changed run artifact: ' + relative)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    identity = manifest['identity']
    for relative, expected in identity['code_hashes'].items():
        if sha256(checked_path(bundle, relative)) != expected:
            raise ValueError('Changed frozen code/config/document: ' + relative)
    config = identity['config']
    source = checked_path(bundle, config['source_root'])
    network = checked_path(bundle, config['network_source_root'])
    if sha256(source / 'pcslice_source_manifest.json') != identity['source_manifest_sha256']:
        raise ValueError('Changed instrument source manifest')
    if sha256(network / 'source_manifest.json') != identity['network_manifest_sha256']:
        raise ValueError('Changed network source manifest')
    source_manifest = json.loads((source / 'pcslice_source_manifest.json').read_text())
    files = source_manifest['files']
    # SurRoL uses a path->hash mapping; NetFI uses a list of path/hash entries.
    entries = files.items() if isinstance(files, dict) else [(x['path'], x['sha256']) for x in files]
    source_count = 0
    for relative, expected in entries:
        if sha256(checked_path(source, relative)) != expected:
            raise ValueError('Changed instrument source file: ' + relative)
        source_count += 1
    for item in json.loads((network / 'source_manifest.json').read_text())['files']:
        if sha256(checked_path(network, item['path'])) != item['sha256']:
            raise ValueError('Changed network source: ' + item['path'])
    if sha256(bundle / 'config/surgical_policy.json') != identity['policy_sha256']:
        raise ValueError('Changed policy specification')
    plan = json.loads((run / 'execution_plan.json').read_text())
    completed = {}
    for stage in ['system', 'policy']:
        expected = set(plan[stage])
        if len(expected) != len(plan[stage]):
            raise ValueError('Duplicate execution ID in ' + stage)
        actual = {p.name for p in (run / stage).iterdir() if p.is_dir()}
        if actual != expected:
            raise ValueError('Missing/unexpected execution directories in ' + stage)
        for execution in sorted(actual):
            folder = run / stage / execution
            marker = json.loads((folder / 'completion.json').read_text())
            if not {'metrics.json', 'events.jsonl.gz', 'trajectory.jsonl.gz'} <= marker.keys():
                raise ValueError('Incomplete marker: ' + str(folder))
            for relative, expected_sha in marker.items():
                artifact = str((folder / relative).relative_to(run))
                if hashes.get(artifact) != expected_sha:
                    raise ValueError('Completion/artifact manifests disagree: ' + artifact)
        completed[stage] = len(actual)
    status = json.loads((run / 'campaign_status.json').read_text())
    if not status['complete_matrix'] or status['completed'] != completed:
        raise ValueError('Campaign status contradicts completed execution IDs')
    return {'integrity': 'PASS', 'completed': completed, 'artifact_hashes_verified': len(hashes),
            'frozen_files_verified': len(identity['code_hashes']), 'instrument_source_files_verified': source_count,
            'checkpoint_weight_files_rehashed_by_this_audit': False,
            'weight_note': 'Run model/checkpoint marker hashes are retained; GPU weight files are not downloaded/rehashed here.',
            'efficacy_or_clinical_safety_established_by_audit': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root), indent=2))


if __name__ == '__main__':
    main()
