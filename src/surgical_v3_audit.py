"""Standard-library integrity audit for the explicitly development-only V3 job."""
import argparse
import json
from pathlib import Path
from .surgical_audit import checked_path,sha256


def audit(root):
    root=Path(root)
    run,bundle=root/'run',root/'bundle'
    hashes=json.loads((run/'development_artifact_manifest.json').read_text())
    for relative,expected in hashes.items():
        if sha256(checked_path(run,relative))!=expected:
            raise ValueError('Changed development artifact: '+relative)
    manifest=json.loads((run/'development_manifest.json').read_text())
    for relative,expected in manifest['source_hashes'].items():
        if sha256(checked_path(bundle,relative))!=expected:
            raise ValueError('Changed frozen implementation: '+relative)
    counts={stage:len(list((run/stage).glob('*/metrics.json'))) for stage in ['systems','policy']}
    status=json.loads((run/'DEVELOPMENT_STATUS.json').read_text())
    if (counts['systems']!=status['expected_system_executions'] or
        counts['policy']!=status['expected_policy_executions'] or
        counts['systems']!=status['system_executions'] or counts['policy']!=status['policy_executions']):
        raise ValueError('Incomplete development counts')
    if not status['development_only'] or status['final_release']:
        raise ValueError('Development/final status inconsistency')
    for stage in ['systems','policy']:
        for metric in (run/stage).glob('*/metrics.json'):
            marker=json.loads((metric.parent/'completion.json').read_text())
            for name,expected in marker.items():
                relative=str((metric.parent/name).relative_to(run))
                if hashes.get(relative)!=expected:
                    raise ValueError('Trial completion/manifest mismatch: '+relative)
    config=manifest['config']
    source=checked_path(bundle,config['source_root'])
    for relative,expected in json.loads((source/'pcslice_source_manifest.json').read_text())['files'].items():
        if sha256(checked_path(source,relative))!=expected:
            raise ValueError('Changed instrument source: '+relative)
    network=checked_path(bundle,config['network_source_root'])
    for item in json.loads((network/'source_manifest.json').read_text())['files']:
        if sha256(checked_path(network,item['path']))!=item['sha256']:
            raise ValueError('Changed network source')
    return {'integrity':'PASS','development_only':True,'counts':counts,
            'artifact_hashes_verified':len(hashes),'frozen_implementation_files_verified':len(manifest['source_hashes']),
            'system_gate_pass':status['system_gate_pass'],'final_release':False,
            'checkpoint_weights_rehashed':False,'scientific_superiority_established_by_audit':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.run_root),indent=2))


if __name__=='__main__':main()
