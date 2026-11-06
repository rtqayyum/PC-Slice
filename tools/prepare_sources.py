"""Acquire only pinned public instrument assets and network parameters."""
import hashlib
import json
from pathlib import Path
import subprocess
from src.prepare_surgical_sources import prepare

REPO='UVA-DSA/telesurgery-qos-analysis'
COMMIT='420e659edd79a0a2bf8755fa424888ad3a555b7e'
FILES=['SurRoL_dVTrainer/tests/dVTrainer/network_conditions.txt',
       'SurRoL_dVTrainer/tests/dVTrainer/NetFI/LICENSE.txt','README.md']

def main():
    prepare()
    root=Path('data/sources/uva-qos-420e659edd79');root.mkdir(parents=True,exist_ok=True)
    manifest=root/'source_manifest.json'
    if manifest.exists():
        recorded=json.loads(manifest.read_text())
        if recorded.get('commit')!=COMMIT:raise ValueError('Wrong network source revision')
        for item in recorded['files']:
            if hashlib.sha256((root/item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Changed source: '+item['path'])
        return
    entries=[]
    for name in FILES:
        url=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}'
        data=subprocess.run(['curl','-fsSL','--retry','2','--max-time','60',url],capture_output=True,check=True).stdout
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        entries.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'url':url})
    manifest.write_text(json.dumps({'repo':REPO,'commit':COMMIT,'scope':'Network configuration only; no third-party replay recording redistributed','files':entries},indent=2)+'\n')

if __name__=='__main__':main()
