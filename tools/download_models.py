"""Download the exact recorded revisions; this does not call hosted inference."""
import hashlib,json
from pathlib import Path
from huggingface_hub import snapshot_download

for entry in json.loads(Path('study/checkpoints.json').read_text()):
    target=Path('models')/entry['name'];target.mkdir(parents=True,exist_ok=True)
    marker=target/'pcslice_checkpoint.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('revision')!=entry['revision'] or old.get('repo_id')!=entry['repo_id']:
            raise ValueError('Existing checkpoint has different identity; use a fresh folder')
    snapshot_download(repo_id=entry['repo_id'],revision=entry['revision'],local_dir=target,
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','README.md','LICENSE*'],max_workers=4)
    weights=list(target.glob('*.safetensors'))
    if not weights:raise ValueError('No downloaded safetensors weights')
    index=target/'model.safetensors.index.json'
    if index.exists() and not all((target/f).is_file() for f in set(json.loads(index.read_text())['weight_map'].values())):
        raise ValueError('Missing weight shard')
    files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*')
           if p.is_file() and '.cache' not in p.relative_to(target).parts and p!=marker}
    marker.write_text(json.dumps({'repo_id':entry['repo_id'],'revision':entry['revision'],'status':'COMPLETE','files':files},indent=2)+'\n')
    print('Checkpoint ready:',entry['name'],entry['revision'])
