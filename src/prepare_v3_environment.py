"""Create a writable V3 venv with read-only access to the existing GPU runtime.

Only the new environment is written. Explicit parent site paths work even when
the existing runtime is itself a venv. Isolated Python probes ignore inherited
PYTHONPATH/user-site directories, including mismatched cluster Python versions.
"""
import argparse
import json
from pathlib import Path
import subprocess


def prepare(parent_python, destination):
    parent_python=Path(parent_python).absolute()
    destination=Path(destination).absolute()
    probe='import json,site,sys; print(json.dumps({"python":sys.executable,"version":list(sys.version_info[:2]),"sites":site.getsitepackages()}))'
    identity=json.loads(subprocess.check_output([str(parent_python),'-I','-c',probe],text=True))
    parent_sites=[str(Path(p).resolve()) for p in identity['sites'] if Path(p).is_dir()]
    if not parent_sites:
        raise ValueError('Existing GPU runtime has no readable site-packages')
    if any('\n' in p or '\r' in p for p in parent_sites):
        raise ValueError('Invalid parent site path')
    identity['sites']=parent_sites
    marker=destination/'.pcslice-v3-parent.json'
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text())!=identity:
            raise ValueError('Existing target is not the matching managed V3 environment: '+str(destination))
    else:
        destination.mkdir(parents=True)
        marker.write_text(json.dumps(identity,indent=2)+'\n')
    python=destination/'bin/python'
    if not python.is_file():
        subprocess.run([str(parent_python),'-I','-m','venv',str(destination)],check=True)
    prefix=json.loads(subprocess.check_output([str(python),'-I','-c',
        'import json,sys,sysconfig; print(json.dumps({"prefix":sys.prefix,"site":sysconfig.get_path("purelib")}))'],text=True))
    local_site=Path(prefix['site']).resolve()
    if Path(prefix['prefix']).resolve()!=destination.resolve() or not local_site.is_relative_to(destination.resolve()):
        raise ValueError('Target Python does not use the writable V3 environment')
    # Plain path entries are appended after local venv packages. They do not
    # execute the parent's .pth files or import global module PYTHONPATH paths.
    (local_site/'pcslice_parent_runtime.pth').write_text('\n'.join(parent_sites)+'\n')
    subprocess.run([str(python),'-I','-m','pip','--version'],check=True,capture_output=True,text=True)
    return {'environment':str(destination),'python':str(python),'parent_sites':parent_sites,
            'parent_environment_modified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-python',required=True)
    p.add_argument('--env-root',required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.parent_python,a.env_root),indent=2))


if __name__=='__main__':main()
