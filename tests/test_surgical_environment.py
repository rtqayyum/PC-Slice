"""Offline regression for inherited read-only packages and a writable V3 venv."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

from src.prepare_v3_environment import prepare


class SurgicalEnvironmentTests(unittest.TestCase):
    def test_readonly_parent_is_reused_without_uninstalling_its_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            parent,overlay=root/'parent',root/'overlay'
            subprocess.run([sys.executable,'-I','-m','venv','--without-pip',str(parent)],check=True)
            parent_python=parent/'bin/python'
            site=Path(subprocess.check_output([str(parent_python),'-I','-c',
                'import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip())
            pkg=site/'pcslice_env_fixture'
            metadata=site/'pcslice_env_fixture-1.0.dist-info'
            pkg.mkdir();metadata.mkdir()
            (pkg/'__init__.py').write_text('VERSION="1.0"\n')
            (metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: pcslice-env-fixture\nVersion: 1.0\n')
            (metadata/'RECORD').write_text('pcslice_env_fixture/__init__.py,,\npcslice_env_fixture-1.0.dist-info/METADATA,,\npcslice_env_fixture-1.0.dist-info/RECORD,,\n')
            files=list(pkg.iterdir())+list(metadata.iterdir())
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            for p in files:p.chmod(stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)
            pkg.chmod(0o555);metadata.chmod(0o555)
            try:
                result=prepare(parent_python,overlay)
                python=Path(result['python'])
                inherited=subprocess.check_output([str(python),'-I','-c',
                    'import pcslice_env_fixture;print(pcslice_env_fixture.VERSION)'],text=True).strip()
                self.assertEqual(inherited,'1.0')
                wheel=root/'pcslice_env_fixture-2.0-py3-none-any.whl'
                members={'pcslice_env_fixture/__init__.py':'VERSION="2.0"\n',
                    'pcslice_env_fixture-2.0.dist-info/METADATA':'Metadata-Version: 2.1\nName: pcslice-env-fixture\nVersion: 2.0\n',
                    'pcslice_env_fixture-2.0.dist-info/WHEEL':'Wheel-Version: 1.0\nGenerator: offline-unit-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
                record=io.StringIO()
                writer=csv.writer(record)
                for name in members:writer.writerow([name,'',''])
                writer.writerow(['pcslice_env_fixture-2.0.dist-info/RECORD','',''])
                members['pcslice_env_fixture-2.0.dist-info/RECORD']=record.getvalue()
                with zipfile.ZipFile(wheel,'w') as stream:
                    for name,value in members.items():stream.writestr(name,value)
                env={k:v for k,v in os.environ.items() if k not in {'PIP_TARGET','PIP_PREFIX','PYTHONPATH','PYTHONHOME'}}
                env.update(PIP_CONFIG_FILE=os.devnull,PIP_USER='false',PIP_REQUIRE_VIRTUALENV='true')
                installed=subprocess.run([str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheel)],
                    env=env,check=True,capture_output=True,text=True)
                self.assertIn('Successfully installed',installed.stdout)
                code='import json,sys,pcslice_env_fixture as p;print(json.dumps({"version":p.VERSION,"path":p.__file__,"prefix":sys.prefix}))'
                active=json.loads(subprocess.check_output([str(python),'-I','-c',code],text=True))
                self.assertEqual(active['version'],'2.0')
                self.assertTrue(Path(active['path']).is_relative_to(overlay))
                self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
                self.assertEqual(prepare(parent_python,overlay)['environment'],str(overlay))
                helper=Path(__file__).resolve().parents[1]/'src/prepare_v3_environment.py'
                cli=subprocess.check_output([sys.executable,str(helper),'--parent-python',str(parent_python),
                    '--env-root',str(overlay)],text=True)
                self.assertEqual(json.loads(cli)['environment'],str(overlay))
            finally:
                pkg.chmod(0o755);metadata.chmod(0o755)
                for p in files:p.chmod(0o644)

    def test_unmanaged_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'existing'
            target.mkdir()
            marker=target/'keep.txt'
            marker.write_text('USER_DATA')
            with self.assertRaisesRegex(ValueError,'not the matching managed'):
                prepare(sys.executable,target)
            self.assertEqual(marker.read_text(),'USER_DATA')


if __name__=='__main__':unittest.main()
