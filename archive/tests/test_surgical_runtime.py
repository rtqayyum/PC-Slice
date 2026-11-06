"""Real articulated physics development checks. No held-out efficacy tuning."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import shlex
import shutil
import subprocess

HAS_PHYSICS = importlib.util.find_spec('pybullet') is not None and importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_PHYSICS, 'Install requirements-surgical.txt')
class SurgicalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('config/surgical_task.json').read_text())
        self.layout = self.config['layouts'][0]  # Development only.
        self.network = {'id': 'development_no_fault', 'kind': 'no_fault', 'parameters': []}

    def test_articulated_motion_and_shadow_isolation(self):
        import numpy as np
        from src.surgical_robot import Instrument, PredictiveChecker
        robot = Instrument(self.config, self.layout)
        checker = PredictiveChecker(self.config, self.layout)
        try:
            self.assertGreater(len(robot.joints), 6)
            self.assertLess(np.linalg.norm(robot.tip() - robot.start), .001)
            before = robot.state()[0].copy()
            checker.check(robot, robot.command(robot.waypoints[0]))
            np.testing.assert_array_equal(before, robot.state()[0])
            robot.set_velocity(robot.command(robot.waypoints[0]))
            for _ in range(24):
                robot.physics_step()
            self.assertGreater(np.linalg.norm(before - robot.state()[0]), 1e-5)
            with self.assertRaises(RuntimeError):
                robot.restore_snapshot(*robot.state(), robot.motor_target)
        finally:
            robot.close()
            checker.close()

    def test_nominal_task_and_completed_hold_not_counted_as_fallback(self):
        from src.surgical_simulation import simulate
        with tempfile.TemporaryDirectory() as folder:
            metrics = simulate(self.config, self.layout, self.network, 24, 8101, 'fixed_checked', folder)
            self.assertTrue(metrics['safe_task_success'])
            self.assertLess(metrics['final_error_m'], self.config['goal_tolerance_m'])
            self.assertGreater(metrics['completed_hold_fraction'], .5)
            self.assertLess(metrics['active_fallback_fraction'], .5)
            self.assertLessEqual(metrics['control_packets'] + metrics['background_packet_quanta'], 24 * self.config['duration_s'])
            with gzip.open(Path(folder) / 'trajectory.jsonl.gz', 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), round(self.config['duration_s'] * self.config['control_hz']))
            self.assertIn('completed_hold', {r['mode'] for r in rows})

    def test_total_outage_has_no_invented_completion(self):
        from src.surgical_simulation import simulate
        outage = {'id': 'development_all_lost', 'kind': 'communication_loss', 'parameters': [1, 100000, 100000, 0]}
        with tempfile.TemporaryDirectory() as folder:
            metrics = simulate(self.config, self.layout, outage, 24, 8101, 'task_checked', folder)
            self.assertFalse(metrics['task_success'])
            self.assertTrue(metrics['timeout'])
            self.assertEqual(metrics['delivered_commands'], 0)
            self.assertEqual(metrics['active_fallback_fraction'], 1.)

    def test_policy_latency_is_applied_on_actual_task_timeline(self):
        from src.surgical_simulation import simulate
        policy = json.loads(Path('config/surgical_policy.json').read_text())
        case = policy['cases'][1]

        def fixture_engine(messages, decoding='strict'):
            time.sleep(.04)
            return {'raw': json.dumps({'action': 'accept', 'service_id': 'control_priority', 'evidence_ids': []}),
                    'source': 'UNIT_TEST_FIXTURE_NOT_LLM_RESULT'}

        with tempfile.TemporaryDirectory() as folder:
            simulate(self.config, self.layout, self.network, 24, 8101, 'task_checked', folder,
                     policy_case=case, policy=policy, engine=fixture_engine, model_id='unit_test_fixture')
            result = json.loads((Path(folder) / 'policy.json').read_text())
            self.assertTrue(result['enforcement']['authorized'])
            self.assertGreater(result['applied_tick'], round(self.config['policy']['request_time_s'] * self.config['control_hz']))
            with gzip.open(Path(folder) / 'trajectory.jsonl.gz', 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows[result['applied_tick'] - 1]['active_service'], 'balanced')
            self.assertEqual(rows[result['applied_tick']]['active_service'], 'control_priority')

    def test_response_after_trial_is_not_an_activation(self):
        from src.surgical_simulation import simulate
        config = {**self.config, 'duration_s': .25,
                  'policy': {**self.config['policy'], 'request_time_s': .05}}
        policy = json.loads(Path('config/surgical_policy.json').read_text())
        case = policy['cases'][0]

        def fixture_engine(messages, decoding='strict'):
            time.sleep(.3)
            return {'raw': json.dumps({'action': 'accept', 'service_id': 'conserve', 'evidence_ids': []}),
                    'source': 'UNIT_TEST_FIXTURE_NOT_LLM_RESULT'}

        with tempfile.TemporaryDirectory() as folder:
            metrics = simulate(config, self.layout, self.network, 24, 8101, 'task_checked', folder,
                     policy_case=case, policy=policy, engine=fixture_engine, model_id='unit_test_fixture')
            self.assertTrue(metrics['policy']['scores']['wrong_service_authorization'])
            self.assertFalse(metrics['policy']['scores']['wrong_service_activation'])
            self.assertFalse(metrics['policy']['activated'])


class SurgicalOrchestrationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('rsync'), 'rsync not installed')
    def test_upload_keeps_nested_sources_and_excludes_weights_and_venv(self):
        text = Path('hpc/SURGICAL_RUN.md').read_text()
        command = text[text.index('rsync -av'):].split('\n```', 1)[0].replace('\\\n', ' ')
        tokens = shlex.split(command)
        excludes = tokens[2:-2]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, destination = root / 'source', root / 'destination'
            source.mkdir()
            destination.mkdir()
            for name in ['data/sources/archive/NetFI/models/model.py', 'data/sources/surrol/pcslice_source_manifest.json',
                         'models/weights.safetensors', '.venv-surgical/bin/python', 'results/old.json', 'envs/python']:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('TRANSFER_FIXTURE')
            result = subprocess.run(['rsync', '-an', '--itemize-changes', *excludes,
                                     str(source) + '/', str(destination) + '/'], check=True, capture_output=True, text=True)
            self.assertIn('NetFI/models/model.py', result.stdout)
            self.assertIn('pcslice_source_manifest.json', result.stdout)
            for excluded in ['weights.safetensors', '.venv-surgical', 'results/old.json', 'envs/python']:
                self.assertNotIn(excluded, result.stdout)

    def test_source_snapshot_and_matrix_are_explicit(self):
        from src.prepare_surgical_sources import verify
        from src.surgical_campaign import load_config, trial_specs, trial_id
        self.assertEqual(verify()['commit'], 'a903fa6de40b41e1cad12ddd2d4a6758ec78e817')
        config = load_config('config/surgical_task.json')
        specs = list(trial_specs(config))
        self.assertEqual(len(specs), 11592)
        self.assertEqual(len({trial_id(s) for s in specs}), len(specs))
        self.assertTrue(all(s['layout']['split'] == 'test' for s in specs))

    def test_exact_slurm_resources_and_no_guessed_walltime(self):
        script = Path('hpc/surgical.slurm').read_text()
        self.assertIn('#SBATCH --cpus-per-task=20', script)
        self.assertIn('#SBATCH --gres=gpu:1', script)
        self.assertIn('#SBATCH --job-name=pcslice_surgical', script)
        self.assertNotIn('#SBATCH --time', script)
        submit = Path('hpc/submit_surgical.sh').read_text()
        self.assertIn('scontrol show partition gpuq', submit)
        self.assertNotIn('scancel', submit)

    def test_partial_trial_is_not_completed(self):
        from src.surgical_campaign import completed
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'metrics.json').write_text('{}')
            self.assertFalse(completed(path))

    def test_completed_trial_hashes_are_checked(self):
        from src.surgical_campaign import completed
        from src.policy_benchmark import digest
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'metrics.json').write_text('{}')
            (path / 'events.jsonl.gz').write_bytes(b'EVENT_HASH_FIXTURE')
            (path / 'trajectory.jsonl.gz').write_bytes(b'TRAJECTORY_HASH_FIXTURE')
            files = ['metrics.json', 'events.jsonl.gz', 'trajectory.jsonl.gz']
            (path / 'completion.json').write_text(json.dumps({name: digest(path / name) for name in files}))
            self.assertTrue(completed(path))
            (path / 'metrics.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                completed(path)


if __name__ == '__main__':
    unittest.main()
