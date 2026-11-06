"""Real-physics development pipeline checks; never opened held-out layouts."""
import importlib.util
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile
import unittest

HAS_PHYSICS = importlib.util.find_spec('pybullet') is not None and importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_PHYSICS, 'Install requirements-surgical.txt')
class SurgicalIntegrationTests(unittest.TestCase):
    def test_process_execution_report_and_hash_verified_resume(self):
        self._exercise_config('config/surgical_task.json')

    def test_packetized_process_report_and_hash_verified_resume(self):
        self._exercise_config('config/surgical_task_v2.json')

    def test_joint_process_report_and_hash_verified_resume(self):
        self._exercise_config('config/surgical_task_v3.json')

    def _exercise_config(self, config_path):
        from src.surgical_campaign import _worker, completed, trial_id, run_trial
        from src.surgical_campaign import load_config
        from src.surgical_report import report
        config = load_config(config_path)
        # Deliberately development-only; this verifies software, not efficacy.
        config['duration_s'] = .5
        config['policy']['request_time_s'] = .05
        layout = config['layouts'][0]
        specs = [{'layout': layout, 'network': {'id': 'development_no_fault', 'kind': 'no_fault', 'parameters': []},
                  'budget': 24, 'seed': 8101, 'method': method, 'stress': 'nominal'}
                 for method in (['fixed_plan_short', 'joint_plan'] if config.get('backend')=='packetized-v3' else
                                ['fixed_plan_long', 'adaptive_plan'] if config.get('backend') == 'packetized-v2' else ['fixed_checked', 'task_checked'])]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = [trial_id(s) for s in specs]
            policy = json.loads(Path(config.get('policy_path','config/surgical_policy.json')).read_text())
            policy_spec = {**specs[1], 'case': policy['cases'][0], 'condition': 'no_rag', 'model': 'deterministic'}
            policy_id = trial_id(policy_spec)
            (root / 'execution_plan.json').write_text(json.dumps({'system': ids, 'policy': [policy_id]}))
            jobs = [(config, s, root / 'system' / i) for s, i in zip(specs, ids)]
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
                outputs = list(pool.map(_worker, jobs))
            for output in outputs:
                self.assertTrue(completed(Path(output)))
            before = (Path(outputs[0]) / 'metrics.json').read_bytes()
            _worker(jobs[0])
            self.assertEqual(before, (Path(outputs[0]) / 'metrics.json').read_bytes())
            run_trial(config, policy_spec, root / 'policy' / policy_id, policy)
            status = report(root)
            self.assertTrue(status['complete_matrix'])
            self.assertFalse(status['physical_safety_proof'])
            self.assertTrue((root / 'task_resource_tradeoff.png').is_file())
            self.assertTrue((root / 'artifact_manifest.json').is_file())
            self.assertTrue((root / 'policy_decision_metrics.csv').is_file())
            self.assertTrue((root / 'policy_by_task.csv').is_file())
            comparisons = json.loads((root / 'paired_comparisons.json').read_text())
            self.assertTrue(any(r['metric'] == 'control_bytes' for r in comparisons))


if __name__ == '__main__':
    unittest.main()
