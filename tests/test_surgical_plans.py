"""Finite-plan transport, physics and policy integration on development cases."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

HAS_PHYSICS = importlib.util.find_spec('pybullet') is not None and importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_PHYSICS, 'Install requirements-surgical.txt')
class SurgicalPlanTests(unittest.TestCase):
    def setUp(self):
        from src.surgical_campaign import load_config
        self.config = load_config('config/surgical_task_v2.json')
        self.layout = self.config['layouts'][0]
        self.network = {'id': 'development_constant_delay', 'kind': 'delay',
                        'parameters': [220, [1], [1e9]]}

    def test_wire_roundtrip_validity_and_future_plan_does_not_displace_current(self):
        import numpy as np
        from src.surgical_plans import Plan, active_plan
        first = Plan(10, 20, 4, np.arange(12, dtype=np.float32).reshape(2, 6))
        next_plan = Plan(30, 40, 4, np.zeros((2, 6), dtype=np.float32))
        restored = Plan.decode(first.encode())
        self.assertEqual(first.wire_bytes, 90)
        np.testing.assert_array_equal(restored.at(25), first.commands[1])
        self.assertIsNone(restored.at(28))
        self.assertIs(active_plan([first, next_plan], 23), first)
        self.assertIsNone(active_plan([first, next_plan], 35))
        self.assertIs(active_plan([first, next_plan], 40), next_plan)
        with self.assertRaises(ValueError):
            Plan.decode(first.encode()[:-1])
        with self.assertRaises(ValueError):
            Plan.decode(b'bad')
        first.commands[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            Plan.decode(first.encode())

    def test_high_delay_task_expiry_and_actual_byte_budget(self):
        from src.surgical_simulation import simulate
        with tempfile.TemporaryDirectory() as folder:
            m = simulate(self.config, self.layout, self.network, 24, 8102,
                         'adaptive_plan', folder)
            self.assertTrue(m['safe_task_success'])
            self.assertLessEqual(m['transmitted_bytes'], 24 * 124 * self.config['duration_s'] + 1e-6)
            self.assertEqual(m['control_bytes'], m['control_packets'] * 474)
            self.assertGreater(m['skipped_plan_prefix_ticks'], 0)
            self.assertGreater(m['delivered_commands'], 0)
            self.assertGreater(m['p95_delivered_delay_s'], self.config['command_ttl_s'])

    def test_total_outage_and_expired_plan_cannot_create_motion(self):
        from src.surgical_simulation import simulate
        network = {'id': 'development_outage', 'kind': 'communication_loss',
                   'parameters': [1, 100000, 100000, 0]}
        c = {**self.config, 'duration_s': 1.5}
        with tempfile.TemporaryDirectory() as folder:
            m = simulate(c, self.layout, network, 24, 8102, 'adaptive_plan', folder)
            self.assertFalse(m['task_success'])
            self.assertEqual(m['delivered_commands'], 0)
            self.assertEqual(m['active_fallback_fraction'], 1.)
            self.assertLess(m['path_length_m'], 1e-8)

    def test_authored_bad_plan_exercises_checker_and_ablation(self):
        from src.surgical_simulation import simulate
        network = {'id': 'development_no_fault', 'kind': 'no_fault', 'parameters': []}
        c = {**self.config, 'duration_s': 6.}
        with tempfile.TemporaryDirectory() as folder:
            a = simulate(c, self.layout, network, 24, 8102, 'adaptive_plan',
                         Path(folder) / 'checked', stress='unsafe_plan')
            b = simulate(c, self.layout, network, 24, 8102, 'adaptive_no_checker',
                         Path(folder) / 'unchecked', stress='unsafe_plan')
            self.assertGreater(a['unsafe_plan_injections'], 0)
            self.assertGreater(a['checker_rejections'], 0)
            self.assertEqual(a['breach_count'], 0)
            self.assertGreater(b['breach_count'], 0)

    def test_policy_response_changes_actual_plan_service_share(self):
        from src.surgical_simulation import simulate
        policy = json.loads(Path('config/surgical_policy.json').read_text())
        case = policy['cases'][1]
        def fixture_engine(messages, decoding='strict'):
            time.sleep(.025)
            return {'raw': json.dumps({'action': 'accept', 'service_id': 'control_priority', 'evidence_ids': []}),
                    'source': 'UNIT_TEST_FIXTURE_NOT_LLM_RESULT'}
        c = {**self.config, 'duration_s': 1.5}
        with tempfile.TemporaryDirectory() as folder:
            simulate(c, self.layout, self.network, 24, 8102, 'adaptive_plan', folder,
                     policy_case=case, policy=policy, engine=fixture_engine, model_id='unit_test_fixture')
            record = json.loads((Path(folder) / 'policy.json').read_text())
            with gzip.open(Path(folder) / 'trajectory.jsonl.gz', 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            applied = record['applied_tick']
            self.assertGreater(applied, round(c['policy']['request_time_s'] * c['control_hz']))
            self.assertEqual(rows[applied - 1]['allocated_control_share'], .75)
            self.assertEqual(rows[applied]['allocated_control_share'], 1.)

    def test_fresh_matrix_and_legacy_entry_point_are_distinct(self):
        from src.surgical_campaign import trial_specs, execution_plan
        policy = json.loads(Path('config/surgical_policy.json').read_text())
        plan = execution_plan(self.config, policy)
        self.assertEqual(len(plan['system']), 8802)
        self.assertEqual(len(plan['policy']), 5472)
        self.assertTrue(all(s['layout']['split'] == 'test' for s in trial_specs(self.config)))
        self.assertEqual(self.config['policy']['models'], ['Qwen2.5-7B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Phi-3.5-mini-instruct'])


if __name__ == '__main__':
    unittest.main()
