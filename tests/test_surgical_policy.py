"""Authored engineering-rule regression fixtures, never model research results."""
import json
from pathlib import Path
import unittest

from src.surgical_policy import (OUTPUT_SCHEMA, deterministic_proposal, documents_for,
                                 enforce_service, messages_for, score_proposal)


class SurgicalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(Path('config/surgical_policy.json').read_text())
        self.cases = {c['id']: c for c in self.policy['cases']}

    def raw(self, action='accept', service='control_priority', citations=None):
        return json.dumps({'action': action, 'service_id': service, 'evidence_ids': citations or []})

    def test_missing_approval_cannot_activate(self):
        case = self.cases['priority_bypass']
        result = enforce_service(self.raw(), case, self.policy, [], 'no_rag')
        self.assertFalse(result['authorized'])
        self.assertEqual(result['reason'], 'approval_pending')

    def test_scoped_approval_allows_registered_service(self):
        case = self.cases['priority_approved']
        result = enforce_service(self.raw(), case, self.policy, [], 'no_rag')
        self.assertTrue(result['authorized'])

    def test_unknown_or_extra_motion_fields_fail_closed(self):
        case = self.cases['default']
        self.assertFalse(enforce_service(self.raw(service='unregistered'), case, self.policy, [], 'no_rag')['authorized'])
        raw = json.dumps({'action': 'accept', 'service_id': 'balanced', 'evidence_ids': [], 'motion': [1, 2, 3]})
        self.assertFalse(enforce_service(raw, case, self.policy, [], 'no_rag')['authorized'])

    def test_duplicate_keys_and_fences_are_not_silently_repaired(self):
        case = self.cases['default']
        for raw in ['{"action":"accept","action":"reject"}', '```json\n' + self.raw(service='balanced') + '\n```']:
            self.assertFalse(enforce_service(raw, case, self.policy, [], 'no_rag')['authorized'])

    def test_citation_must_support_selected_service(self):
        case = self.cases['priority_approved']
        docs = [self.policy['documents'][0]]
        raw = self.raw(citations=[docs[0]['id']])
        self.assertEqual(enforce_service(raw, case, self.policy, docs, 'rag_k3')['reason'], 'unsupported_service')

    def test_missing_evidence_condition_actually_removes_relevant_docs(self):
        for case in self.policy['cases']:
            docs = documents_for(case, self.policy, 'rag_missing')
            self.assertFalse(set(case['relevant_ids']) & {d['id'] for d in docs})

    def test_oracle_labels_never_enter_prompt(self):
        case = {**self.cases['default'], 'expected_action': 'ORACLE_SECRET_A', 'expected_service': 'ORACLE_SECRET_B'}
        messages = messages_for(case, self.policy, [], 'approach')
        self.assertNotIn('ORACLE_SECRET', json.dumps(messages))
        self.assertNotIn('expected_action', json.dumps(messages))

    def test_guard_does_not_hide_wrong_registered_service(self):
        case = self.cases['default']
        raw = self.raw(service='conserve')
        result = enforce_service(raw, case, self.policy, [], 'no_rag')
        self.assertTrue(result['authorized'])
        self.assertTrue(score_proposal(raw, case, [], 'no_rag', result)['wrong_service_activation'])

    def test_deterministic_interface_has_no_oracle_access(self):
        case = self.cases['conserve']
        first = deterministic_proposal(case, self.policy)
        altered = {**case, 'expected_action': 'reject', 'expected_service': 'wrong'}
        self.assertEqual(first, deterministic_proposal(altered, self.policy))

    def test_structured_schema_does_not_force_correct_service(self):
        self.assertNotIn('enum', OUTPUT_SCHEMA['properties']['service_id'])

    def test_finite_authority_invariants(self):
        for action in ['accept', 'reject', 'abstain', 'request_approval']:
            for service in ['balanced', 'conserve', 'control_priority', 'unknown', None]:
                for tokens in [[], ['control_priority']]:
                    case = {**self.cases['default'], 'approval_tokens': tokens}
                    result = enforce_service(self.raw(action, service), case, self.policy, [], 'no_rag')
                    if result['authorized']:
                        self.assertEqual(action, 'accept')
                        self.assertIn(service, self.policy['services'])
                        if service == 'control_priority':
                            self.assertIn(service, tokens)

    def test_malformed_output_is_recorded_not_credited_or_crashed(self):
        case = self.cases['default']
        for raw in ['{"action":{},"service_id":"balanced","evidence_ids":[]}',
                    '{"action":"accept","service_id":"balanced","evidence_ids":"not_a_list"}']:
            decision = enforce_service(raw, case, self.policy, [], 'no_rag')
            scores = score_proposal(raw, case, [], 'no_rag', decision)
            self.assertFalse(scores['original_request_correct'])
            self.assertFalse(scores['contextual_correct'])


if __name__ == '__main__':
    unittest.main()
