"""V3 causality, byte accounting and articulated development checks."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

HAS_PHYSICS=importlib.util.find_spec('pybullet') is not None and importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_PHYSICS,'Install surgical physics dependencies')
class SurgicalJointTests(unittest.TestCase):
    def setUp(self):
        from src.surgical_campaign import load_config
        self.c=load_config('config/surgical_task_v3.json')
        self.layout=self.c['layouts'][0]
        self.network={'id':'dev_no_fault','kind':'no_fault','parameters':[]}

    def test_feedback_wire_cost_and_unknown_ack_rejection(self):
        import numpy as np
        from src.surgical_feedback import encode_feedback,decode_feedback
        from src.surgical_plans import Plan
        q=np.arange(12)/100
        plan=Plan(7,20,4,np.zeros((3,6),dtype=np.float32))
        payload=encode_feedback(8,(q,q,q,np.zeros(3),19),[plan],5)
        self.assertEqual(len(payload)+28,215)  # 27-byte version/progress header + state/ACK + IP/UDP.
        sequence,snapshot,delay=decode_feedback(payload,{7:plan})
        self.assertEqual((sequence,snapshot[4],delay),(8,19,5))
        np.testing.assert_allclose(q,snapshot[0],atol=1e-8)
        with self.assertRaises(ValueError):
            decode_feedback(payload,{})
        with self.assertRaises(ValueError):
            decode_feedback(payload[:-1],{7:plan})

    def test_lost_feedback_does_not_leak_current_robot_state(self):
        import numpy as np
        from src.surgical_joint_transport import JointTransport
        from src.surgical_robot import Instrument
        r=Instrument(self.c,self.layout)
        trace=[{'arrival_tick':None,'dropped':True} for _ in range(60)]
        t=JointTransport(self.c,self.layout,'joint_plan',24,trace,trace)
        try:
            first=t.observe(r,0,False,lambda *args,**kwargs:None)
            r.set_velocity(r.command(r.waypoints[0]))
            for tick in range(1,40):
                for _ in range(r.substeps):r.physics_step()
                latest=t.observe(r,tick,False,lambda *args,**kwargs:None)
            self.assertEqual(latest[4],0)
            np.testing.assert_array_equal(first[3],latest[3])
            self.assertGreater(np.linalg.norm(r.tip()-latest[3]),.005)
            self.assertGreater(t.feedback_bytes,0)
            self.assertEqual(t.feedback_received,0)
        finally:
            t.close();r.close()

    def test_nominal_task_and_aggregate_byte_budget(self):
        from src.surgical_simulation import simulate
        with tempfile.TemporaryDirectory() as folder:
            m=simulate(self.c,self.layout,self.network,24,31001,'joint_plan',folder)
            self.assertTrue(m['safe_task_success'])
            self.assertGreater(m['feedback_bytes'],0)
            self.assertEqual(m['management_bytes'],m['control_bytes']+m['feedback_bytes'])
            self.assertLessEqual(m['transmitted_bytes'],24*124*self.c['duration_s']+1e-6)
            self.assertTrue(set(m['selected_horizons_ticks'])<=set(self.c['plan_horizons_ticks']))

    def test_planning_latency_prevents_premature_transmission(self):
        from src.surgical_joint_planner import CandidatePlanner
        from src.surgical_simulation import simulate
        original=CandidatePlanner.rollout
        def slow(self,*args,**kwargs):
            time.sleep(.04)
            return original(self,*args,**kwargs)
        c={**self.c,'duration_s':1.}
        with tempfile.TemporaryDirectory() as folder,patch.object(CandidatePlanner,'rollout',slow):
            simulate(c,self.layout,self.network,24,31001,'joint_plan',folder)
            with gzip.open(Path(folder)/'events.jsonl.gz','rt') as stream:
                rows=[json.loads(line) for line in stream]
            planning=[r for r in rows if r['kind']=='planning_started']
            sent=[r for r in rows if r['kind']=='plan_sent']
            self.assertTrue(planning and sent)
            self.assertGreaterEqual(planning[0]['ready_tick']-planning[0]['tick'],3)
            self.assertGreaterEqual(sent[0]['tick'],planning[0]['ready_tick'])

    def test_total_outage_has_no_fictitious_completion(self):
        from src.surgical_simulation import simulate
        lost={'id':'dev_all_lost','kind':'communication_loss','parameters':[1,100000,100000,0]}
        c={**self.c,'duration_s':1.}
        with tempfile.TemporaryDirectory() as folder:
            m=simulate(c,self.layout,lost,24,31001,'joint_plan',folder)
            self.assertFalse(m['task_success'])
            self.assertEqual(m['feedback_received'],0)
            self.assertEqual(m['delivered_commands'],0)
            self.assertLess(m['path_length_m'],1e-8)
            self.assertLessEqual(m['transmitted_bytes'],24*124+1e-6)

    def test_reduced_mpc_optimization_is_feasible_and_shadow_only(self):
        import numpy as np
        from src.surgical_robot import Instrument
        from src.surgical_joint_planner import CandidatePlanner
        r=Instrument(self.c,self.layout)
        planner=CandidatePlanner(self.c,self.layout)
        try:
            q,dq=r.state()
            snapshot=(q,dq,r.motor_target.copy(),r.tip(),0)
            candidate,solver=planner.mpc(snapshot,[],0,0,2,36,4)
            self.assertTrue(solver['success'])
            self.assertGreaterEqual(solver['minimum_constraint'],-1e-6)
            self.assertLessEqual(solver['objective'],solver['initial_objective']+1e-8)
            self.assertEqual(candidate.plan.end,38)
            np.testing.assert_array_equal(q,r.state()[0])
        finally:
            planner.close();r.close()

    def test_improved_service_prompt_has_no_oracle_labels(self):
        from src.surgical_policy import messages_for,documents_for,enforce_service
        policy=json.loads(Path(self.c['policy_path']).read_text())
        case={**policy['cases'][1],'expected_action':'SECRET_ORACLE_ACTION','expected_service':'SECRET_ORACLE_SERVICE'}
        selected=documents_for(case,policy,'rag_policy_k2')
        prompt=json.dumps(messages_for(case,policy,selected,'approach'))
        self.assertNotIn('SECRET_ORACLE',prompt)
        self.assertTrue(any(d['id'].startswith('policy.invariants.') for d in selected))
        service=next(d for d in selected if d.get('service_id')=='control_priority')
        raw=json.dumps({'action':'accept','service_id':'control_priority','evidence_ids':[service['id']]})
        self.assertTrue(enforce_service(raw,case,policy,selected,'rag_policy_k2')['authorized'])
        self.assertFalse(enforce_service(raw,{**case,'approval_tokens':[]},policy,selected,'rag_policy_k2')['authorized'])

    def test_unreleased_final_matrix_cannot_be_submitted_by_runner(self):
        from src.surgical_campaign import run
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'development only'):
            run('config/surgical_task_v3.json',folder,1)

    def test_rag_policy_response_changes_actual_shared_link_allocation(self):
        from src.surgical_simulation import simulate
        policy=json.loads(Path(self.c['policy_path']).read_text())
        case=policy['cases'][1]
        def fixture(messages,decoding='strict'):
            time.sleep(.025)
            return {'raw':json.dumps({'action':'accept','service_id':'control_priority',
                                     'evidence_ids':['service.control_priority.v1']}),
                    'source':'UNIT_TEST_FIXTURE_NOT_MODEL_RESULT'}
        c={**self.c,'duration_s':1.2}
        with tempfile.TemporaryDirectory() as folder:
            simulate(c,self.layout,self.network,24,31001,'joint_plan',folder,
                policy_case=case,policy=policy,engine=fixture,condition='rag_policy_k2',model_id='unit_test_fixture')
            p=json.loads((Path(folder)/'policy.json').read_text())
            self.assertTrue(p['enforcement']['authorized'])
            with gzip.open(Path(folder)/'trajectory.jsonl.gz','rt') as stream:
                rows=[json.loads(line) for line in stream]
            applied=p['applied_tick']
            self.assertAlmostEqual(rows[applied-1]['allocated_control_share'],.75*(1-c['feedback_share']))
            self.assertAlmostEqual(rows[applied]['allocated_control_share'],1-c['feedback_share'])


if __name__=='__main__':unittest.main()
