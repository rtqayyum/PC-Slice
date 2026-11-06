"""Task-progress and expired-plan regression fixtures, not research results."""
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.surgical_progress import ExecutedTaskProgress, ProgressState


class ExecutedProgressTests(unittest.TestCase):
    def setUp(self):
        self.c = {'waypoint_tolerance_m':.004,'goal_tolerance_m':.005,
                  'goal_dwell_s':.25,'control_hz':60}
        self.waypoints = [(0.,0.,0.),(.05,0.,0.),(.1,0.,0.)]
        self.p = ExecutedTaskProgress(self.c,self.waypoints,self.waypoints[-1])

    def test_milestones_survive_motion_between_feedback_samples(self):
        self.assertEqual(self.p.update(self.waypoints[0],0).waypoint,1)
        self.assertEqual(self.p.update((.025,.01,0.),1).waypoint,1)
        self.assertEqual(self.p.update(self.waypoints[1],2).waypoint,2)
        for tick in range(3,17):
            self.assertIsNone(self.p.update(self.waypoints[2],tick).completion_tick)
        self.assertEqual(self.p.update(self.waypoints[2],17).completion_tick,18)
        self.assertEqual(self.p.update((1.,1.,1.),18),ProgressState(2,18))

    def test_near_goal_without_ordered_waypoints_is_not_completion(self):
        for tick in range(30):
            self.p.update(self.waypoints[2],tick)
        self.assertEqual(self.p.state,ProgressState())

    def test_goal_dwell_requires_consecutive_samples(self):
        self.p.update(self.waypoints[0],0)
        self.p.update(self.waypoints[1],1)
        for tick in range(2,12):
            self.p.update(self.waypoints[2],tick)
        self.p.update((.12,0.,0.),12)
        for tick in range(13,27):
            self.assertIsNone(self.p.update(self.waypoints[2],tick).completion_tick)
        self.assertEqual(self.p.update(self.waypoints[2],27).completion_tick,28)

    def test_duplicate_skipped_and_nonfinite_samples_are_rejected(self):
        for tick,tip in [(1,self.waypoints[0]),(0,(math.nan,0.,0.))]:
            with self.assertRaises(ValueError):
                self.p.update(tip,tick)
        self.p.update(self.waypoints[0],0)
        with self.assertRaises(ValueError):
            self.p.update(self.waypoints[1],0)


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('pybullet'),
                     'Install surgical physics dependencies')
class ProgressTransportTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from src.surgical_campaign import load_config
        self.np = np
        self.c = load_config('config/surgical_task_v3.json')
        self.robot = SimpleNamespace(state=lambda:(np.zeros(12),np.zeros(12)),
            motor_target=np.zeros(12),tip=lambda:np.zeros(3),clearance=lambda tip:1.)
        self.trace = [{'arrival_tick':k,'dropped':False} for k in range(120)]
        self.events = []

    def event(self,tick,kind,**fields):
        self.events.append(dict(tick=tick,kind=kind,**fields))

    def transport(self,method='fixed_plan_short'):
        from src.surgical_joint_transport import JointTransport
        with patch('src.surgical_joint_transport.CandidatePlanner') as factory:
            t = JointTransport(self.c,self.c['layouts'][0],method,24,self.trace,self.trace)
        t.credit=t.control_credit=1500.
        t.observe(self.robot,0,False,self.event)
        def candidate(snapshot,plans,tick,waypoint,lead,horizon,stride):
            from src.surgical_plans import Plan
            plan=Plan(tick,tick+lead,stride,self.np.zeros((horizon//stride,6)),self.np.zeros((horizon,3)))
            return SimpleNamespace(plan=plan,goal_tick=None,feasible=lambda *args:True)
        t.planner.rollout.side_effect=candidate
        t.planner.mpc.side_effect=lambda *args:(candidate(*args),{'success':True})
        return t

    def test_progress_wire_roundtrip_and_validation(self):
        from src.surgical_feedback import encode_feedback,decode_feedback,HEADER
        state=(self.np.zeros(12),)*3+(self.np.zeros(3),20)
        payload=encode_feedback(2,state,[],3,ProgressState(2,18))
        self.assertEqual(HEADER.size,self.c['feedback_header_bytes'])
        self.assertEqual(len(payload)+28,211)
        self.assertEqual(decode_feedback(payload,{})[1][6],ProgressState(2,18))
        for bad in [ProgressState(0,18),ProgressState(2,21),ProgressState(3,None)]:
            with self.assertRaises(ValueError):
                encode_feedback(2,state,[],3,bad)
        with self.assertRaises(ValueError):
            decode_feedback(b'OLD!'+payload[4:],{})

    def test_reordered_feedback_cannot_erase_latched_progress(self):
        import heapq
        from src.surgical_feedback import encode_feedback
        t=self.transport()
        # Authored packet fixtures: the newer state arrives first.
        for seq,sampled,arrival,phase in [(20,10,15,1),(21,12,14,2)]:
            snapshot=(self.np.zeros(12),)*3+(self.np.zeros(3),sampled)
            heapq.heappush(t.feedback_pending,(arrival,sampled,
                encode_feedback(seq,snapshot,[],0,ProgressState(phase))))
        self.assertEqual(t.observe(self.robot,13,False,self.event,ProgressState(2))[6].waypoint,0)
        self.assertEqual(t.observe(self.robot,14,False,self.event,ProgressState(2))[6].waypoint,2)
        self.assertEqual(t.observe(self.robot,15,False,self.event,ProgressState(2))[6].waypoint,2)

    def test_terminal_feedback_is_charged_and_not_a_perfect_sender_signal(self):
        from src.surgical_joint_transport import JointTransport
        lost=[{'arrival_tick':None,'dropped':True} for k in self.trace]
        with patch('src.surgical_joint_transport.CandidatePlanner'):
            t=JointTransport(self.c,self.c['layouts'][0],'joint_plan',24,self.trace,lost)
        t.credit=t.feedback_credit=1000.
        t.observe(self.robot,0,False,self.event)
        t.observe(self.robot,12,True,self.event,ProgressState(2,10))
        self.assertEqual(t.observed[6],ProgressState())
        self.assertEqual(t.feedback_received,0)
        self.assertEqual(t.feedback_bytes,422)  # Initial + repeated terminal frame.

    def test_expired_plan_is_not_transmitted_and_replans_latest_received_state(self):
        from src.surgical_plans import Plan
        methods=self.c['methods']
        for method in methods:
            with self.subTest(method=method):
                t=self.transport(method)
                old=Plan(2,3,1,self.np.zeros((4,6)))
                t.ready=(10,old)
                t.busy_until=10
                observed=t.observed
                t.step(self.robot,10,0,.75,False,'nominal',self.event)
                self.assertEqual(t.expired_unsent_plans,1)
                self.assertEqual(t.sent,0)
                self.assertEqual(t.control_bytes,0)
                self.assertEqual((old.sequence,old.start,old.end),(2,3,7))
                call=t.planner.mpc.call_args if 'mpc' in method else t.planner.rollout.call_args
                self.assertIs(call.args[0],observed)
                self.assertEqual(call.args[2],10)
                self.assertGreater(t.ready[1].start,10)

    def test_predicted_expiry_while_waiting_for_credit_discards_without_charging(self):
        from src.surgical_plans import Plan
        t=self.transport()
        t.ready=(9,Plan(2,11,1,self.np.zeros((4,6))))  # End15, estimated arrival16.
        t.credit=t.control_credit=0.
        t.step(self.robot,10,0,.75,False,'nominal',self.event)
        self.assertEqual(t.predicted_expired_unsent_plans,1)
        self.assertIsNone(t.ready)
        self.assertEqual(t.control_bytes,0)

    def test_usable_late_prefix_keeps_its_original_timestamp(self):
        from src.surgical_plans import Plan
        t=self.transport()
        old=Plan(2,5,4,self.np.zeros((9,6)))
        t.ready=(10,old)
        t.step(self.robot,10,0,.75,False,'nominal',self.event)
        self.assertEqual(t.sent,1)
        self.assertEqual(t.known[2].start,5)
        sent=next(e for e in self.events if e['kind']=='plan_sent')
        self.assertEqual(Plan.decode(bytes.fromhex(sent['payload_hex'])).start,5)
        self.assertEqual(t.skipped_prefix_ticks,5)

    def test_only_received_completion_stops_sender_planning(self):
        from src.surgical_plans import Plan
        t=self.transport()
        t.ready=(10,Plan(2,5,4,self.np.zeros((9,6))))
        t.step(self.robot,10,0,.75,True,'nominal',self.event)
        self.assertEqual(t.sent,1)  # Receiver knows, sender still does not.
        self.assertEqual(t.delivered,0)  # Completed receiver rejects the packet.
        t.observed=(*t.observed[:6],ProgressState(2,9))
        t.ready=(11,Plan(3,15,4,self.np.zeros((9,6))))
        t.step(self.robot,11,2,.75,True,'nominal',self.event)
        self.assertEqual(t.sent,1)
        self.assertIsNone(t.ready)
        t.planner.rollout.assert_not_called()


if __name__=='__main__':
    unittest.main()
