"""Shared v3 byte link, impaired feedback, compute availability and decisions."""
import heapq
import math
import time
import numpy as np
from .surgical_feedback import encode_feedback, decode_feedback, HEADER
from .surgical_joint_planner import CandidatePlanner
from .surgical_plans import Plan, active_plan
from .surgical_progress import ProgressState


class JointTransport:
    def __init__(self, config, layout, method, budget, trace, feedback_trace):
        if config['feedback_header_bytes']!=HEADER.size:
            raise ValueError('Feedback header configuration does not match P3P1 wire format')
        self.config,self.method,self.budget,self.trace = config,method,budget,trace
        self.feedback_trace = feedback_trace
        self.planner = CandidatePlanner(config,layout)
        self.solver_load_seconds=0.
        if method in {'et_mpc_adapted','periodic_mpc_adapted'}:
            startup=time.perf_counter()
            from scipy.optimize import minimize
            self.solver_load_seconds=time.perf_counter()-startup
        self.pending,self.feedback_pending,self.buffer = [],[],[]
        self.known,self.forecasts = {},{}
        self.credit=self.control_credit=self.feedback_credit=0.
        self.sent=self.delivered=self.dropped=self.stale=self.video=0
        self.control_bytes=self.feedback_bytes=0
        self.planning_seconds=0.
        self.skipped_prefix_ticks=self.future_received=self.expired_arrivals=0
        self.buffer_empty_ticks=self.budget_fault_injections=0
        self.delays,self.feedback_delays,self.delay_estimates = [],[],[]
        self.feedback_sent=self.feedback_received=self.feedback_dropped=0
        self.observed=None
        self.latest_feedback=-1
        self.last_sent=-10**9
        self.ready=None
        self.busy_until=0
        self.compute_estimate=config['initial_compute_ticks']
        self.compute_late_plans=self.planning_jobs=self.no_feasible_candidate=0
        self.solver_calls=self.solver_failures=0
        self.selected_horizons=[]
        self.awaiting_ack=None
        self.expired_unsent_plans=self.predicted_expired_unsent_plans=0

    def close(self):
        self.planner.close()

    def observe(self, robot, tick, complete, event, progress=ProgressState()):
        c,hz=self.config,self.config['control_hz']
        max_size=42+24*math.ceil(max(c['plan_horizons_ticks'])/c['plan_stride_ticks'])
        cap=4*max_size
        self.credit=min(cap,self.credit+self.budget*c['packet_bytes']/hz)
        self.feedback_credit=min(2*max_size,self.feedback_credit+self.budget*c['packet_bytes']*c['feedback_share']/hz)
        q,dq=robot.state()
        current=(q,dq,robot.motor_target.copy(),robot.tip(),tick)
        if self.observed is None:
            # Common initialized state, not perfect subsequent sender feedback.
            self.observed=(*current,tuple(),ProgressState())
        self.buffer=[p for p in self.buffer if p.end>tick]
        # Repeated terminal telemetry uses the sampled, impaired, charged link;
        # there is no free completion notification to the sender.
        if tick % c['feedback_period_ticks']==0:
            payload=encode_feedback(self.feedback_sent,current,self.buffer,self.delays[-1] if self.delays else 0,progress)
            size=len(payload)+28
            if min(self.credit,self.feedback_credit)+1e-8>=size:
                self.credit-=size
                self.feedback_credit-=size
                self.feedback_bytes+=size
                self.feedback_sent+=1
                arrival=self.feedback_trace[tick]['arrival_tick']
                if arrival is not None and arrival<len(self.trace):
                    heapq.heappush(self.feedback_pending,(arrival,tick,payload))
                else:
                    self.feedback_dropped+=1
                event(tick,'feedback_sent',wire_bytes=size,payload_hex=payload.hex(),arrival_tick=arrival)
        while self.feedback_pending and self.feedback_pending[0][0]<=tick:
            arrival,transmitted,payload=heapq.heappop(self.feedback_pending)
            sequence,observed,delay=decode_feedback(payload,self.known)
            if sequence>self.latest_feedback and observed[4]>=self.observed[4]:
                previous=self.observed[6]
                if (observed[6].waypoint<previous.waypoint or
                    (previous.completion_tick is not None and observed[6].completion_tick!=previous.completion_tick)):
                    raise ValueError('Received progress acknowledgment regressed')
                self.latest_feedback=sequence
                self.observed=observed
                self.feedback_received+=1
                self.feedback_delays.append(tick-transmitted)
                if observed[5]:
                    self.delay_estimates.append(delay)
                event(tick,'feedback_received',sequence=sequence,sampled_tick=observed[4],age_s=(tick-observed[4])/hz,
                      executed_waypoint=observed[6].waypoint,completion_tick=observed[6].completion_tick)
        return self.observed

    def step(self, robot, tick, waypoint, share_cap, complete, stress, event):
        c,hz=self.config,self.config['control_hz']
        observed=self.observed
        if observed is None:
            raise RuntimeError('Observe the impaired return link before deciding')
        max_size=42+24*math.ceil(max(c['plan_horizons_ticks'])/c['plan_stride_ticks'])
        share=min(1.,share_cap)*(1-c['feedback_share'])
        self.control_credit=min(2*max_size,self.control_credit+self.budget*c['packet_bytes']*share/hz)
        lead=max(1,min(c['max_lead_ticks'],math.ceil(np.quantile(self.delay_estimates[-32:],.8))
                     if self.delay_estimates else c['initial_delay_estimate_ticks']))
        compute=int(math.ceil(self.compute_estimate))
        sender_complete=observed[6].completion_tick is not None
        replan=False
        if self.ready is not None and tick>=self.busy_until:
            _,plan=self.ready
            reason=None
            if sender_complete:
                reason='completion_acknowledged'
            elif tick>=plan.end:
                reason='expired_before_transmission'
                self.expired_unsent_plans+=1
            elif tick+lead>=plan.end:
                reason='predicted_expired_on_arrival'
                self.predicted_expired_unsent_plans+=1
            if reason:
                self.ready=None
                replan=not sender_complete
                event(tick,'plan_discarded',sequence=plan.sequence,start_tick=plan.start,end_tick=plan.end,
                      reason=reason,estimated_arrival_tick=tick+lead)
        old=active_plan(observed[5],observed[4])
        error=0.
        if old and old.prediction is not None:
            index=observed[4]-old.start
            if 0<=index<len(old.prediction):
                error=float(np.linalg.norm(observed[3]-old.prediction[index]))
        remaining=max((p.end-tick for p in observed[5]),default=0)
        coverage_deadline=lead+compute+c['refresh_guard_ticks']
        joint=self.method.startswith('joint_')
        period=c['fixed_short_period_ticks'] if self.method=='fixed_plan_short' else c['fixed_long_period_ticks']
        if self.method=='fixed_plan_extended':
            period=max(c['plan_horizons_ticks'])//2
        horizon=(c['short_plan_ticks'] if self.method in {'fixed_plan_short','joint_fixed_short'} else
                 max(c['plan_horizons_ticks']) if self.method=='fixed_plan_extended' else c['long_plan_ticks'])
        size=42+24*math.ceil(horizon/c['plan_stride_ticks'])
        refill=math.ceil(max(0.,size-self.control_credit)/(self.budget*c['packet_bytes']*max(share,.001))*hz)
        due=tick-self.last_sent>=period
        if self.method in {'event_plan','et_mpc_adapted'} or joint:
            threshold=c['event_error_m'] if not joint else max(c['refresh_error_floor_m'],
                min(c['event_error_m'],robot.clearance(observed[3])*.25))
            due=remaining<=coverage_deadline+refill or error>threshold
        if self.method in {'event_plan','et_mpc_adapted'} or joint:
            if self.awaiting_ack is not None:
                sequence,deadline=self.awaiting_ack
                if any(p.sequence>=sequence for p in observed[5]) or tick>=deadline:
                    self.awaiting_ack=None
                else:
                    due=False
        due=due or replan
        # One outstanding planning job; its measured compute time advances
        # simulated availability while the existing actuator buffer continues.
        if due and not sender_complete and self.ready is None and tick>=self.busy_until:
            minimum_size=42+24*c['short_plan_ticks']//c['plan_stride_ticks'] if joint else size
            if min(self.credit,self.control_credit)+1e-8>=minimum_size:
                started=time.perf_counter()
                horizons=([c['short_plan_ticks']] if self.method=='joint_fixed_short' else c['plan_horizons_ticks']) if joint else [horizon]
                options=[]
                solver=None
                for h in horizons:
                    wire=42+24*h//c['plan_stride_ticks']
                    token_wait=math.ceil(max(0.,wire-min(self.credit,self.control_credit))/
                        (self.budget*c['packet_bytes']*max(share,.001))*hz)
                    if token_wait and (not joint or remaining<token_wait+coverage_deadline):
                        continue
                    if self.method in {'et_mpc_adapted','periodic_mpc_adapted'}:
                        candidate,solver=self.planner.mpc(observed,observed[5],tick,waypoint,lead+compute,h,c['plan_stride_ticks'])
                        self.solver_calls+=1
                        self.solver_failures+=not solver['success']
                    else:
                        candidate=self.planner.rollout(observed,observed[5],tick,waypoint,lead+compute+token_wait,h,c['plan_stride_ticks'])
                    feasible=candidate.feasible(c['checker_margin_m'],error,c['uncertainty_growth_m_s'] if joint else 0.,hz)
                    # A long zero-motion tail is charged but not useful coverage.
                    useful=min(h,candidate.goal_tick) if candidate.goal_tick is not None else h
                    useful=max(1,useful-coverage_deadline)
                    score=wire/useful
                    options.append((score,candidate,feasible,token_wait))
                elapsed=time.perf_counter()-started
                self.planning_seconds+=elapsed
                self.planning_jobs+=1
                actual_compute=max(1,math.ceil(elapsed*hz))
                self.compute_estimate=.8*self.compute_estimate+.2*actual_compute
                self.busy_until=tick+actual_compute
                feasible_options=[x for x in options if x[2]]
                if feasible_options:
                    _,candidate,_,token_wait=min(feasible_options,key=lambda x:x[0])
                    self.ready=(self.busy_until+token_wait,candidate.plan)
                    self.selected_horizons.append(len(candidate.plan.prediction))
                    event(tick,'planning_started',ready_tick=self.busy_until,measured_seconds=elapsed,
                        sampled_tick=observed[4],acknowledged_waypoint=observed[6].waypoint,
                        selected_horizon=len(candidate.plan.prediction),estimated_compute_ticks=compute,
                        candidates=[{'horizon':len(x.plan.prediction),'cost_bytes_per_tick':v,'feasible':f,'token_wait_ticks':w}
                                    for v,x,f,w in options],solver=solver)
                else:
                    self.no_feasible_candidate+=1
                    event(tick,'planning_infeasible',ready_tick=self.busy_until,measured_seconds=elapsed,solver=solver,
                          sampled_tick=observed[4],acknowledged_waypoint=observed[6].waypoint)
        if self.ready is not None and self.ready[0]<=tick and not sender_complete:
            _,plan=self.ready
            wire=plan.wire_bytes
            if min(self.credit,self.control_credit)+1e-8>=wire:
                self.ready=None
                self.compute_late_plans+=tick>plan.start
                if stress=='unsafe_plan' and hz<=tick<3*hz:
                    plan.commands[:]=robot.command(robot.center).astype(np.float32)
                    self.budget_fault_injections+=1
                    event(tick,'authored_unsafe_plan_injection',sequence=plan.sequence)
                payload=plan.encode()
                self.known[plan.sequence]=plan
                self.control_credit-=wire
                self.credit-=wire
                self.control_bytes+=wire
                self.sent+=1
                self.last_sent=tick
                feedback_lead=math.ceil(np.quantile(self.feedback_delays[-32:],.8)) if self.feedback_delays else lead
                self.awaiting_ack=(plan.sequence,tick+lead+feedback_lead+c['feedback_period_ticks']+c['refresh_guard_ticks'])
                arrival=self.trace[tick]['arrival_tick']
                if arrival is not None and arrival<len(self.trace):
                    heapq.heappush(self.pending,(arrival,tick,payload))
                else:
                    self.dropped+=1
                event(tick,'plan_sent',sequence=plan.sequence,start_tick=plan.start,end_tick=plan.end,
                      wire_bytes=wire,payload_hex=payload.hex(),arrival_tick=arrival,share=share,
                      estimated_arrival_tick=tick+lead)
        while self.pending and self.pending[0][0]<=tick:
            arrival,transmitted,payload=heapq.heappop(self.pending)
            plan=Plan.decode(payload)
            self.delays.append(tick-transmitted)
            if complete or plan.end<=tick:
                self.stale+=1
                self.expired_arrivals+=plan.end<=tick
                continue
            self.buffer.append(plan)
            self.delivered+=1
            self.future_received+=tick<plan.start
            self.skipped_prefix_ticks+=max(0,tick-plan.start)
            # Wire decoding creates commands only. Sender-only forecasts are
            # reconstructed for ACKs from actually transmitted plans.
            event(tick,'plan_received',sequence=plan.sequence,age_s=(tick-transmitted)/hz)
        if sender_complete:
            self.control_credit=0.
        video=int((max(0.,self.credit-self.control_credit-self.feedback_credit)+1e-8)/c['packet_bytes'])
        self.credit-=video*c['packet_bytes']
        self.video+=video
        if video:
            event(tick,'background_service',packet_quanta=video)
        chosen=active_plan(self.buffer,tick)
        applied=chosen.at(tick) if chosen is not None and not complete else np.zeros(6)
        mode='completed_hold' if complete else 'remote' if chosen is not None else 'fallback'
        self.buffer_empty_ticks+=mode=='fallback'
        age=(tick-chosen.sequence)/hz if chosen else c['duration_s']
        return applied,age,mode,share
