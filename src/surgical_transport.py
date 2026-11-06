"""Shared finite-plan protocol and byte scheduler for the v2 comparisons.

Reliable delayed state/buffer acknowledgments are a declared simulation
assumption, shared by every comparator. No future impairment is visible to
the sender. Propagation delay is exogenous, never shortened by allocation.
"""
import heapq
import math
import time
import numpy as np

from .surgical_plans import Plan, Planner, active_plan


class PlanTransport:
    def __init__(self, config, layout, method, budget, trace):
        self.config, self.method, self.budget, self.trace = config, method, budget, trace
        self.planner = Planner(config, layout)
        self.pending, self.buffer, self.history = [], [], []
        self.credit = self.control_credit = 0.
        self.last_sent = -10**9
        self.delays = []
        self.sent = self.delivered = self.dropped = self.stale = self.video = 0
        self.control_bytes = self.planning_seconds = 0.
        self.skipped_prefix_ticks = self.future_received = self.expired_arrivals = 0
        self.buffer_empty_ticks = self.budget_fault_injections = 0
        self.forecasts = {}

    def close(self):
        self.planner.close()

    def step(self, robot, tick, waypoint, share_cap, complete, stress, event):
        c, hz = self.config, self.config['control_hz']
        q, dq = robot.state()
        self.buffer = [p for p in self.buffer if p.end > tick]
        # Snapshot is captured before this tick's packet processing. A delay of
        # one tick exposes only prior receiver state, including prior arrivals.
        self.history.append((q.copy(), dq.copy(), robot.motor_target.copy(), robot.tip().copy(),
                             tick, tuple(self.buffer), tuple(self.delays[-32:])))
        observed = self.history[max(0, tick - c['telemetry_delay_ticks'])]
        samples = observed[6]
        lead = max(1, min(c['max_lead_ticks'], math.ceil(np.quantile(samples, .8))
                         if samples else c['initial_delay_estimate_ticks']))
        old = active_plan(observed[5], observed[4])
        error = 0.
        if old is not None:
            forecast = self.forecasts.get(old.sequence)
            index = observed[4] - old.start
            if forecast is not None and 0 <= index < len(forecast):
                error = float(np.linalg.norm(observed[3] - forecast[index]))
        remaining = max((p.end - tick for p in observed[5]), default=0)
        adaptive = self.method in {'adaptive_plan', 'adaptive_no_checker'}
        # Conventional event control uses a constant Cartesian forecast-error
        # threshold. Proposed scheduling scales this by observed clearance.
        clearance = max(0., robot.clearance(observed[3]))
        threshold = c['event_error_m']
        if adaptive:
            threshold = min(threshold, max(c['adaptive_min_error_m'], clearance * .25))
        share = min(share_cap, 1.)  # Matched service-policy caps for all methods.
        horizon = c['short_plan_ticks'] if self.method == 'fixed_plan_short' else c['long_plan_ticks']
        size = 42 + math.ceil(horizon / c['plan_stride_ticks']) * 24
        period = c['fixed_short_period_ticks'] if self.method == 'fixed_plan_short' else c['fixed_long_period_ticks']
        due = tick - self.last_sent >= period
        if self.method in {'event_plan', 'adaptive_plan', 'adaptive_no_checker'}:
            # Include byte-token refill time in the refresh deadline; this
            # prevents an already depleted buffer from being the only trigger.
            refill = math.ceil(max(0., size - self.control_credit) /
                               (self.budget * max(share, .001) * c['packet_bytes']) * hz)
            due = remaining <= lead + c['refresh_guard_ticks'] + refill or error > threshold
        cap = 2 * (42 + math.ceil(c['long_plan_ticks'] / c['plan_stride_ticks']) * 24)
        self.credit = min(cap, self.credit + self.budget * c['packet_bytes'] / hz)
        self.control_credit = min(cap, self.control_credit + self.budget * c['packet_bytes'] * share / hz)
        # Release control reservations after completion. Background otherwise
        # consumes only unreserved bytes, so larger datagrams can accumulate.
        if complete:
            self.control_credit = 0.
        if due and not complete and min(self.credit, self.control_credit) + 1e-8 >= size:
            started = time.perf_counter()
            plan = self.planner.make(observed, observed[5], tick, waypoint, lead,
                                     horizon, c['plan_stride_ticks'])
            self.planning_seconds += time.perf_counter() - started
            self.forecasts[plan.sequence] = plan.prediction
            # Controlled command-source fault, identical across comparators.
            # Explicit stress only; no LLM generates motion or fault values.
            if stress == 'unsafe_plan' and hz <= tick < 3 * hz:
                plan.commands[:] = robot.command(robot.center).astype(np.float32)
                self.budget_fault_injections += 1
                event(tick, 'authored_unsafe_plan_injection', sequence=plan.sequence)
            payload = plan.encode()
            assert plan.wire_bytes == size
            self.credit -= size
            self.control_credit -= size
            self.control_bytes += size
            self.last_sent = tick
            self.sent += 1
            impairment = self.trace[tick]  # Consult only AFTER deciding/sending.
            arrival = impairment['arrival_tick']
            if arrival is not None and arrival < len(self.trace):
                heapq.heappush(self.pending, (arrival, tick, payload))
            else:
                self.dropped += 1
            event(tick, 'plan_sent', sequence=plan.sequence, start_tick=plan.start,
                  end_tick=plan.end, stride=plan.stride, wire_bytes=size,
                  command_joint_velocities=plan.commands.tolist(), arrival_tick=arrival,
                  dropped=impairment['dropped'], share=share, estimated_delay_ticks=lead,
                  observed_forecast_error_m=error, refresh_threshold_m=threshold)
        available_background = max(0., self.credit - self.control_credit)
        video = int((available_background + 1e-8) / c['packet_bytes'])
        self.credit -= video * c['packet_bytes']
        self.video += video
        if video:
            event(tick, 'background_service', packet_quanta=video)
        while self.pending and self.pending[0][0] <= tick:
            arrival, sent, payload = heapq.heappop(self.pending)
            plan = Plan.decode(payload)
            self.delays.append(tick - sent)  # Includes expired arrivals as observations.
            if complete or plan.end <= tick:
                self.stale += 1
                self.expired_arrivals += plan.end <= tick
                event(tick, 'plan_rejected', sequence=plan.sequence, reason='completed_or_expired')
                continue
            self.buffer.append(plan)
            self.delivered += 1
            self.future_received += tick < plan.start
            self.skipped_prefix_ticks += max(0, tick - plan.start)
            event(tick, 'plan_received', sequence=plan.sequence, start_tick=plan.start,
                  end_tick=plan.end, skipped_ticks=max(0, tick-plan.start), age_s=(tick-sent)/hz)
        chosen = active_plan(self.buffer, tick)
        applied = chosen.at(tick) if chosen else np.zeros(6)
        if complete:
            applied = np.zeros(6)
        age = (tick - chosen.sequence) / hz if chosen else c['duration_s']
        mode = 'completed_hold' if complete else ('remote' if chosen else 'fallback')
        self.buffer_empty_ticks += mode == 'fallback'
        # Bounded storage, including during long runs/outages.
        live_sequences = {p.sequence for p in self.buffer} | {p.sequence for p in observed[5]}
        self.forecasts = {k: v for k, v in self.forecasts.items() if k in live_sequences or k == self.last_sent or tick-k < 2*hz}
        return applied, age, mode, share
