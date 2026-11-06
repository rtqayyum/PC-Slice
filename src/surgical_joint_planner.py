"""Finite-horizon candidate rollouts and reduced-shooting MPC comparator.

The shadow is nonlinear articulated physics. Prefix feasibility is empirical
and conditional on an assumed uncertainty allowance, never a formal proof.
The MPC comparator adapts OCP (2) of Puettschneider et al. (ICCA 2025) to a
three-block gain basis of the existing bounded differential-IK controller.
It is not a reproduction of that paper's Furuta plant or radio experiment.
"""
from dataclasses import dataclass
import numpy as np
from .surgical_plans import Planner, Plan, active_plan


@dataclass
class Candidate:
    plan: Plan
    clearances: np.ndarray
    brake_clearance: float
    goal_tick: int | None
    terminal_error: float
    tracking_cost: float

    def feasible(self, margin, residual, growth, hz):
        allowance = residual + growth * np.arange(1, len(self.clearances)+1) / hz
        return bool(np.all(self.clearances - allowance >= margin) and
                    self.brake_clearance - residual - growth*len(self.clearances)/hz >= margin)


class CandidatePlanner(Planner):
    def rollout(self, snapshot, plans, tick, waypoint, lead, horizon, stride, gains=None):
        r, c = self.shadow, self.config
        r.restore_snapshot(*snapshot[:3])
        start = tick + lead
        for future in range(snapshot[4], start):
            old = active_plan(plans, future)
            r.brake() if old is None else r.set_velocity(old.at(future))
            for _ in range(r.substeps):
                r.physics_step()
            if waypoint < 2 and np.linalg.norm(r.tip()-r.waypoints[waypoint]) <= c['waypoint_tolerance_m']:
                waypoint += 1
        commands, prediction, clearances = [], [], []
        goal_tick, dwell, tracking = None, 0, 0.
        gains = np.ones(3) if gains is None else np.asarray(gains)
        for k in range(horizon):
            tip = r.tip()
            prediction.append(tip)
            if waypoint < 2 and np.linalg.norm(tip-r.waypoints[waypoint]) <= c['waypoint_tolerance_m']:
                waypoint += 1
            if k % stride == 0:
                block = min(len(gains)-1, k*len(gains)//horizon)
                command = (r.command(r.waypoints[waypoint])*gains[block]).astype(np.float32)
                commands.append(command)
            r.set_velocity(command)
            minimum = r.clearance()
            for _ in range(r.substeps):
                minimum = min(minimum, r.physics_step())
            clearances.append(minimum)
            error = float(np.linalg.norm(r.tip()-r.goal))
            tracking += error**2 + 1e-6*float(np.dot(command, command))
            dwell = dwell+1 if waypoint == 2 and error <= c['goal_tolerance_m'] else 0
            if goal_tick is None and dwell >= round(c['goal_dwell_s']*c['control_hz']):
                goal_tick = k+1
        terminal = float(np.linalg.norm(r.tip()-r.goal))
        r.brake()
        brake = r.clearance()
        for _ in range(c['brake_ticks']*r.substeps):
            brake = min(brake, r.physics_step())
        return Candidate(Plan(tick, start, stride, np.array(commands), np.array(prediction)),
                         np.array(clearances), brake, goal_tick, terminal, tracking)

    def mpc(self, snapshot, plans, tick, waypoint, lead, horizon, stride):
        from scipy.optimize import minimize
        cache = {}
        def evaluate(gains):
            key = tuple(np.round(gains, 10))
            if key not in cache:
                cache[key] = self.rollout(snapshot, plans, tick, waypoint, lead, horizon, stride, gains)
            return cache[key]
        def objective(gains):
            x = evaluate(gains)
            return x.tracking_cost / horizon + 10*x.terminal_error**2
        def constraints(gains):
            x = evaluate(gains)
            return np.r_[x.clearances-self.config['checker_margin_m'],
                         x.brake_clearance-self.config['checker_margin_m']]
        result = minimize(objective, np.ones(3), method='SLSQP', bounds=[(.35,1.)]*3,
            constraints=[{'type':'ineq','fun':constraints}],
            options={'maxiter':self.config['mpc_max_iterations'], 'ftol':1e-8, 'eps':1e-3})
        candidate = evaluate(result.x)
        return candidate, {'solver':'SLSQP', 'success':bool(result.success),
                           'message':str(result.message), 'evaluations':len(cache),
                           'gains':result.x.tolist(), 'objective':float(result.fun),
                           'initial_objective':float(objective(np.ones(3))),
                           'minimum_constraint':float(constraints(result.x).min()),
                           'adaptation':'ICCA2025 OCP(2), reduced 3-block gain basis; no terminal invariant-set proof'}
