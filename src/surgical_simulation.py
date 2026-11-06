"""Single-timeline instrument, scheduler, packet and on-demand policy execution.

Reference motion and restricted geometry are authored simulation tasks, not
recorded surgeon motions. The simulator never consumes oracle outcome labels.
"""
import gzip
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np

from .network_replay import impairments
from .surgical_robot import Instrument, PredictiveChecker
from .surgical_policy import (documents_for, messages_for, enforce_service,
                              deterministic_proposal, score_proposal)


def simulate(config, layout, network, budget, seed, method, output,
             stress="nominal", policy_case=None, policy=None, engine=None,
             condition="no_rag", model_id="none", decoding="strict"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    robot = Instrument(config, layout)
    v3 = config.get('backend') == 'packetized-v3'
    packetized = config.get("backend") in {"packetized-v2", "packetized-v3"} and method != "instant_event_checked"
    checked = method not in {"fixed", "event", "no_receiver_check", "adaptive_no_checker", "joint_no_checker"}
    checker = PredictiveChecker(config, layout) if checked else None
    if stress == "weaker_actuation":
        robot.force_scale = 0.5  # Explicit out-of-checker-model stress.
    hz = config["control_hz"]
    ticks = round(config["duration_s"] * hz)
    trace = impairments(network, [1000 * k / hz for k in range(ticks)], seed)
    from .surgical_transport import PlanTransport
    if v3:
        from .surgical_joint_transport import JointTransport
        feedback_trace = impairments(network, [1000*k/hz for k in range(ticks)], seed ^ 0x5DAE71)
        transport = JointTransport(config, layout, method, budget, trace, feedback_trace)
    else:
        transport = PlanTransport(config, layout, method, budget, trace) if packetized else None
    pending = []
    latest = -1
    received_tick, source_tick = -10**9, -10**9
    held, previous_desired = np.zeros(6), np.zeros(6)
    motor_minimum, waypoint, dwell = robot.clearance(), 0, 0
    completion_time = None
    progress = None
    if v3:
        from .surgical_progress import ExecutedTaskProgress
        progress = ExecutedTaskProgress(config,robot.waypoints,robot.goal)
    command_queue = None
    credit, command_credit = 0., 0.
    service_id = "balanced"
    policy_due, policy_record = None, None
    request_tick = round(config["policy"]["request_time_s"] * hz)
    history = []
    sent = delivered = dropped = replaced = stale = rejected = fallback_ticks = video = 0
    breaches = 0
    previous_breach = False
    breach_duration = maximum_penetration = 0.
    path_length = 0.
    previous_tip = robot.tip()
    clearance_min = robot.clearance()
    tracking_squared = 0.
    maximum_age = 0.
    mode_changes = recoveries = fallback_entries = 0
    last_mode = "initial"
    ages, active_ages, delays = [], [], []
    completed_hold_ticks = 0
    fallback_breach_s = 0.
    fallback_start = None
    recovery_durations = []
    checker_seconds = 0.
    checker_deadline_misses = 0
    events_path = output / "events.jsonl.gz"
    trajectory_path = output / "trajectory.jsonl.gz"
    wall_start = time.perf_counter()
    metrics = None
    try:
        with gzip.open(events_path, "wt") as events, gzip.open(trajectory_path, "wt") as trajectories:
            def event(tick, kind, **fields):
                events.write(json.dumps({"tick": tick, "time_s": tick / hz, "kind": kind, **fields}) + "\n")

            event(0, "trial_start", layout=layout, network=network, seed=seed, budget_packets_s=budget,
                  method=method, stress=stress, initial_tip_m=robot.initial_tip.tolist(),
                  goal_m=robot.goal.tolist(), restricted_center_m=robot.center.tolist(),
                  restricted_radius_m=config["restricted_radius_m"], tip_radius_m=config["tip_radius_m"],
                  waypoints_m=[x.tolist() for x in robot.waypoints])
            for tick in range(ticks):
                q, dq = robot.state()
                tip = robot.tip()
                history.append((q.copy(), dq.copy(), robot.motor_target.copy(), tip.copy()))
                observed = history[max(0, tick - config["telemetry_delay_ticks"])]
                if v3:
                    observed = transport.observe(robot, tick, completion_time is not None, event,progress.state)
                remote_tip = observed[3]
                if v3:
                    waypoint = observed[6].waypoint
                elif waypoint < len(robot.waypoints) - 1 and np.linalg.norm(remote_tip - robot.waypoints[waypoint]) <= config["waypoint_tolerance_m"]:
                    waypoint += 1
                    event(tick, "waypoint_change", waypoint=waypoint)
                phase = ["approach", "traverse", "placement"][waypoint]
                if policy_case and tick == request_tick:
                    start = time.perf_counter()
                    selected = documents_for(policy_case, policy, condition)
                    retrieval_seconds = time.perf_counter() - start
                    messages = messages_for(policy_case, policy, selected, phase)
                    if engine is None:
                        response = {"raw": deterministic_proposal(policy_case, policy),
                                    "generation_seconds": 0., "source": "deterministic_interface"}
                    else:
                        response = engine(messages, decoding=decoding)
                    validation_start = time.perf_counter()
                    enforcement = enforce_service(response["raw"], policy_case, policy, selected, condition)
                    validation_seconds = time.perf_counter() - validation_start
                    elapsed = time.perf_counter() - start
                    policy_record = {"case_id": policy_case["id"], "model": model_id,
                        "condition": condition, "decoding": decoding, "query": policy_case["query"], "messages": messages,
                        "selected_ids": [d["id"] for d in selected], "response": response,
                        "measured_request_seconds": elapsed, "enforcement": enforcement,
                        "retrieval_seconds": retrieval_seconds, "validation_seconds": validation_seconds,
                        "scores": score_proposal(response["raw"], policy_case, selected, condition, enforcement)}
                    policy_due = tick + max(1, math.ceil(elapsed * hz))
                    event(tick, "policy_requested", case_id=policy_case["id"], response_due_tick=policy_due,
                          measured_seconds=elapsed, note="Measured response latency injected into continuing simulated dynamics")
                if policy_due == tick:
                    decision = policy_record["enforcement"]
                    if decision["authorized"]:
                        service_id = decision["service_id"]
                    event(tick, "policy_response", enforcement=decision, active_service=service_id)
                    policy_record["applied_tick"] = tick
                    policy_record["task_already_complete"] = completion_time is not None

                if transport:
                    cap = policy["services"][service_id]["max_control_share"] if policy else .75
                    applied, age, mode, share = transport.step(robot, tick, waypoint, cap,
                        completion_time is not None, stress, event)
                    sent, delivered, dropped, stale, video = (transport.sent, transport.delivered,
                        transport.dropped, transport.stale, transport.video)
                    delays = [d / hz for d in transport.delays]
                else:
                    desired = robot.command(robot.waypoints[waypoint], observed[0], observed[3])
                    if completion_time is not None:
                        desired = np.zeros(6)
                    clearance = robot.clearance()
                    near = clearance < 0.010
                    turning = waypoint != 1
                    adaptive = method in {"task_checked", "no_task_awareness", "no_receiver_check", "no_expiry_fallback"}
                    share_cap = policy["services"][service_id]["max_control_share"] if policy else 1.0
                    desired_share = (1.0 if near or (turning and method != "no_task_awareness") else 0.5) if adaptive else 0.75
                    share = min(share_cap, desired_share)
                    credit = min(4., credit + budget / hz)
                    command_credit = min(2., command_credit + budget * share / hz)
                    normalized_change = np.linalg.norm((desired - previous_desired) / np.maximum(robot.velocity_limits[:6], 1e-3))
                    fixed_period = max(1, math.ceil(hz / (budget * 0.75)))
                    period = (1 if near or (turning and method != "no_task_awareness") else 6) if adaptive else fixed_period
                    emit = tick % period == 0
                    if method in {"event", "event_checked", "instant_event_checked"}:
                        emit = tick == 0 or normalized_change > 0.08 or tick - source_tick > hz * 0.1
                    if emit and completion_time is None:
                        replaced += command_queue is not None
                        command_queue = (tick, desired.copy())
                        previous_desired = desired.copy()
                    if command_queue is not None and credit >= 1 and command_credit >= 1:
                        sampled, command = command_queue
                        credit -= 1
                        command_credit -= 1
                        sent += 1
                        impairment = trace[tick]
                        if impairment["arrival_tick"] is not None and impairment["arrival_tick"] < ticks:
                            heapq.heappush(pending, (impairment["arrival_tick"], sampled, tick, command.tolist()))
                        else:
                            dropped += 1
                        event(tick, "packet_sent", sampled_tick=sampled, command_joint_velocity=command.tolist(),
                              arrival_tick=impairment["arrival_tick"], dropped=impairment["dropped"],
                              share=share, active_service=service_id)
                        command_queue = None
                    # Saturated background-video queue shares the same finite link
                    # service. These are fixed-size service quanta, not encoded video.
                    served_video = int(credit)
                    credit -= served_video
                    video += served_video
                    if served_video:
                        event(tick, "background_service", packet_quanta=served_video)
                    while pending and pending[0][0] <= tick:
                        arrival, sampled, transmitted, command = heapq.heappop(pending)
                        if sampled <= latest or completion_time is not None:
                            stale += 1
                            event(tick, "packet_rejected", reason="sequence_or_completed", sampled_tick=sampled)
                            continue
                        latest = sampled
                        age = (tick - sampled) / hz
                        if checked and age > config["command_ttl_s"]:
                            stale += 1
                            event(tick, "packet_rejected", reason="expired_on_arrival", age_s=age)
                            continue
                        held = np.array(command)
                        received_tick, source_tick = tick, sampled
                        delivered += 1
                        delays.append((tick - transmitted) / hz)
                        event(tick, "packet_received", sampled_tick=sampled, age_s=age)
                    age = (tick - source_tick) / hz if source_tick >= 0 else config["duration_s"]
                    mode, applied = "remote", held.copy()
                    expired = source_tick < 0 or age > config["command_ttl_s"]
                    if ((checked or method == "no_receiver_check") and method != "no_expiry_fallback" and expired) or completion_time is not None:
                        mode, applied = "fallback", np.zeros(6)
                    if completion_time is not None:
                        mode = "completed_hold"
                if transport is None or mode == 'remote':
                    ages.append(age)
                    if completion_time is None:
                        active_ages.append(age)
                    maximum_age = max(maximum_age, age)
                predicted_margin = None
                if checker:
                    check_start = time.perf_counter()
                    snapshot = history[max(0, tick - 3)] if stress == "delayed_checker" else None
                    # Optimistic clearance offset intentionally breaks the nominal
                    # observation assumption. It is NOT a robust-noise guarantee.
                    bias = -0.002 if stress == "state_bias" else 0.
                    valid, predicted_margin = checker.check(robot, applied, snapshot, bias)
                    check_elapsed = time.perf_counter() - check_start
                    if v3 and check_elapsed > 1/hz:
                        valid = False
                        checker_deadline_misses += 1
                        event(tick, 'checker_deadline_miss', measured_seconds=check_elapsed)
                    if not valid:
                        rejected += 1
                        mode, applied = ("completed_hold" if completion_time is not None else "fallback"), np.zeros(6)
                        event(tick, "command_rejected", predicted_margin_m=predicted_margin)
                    checker_seconds += time.perf_counter() - check_start
                if mode != last_mode:
                    mode_changes += 1
                    fallback_entries += mode == "fallback"
                    recoveries += last_mode == "fallback" and mode == "remote"
                    if mode == "fallback":
                        fallback_start = tick
                    elif last_mode == "fallback" and mode == "remote" and fallback_start is not None:
                        recovery_durations.append((tick - fallback_start) / hz)
                        fallback_start = None
                    event(tick, "mode_change", previous=last_mode, mode=mode)
                if mode != "remote":
                    robot.brake()
                    fallback_ticks += mode == "fallback"
                    completed_hold_ticks += mode == "completed_hold"
                else:
                    robot.set_velocity(applied)
                last_mode = mode
                substep_min = robot.clearance()
                for _ in range(robot.substeps):
                    margin = robot.physics_step()
                    substep_min = min(substep_min, margin)
                    violated = margin < 0
                    breaches += violated and not previous_breach
                    previous_breach = violated
                    breach_duration += violated * robot.dt
                    fallback_breach_s += (violated and mode == "fallback") * robot.dt
                    maximum_penetration = max(maximum_penetration, -margin)
                clearance_min = min(clearance_min, substep_min)
                next_tip = robot.tip()
                path_length += float(np.linalg.norm(next_tip - previous_tip))
                previous_tip = next_tip
                error = float(np.linalg.norm(next_tip - robot.goal))
                tracking_squared += error * error
                if progress is not None:
                    previous_waypoint=progress.state.waypoint
                    state=progress.update(next_tip,tick)
                    if state.waypoint!=previous_waypoint:
                        event(tick,'executed_waypoint_change',waypoint=state.waypoint,observed_tick=tick+1)
                    if completion_time is None and state.completion_tick is not None:
                        completion_time=state.completion_tick/hz
                        event(tick,'task_completed',completion_time_s=completion_time,source='executed_receiver_progress')
                else:
                    dwell = dwell + 1 if error <= config["goal_tolerance_m"] and waypoint == 2 else 0
                    if completion_time is None and dwell >= math.ceil(config["goal_dwell_s"] * hz):
                        completion_time = (tick + 1) / hz
                        event(tick, "task_completed", completion_time_s=completion_time)
                trajectories.write(json.dumps({"tick": tick, "time_s": (tick + 1) / hz,
                    "tip_m": next_tip.tolist(), "joint_position": robot.state()[0].tolist(),
                    "joint_velocity": robot.state()[1].tolist(), "applied_joint_velocity": applied.tolist(),
                    "goal_error_m": error, "min_substep_clearance_m": substep_min,
                    "predicted_margin_m": predicted_margin,
                    "phase": ["approach","traverse","placement"][progress.state.waypoint] if v3 else phase,
                    "waypoint": progress.state.waypoint if v3 else waypoint,
                    **({'sender_waypoint':waypoint,'sender_phase':phase} if v3 else {}),
                    "mode": mode, "command_age_s": age if transport is None or mode == 'remote' else None,
                    "active_service": service_id, "allocated_control_share": share,
                    **({"receiver_buffer_plans": len(transport.buffer),
                        "receiver_buffer_end_tick": max((p.end for p in transport.buffer), default=None),
                        "remaining_byte_credit": transport.credit,
                        "remaining_control_byte_credit": transport.control_credit} if transport else {})}) + "\n")
            if policy_record is not None:
                policy_record["response_within_trial"] = policy_due < ticks
                policy_record["activated"] = policy_due < ticks and policy_record["enforcement"]["authorized"]
                policy_record["scores"]["wrong_service_authorization"] = policy_record["scores"]["wrong_service_activation"]
                policy_record["scores"]["wrong_service_activation"] &= policy_due < ticks
                (output / "policy.json").write_text(json.dumps(policy_record, indent=2) + "\n")
            metrics = {"layout": layout["id"], "split": layout["split"], "network": network["id"],
                "budget_packets_s": budget, "seed": seed, "method": method, "stress": stress,
                "model": model_id, "condition": condition, "decoding": decoding,
                "case_id": policy_case["id"] if policy_case else "none",
                "task_success": completion_time is not None,
                "safe_task_success": completion_time is not None and breaches == 0,
                "completion_time_s": completion_time, "timeout": completion_time is None,
                "final_error_m": float(np.linalg.norm(robot.tip() - robot.goal)),
                "goal_distance_rmse_m": math.sqrt(tracking_squared / ticks),
                "minimum_clearance_m": clearance_min, "breach_count": int(breaches),
                "breach_duration_s": breach_duration, "maximum_penetration_m": maximum_penetration,
                "path_length_m": path_length, "control_packets": sent, "background_packet_quanta": video,
                "transmitted_bytes": (transport.control_bytes + video * config["packet_bytes"]) if transport else (sent + video) * config["packet_bytes"],
                "control_bytes": transport.control_bytes if transport else sent * config["packet_bytes"], "delivered_commands": delivered,
                "lost_or_beyond_trial_packets": dropped, "replaced_unsent_commands": int(replaced),
                "stale_packets_rejected": stale, "checker_rejections": rejected,
                "fallback_fraction": fallback_ticks / ticks, "fallback_entries": fallback_entries,
                "active_fallback_fraction": fallback_ticks / max(1, ticks - completed_hold_ticks),
                "completed_hold_fraction": completed_hold_ticks / ticks,
                "fallback_breach_duration_s": fallback_breach_s,
                "fallback_to_remote_durations_s": recovery_durations,
                "recoveries": recoveries, "mode_changes": mode_changes,
                "p95_age_s": float(np.quantile(ages, .95)) if ages else None,
                "max_age_s": maximum_age if ages else None,
                "p95_active_age_s": float(np.quantile(active_ages, .95)) if active_ages else None,
                "p95_delivered_delay_s": float(np.quantile(delays, .95)) if delays else None,
                "checker_seconds": checker_seconds, "checker_calls": checker.calls if checker else 0,
                "wall_seconds": time.perf_counter() - wall_start,
                "initialization_error_m": float(np.linalg.norm(robot.initial_tip - robot.start)),
                "policy": {"enforcement": policy_record["enforcement"], "scores": policy_record["scores"],
                    "measured_request_seconds": policy_record["measured_request_seconds"],
                    "retrieval_seconds": policy_record["retrieval_seconds"],
                    "validation_seconds": policy_record["validation_seconds"],
                    "response_within_trial": policy_record["response_within_trial"],
                    "task_already_complete": policy_record.get("task_already_complete", False),
                    "activated": policy_record["activated"]} if policy_record else None}
            if transport:
                metrics.update({"command_protocol": "finite_timestamped_velocity_plan",
                    "planning_seconds": transport.planning_seconds,
                    "expired_plan_arrivals": transport.expired_arrivals,
                    "skipped_plan_prefix_ticks": transport.skipped_prefix_ticks,
                    "future_plans_received": transport.future_received,
                    "buffer_empty_ticks": transport.buffer_empty_ticks,
                    "unsafe_plan_injections": transport.budget_fault_injections})
            if v3:
                metrics.update({'feedback_bytes':transport.feedback_bytes,
                    'management_bytes':transport.control_bytes+transport.feedback_bytes,
                    'transmitted_bytes':transport.control_bytes+transport.feedback_bytes+video*config['packet_bytes'],
                    'feedback_packets':transport.feedback_sent,
                    'feedback_received':transport.feedback_received,
                    'feedback_lost_or_beyond_trial':transport.feedback_dropped,
                    'p95_feedback_delay_s':float(np.quantile(transport.feedback_delays,.95))/hz if transport.feedback_delays else None,
                    'planning_jobs':transport.planning_jobs,
                    'planning_available_after_start':transport.compute_late_plans,
                    'planning_infeasible':transport.no_feasible_candidate,
                    'selected_horizons_ticks':transport.selected_horizons,
                    'checker_deadline_misses':checker_deadline_misses,
                    'feedback_model':'serialized impaired return frames on shared aggregate byte budget',
                    'planning_timing':'measured compute time injected into plan availability'})
                metrics['solver_startup_seconds']=transport.solver_load_seconds
                metrics['solver_calls']=transport.solver_calls
                metrics['solver_failures']=transport.solver_failures
                metrics['expired_unsent_plans']=transport.expired_unsent_plans
                metrics['predicted_expired_unsent_plans']=transport.predicted_expired_unsent_plans
                metrics['final_executed_waypoint']=progress.state.waypoint
                metrics['final_sender_waypoint']=waypoint
                metrics['completion_acknowledged_by_sender']=transport.observed[6].completion_tick is not None
                metrics['task_progress_semantics']='receiver-latched executed progress; impaired charged P3P1 feedback'
            event(ticks, "trial_finished", metrics=metrics)
    finally:
        robot.close()
        if transport:
            transport.close()
        if checker:
            checker.close()
    # Completion marker written last. A partial trajectory is never resumed as
    # a finished trial. Failed task outcomes still count as completed experiments.
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
