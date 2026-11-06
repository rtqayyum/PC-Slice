"""Articulated SurRoL PSM adapter, SI units. No state reset after initialization.

We load upstream URDF/assets directly, not the legacy Gym/RL training stack.
Motor, waypoint and restricted-region settings are experimental extensions.
Only the tool-tip spherical exclusion and workspace are evaluated, not shaft
collision, tissue contact forces, patient injury or physical robot assurance.
"""
from pathlib import Path
import math
import numpy as np
import pybullet as p


class Instrument:
    def __init__(self, config, layout, shadow=False):
        self.config, self.layout = config, layout
        self.client = p.connect(p.DIRECT)
        self.dt = 1 / config["physics_hz"]
        self.substeps = config["physics_hz"] // config["control_hz"]
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1,
                                    physicsClientId=self.client)
        urdf = Path(config["source_root"]) / "surrol/assets/psm/psm.urdf"
        self.body = p.loadURDF(str(urdf.resolve()), [0.05, 0.24, 0.8524],
                              p.getQuaternionFromEuler([0, 0, -110 * math.pi / 180]),
                              useFixedBase=True, flags=p.URDF_MAINTAIN_LINK_ORDER,
                              physicsClientId=self.client)
        self.base_rotation = np.array(p.getMatrixFromQuaternion(
            p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)[1])).reshape(3, 3)
        self.joints = [j for j in range(p.getNumJoints(self.body, physicsClientId=self.client))
                       if p.getJointInfo(self.body, j, physicsClientId=self.client)[2] != p.JOINT_FIXED]
        infos = [p.getJointInfo(self.body, j, physicsClientId=self.client) for j in self.joints]
        self.lower = np.array([i[8] for i in infos])
        self.upper = np.array([i[9] for i in infos])
        self.forces = np.array([max(0.1, i[10]) for i in infos])
        self.velocity_limits = np.array([max(0.001, i[11]) for i in infos])
        self.workspace = np.array([[0.50, 0.60], [-0.05, 0.05], [0.6912, 0.7552]])
        self.center = np.array([0.55, 0, 0.7212 + layout["z_shift_m"]])
        self.start = self.center + [-0.032, 0, 0]
        self.goal = self.center + [0.032, 0, 0]
        offset = np.zeros(3)
        offset[1 if layout["axis"] == "y" else 2] = layout["sign"] * layout.get("waypoint_offset_m", 0.024)
        self.waypoints = [self.center + [-0.022, 0, 0] + offset,
                          self.center + [0.022, 0, 0] + offset, self.goal.copy()]
        initial = p.calculateInverseKinematics(self.body, 8, self.start,
                     maxNumIterations=1000,
                     residualThreshold=1e-9, physicsClientId=self.client)
        initial = np.clip(np.array(initial), self.lower, self.upper)
        for j, value in zip(self.joints, initial):
            p.resetJointState(self.body, j, float(value), physicsClientId=self.client)
        self.motor_target = initial.copy()
        self.force_scale = 1.0
        self.set_velocity(np.zeros(6))
        # Initialization only; no teleporting/clamping of the live trajectory.
        for _ in range(24):
            p.stepSimulation(physicsClientId=self.client)
        self.motor_target = self.state()[0].copy()
        self.initial_tip = self.tip()
        self.shadow = shadow

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def state(self):
        states = p.getJointStates(self.body, self.joints, physicsClientId=self.client)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def tip(self):
        return np.array(p.getLinkState(self.body, 8, computeForwardKinematics=True,
                                      physicsClientId=self.client)[4])

    def restore_snapshot(self, q, dq, motor_target):
        if not self.shadow:
            raise RuntimeError("Live instrument state may not be reset")
        for j, value, speed in zip(self.joints, q, dq):
            p.resetJointState(self.body, j, float(value), float(speed), physicsClientId=self.client)
        self.motor_target = np.array(motor_target).copy()

    def clearance(self, tip=None):
        tip = self.tip() if tip is None else np.array(tip)
        sphere = np.linalg.norm(tip - self.center) - self.config["restricted_radius_m"] - self.config["tip_radius_m"]
        workspace = min(np.min(tip - self.workspace[:, 0]), np.min(self.workspace[:, 1] - tip))
        return float(min(sphere, workspace))

    def command(self, target, observed_q=None, observed_tip=None):
        q, _ = self.state()
        if observed_q is not None:
            q = np.array(observed_q)
        # calculateJacobian accepts the delayed joint positions without
        # resetting the live body. Both q and Cartesian observation may lag.
        tip = self.tip() if observed_tip is None else np.array(observed_tip)
        desired = (np.array(target) - tip) * 4.0
        norm = np.linalg.norm(desired)
        if norm > self.config["max_cartesian_speed_m_s"]:
            desired *= self.config["max_cartesian_speed_m_s"] / norm
        jacobian, angular = p.calculateJacobian(self.body, 8, [0, 0, 0], q.tolist(),
                         [0.] * len(q), [0.] * len(q), physicsClientId=self.client)
        # Bullet Jacobians are expressed in the fixed base frame, while task
        # coordinates are world SI coordinates. Rotate BOTH linear/angular rows.
        jacobian = self.base_rotation @ np.array(jacobian)[:, :6]
        # Damped least squares; zero angular objective limits orientation drift.
        full = np.vstack([jacobian, (self.base_rotation @ np.array(angular)[:, :6]) * 0.05])
        velocity = full.T @ np.linalg.solve(full @ full.T + np.eye(6) * 1e-6,
                                            np.r_[desired, np.zeros(3)])
        limits = np.minimum(self.velocity_limits[:6], [0.6, 0.6, 0.03, 1, 1, 1])
        return np.clip(velocity, -limits, limits)

    def set_velocity(self, velocity):
        self.active_velocity = np.array(velocity, dtype=float)

    def physics_step(self):
        self.motor_target[:6] += self.active_velocity * self.dt
        # Targets, not physical states, are actuator-limited.
        self.motor_target = np.clip(self.motor_target, self.lower, self.upper)
        q_pitch = self.motor_target[1]
        for index, joint in enumerate(self.joints):
            if joint in {9, 10, 11, 12}:
                self.motor_target[index] = q_pitch if joint in {9, 12} else -q_pitch
            if joint in {6, 7}:
                self.motor_target[index] = 0
        p.setJointMotorControlArray(self.body, self.joints, p.POSITION_CONTROL,
                       targetPositions=self.motor_target.tolist(), forces=(self.forces * self.force_scale).tolist(),
                       positionGains=[0.1] * len(self.joints), velocityGains=[1.] * len(self.joints),
                       physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        return self.clearance()

    def brake(self):
        # Hold measured joint positions; inertia still evolves in PyBullet.
        self.motor_target = self.state()[0].copy()
        self.set_velocity(np.zeros(6))


class PredictiveChecker:
    """Independent shadow execution; empirical prediction, not a formal proof.

    Checks each physics sample over command then braking rollout. A finite
    same-engine prediction does not prove continuous-time or mismatched safety.
    """
    def __init__(self, config, layout):
        self.shadow = Instrument(config, layout, shadow=True)
        self.config = config
        self.calls = 0

    def check(self, robot, velocity, delayed_snapshot=None, bias_m=0.0):
        q, dq = robot.state() if delayed_snapshot is None else delayed_snapshot[:2]
        target = robot.motor_target if delayed_snapshot is None else delayed_snapshot[2]
        self.shadow.restore_snapshot(q, dq, target)
        if np.linalg.norm(velocity) == 0:
            self.shadow.brake()
        else:
            self.shadow.set_velocity(velocity)
        minimum = self.shadow.clearance() - bias_m
        for _ in range(self.config["prediction_ticks"] * robot.substeps):
            minimum = min(minimum, self.shadow.physics_step() - bias_m)
        self.shadow.brake()
        for _ in range(self.config["brake_ticks"] * robot.substeps):
            minimum = min(minimum, self.shadow.physics_step() - bias_m)
        self.calls += 1
        return minimum >= self.config["checker_margin_m"], float(minimum)

    def close(self):
        self.shadow.close()
