"""Finite timestamped velocity sequences; known PPC idea, not claimed novel.

The receiver never plans a task or creates motion from LLM output. It consumes
only a finite transmitted sequence, using its time index, and brakes on expiry.
Both checked baselines and the proposed scheduler use this same representation.
"""
from dataclasses import dataclass
import struct
import numpy as np

from .surgical_robot import Instrument


@dataclass
class Plan:
    sequence: int
    start: int
    stride: int
    commands: np.ndarray
    prediction: np.ndarray | None = None  # Sender-only forecast; never receiver motion.

    @property
    def end(self):
        return self.start + self.stride * len(self.commands)

    def at(self, tick):
        if self.start <= tick < self.end:
            return self.commands[(tick - self.start) // self.stride].astype(float)
        return None

    def encode(self):
        return struct.pack("!IIIH", self.sequence, self.start, self.stride, len(self.commands)) + self.commands.astype(
            ">f4").tobytes()

    @classmethod
    def decode(cls, payload):
        if len(payload) < 14:
            raise ValueError('Truncated command sequence header')
        sequence, start, stride, count = struct.unpack("!IIIH", payload[:14])
        if stride < 1 or count < 1 or len(payload) != 14 + count * 24:
            raise ValueError("Malformed command sequence")
        commands = np.frombuffer(payload[14:], dtype=">f4").reshape(count, 6).astype(np.float32)
        if not np.isfinite(commands).all():
            raise ValueError("Nonfinite command sequence")
        return cls(sequence, start, stride, commands)

    @property
    def wire_bytes(self):
        return len(self.encode()) + 28  # Declared IPv4(20)+UDP(8), no link-layer overhead.


def active_plan(plans, tick):
    valid = [p for p in plans if p.start <= tick < p.end]
    return max(valid, key=lambda p: p.sequence, default=None)


class Planner:
    def __init__(self, config, layout):
        self.shadow = Instrument(config, layout, shadow=True)
        self.config = config

    def make(self, snapshot, plans, tick, waypoint, lead_ticks, horizon_ticks, stride):
        r = self.shadow
        r.restore_snapshot(snapshot[0], snapshot[1], snapshot[2])
        # Account for the known age of the returned state and for commands that
        # the acknowledged actuator buffer is predicted to execute in transit.
        start = tick + lead_ticks
        for future in range(snapshot[4], start):
            old = active_plan(plans, future)
            if old is None:
                r.brake()
            else:
                r.set_velocity(old.at(future))
            for _ in range(r.substeps):
                r.physics_step()
            if waypoint < 2 and np.linalg.norm(r.tip() - r.waypoints[waypoint]) <= self.config["waypoint_tolerance_m"]:
                waypoint += 1
        commands, prediction = [], []
        for k in range(horizon_ticks):
            prediction.append(r.tip().copy())
            if waypoint < 2 and np.linalg.norm(r.tip() - r.waypoints[waypoint]) <= self.config["waypoint_tolerance_m"]:
                waypoint += 1
            if k % stride == 0:
                command = r.command(r.waypoints[waypoint]).astype(np.float32)
                commands.append(command)
            r.set_velocity(command)
            for _ in range(r.substeps):
                r.physics_step()
        return Plan(tick, start, stride, np.array(commands, dtype=np.float32), np.array(prediction))

    def close(self):
        self.shadow.close()
