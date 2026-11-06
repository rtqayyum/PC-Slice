"""Executed progress for the authored reaching task, not a safety proof."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProgressState:
    waypoint: int = 0
    completion_tick: int | None = None


class ExecutedTaskProgress:
    """Only consecutive executed tip samples advance task milestones/dwell."""
    def __init__(self, config, waypoints, goal):
        if len(waypoints) != 3:
            raise ValueError('The reaching task requires two route waypoints and one goal')
        self.waypoints, self.goal = waypoints, goal
        self.waypoint_tolerance = config['waypoint_tolerance_m']
        self.goal_tolerance = config['goal_tolerance_m']
        self.required_dwell = math.ceil(config['goal_dwell_s']*config['control_hz'])
        self.state = ProgressState()
        self.dwell = 0
        self.last_tick = -1

    def update(self, tip, tick):
        """Consume the end-of-control-tick position exactly once, in order."""
        if tick != self.last_tick+1:
            raise ValueError('Executed progress requires consecutive control ticks')
        if not all(math.isfinite(float(x)) for x in tip):
            raise ValueError('Nonfinite executed tip position')
        self.last_tick = tick
        if self.state.completion_tick is not None:
            return self.state
        waypoint = self.state.waypoint
        if waypoint < 2 and math.dist(tip,self.waypoints[waypoint]) <= self.waypoint_tolerance:
            waypoint += 1
        self.dwell = self.dwell+1 if waypoint==2 and math.dist(tip,self.goal)<=self.goal_tolerance else 0
        completion = tick+1 if self.dwell >= self.required_dwell else None
        self.state = ProgressState(waypoint,completion)
        return self.state
