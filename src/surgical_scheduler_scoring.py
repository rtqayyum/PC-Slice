"""Exploratory waypoint-aware candidate ranking for development experiments.

This is a heuristic selection cost, not a safety barrier or success guarantee.
Receiver checking and physical task thresholds remain unchanged.
"""
import numpy as np


def progress_weighted_cost(wire_bytes, useful_ticks, prediction, target, initial_tip,
                           tolerance_m, deficit_weight):
    """Return (bytes-per-useful-tick weighted score, normalized waypoint deficit).

    A predicted visit to the currently acknowledged waypoint has zero deficit.
    An unreachable/unchanged waypoint has near-unit deficit. The score is used
    only to rank plans already assessed as feasible by the existing checker.
    """
    if useful_ticks < 1 or wire_bytes <= 0 or tolerance_m <= 0 or deficit_weight < 0:
        raise ValueError("Invalid progress-score parameters")
    points = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    initial_tip = np.asarray(initial_tip, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Prediction must contain 3D tip positions")
    initial_distance = float(np.linalg.norm(initial_tip - target))
    closest = float(np.min(np.linalg.norm(points - target, axis=1)))
    if not np.isfinite(closest + initial_distance):
        raise ValueError("Nonfinite waypoint distance")
    unmet = max(0.0, closest - tolerance_m)
    total = max(tolerance_m, initial_distance - tolerance_m)
    deficit = min(1.0, unmet / total)
    return float(wire_bytes / useful_ticks * (1.0 + deficit_weight * deficit)), float(deficit)
