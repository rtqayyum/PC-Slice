"""Source-parameterized potential-packet impairments, not measured network traces.

Python RNG implements the archived distributions, not identical upstream draws.
Loss evolves at every potential packet opportunity, even when a scheduler does
not transmit. This explicit common-random-number abstraction pairs baselines;
it is not a claim of bit-identical packet-thinned NetFI execution.
"""
from bisect import bisect_left
import random


def impairments(profile, times_ms, seed):
    rng = random.Random(seed)
    state, burst, remaining, cooldown = 0, None, 0, 0
    result = []
    for tick, time_ms in enumerate(times_ms):
        delay, dropped = 0.0, False
        kind, params = profile["kind"], profile["parameters"]
        if kind == "delay":
            lower, weights, rates = params
            rate = rng.choices(rates, weights=weights)[0]
            delay = lower + rng.expovariate(rate)
        elif kind == "packet_loss":
            good, bad, intermediate1, intermediate2 = params
            if any(intermediate1) or any(intermediate2) or good[1] or good[3] or bad[1] or bad[3]:
                raise ValueError("Adapter supports only the published two-reachable-state configurations")
            values = good if state == 0 else bad
            if rng.random() >= values[0]:
                state = 1 - state
            values = good if state == 0 else bad
            if burst is None:
                alpha, scale = values[4:6]
                burst = int(scale * ((1 - rng.random()) ** (-1 / alpha) - 1))
            if burst > 0:
                dropped, burst = True, burst - 1
            else:
                burst = None
        elif kind == "communication_loss":
            probability, minimum, maximum, cool = params
            if remaining > 0:
                dropped, remaining = True, remaining - 1
                if remaining == 0:
                    cooldown = int(cool)
            elif cooldown > 0:
                cooldown -= 1
            elif rng.random() < probability:
                dropped = True
                length = rng.randint(int(minimum), int(maximum))
                remaining = max(0, length - 1)
                if remaining == 0:
                    cooldown = int(cool)
        elif kind != "no_fault":
            raise ValueError("Unknown published impairment kind")
        arrival = None if dropped else bisect_left(times_ms, time_ms + delay, lo=tick)
        result.append({"send_tick": tick, "arrival_tick": arrival,
                       "dropped": dropped, "delay_ms": delay})
    return result
