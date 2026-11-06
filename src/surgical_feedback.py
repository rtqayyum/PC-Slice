"""Versioned, charged state/plan/progress frames for the V3 return link."""
import struct
import numpy as np
from .surgical_progress import ProgressState


# Explicit revision: never interpret a frozen 620616 frame as a repaired frame.
MAGIC = b'P3P1'
HEADER = struct.Struct('!4sIIIIHBI')
NO_COMPLETION = 0xffffffff


def _validate_progress(waypoint, completion, sampled_tick):
    if waypoint not in (0,1,2):
        raise ValueError('Invalid executed waypoint')
    if completion is not None and (waypoint != 2 or not 0 < completion <= sampled_tick):
        raise ValueError('Invalid executed completion acknowledgment')


def encode_feedback(sequence, snapshot, plans, delay_ticks, progress=ProgressState()):
    q,dq,target,tip,tick = snapshot[:5]
    _validate_progress(progress.waypoint,progress.completion_tick,tick)
    header = HEADER.pack(MAGIC,sequence,tick,len(q),delay_ticks,len(plans),
                         progress.waypoint,NO_COMPLETION if progress.completion_tick is None else progress.completion_tick)
    vectors = np.concatenate([q,dq,target,tip]).astype('>f4').tobytes()
    acknowledgments = b''.join(struct.pack('!I',p.sequence) for p in plans)
    return header+vectors+acknowledgments


def decode_feedback(payload, known):
    if len(payload)<HEADER.size:
        raise ValueError('Truncated feedback frame')
    magic,sequence,tick,n,delay,count,waypoint,completion = HEADER.unpack(payload[:HEADER.size])
    if magic != MAGIC:
        raise ValueError('Unsupported feedback frame version')
    completion = None if completion==NO_COMPLETION else completion
    _validate_progress(waypoint,completion,tick)
    vector_end = HEADER.size+4*(3*n+3)
    if n<6 or len(payload)!=vector_end+4*count:
        raise ValueError('Malformed feedback frame')
    x = np.frombuffer(payload[HEADER.size:vector_end],dtype='>f4').astype(float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite feedback state')
    ids = [struct.unpack('!I',payload[vector_end+4*k:vector_end+4*k+4])[0] for k in range(count)]
    if any(k not in known for k in ids):
        raise ValueError('Acknowledgment for an unsent plan')
    return sequence,(x[:n],x[n:2*n],x[2*n:3*n],x[3*n:3*n+3],tick,
                     tuple(known[k] for k in ids),ProgressState(waypoint,completion)),delay
