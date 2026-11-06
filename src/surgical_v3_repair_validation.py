"""Fixed local DEVELOPMENT controls for progress/expiry repairs; no inference."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import shutil

from .policy_benchmark import digest
from .surgical_campaign import load_config,networks,run_trial,runtime,trial_id
from .surgical_feedback import decode_feedback
from .surgical_plans import Plan
from .surgical_progress import ExecutedTaskProgress


def read(path):
    return json.loads(path.read_text())


def check_trial(folder, config, layout):
    """Check functional invariants, independently of comparative task scores."""
    m=read(folder/'metrics.json')
    with gzip.open(folder/'events.jsonl.gz','rt') as stream:
        events=[json.loads(line) for line in stream]
    with gzip.open(folder/'trajectory.jsonl.gz','rt') as stream:
        trajectory=[json.loads(line) for line in stream]
    errors=[]
    def require(ok,reason):
        if not ok:errors.append(reason)
    hz=config['control_hz']
    require(len(trajectory)==round(config['duration_s']*hz),'trajectory count')
    sent=[e for e in events if e['kind']=='plan_sent']
    returned=[e for e in events if e['kind']=='feedback_sent']
    require(sum(e['wire_bytes'] for e in sent)==m['control_bytes'],'control bytes')
    require(sum(e['wire_bytes'] for e in returned)==m['feedback_bytes'],'feedback bytes')
    require(m['management_bytes']==m['control_bytes']+m['feedback_bytes'],'management bytes')
    require(m['transmitted_bytes']<=m['budget_packets_s']*config['packet_bytes']*config['duration_s']+1e-7,'aggregate byte budget')
    known={Plan.decode(bytes.fromhex(e['payload_hex'])).sequence:Plan.decode(bytes.fromhex(e['payload_hex'])) for e in sent}
    for e in sent:
        require(e['tick']<e['end_tick'],'already-expired transmission')
        require(e['estimated_arrival_tick']<e['end_tick'],'predicted-expired transmission')
    for e in sent+returned:
        require(len(bytes.fromhex(e['payload_hex']))+28==e['wire_bytes'],'wire payload length')
    frames={}
    for e in returned:
        sequence,snapshot,delay=decode_feedback(bytes.fromhex(e['payload_hex']),known)
        frames[sequence]=snapshot
    received_by_tick={}
    for e in events:
        if e['kind']=='feedback_received':
            received_by_tick.setdefault(e['tick'],[]).append(e)
    sender_waypoint=0
    progress=ExecutedTaskProgress(config,events[0]['waypoints_m'],events[0]['goal_m'])
    for row in trajectory:
        tick=row['tick']
        for event in received_by_tick.get(tick,[]):
            require(event['sequence'] in frames,'received unsent feedback')
            snapshot=frames[event['sequence']]
            require(snapshot[4]<=tick,'future feedback')
            require(snapshot[6].waypoint>=sender_waypoint,'sender progress regressed')
            sender_waypoint=snapshot[6].waypoint
        require(row['sender_waypoint']==sender_waypoint,'sender phase without received acknowledgment')
        state=progress.update(row['tip_m'],tick)
        require(row['waypoint']==state.waypoint,'executed progress')
        if state.completion_tick is not None and tick>=state.completion_tick:
            require(row['mode']=='completed_hold','no hold after completion')
            require(all(v==0 for v in row['applied_joint_velocity']),'motion command after completion')
    expected=None if progress.state.completion_tick is None else progress.state.completion_tick/hz
    require(expected==m['completion_time_s'],'completion disagrees with executed samples')
    # Feedback sampled at tick k must describe the completed state after k-1.
    for snapshot in frames.values():
        tick=snapshot[4]
        if tick:
            require(snapshot[6].waypoint==trajectory[tick-1]['waypoint'],'feedback phase sampled from wrong time')
        completion=None if m['completion_time_s'] is None else round(m['completion_time_s']*hz)
        require(snapshot[6].completion_tick==(completion if completion is not None and completion<=tick else None),'feedback completion sampled from wrong time')
    return {'trial_id':folder.name,'invariant_errors':errors,'metrics':m}


def validate(output):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    config=load_config('config/surgical_task_v3.json')
    if 'progress-repair' not in config['version'] or any(l['split']!='development' for l in config['layouts']):
        raise ValueError('Only the repaired development configuration is allowed')
    from .prepare_surgical_sources import verify
    verify(config['source_root'])
    profiles=networks(config)
    layout=config['layouts'][0]
    methods=['fixed_plan_short','event_plan','joint_plan','joint_fixed_short']
    specs=[dict(layout=layout,network=profiles[n],budget=24,seed=31021,method=m,stress='nominal')
           for n in ['freefault1','delay2'] for m in methods]
    specs += [dict(layout=layout,network=profiles['freefault1'],budget=24,seed=31021,method=m,stress='nominal')
              for m in ['et_mpc_adapted','periodic_mpc_adapted']]
    specs.append(dict(layout=layout,network={'id':'repair_validation_all_lost','kind':'communication_loss',
        'parameters':[1,100000,100000,0]},budget=24,seed=31021,method='joint_plan',stress='nominal'))
    bundle=root/'bundle'
    bundle.mkdir()
    for name in ['src','config','tests']:
        shutil.copytree(name,bundle/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['experiments/SURGICAL_TASK_V3.md','hpc/validate_surgical_v3_repairs.sh',
                 'AGENTS.md','PROJECT_CHARTER.md','EVALUATION_PROTOCOL.md','IMPLEMENTATION_STATUS.md']:
        destination=bundle/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,destination)
    identity={'scope':'Software/causality validation only; no inference or final efficacy',
        'started_utc':datetime.now(timezone.utc).isoformat(),'config':config,'specs':specs,'runtime':runtime(),
        'frozen_files':{str(p.relative_to(bundle)):digest(p) for p in bundle.rglob('*') if p.is_file()},
        'source_manifest_sha256':digest(Path(config['source_root'])/'pcslice_source_manifest.json'),
        'network_manifest_sha256':digest(Path(config['network_source_root'])/'source_manifest.json')}
    (root/'validation_plan.json').write_text(json.dumps(identity,indent=2)+'\n')
    results=[]
    for index,spec in enumerate(specs,1):
        folder=root/'trials'/trial_id(spec)
        run_trial(config,spec,folder)
        result=check_trial(folder,config,layout)
        results.append(result)
        m=result['metrics']
        print(f"[{index}/{len(specs)}] {m['network']} {m['method']}: "
              f"safe_completion={m['safe_task_success']} breaches={m['breach_count']} "
              f"management_bytes={m['management_bytes']} invariant_errors={len(result['invariant_errors'])}",flush=True)
    nominal=[r for r in results if r['metrics']['network']=='freefault1' and r['metrics']['method'] in methods]
    lost=results[-1]['metrics']
    passed=(not any(r['invariant_errors'] for r in results) and all(r['metrics']['safe_task_success'] for r in nominal)
            and not lost['task_success'] and lost['delivered_commands']==0 and lost['feedback_received']==0)
    status={'scope':identity['scope'],'functional_validation_pass':passed,'development_only':True,
        'final_release':False,'inference_executed':False,'executions':len(results),
        'results':results,'note':'Task outcomes and MPC failures retained. Passing software controls does not establish allocation advantage.'}
    (root/'VALIDATION_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    hashes={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    (root/'validation_artifact_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Functional validation:', 'PASS' if passed else 'FAIL', '; final_release=False',flush=True)
    print('Artifacts:',root.resolve(),flush=True)
    return passed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    if not validate(args.output_dir):
        raise SystemExit(1)


if __name__=='__main__':
    main()
