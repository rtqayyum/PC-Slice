"""Development-only joint controller gate. No test layouts or model inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import shutil
from pathlib import Path
from .surgical_campaign import load_config, networks, _worker, trial_id, digest


def evaluate(output, profiles_to_run, seeds, workers=4):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    config=load_config('config/surgical_task_v3.json')
    from .prepare_surgical_sources import verify
    from .surgical_campaign import runtime
    verify(config['source_root'])
    bundle=root/'bundle'
    bundle.mkdir()
    for directory in ['src','config','tests']:
        shutil.copytree(directory,bundle/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['experiments/SURGICAL_TASK_V3.md','requirements-surgical-v3.txt']:
        destination=bundle/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,destination)
    profiles=networks(config)
    specs=[]
    for layout in config['layouts']:
        if layout['split']!='development':
            raise ValueError('Development gate must not open final layouts')
        for network in profiles_to_run:
            for seed in seeds:
                for method in config['methods']:
                    specs.append(dict(layout=layout,network=profiles[network],budget=24,seed=seed,method=method,stress='nominal'))
    (root/'development_plan.json').write_text(json.dumps({'config':config,'specs':specs,
        'code_hashes':{p.name:digest(p) for p in Path('src').glob('surgical*.py')},
        'runtime':runtime()},indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[(config,s,root/trial_id(s)) for s in specs]
        for folder in pool.map(_worker,jobs):
            m=json.loads((Path(folder)/'metrics.json').read_text())
            print(m['layout'],m['network'],m['method'],'success=',m['safe_task_success'],
                  'bytes=',m['management_bytes'],'completion=',m['completion_time_s'],
                  'planning=',round(m['planning_seconds'],3),'deadline_misses=',m['checker_deadline_misses'],flush=True)
    rows=[json.loads((root/trial_id(s)/'metrics.json').read_text()) for s in specs]
    index={(r['layout'],r['network'],r['seed'],r['method']):r for r in rows}
    comparisons=[]
    for baseline in config['comparison_baselines']:
        pairs=[(r,index[r['layout'],r['network'],r['seed'],baseline]) for r in rows if r['method']=='joint_plan']
        comparisons.append({'baseline':baseline,'pairs':len(pairs),
            'candidate_safe_successes':sum(a['safe_task_success'] for a,b in pairs),
            'baseline_safe_successes':sum(b['safe_task_success'] for a,b in pairs),
            'candidate_only_failures':sum(not a['safe_task_success'] and b['safe_task_success'] for a,b in pairs),
            'candidate_breaches':sum(a['breach_count']>0 for a,b in pairs),
            'mean_management_byte_difference':sum(a['management_bytes']-b['management_bytes'] for a,b in pairs)/len(pairs),
            'mean_success_completion_difference_s':sum((a['completion_time_s']-b['completion_time_s']) for a,b in pairs if a['task_success'] and b['task_success'])/max(1,sum(a['task_success'] and b['task_success'] for a,b in pairs))})
    # A strict no-regression development gate, not a statistical superiority test.
    eligible=[r for r in comparisons if r['candidate_only_failures']==0 and r['candidate_breaches']==0
              and (r['mean_management_byte_difference']<0 or r['candidate_safe_successes']>r['baseline_safe_successes'])]
    passed=len(eligible)==len(comparisons)
    result={'scope':'development only; not final efficacy','gate_pass':passed,
            'comparisons':comparisons,'rows':rows,'no_hpc_submission':True}
    (root/'development_assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DEVELOPMENT GATE:', 'PASS' if passed else 'NOT PASSED; do not launch a final sweep',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--profiles',nargs='+',default=['freefault1','delay1','delay2'])
    p.add_argument('--seeds',nargs='+',type=int,default=[31001])
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=20:
        p.error('Use 1--20 workers')
    evaluate(a.output_dir,a.profiles,a.seeds,a.workers)


if __name__=='__main__':
    main()
