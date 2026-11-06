"""One HPC development allocation: comparable CPU gate plus real LLM policies.

All layouts and requests are declared DEVELOPMENT inputs. This never releases
a final matrix or upgrades prior open test cases into held-out evidence.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path

from .policy_benchmark import digest
from .surgical_campaign import load_config, networks, preflight_models, run_trial, trial_id, runtime
from .surgical_v3_development import evaluate


def run(output, workers):
    config=load_config('config/surgical_task_v3.json')
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    policy=json.loads(Path(config['policy_path']).read_text())
    identity={'config':config,'runtime':runtime(),
              'scope':'development only; opened prior requests; no final research release',
              'started_utc':datetime.now(timezone.utc).isoformat(),
              'source_hashes':{str(p):digest(p) for folder in ['src','config','tests','hpc']
                              for p in Path(folder).glob('*') if p.is_file()}}
    (root/'development_manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    preflight_models(config)
    import torch
    import scipy,transformers
    if not torch.__version__.startswith('2.7.1') or transformers.__version__!='4.51.3' or scipy.__version__!='1.16.3':
        raise ValueError('Expected torch 2.7.1 / transformers 4.51.3 / scipy 1.16.3 runtime')
    from .surgical_policy import LLMEngine,documents_for,messages_for
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('One visible CUDA GPU required for this development allocation')
    (root/'hardware.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),
        'visible_gpus':torch.cuda.device_count(),'allocated_cpus':os.environ.get('SLURM_CPUS_PER_TASK')},indent=2)+'\n')
    for name in config['policy']['models']:
        engine=LLMEngine(name,config['policy'])
        try:
            case=policy['cases'][0]
            evidence=documents_for(case,policy,'rag_policy_k2')
            response=engine(messages_for(case,policy,evidence,'approach'),decoding='constrained')
            if not response['raw'].strip():
                raise ValueError('Empty GPU generation')
            from .policy_benchmark import parse_output
            parsed=parse_output(response['raw'])
            if set(parsed)!={'action','service_id','evidence_ids'}:
                raise ValueError('Structured-generation infrastructure check failed')
            (root/('gpu_check_'+name+'.json')).write_text(json.dumps(response,indent=2)+'\n')
        finally:
            engine.close()
        print('GPU development generation check:',name,flush=True)
    systems=evaluate(root/'systems',config['network_profiles'],[31011,31012],workers)
    # One declared task setting keeps this a bounded development diagnostic.
    # It provides real request latency/activation evidence, not broad efficacy.
    profiles=networks(config)
    base={'layout':config['layouts'][0],'network':profiles['delay2'],
          'budget':24,'seed':31013,'method':'joint_plan','stress':'nominal'}
    paths=[]
    for case in policy['cases']:
        spec={**base,'case':case,'condition':'no_rag','model':'deterministic'}
        paths.append(run_trial(config,spec,root/'policy'/trial_id(spec),policy))
    for name in config['policy']['models']:
        engine=LLMEngine(name,config['policy'])
        try:
            settings=list(itertools.product(policy['cases'],config['policy']['conditions'],config['policy']['decoding']))
            for index,(case,condition,decoding) in enumerate(settings,1):
                spec={**base,'case':case,'condition':condition,'model':name,'decoding':decoding}
                paths.append(run_trial(config,spec,root/'policy'/trial_id(spec),policy,engine))
                if index%12==0:
                    print('LLM development cases:',name,index,'/',len(settings),flush=True)
        finally:
            engine.close()
    rows=[json.loads((Path(p)/'metrics.json').read_text()) for p in paths]
    groups=[]
    for key in sorted({(r['model'],r['condition'],r['decoding']) for r in rows}):
        selected=[r for r in rows if (r['model'],r['condition'],r['decoding'])==key]
        groups.append({'model':key[0],'condition':key[1],'decoding':key[2],'n':len(selected),
            'contextual_correct':sum(r['policy']['scores']['contextual_correct'] for r in selected)/len(selected),
            'safe_task_success':sum(r['safe_task_success'] for r in selected)/len(selected),
            'wrong_activations':sum(r['policy']['scores']['wrong_service_activation'] for r in selected),
            'mean_request_seconds':sum(r['policy']['measured_request_seconds'] for r in selected)/len(selected)})
    expected_system=len(config['layouts'])*len(config['network_profiles'])*2*len(config['methods'])
    expected_policy=len(policy['cases'])*(1+len(config['policy']['models'])*len(config['policy']['conditions'])*len(config['policy']['decoding']))
    if len(systems['rows'])!=expected_system or len(rows)!=expected_policy:
        raise ValueError('Incomplete development execution matrix')
    status={'development_only':True,'system_executions':len(systems['rows']),
            'policy_executions':len(rows),'system_gate_pass':systems['gate_pass'],
            'expected_system_executions':expected_system,'expected_policy_executions':expected_policy,
            'final_release':False,'held_out_llm_evidence':False,'policy_groups':groups,
            'next':'Review development failures and comparator fidelity; no automatic final submission.'}
    (root/'DEVELOPMENT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    hashes={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    (root/'development_artifact_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Development complete; gate_pass=',systems['gate_pass'],'final_release=False',flush=True)
    print('Status:',root/'DEVELOPMENT_STATUS.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--workers',type=int,default=20)
    args=p.parse_args()
    if not 1<=args.workers<=20:p.error('Use 1--20 workers')
    run(args.output_dir,args.workers)


if __name__=='__main__':main()
