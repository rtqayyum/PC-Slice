"""Derive comparison-figure tables from archived observations; no new trials.

Pure Python; distributed observations suffice, without HPC/model access.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

METHODS=['fixed_plan_short','fixed_plan_long','fixed_plan_extended','event_plan',
         'et_mpc_adapted','periodic_mpc_adapted','joint_plan','joint_fixed_short','joint_no_checker']
CODES=['FS','FL','FX','EV','EM','PM','JP','JS','NC']
CONDITIONS=['no_rag','rag_k3','rag_policy_k2','rag_missing']
MODELS=['Qwen2.5-7B-Instruct','Mistral-7B-Instruct-v0.3','Phi-3.5-mini-instruct']
PROFILES=['freefault1','delay1','delay2','delay3','packetloss1','packetloss3','communicationloss2','communicationloss3']

def quantile(values,p):
    v=sorted(values);idx=(len(v)-1)*p;lo=int(idx);hi=math.ceil(idx)
    return v[lo]+(v[hi]-v[lo])*(idx-lo)

def generate(records,output):
    rows=json.loads(records.read_text());output.mkdir(parents=True,exist_ok=True)
    systems=[r for r in rows if r['stage']=='systems']
    policy=[r for r in rows if r['stage']=='policy']
    assert len(systems)==288 and len(policy)==300
    def dat(name,header,values):
        (output/name).write_text(' '.join(header)+'\n'+'\n'.join(' '.join(str(x) for x in row) for row in values)+'\n')
    summary=[];dist=[];net=[];horizons=[]
    for j,(method,code) in enumerate(zip(METHODS,CODES)):
        rr=[r for r in systems if r['method']==method];assert len(rr)==32
        good=[r for r in rr if r['safe_task_success']]
        summary.append([j,code,100*len(good)/32,sum(r['breach_count']>0 for r in rr),
            mean(r['management_bytes'] for r in rr)/1000,mean(r['control_bytes'] for r in rr)/1000,
            mean(r['feedback_bytes'] for r in rr)/1000,mean(r['planning_seconds'] for r in rr),
            sum(r['planning_available_after_start'] for r in rr),sum(r['control_packets'] for r in rr),
            sum(r['expired_unsent_plans'] for r in rr),sum(r['solver_calls'] for r in rr),sum(r['solver_failures'] for r in rr)])
        ts=sorted(r['completion_time_s'] for r in good)
        # Cumulative safe completion uses all 32 trials as denominator; failures do not disappear.
        dat('completion_'+code+'.dat',['t','completed'],[[0,0]]+[[t,100*(i+1)/32] for i,t in enumerate(ts)]+[[16,100*len(good)/32]])
        for metric,scale in [('final_error_m',1000),('active_fallback_fraction',100),('minimum_clearance_m',1000),('recoveries',1),('planning_seconds_p95',1000)]:
            v=[scale*r[metric] for r in rr]
            dist.append([j,metric,min(v),quantile(v,.25),quantile(v,.5),quantile(v,.75),max(v),mean(v)])
        for i,profile in enumerate(PROFILES):
            pp=[r for r in rr if r['network']==profile];assert len(pp)==4
            net.append([i,j,sum(r['safe_task_success'] for r in pp),100*sum(r['safe_task_success'] for r in pp)/4])
        hh=[h for r in rr for h in r['selected_horizons_ticks']]
        horizons.append([j]+[100*hh.count(h)/len(hh) if hh else 0 for h in [36,72,144]]+[len(hh)])
    dat('comparison_summary.dat',['id','code','success','breaches','management','control','feedback','planning','late','sent','discarded','solvercalls','solverfailed'],summary)
    for metric in ['final_error_m','active_fallback_fraction','minimum_clearance_m','recoveries','planning_seconds_p95']:
        dat('distribution_'+metric+'.dat',['id','low','q1','median','q3','high','mean'],[[r[0]]+r[2:] for r in dist if r[1]==metric])
    dat('network_counts.dat',['profile','method','successes','percentage'],net)
    dat('horizon_choices.dat',['id','short','long','extended','n'],horizons)
    dat('planning_transport.dat',['id','late','discarded','solverfailed'],[[r[0],100*r[8]/r[9] if r[9] else 0,r[10],100*r[12]/r[11] if r[11] else 0] for r in summary])
    vals=[];activation=[];rag=[];latency=[]
    for mi,model in enumerate(MODELS):
        for ci,condition in enumerate(CONDITIONS):
            for di,decoder in enumerate(['strict','constrained']):
                rr=[r for r in policy if (r['model'],r['condition'],r['decoding'])==(model,condition,decoder)];assert len(rr)==12
                details=[r['policy_detail'] for r in rr]
                correct=sum(d['scores']['contextual_correct'] for d in details)
                schema=sum(d.get('schema_valid',False) for d in details)
                activated=sum(d['activated'] for d in details)
                assert all(not d['scores']['wrong_service_activation'] for d in details)
                vals.append([ci,mi,di,correct,schema,100*correct/12,100*schema/12])
                activation.append([ci,mi*2+di,activated,100*activated/5])
                secs=[d['request_seconds'] for d in details]
                latency.append([mi*4+ci,mi,ci,di,quantile(secs,.5),quantile(secs,.95),min(secs),max(secs)])
                if mi==0 and di==0:
                    rag.append([ci,100*mean(d['scores']['retrieval_recall'] for d in details),100*mean(d['scores']['retrieval_precision'] for d in details)])
    for di,decoder in enumerate(['strict','constrained']):
        dat('semantic_'+decoder+'.dat',['condition','model','correct','schema','accuracy','validity'],[[r[0],r[1]]+r[3:] for r in vals if r[2]==di])
        dat('latency_'+decoder+'.dat',['id','model','condition','median','p95','low','high'],[[r[0],r[1],r[2]]+r[4:] for r in latency if r[3]==di])
    dat('retrieval_metrics.dat',['condition','recall','precision'],rag)
    dat('retrieval_nonempty.dat',['condition','recall','precision'],[r for r in rag if r[0]>0])
    dat('activation_counts.dat',['condition','group','activated','coverage'],activation)
    dat('activation_ordered.dat',['condition','group','activated','coverage'],sorted(activation,key=lambda r:(r[1],r[0])))
    (output/'comparison_definitions.json').write_text(json.dumps({'observed_systems':288,'observed_policy':300,
        'method_order':dict(zip(CODES,METHODS)),'network_profile_order':PROFILES,
        'intervals':'Median, interquartile interval, and observed min/max; not confidence intervals',
        'completion_curves':'Safe completions by t divided by all 32 matched trials; failures retained at horizon',
        'planning_p95':'Per-trial 95th percentile planning-call wall time, then summarized over 32 trials',
        'activation_coverage':'Correct original-request service activations out of five valid selection requests; missing-evidence abstention may be appropriate',
        'no_new_experiments':True},indent=2)+'\n')
    print('Generated observed comparison data:',output)

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=here/'data/observations.json')
    p.add_argument('--output',type=Path,default=here/'data')
    a=p.parse_args();generate(a.records,a.output)
