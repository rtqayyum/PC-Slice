"""Read-only consistency checks for the published data and source snapshot."""
import hashlib,json,math
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'study/observations.json').read_text())
summary=json.loads((root/'study/measured_summary.json').read_text())
systems=[r for r in rows if r['stage']=='systems'];policies=[r for r in rows if r['stage']=='policy']
assert (len(systems),len(policies))==(288,300)
comparisons=0
for method,expected in summary['methods'].items():
    rr=[r for r in systems if r['method']==method]
    measured={'n':len(rr),'safe_successes':sum(r['safe_task_success'] for r in rr),
              'breach_trials':sum(r['breach_count']>0 for r in rr),
              'mean_management_bytes':mean(r['management_bytes'] for r in rr),
              'mean_final_error_mm':1000*mean(r['final_error_m'] for r in rr),
              'minimum_clearance_mm':1000*min(r['minimum_clearance_m'] for r in rr)}
    for k,v in measured.items():assert math.isclose(v,expected[k],rel_tol=1e-11,abs_tol=1e-10),(method,k);comparisons+=1
for group in summary['policy_groups']:
    rr=[r for r in policies if (r['model'],r['condition'],r['decoding'])==(group['model'],group['condition'],group['decoding'])]
    assert len(rr)==group['n']
    assert sum(r['policy_detail']['scores']['contextual_correct'] for r in rr)==group['contextual_correct']
    assert sum(r['policy_detail']['activated'] for r in rr)==group['activated']
    comparisons+=3
sources=json.loads((root/'study/frozen_source_hashes.json').read_text())
for f,entry in sources['files'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==entry['published_sha256'],f
print(json.dumps({'status':'PASS','systems':len(systems),'policy':len(policies),'summary_cross_checks':comparisons,
 'source_files_verified':len(sources['files']),'new_simulation_or_inference':False},indent=2))
