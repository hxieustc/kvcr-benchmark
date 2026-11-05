"""Analyze saved metrics without confusing interval counters with transfers."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
out=a.run/'analysis';out.mkdir(exist_ok=True)
requests=list(csv.DictReader((out/'requests.csv').open()))
starts=[int(r['request_start_ns'])/1e9 for r in requests if r['request_start_ns']]
assert starts,'Need successful profiling requests to bound server analysis'
begin=min(starts);end=max(float(r['request_start_ns'])/1e9+float(r['request_latency'])/1000 for r in requests)
selected=['sglang:num_running_reqs','sglang:num_queue_reqs','sglang:token_usage','sglang:full_token_usage','sglang:mamba_usage','sglang:kv_used_tokens','sglang:kv_evictable_tokens','sglang:kv_available_tokens','sglang:gen_throughput','sglang:cache_hit_rate']
samples=[];tier_rows=[];hca=[]
with gzip.open(a.run/'telemetry.jsonl.gz','rt') as f:
    for line in f:
        record=json.loads(line);t=record['utc']
        if not begin-6<=t<=end+6:continue
        for w in record['workers']:
            vals=Counter();tiers=Counter()
            for l in w.get('text','').splitlines():
                if l.startswith('#'):continue
                m=re.match(r'([^\s{]+)(?:\{(.*?)\})?\s+([^\s]+)',l)
                if not m:continue
                name,labels,value=m.groups()
                try:value=float(value)
                except ValueError:continue
                if name in selected:vals[name]+=value
                if name=='sglang:prefill_effective_tokens_total':
                    mode=re.search(r'(?:^|,)mode="([^"]+)"',labels or '')
                    if mode:tiers[mode[1]]+=value
            samples.append(dict(utc=t,rank=w['rank'],**dict(vals)))
            if tiers:tier_rows.append(dict(utc=t,rank=w['rank'],**dict(tiers)))
        c=record.get('hca_port_counters',{})
        if 'port_xmit_data' in c and 'port_rcv_data' in c:
            hca.append((t,c['port_xmit_data']*4,c['port_rcv_data']*4))
def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
write_csv('worker-metrics.csv',samples);write_csv('tier-token-counters.csv',tier_rows)
totals=Counter();per_worker={}
for rank in range(4):
    rows=[r for r in tier_rows if r['rank']==rank]
    if len(rows)<2:continue
    d={mode:rows[-1].get(mode,0)-rows[0].get(mode,0) for mode in ('input','device_hit','host_hit','storage_hit')}
    assert min(d.values())>=0,('counter reset',rank,d)
    denominator=sum(d.values());totals.update(d)
    per_worker[str(rank)]={'tokens':d,'total':denominator,'fraction':{k:v/denominator if denominator else None for k,v in d.items()}}
denominator=sum(totals.values())
kv=Counter();gauges=[]
for rank in range(4):
    for line in (a.run/f'worker-{rank}.log').read_text(errors='replace').splitlines():
        m=re.search(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z).*KVCR telemetry rank=\d+ interval_s=(\S+) stats=(\{.*\})',line)
        if not m:continue
        t=datetime.fromisoformat(m[1].replace('Z','+00:00')).timestamp()
        if not begin<=t<=end:continue
        stats=json.loads(m[3]);kv.update({k:v for k,v in stats.items() if k.endswith(('.counter','.count','.sum'))})
        gauges.append({'utc':t,'rank':rank,**{k:v for k,v in stats.items() if k.endswith('.gauge')}})
write_csv('kvcr-state.csv',gauges)
result={'profiling_start_utc':begin,'profiling_end_utc':end,
        'sample_boundary_note':'Server sample/interval boundaries approximate client profiling boundaries within roughly one 5-second interval.',
        'tier_tokens':dict(totals),'tier_total':denominator,'tier_fractions':{k:v/denominator if denominator else None for k,v in totals.items()},
        'per_worker':per_worker,'kvcr_interval_totals':dict(kv),
        'cache_tier_note':'For this direct-linker integration, device_hit=HBM and host_hit=KVCR-owned DRAM, combining local and remote. UnifiedCacheLinker.match exposes the KVCR hit as host_hit_length; load_back materializes it and schedule_policy records host_hit. The independent framework host controller is disabled. storage_hit is the separate storage-prefetch accounting path, not the direct-linker DRAM tier. Input is recomputed tokens. KVCR piece/byte counters cannot be converted into logical token hit rates.',
        'tier_semantics':{'input':'recomputed tokens','device_hit':'HBM','host_hit':'KVCR-owned DRAM (local plus remote)','storage_hit':'separate storage-prefetch path'},
        'transfer_note':'Interval histogram count/sum/max cannot reconstruct individual transfer size/bandwidth CDFs. Byte totals divided by summed durations are byte-time ratios, not arithmetic means of per-transfer bandwidth.',
        'hca_samples':len(hca),
        'hca_note':('Physical-port counters use 4-byte units and include other processes sharing mlx5_0.' if hca else 'Physical-port counters are not exposed in this pod; physical NIC bandwidth is unavailable, not zero. KVCR byte/duration telemetry remains available.')}
(out/'system-summary.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
for rank in range(4):
    rs=[r for r in samples if r['rank']==rank]
    for ax,key,title in zip(axes,('sglang:num_running_reqs','sglang:num_queue_reqs','sglang:token_usage'),('Running requests','Queued requests','HBM token utilization fraction')):
        points=[((r['utc']-begin)/60,r[key]) for r in rs if key in r]
        if points:ax.plot(*zip(*points),label=f'worker {rank}')
        ax.set(ylabel=title);ax.grid(alpha=.25)
for ax in axes:ax.legend()
axes[-1].set_xlabel('Minutes from first profiling request');fig.tight_layout();fig.savefig(out/'worker-load.png',dpi=150);plt.close(fig)
with (out/'report.md').open('a') as f:
    f.write('\n## Server-side cache and load\n\n'+result['sample_boundary_note']+' '+result['cache_tier_note']+'\n\n')
    f.write('| Tier | Tokens | Fraction |\n|---|---:|---:|\n')
    for k,v in totals.items():f.write(f'| {k} | {v:.0f} | {v/denominator:.4%} |\n' if denominator else f'| {k} | {v:.0f} | unavailable |\n')
    f.write('\n![Worker load and utilization](worker-load.png)\n\n'+result['transfer_note']+' '+result['hca_note']+' See `system-summary.json` and the saved counter/state CSVs.\n')
print(json.dumps({'tier_tokens':dict(totals),'metric_samples':len(samples),'kvcr_intervals':len(gauges)}))
