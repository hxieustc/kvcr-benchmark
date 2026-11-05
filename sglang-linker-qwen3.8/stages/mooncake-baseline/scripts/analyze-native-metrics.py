"""Summarize native API counters and store occupancy from saved snapshots."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
out=a.run/'analysis';system=json.loads((out/'system-summary.json').read_text())
begin,end=system['profiling_start_utc'],system['profiling_end_utc']
def parse(text):
    values={}
    for line in text.splitlines():
        if line.startswith('#'):continue
        m=re.match(r'([^\s{]+)(?:\{(.*?)\})?\s+([^\s]+)',line)
        if not m:continue
        try:value=float(m[3])
        except ValueError:continue
        labels=tuple(sorted(re.findall(r'(\w+)="([^"]*)"',m[2] or '')))
        values[(m[1],labels)]=value
    return values
native=defaultdict(list);occupancy=[];gpu=defaultdict(list);memory=[];events=[]
with gzip.open(a.run/'telemetry.jsonl.gz','rt') as stream:
    for line in stream:
        row=json.loads(line);t=row['utc']
        if not begin-6<=t<=end+6:continue
        for entry in row.get('mooncake_metrics',[]):
            values=parse(entry.get('text',''));port=entry['port']
            if port==25002:
                allocated=values.get(('master_allocated_bytes',()),0)
                capacity=values.get(('master_total_capacity_bytes',()),0)
                occupancy.append({'utc':t,'allocated_bytes':allocated,'capacity_bytes':capacity,
                                  'utilization':allocated/capacity if capacity else None})
            else:native[port].append((t,values))
        if begin<=t<=end:
            for values in csv.reader(row.get('gpu_csv','').splitlines()):
                if len(values)==6:gpu[int(values[0])].append(float(values[2]))
            memory.append(row['cgroup_memory_current_bytes']/1024**3)
            events.append({k:int(v) for k,v in (l.split() for l in row['cgroup_memory_events'].splitlines())})
aggregate=defaultdict(lambda:defaultdict(float));resets=[]
for port,rows in native.items():
    if len(rows)<2:continue
    first,last=rows[0][1],rows[-1][1]
    for (metric,labels),value in last.items():
        prefix='mooncake_transfer_'
        if not metric.startswith(prefix):continue
        tail=metric[len(prefix):]
        for direction in ('read','write'):
            stem=direction+'_operation_'
            if tail not in (stem+'count',stem+'bytes',stem+'latency_sum'):continue
            delta=value-first.get((metric,labels),0)
            if delta<0:resets.append({'port':port,'metric':metric,'labels':dict(labels)});continue
            operation=dict(labels).get('op_name','unknown')
            field={'count':'count','bytes':'bytes','latency_sum':'latency_us'}[tail[len(stem):]]
            aggregate[direction+':'+operation][field]+=delta
table=[]
for operation,item in sorted(aggregate.items()):
    n=item.get('count',0);size=item.get('bytes',0);seconds=item.get('latency_us',0)/1e6
    if n==0 and size==0:continue
    table.append(dict(operation=operation,count=n,bytes=size,summed_seconds=seconds,
        mean_duration_ms=seconds/n*1000 if n else None,
        byte_time_ratio_GBps=size/seconds/1e9 if size and seconds else None))
if occupancy:
    with (out/'store-occupancy.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=occupancy[0]);writer.writeheader();writer.writerows(occupancy)
result={'native_cpp_api_aggregates':table,'counter_resets':resets,
        'gpu_utilization_mean_pct':{str(k):statistics.fmean(v) for k,v in gpu.items()},
        'peak_cgroup_memory_GiB':max(memory) if memory else None,
        'cgroup_event_delta':{k:events[-1][k]-events[0].get(k,0) for k in events[-1]} if events else {},
        'peak_store_utilization':max((x['utilization'] for x in occupancy if x['utilization'] is not None),default=None),
        'last_store_occupancy':occupancy[-1] if occupancy else None,
        'scope':'Native C++ store API counters, combining local and remote, not pure remote RDMA. Latencies use exporter microseconds. Byte/time ratios are aggregate ratios; they are not mean per-transfer bandwidth. Sample boundaries approximate the client window. Master replica lookup counters are not token cache-hit rates.'}
(out/'native-metrics-summary.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,4))
ax.plot([(x['utc']-begin)/60 for x in occupancy],[100*x['utilization'] if x['utilization'] is not None else float('nan') for x in occupancy])
ax.set(xlabel='Minutes from first profiling request',ylabel='Mooncake store occupancy (%)',title='Private master: allocated / registered bytes')
ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'store-occupancy.png',dpi=160);plt.close(fig)
lines=['\n## Native API counters and store utilization\n',result['scope'],'',
       '| Native operation | Count | Bytes | Mean duration (ms) | Bytes / summed seconds (GB/s) |',
       '|---|---:|---:|---:|---:|']
def fmt(value):return 'unavailable' if value is None else f'{value:.6g}'
for row in table:lines.append('| '+row['operation']+' | '+' | '.join(fmt(row[k]) for k in ('count','bytes','mean_duration_ms','byte_time_ratio_GBps'))+' |')
lines+=['','![Mooncake store occupancy](store-occupancy.png)','',
        'Mean sampled GPU utilization (%): `'+json.dumps(result['gpu_utilization_mean_pct'])+'`. '
        'Cgroup event deltas: `'+json.dumps(result['cgroup_event_delta'])+'`. '
        'Full resource/counter summary: `native-metrics-summary.json`.\n']
with (out/'report.md').open('a') as stream:stream.write('\n'.join(lines))
print(json.dumps({'native_operations':len(table),'counter_resets':len(resets),'occupancy_samples':len(occupancy)}))
