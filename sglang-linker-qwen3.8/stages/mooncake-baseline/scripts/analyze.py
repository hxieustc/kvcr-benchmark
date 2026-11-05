#!/usr/bin/env python3
"""Summarize official AgentX profiling exports without mixing warmup samples."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
artifact=a.run/'aiperf_artifacts'/'profile_export_aiperf.json'
d=json.loads(artifact.read_text())
keys=['time_to_first_token','inter_token_latency','full_response_inter_token_latency','request_latency','request_throughput','output_token_throughput','total_token_throughput','effective_concurrency','overall_usage_prompt_cache_read_pct','total_usage_prompt_cache_read_tokens','total_usage_prompt_tokens','request_count']
summary={k:d[k] for k in keys if k in d}
summary.update({'input_config':d['input_config'],'error_summary':d.get('error_summary'),'was_cancelled':d.get('was_cancelled'),'benchmark_duration':d.get('benchmark_duration'),'source_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()})
out=a.run/'analysis';out.mkdir(exist_ok=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for key in keys:
    value=d.get(key)
    if isinstance(value,dict):
        rows.append('| '+key+' | '+' | '.join(str(value.get(k,'—')) for k in ['unit','avg','p50','p90','p99'])+' |')
text='''# Four-worker SGLang/Mooncake direct-linker AgentX results

This run uses four B200 TP1 workers, a total client concurrency of 48, 224 GiB Mooncake DRAM per worker, five extra warmup requests per lane and 1,800 seconds of profiling. The original CI run used one GB300 and 3,600 seconds; performance is not a hardware-matched speedup comparison. NEXTN uses the CI synthetic acceptance setting, so this benchmark does not measure model quality.

| Metric | Unit | Mean | p50 | p90 | p99 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n\nSee `summary.json` for completion/error counts and the complete resolved client configuration. Warmup metrics are excluded from this table. Per-worker raw Prometheus snapshots, GPU samples and physical-HCA counters are in `telemetry.jsonl.gz`; HCA counters include other processes sharing the port. Mooncake interval telemetry and transfer errors remain in each worker log. Framework HiCache host cache is disabled; Mooncake owns the DRAM tier.\n'
(out/'report.md').write_text(text)
raw=a.run/'aiperf_artifacts/profile_export.jsonl'
if raw.exists():
    samples=[]
    counts=defaultdict(Counter)
    for line in raw.open():
        record=json.loads(line);meta=record.get('metadata',{});metrics=record.get('metrics',{})
        phase=meta.get('phase_kind','unknown')
        counts[phase]['records']+=1
        if meta.get('was_cancelled'):counts[phase]['cancelled']+=1
        elif meta.get('context_overflow_skip'):counts[phase]['context_overflow_skipped']+=1
        elif record.get('error'):counts[phase]['errors_in_request_export']+=1
        else:counts[phase]['not_cancelled_or_skipped_or_error']+=1
        if meta.get('phase_kind')!='profiling' or meta.get('was_cancelled') or record.get('error'):
            continue
        sample={'request_id':meta.get('x_request_id'),'request_start_ns':meta.get('request_start_ns')}
        for name in ['time_to_first_token','inter_token_latency','request_latency','usage_prompt_tokens','usage_prompt_cache_read_tokens','usage_completion_tokens']:
            sample[name]=metrics.get(name,{}).get('value')
        if sample['time_to_first_token'] is not None:samples.append(sample)
    if samples:
        with (out/'requests.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=samples[0]);writer.writeheader();writer.writerows(samples)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,name,title,scale,unit in [(axes[0],'time_to_first_token','TTFT',0.001,'s'),(axes[1],'inter_token_latency','TPOT',1,'ms')]:
            values=sorted(row[name]*scale for row in samples if row[name] is not None)
            ax.step(values,[(i+1)/len(values) for i in range(len(values))],where='post')
            ax.set(title=f'{title} CDF (n={len(values)})',xlabel=f'{title} ({unit})',ylabel='CDF',ylim=(0,1))
            ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/'latency-cdfs.png',dpi=160);plt.close(fig)
        with (out/'report.md').open('a') as f:f.write('\n![Profiling latency CDFs](latency-cdfs.png)\n')
    summary['raw_request_export_counts']={phase:dict(value) for phase,value in counts.items()}
    summary['profiling_latency_samples']=len(samples)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'report.md').open('a') as f:
        f.write('\nRaw request-export accounting (separate from the official error summary):\n\n```json\n'+json.dumps(summary['raw_request_export_counts'],indent=2)+'\n```\n')
print(json.dumps({k:summary[k] for k in ['time_to_first_token','inter_token_latency','request_count','error_summary'] if k in summary},indent=2))
