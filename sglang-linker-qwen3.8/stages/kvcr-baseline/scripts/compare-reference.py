"""Compare official profiling metrics with the preserved InferenceX artifact."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
root = a.run.parent.parent
reference = json.loads((root / 'sources/reference-artifact/aiperf_artifacts/profile_export_aiperf.json').read_text())
current = json.loads((a.run / 'aiperf_artifacts/profile_export_aiperf.json').read_text())
out = a.run / 'analysis'
out.mkdir(exist_ok=True)
lines = [
    '# Comparison with the InferenceX reference', '',
    'The reference uses one GB300 worker; this reproduction uses four B200 TP1 workers. '
    'Both use total client concurrency 48. The reference profiles for 3,600 seconds '
    'and uses ten extra warmup requests per lane; this run uses 1,800 seconds and five. '
    'These differences prevent attributing the observed ratios solely to KVCR.', '',
    '| Metric | Statistic | Unit | Reference | This run |',
    '|---|---|---|---:|---:|',
]
for name in ('time_to_first_token', 'inter_token_latency', 'request_latency',
             'request_throughput', 'output_token_throughput', 'effective_concurrency',
             'overall_usage_prompt_cache_read_pct', 'request_count'):
    r, c = reference.get(name), current.get(name)
    if not isinstance(r, dict) or not isinstance(c, dict):
        continue
    for stat in ('avg', 'p50', 'p90', 'p99'):
        if stat not in r and stat not in c:
            continue
        lines.append(f"| {name} | {stat} | {c.get('unit', r.get('unit', ''))} | {r.get(stat, '—')} | {c.get(stat, '—')} |")
lines += ['', '## Validation and interpretation', '',
    '- Preflight must demonstrate source DRAM export, semantic code readback, and positive physical cross-worker transfer counters. Validation prefixes are cleared before AgentX warmup.',
    '- NEXTN synthetic acceptance is 2.32 using real draft tokens. This is a performance experiment, not a model-quality evaluation.',
    '- Each KVCR worker owns 224 GiB of DRAM; framework HiCache host storage is disabled. The pinned PLE host table adds a separate rounded 64 GiB per worker.',
    '- The QSA/Mamba fixture transfers full-attention KV, compressed indexer state, Mamba checkpoints, PLE history and draft state. Actual token capacity differs from a full-attention-only cache with the same byte budget.',
    '- Server interval counters combine local and remote KVCR token reuse. They do not establish individual-transfer CDFs or separate logical-token hit rates for local versus remote storage.',
    '- Physical HCA counter files are not exposed in this pod; their empty maps do not mean zero traffic. KVCR transfer counters remain available. Same-node worker RDMA is not a cross-node bandwidth measurement.', '',
    '## Official completion metadata', '', '```json',
    json.dumps({label: {key: obj.get(key) for key in ('request_count', 'error_summary', 'was_cancelled', 'benchmark_duration')}
                for label, obj in (('reference', reference), ('this_run', current))}, indent=2),
    '```', '',
    'Exact deployment settings, source pins, scripts, canary responses and raw exports accompany this report. '
    'See `report.md`, `system-summary.json`, the run’s `setup.resolved.json` and top-level `startup-investigation.md`.', '',
]
(out / 'reference-comparison.md').write_text('\n'.join(lines))
print(out / 'reference-comparison.md')
