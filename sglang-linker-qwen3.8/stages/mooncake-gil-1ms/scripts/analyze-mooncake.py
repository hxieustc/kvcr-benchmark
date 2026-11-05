#!/usr/bin/env python3
"""Offline empirical Python/native API distributions; never remote-only RDMA."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


BASELINE = Path('/workspace/sglang-linker-qwen38-agentx-20261031-r1/runs/r16-agentx/analysis')
MIRROR = (Path(__file__).resolve().parents[2] / 'sglang-linker-qwen38-agentx/results'
          / 'results-r16-agentx-20261031T032848Z/runs/r16-agentx/analysis')
BEGIN = '<!-- BEGIN MOONCAKE OFFLINE ANALYSIS -->'
END = '<!-- END MOONCAKE OFFLINE ANALYSIS -->'
SCOPE = ('All timings measure Python/native API calls, including local and remote work; '
         'completed bytes are not pure remote bytes. No additional metadata calls were collected. '
         'Physical objects include pool/indexer/state; repeated per-layer GET keys are not '
         'logical token pages. PUT bytes are accepted payload sizes, not measured wire bytes; existing objects may be no-ops. No remote-only or native C++ CDF can be inferred.')
METRICS = ('requested_bytes', 'completed_bytes', 'objects', 'pieces', 'duration_ns', 'gbps')
METHODS = {'batch_put_from_multi_buffers': 'batch_put', 'batch_put_from': 'batch_put',
           'batch_get_into_multi_buffer_ranges': 'range_get'}
FIELDS = ('rank', 'utc', 'end_utc', 'operation', 'method', 'session_id', 'objects', 'pieces',
          'requested_bytes', 'completed_bytes', 'duration_ns', 'gbps', 'success',
          'error', 'first_key', 'scope')


def load(path):
    return json.loads(path.read_text())


def distribution(values):
    values = sorted(v for v in values if v is not None)
    result = {'count': len(values), 'mean': None, 'p50': None, 'p90': None, 'p99': None}
    if values:
        result['mean'] = math.fsum(values) / len(values)
        for key, q in [('p50', .5), ('p90', .9), ('p99', .99)]:
            index = (len(values) - 1) * q
            low, high = math.floor(index), math.ceil(index)
            result[key] = values[low] + (values[high] - values[low]) * (index - low)
    return result


def transfers(run, begin, end):
    rows, exclusions, workers = [], Counter(), {}
    for rank in range(4):
        path = run / f'mooncake-operations-{rank}.jsonl'
        counts = Counter()
        with path.open() as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                counts['raw_calls'] += 1
                operation = METHODS.get(r['method'])
                if operation is None:
                    exclusions['other_method'] += 1
                    continue
                utc, duration = r['utc'], r['duration_ns']
                numbers = [utc, duration] + [r[k] for k in METRICS[:4]]
                if any(not math.isfinite(v) or v < 0 for v in numbers):
                    raise ValueError(f'{path}:{line_no}: invalid numeric measurement')
                finish = utc + duration / 1e9
                if utc < begin or finish > end:
                    exclusions['outside_window'] += 1
                    continue
                if r['success'] is not True or r['error'] is not None:
                    exclusions['unsuccessful'] += 1
                    continue
                # Decimal GB/s = (bytes / 1e9) / (ns / 1e9).
                rows.append(dict(r, rank=rank, operation=operation, end_utc=finish,
                                 gbps=r['completed_bytes'] / duration if duration else None))
                counts[operation] += 1
        workers[str(rank)] = dict(counts)
    return rows, dict(exclusions), workers


def plot_cdfs(rows, path):
    # Keep --help and data loading usable without plotting dependencies.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, METRICS):
        for operation, label in [('batch_put', 'Successful batch PUT'),
                                 ('range_get', 'Successful range GET')]:
            values = sorted(r[key] for r in rows
                            if r['operation'] == operation and r[key] is not None)
            if values:
                ax.step(values, [(i + 1) / len(values) for i in range(len(values))],
                        where='post', label=f'{label} (n={len(values)})')
        ax.set(xlabel={'duration_ns': 'API duration (ns)',
                       'gbps': 'Completed bytes / API duration (GB/s, decimal)'}.get(key, key),
               ylabel='Empirical CDF', ylim=(0, 1))
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'No successful calls', transform=ax.transAxes, ha='center')
    fig.suptitle('Profiling window: Python/native API, local + remote combined')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(value):
    return 'unavailable' if value is None else f'{value:.6g}'


def comparison(client, system, baseline):
    reference, server = load(baseline / 'summary.json'), load(baseline / 'system-summary.json')
    text = ('# Mooncake versus KVCR AgentX\n\n'
            f'KVCR source files: `{baseline.resolve()}`. '
            'Mooncake source: this run’s `summary.json` and `system-summary.json`.\n\n'
            'Both setups: four B200 workers, 224 GiB store DRAM per worker, client concurrency '
            '48, five extra warmup requests per lane, 1,800 seconds profiling. These are setup '
            'parameters, not a guarantee of identical realized concurrency or request mix.\n\n'
            '| Metric | Statistic | Mooncake | KVCR | Unit |\n'
            '|---|---|---:|---:|---|\n')
    metrics = [('time_to_first_token', 'TTFT'), ('inter_token_latency', 'TPOT'),
               ('request_throughput', 'Request throughput'),
               ('output_token_throughput', 'Output token throughput'),
               ('total_token_throughput', 'Total token throughput'),
               ('overall_usage_prompt_cache_read_pct', 'Client prompt cache read')]
    for key, label in metrics:
        left, right = client.get(key, {}), reference.get(key, {})
        units_match = left.get('unit') == right.get('unit')
        unit = left.get('unit') or right.get('unit', 'unavailable')
        if left and right and not units_match:
            unit = f"Mooncake {left.get('unit')}; KVCR {right.get('unit')}"
        for stat, name in [('avg', 'mean'), ('p50', 'p50'), ('p90', 'p90'), ('p99', 'p99')]:
            text += f'| {label} | {name} | {fmt(left.get(stat))} | {fmt(right.get(stat))} | {unit} |\n'
    text += ('\n| Server cache tier | Mooncake tokens | Mooncake fraction | KVCR tokens | KVCR fraction |\n'
             '|---|---:|---:|---:|---:|\n')
    for key, label in [('input', 'input (recomputed)'), ('device_hit', 'device_hit (HBM)'),
                       ('host_hit', 'host_hit (store DRAM, local + remote)'),
                       ('storage_hit', 'storage_hit (separate storage-prefetch path)')]:
        values = [fmt(s.get(field, {}).get(key)) for s in (system, server)
                  for field in ('tier_tokens', 'tier_fractions')]
        text += f'| {label} | ' + ' | '.join(values) + ' |\n'
    return text + ('\nFractions are 0–1. Server token deltas and client prompt/cache accounting '
                   'have different populations and boundaries; they need not agree. Server '
                   'sampling boundaries are approximate. Unreported throughput percentiles '
                   'remain unavailable; no distribution is inferred from an aggregate.\n\n'
                   'Native DMA, inclusive source operations, and Python/native store API calls '
                   'measure different boundaries. Read current run counters rather than '
                   'substituting historical bandwidth numbers.\n\n' + SCOPE + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--kvcr-analysis', type=Path,
                        help='KVCR analysis directory; default canonical workspace path or saved local mirror')
    args = parser.parse_args()
    out = args.run / 'analysis'
    client, system = load(out / 'summary.json'), load(out / 'system-summary.json')
    with (out / 'requests.csv').open() as source:
        requests = list(csv.DictReader(source))
    begin = min(int(r['request_start_ns']) / 1e9 for r in requests)
    end = max(int(r['request_start_ns']) / 1e9 + float(r['request_latency']) / 1000
              for r in requests)
    if not math.isfinite(begin) or not math.isfinite(end) or end <= begin:
        raise ValueError('Invalid profiling request window')
    for key, value in [('profiling_start_utc', begin), ('profiling_end_utc', end)]:
        if not math.isclose(system[key], value, rel_tol=0, abs_tol=.001):
            raise ValueError(f'{key}: system-summary.json disagrees with requests.csv')
    rows, exclusions, workers = transfers(args.run, begin, end)
    paired = args.run.parents[2] / 'kvcr/runs' / args.run.name / 'analysis'
    baseline = args.kvcr_analysis or (paired if paired.is_dir() else (BASELINE if BASELINE.is_dir() else MIRROR))
    comparison_text = comparison(client, system, baseline)
    result = {'profiling_start_utc': begin, 'profiling_end_utc': end,
              'window_note': 'Same request-derived window as system analysis, without sample padding. '
              'Only calls fully contained from utc start through duration_ns are included. '
              'Timestamp filtering cannot attribute individual calls to request phases; '
              'concurrent work inside this window cannot be separated. Request completions '
              'can extend into the client grace period.',
              'scope': SCOPE, 'percentiles': 'Linear interpolation at (n-1)*q; equal weight per call.',
              'gbps_note': 'completed_bytes / duration_ns, decimal GB/s. Zero-duration calls '
              'are retained but excluded from GB/s statistics/CDFs.',
              'exclusions': exclusions, 'workers': workers, 'operations': {}}
    table = '| Operation | Measurement | n | Mean | p50 | p90 | p99 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for operation in ('batch_put', 'range_get'):
        group = [r for r in rows if r['operation'] == operation]
        stats = {key: distribution([r[key] for r in group]) for key in METRICS}
        result['operations'][operation] = {'calls': len(group), 'distributions': stats}
        for key, stat in stats.items():
            table += f'| {operation} | {key} | {stat["count"]} | ' + ' | '.join(
                fmt(stat[k]) for k in ('mean', 'p50', 'p90', 'p99')) + ' |\n'
    report = (out / 'report.md').read_text()
    if report.count(BEGIN) != report.count(END) or report.count(BEGIN) > 1:
        raise ValueError('Malformed Mooncake report section markers')
    section = (BEGIN + '\n\n## Mooncake API calls in the profiling window\n\n' + SCOPE
               + '\n\n' + result['window_note'] + '\n\n' + result['gbps_note'] + '\n\n'
               + result['percentiles'] + '\n\n' + table
               + '\n![Mooncake API CDFs](mooncake-cdfs.png)\n\n'
               + 'Raw selected calls: `mooncake-transfers.csv`; machine summary: '
               + '`mooncake-summary.json`; [KVCR comparison](kvcr-comparison.md).\n\n' + END)
    if BEGIN in report:
        start, finish = report.index(BEGIN), report.index(END) + len(END)
        if finish < start:
            raise ValueError('Reversed Mooncake report markers')
        report = report[:start] + section + report[finish:]
    else:
        report += '\n\n' + section + '\n'
    plot_cdfs(rows, out / 'mooncake-cdfs.png')
    with (out / 'mooncake-transfers.csv').open('w', newline='') as dest:
        writer = csv.DictWriter(dest, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (out / 'mooncake-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'kvcr-comparison.md').write_text(comparison_text)
    (out / 'report.md').write_text(report)
    print(json.dumps({'selected_calls': len(rows), 'exclusions': exclusions, 'kvcr_analysis': str(baseline)}))


if __name__ == '__main__':
    main()
