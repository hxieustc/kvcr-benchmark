<!-- Standalone presentation of historical report; plot links omitted. Original raw/results are unchanged. -->
# Native HiCache: 3 selected closed arm(s)

| Arm / attempt | TTFT mean / p50 / p90 / p99 (ms) | ITL mean (ms) | Output tokens/s | Profile outcomes |
|---|---|---:|---:|---|
| kvcr / hicache-1800-r5 | 850.8899132061855 / 358.952067 / 1760.2828045000035 / 8420.25006182 | 6.778140810200842 | 2030.7275655142996 | {'completed': 4074, 'cancelled': 0, 'errors': 0} |
| mooncake-random / hicache-1800-r5 | 764.7420214592411 / 341.286207 / 1575.787285799998 / 6913.960222679998 | 6.734983825913536 | 2034.1566934262785 | {'completed': 4085, 'cancelled': 0, 'errors': 0} |
| mooncake-local-first / hicache-1800-r5 | 819.7665277363883 / 339.593575 / 1740.0644202000008 / 8355.628751380007 | 6.775479557634185 | 2042.7945480201467 | {'completed': 4059, 'cancelled': 0, 'errors': 0} |

Official client metrics above; ITL is the reported interval metric. Derived per-request response TPOT is separately labeled in each client summary and is not GPU decode time. Warmup286 is verified; request-export counts do not establish zero omitted deadline cancellations.

| Arm | HBM / HiCache host / backend storage / recompute token fractions | Cgroup CPU cores / covered seconds | Observer lossless |
|---|---|---|---|
| kvcr | 0.9079535108959164 / 0.001749633374988652 / 2.7667048010011186e-06 / 0.090294089024294 | 9.229382062008408 / 1825.029601097107 | None |
| mooncake-random | 0.912603849052618 / 0.0014126422230939211 / 1.6108539685584463e-05 / 0.08596740018460243 | 5.1806479850561615 / 1825.0286464691162 | None |
| mooncake-local-first | 0.9084594825218741 / 0.0014117869290095801 / 1.6525675681438035e-05 / 0.09011220487343478 | 5.166598898553891 / 1805.028046131134 | None |

HiCache tier semantics: device=HBM, host=persistent framework host, storage=backend read/prefetch (local+remote), input=recompute. These are pooled token-weighted counter deltas, never physical-object hit rates. Indexer host occupancy follows primary ownership, with distinct physical bytes.

Physical pool reservation is not utilization or RSS; observed allocation is censored by fixed capacity. KVCR slot allocation includes FILLING/READY/claims; it cannot establish exact READY duplicates. Moon occupancy comes from the global master when available; per-worker backend pool usage is unavailable without object/group ownership data.

All operation CDFs and size bins have explicit populations. Backend API bytes are logical completion, NOT native copied traffic: duplicate KVCR PUT may be a no-op; Moon SDK filters missing objects, so API volumes are not comparable. Native/source CDFs include positive successful different-agent WRITEs only (source requires unique native join); self/unknown routes and READs are excluded. Backend API wall time includes host work/lock/native completion; NIXL telemetry durations are microseconds, source lifetimes are ns. Descriptors are input references; SDK descCount is retained separately and is not assumed post-merge. Logical page hashes, physical keys/objects and pieces have different units; no uniform blocks-to-bytes conversion. Same-node/different-agent is not proof of remote-only wire traffic.

CPU is measured cgroup CPU over valid adjacent endpoints<=15s, including framework/observer/collector work. Global thread/process CPU is not inferred from raced snapshots. Physical RDMA counters use4-byte data units, include other HCA users and are not transfer CDFs. No GPU compute-time or latency-stage attribution.

{'checked': True, 'cases': ['kvcr', 'mooncake-random', 'mooncake-local-first'], 'scope': 'Matched declared hardware/workload/source settings. No identical timing/cache/batch history, causal allocation gain or inference parity inferred.'}

Every arm uses fresh services, framework-only clear and a small unrelated backend canary residue. Configured Moon policy does not prove each object location or explain a latency delta. Old original KVCR, pool-index and historical Moon campaigns are deliberately not used as matched-speedup denominators.

## kvcr

Source/input scope and size bins: [kvcr/summary.json](historical-summary.json). Compressed CSVs retain full operation rows without raw keys/payloads.

## mooncake-random

Source/input scope and size bins: [mooncake-random/summary.json](historical-summary.json). Compressed CSVs retain full operation rows without raw keys/payloads.

## mooncake-local-first

Source/input scope and size bins: [mooncake-local-first/summary.json](historical-summary.json). Compressed CSVs retain full operation rows without raw keys/payloads.

