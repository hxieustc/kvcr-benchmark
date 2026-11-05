# GB300 native HiCache: KVCR versus Mooncake

This compares the three completed `hicache-1800-r5` arms, all using native
SGLang HiCache: **KVCR**, **Mooncake random**, and **Mooncake local-first**.
These are separate recorded replays, not a new benchmark or an optimization
sweep. The quantitative source is [historical-summary.json](results/historical-summary.json);
[the original report](results/historical-report.md) preserves additional context.

## 1. What the measurements show

Mooncake random has the lowest recorded mean/p90/p99 TTFT. KVCR throughput is
within 1% of either Mooncake arm, and mean decode-side client latencies are
similar. Backend storage reuse accounts for less than 0.002% of accounted input
tokens in every arm; most reuse comes from HBM. These results therefore do not
establish a backend GET-bandwidth explanation for the TTFT gap.

The completed HiCache campaign has no measured KVCR optimization sequence.
The linker pool-index, Mamba-capacity, GIL and native-poll optimizations belong
to separate experiments and are not incorporated in these three arms.

## 2. Matched setup and source identities

| Component | Shared configuration |
|---|---|
| Hardware | Four GB300 GPUs, ARM64, four one-GPU workers on one node; one assigned 400 Gb/s RoCE port. |
| Parallelism | Each worker TP=DP=EP=PP=CP=1; four independent router destinations. |
| Historical pod | 4 GPUs, 40 requested / 48 maximum CPUs, 850 GiB RAM; retained allocation receipts in reference/. |
| Model | nvidia/Qwen3.8-Flash-Next-NVFP4, revision fc694b54fb0174e0913e6adf86691ef85a4ead47; weights read-only. |
| SGLang | 081c4c5da234deb73078acfc09174286d5b3af05; reconstructed HiCache PR45647-base compatibility fixes. |
| KVCR | 5ce23b1527725cf623a3fc7feb94dc6d3d5fafaa; clean native core, default num_threads=4. |
| Dynamo | f28d277edaf403ba5cae65bece687f7b29c0eba3; PR17655 router plus recorded cache-key handshake. |
| AgentX/AIPerf | AgentX1.0.12 locked wheel; release source e4bb2f835ffedbbe6b76252ea5c93caec68c5faf. |
| Mooncake engine | Unmodified 0.3.13; shared SGLang adapter contains complete-state compatibility fixes. |
| Router | dynamo-two-tier-cost-fn, cache threshold0.5, balance absolute4 / relative1.1, file discovery, TCP requests, ZMQ KV events. |
| SGLang controls | Context262144, page64, BF16 KV/state, max-running96, prefill/chunk16384, memory fraction0.88. |
| Mamba controls | GPU maximum480 checkpoints, tracking interval256; host checkpoint capacity described below. |
| Decode | NEXTN3 steps, top-k1, 4 draft tokens, synthetic acceptance2.32, match-expected/real-draft-token; a serving control, not quality evaluation. |
| HiCache | Persistent host cache, write_through, wait_complete prefetch, page_first_direct, direct host IO. |
| Trace | semianalysisai/cc-traces-weka-062126-256k, revision8fecd2fc56694469f758f0afbbb6335ad3043740; 568864747-byte traces.jsonl. |
| Client | 48 total closed-loop lanes, seed42; five extra warmups/lane plus mandatory primers, 286 observed warmups; 1800s sending +30s grace. |
| Replay | Trajectory ratios0.25–0.75, idle cap300s, system idle cap10s; use full rendered argv. |
| Network | NIXL_UCX_SGL_ENABLE=1; UCX_TLS=rc,self,cuda_copy; assigned RNIC:1; RNDV/EAGER/RMA rails=1; UCX_MEMTYPE_CACHE=n. |
| Mooncake DMA-BUF | WITH_NVIDIA_PEERMEM=0, protocol=rdma; no forced TCP. |

Full image/native package/wheel pins and compatibility patch details are in the
[KVCR Repo guide](sglang-hicache-qwen3.8-kvcr.md) and
[Mooncake Repo guide](sglang-hicache-qwen3.8-mooncake.md). Both arms share identical
router source/build controls. A rebuilt wheel must be shared across fresh
controls; source equivalence alone does not guarantee historical wheel-byte identity.
The bounded runtime observer is part of the measured setup for all arms.

KVCR uses native in-process KVCRStore with remote hints and inventory publication;
there is no vLLM Guard or external linker. Mooncake uses one metadata/master
service and four contributed segments; allocation_strategy is random or
local_first. A clean KVCR core or unpatched Mooncake engine does not imply an
unmodified SGLang integration: full Mamba metadata and router-key fixtures apply
to both adapters.

## 3. Memory layout, budgets and cache semantics

Both backends restore storage GETs into **framework host DRAM**, then SGLang
restores host state to HBM. This is not the earlier Mooncake linker direct-to-HBM route.

| Per-worker quantity | Common / KVCR | Mooncake |
|---|---:|---:|
| Combined host + backend budget | 137438953472 B (128 GiB) | same |
| Framework host request, 20% | 27487790694 B | same |
| CLI hicache-size, decimal GB | 27.487790694 | same |
| Backend request, 80% | 109951162778 B | same |
| Actual rounded host allocation | 27389719952 B | same |
| Actual backend reservation | 109934192208 B | 109951162778 B |
| PLE model host storage, separate | requested64 GiB | same |

The four-worker cache budget is512 GiB, excluding PLE and other framework/native
memory. Backend reserved capacity is not equivalent to READY occupancy or RSS.
The historical host geometry per worker is:

| Pool | Slots | Bytes per slot | Reserved bytes |
|---|---:|---:|---:|
| KV | 12598 | 1703936 | 21466185728 |
| Indexer | 12598 | 53248 | 670818304 |
| Complete Mamba checkpoint | 89 | 59019280 | 5252715920 |

KV/indexer include12 target full-attention layers plus the packed draft layer;
Mamba includes36 target linear-attention layers with no draft-Mamba requirement.
The metadata accessor exports all complete checkpoint components, including
mixed dtypes and nonempty temporal/conv/PLE/slot/history components. These units
have different sizes and reuse lifetimes; a checkpoint is not one uniform KV page.
The three host-pool reservations sum to27389719952B. Indexer ownership follows
KV slots, but occupies separate bytes.

HiCache frontend and backend capacity are distinct budgets: the framework host
retains the typed page/checkpoint pools, while the KVCR backend owns registered
regions described by their indexed layouts and Mooncake owns object-store
segments. Both integrations still depend on the framework's complete-state
semantics. Mooncake does not eliminate the model's Mamba/KV/indexer requirements.

All three arms use a framework-only reset after real peer preflight. KVCR backend
clear is a no-op, so Mooncake master remove_all is not used as an asymmetric
reset. Each arm starts fresh services/store before its own preflight, although a
small unrelated preflight canary can remain. No arm is claimed perfectly empty
at the beginning of warmup.

## 4. Client latency, throughput and queueing context

### TTFT distribution

All latency values below are milliseconds. These are official successful
profiling exports; every arm recorded zero profiling errors/cancellations.

| Arm | n | Mean | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| KVCR | 4074 | 850.890 | 358.952 | 1760.283 | 8420.250 |
| Mooncake random | 4085 | 764.742 | 341.286 | 1575.787 | 6913.960 |
| Mooncake local-first | 4059 | 819.767 | 339.594 | 1740.064 | 8355.629 |

### TPOT / client inter-token latency

| Arm | n | Mean ms | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| KVCR | 4074 | 6.778 | 5.955 | 9.618 | 22.465 |
| Mooncake random | 4085 | 6.735 | 5.854 | 9.489 | 19.193 |
| Mooncake local-first | 4059 | 6.775 | 5.849 | 9.218 | 20.338 |

Derived per-request TPOT is `(request_latency − TTFT)/(completion_tokens − 1)`
for successful requests with more than one completion token. It numerically
matches exported inter_token_latency in these records, but is client response
timing, not a measurement of GPU decode kernel time. Full-response ITL is a
separate field retained in the summary.

### Throughput and active concurrency

| Arm | Completed | Output tokens/s | Requests/s | Effective concurrency mean / p90 / p99 |
| --- | --- | --- | --- | --- |
| KVCR | 4074 | 2030.728 | 2.226 | 14.664 / 25.000 / 36.000 |
| Mooncake random | 4085 | 2034.157 | 2.232 | 14.447 / 25.000 / 37.000 |
| Mooncake local-first | 4059 | 2042.795 | 2.247 | 14.571 / 25.000 / 36.000 |

48 lanes is a closed-loop workload limit, not48 continuously active requests.
Effective concurrency includes arrival/replay pacing and service effects; it is
not a router queue-depth counter. The summary has no exclusive per-request
router-queue, worker-scheduler, recompute and storage critical-path timeline.
TTFT cannot be decomposed by subtracting unrelated operation means or percentiles.

| KVCR relative to | TTFT mean | TTFT p90 | TTFT p99 | Output throughput |
| --- | --- | --- | --- | --- |
| Mooncake random | +11.26% | +11.71% | +21.79% | -0.17% |
| Mooncake local-first | +3.80% | +1.16% | +0.77% | -0.59% |

Positive TTFT differences mean KVCR is slower. Mooncake random is the strongest
observed TTFT control here; local-first has slightly greater throughput but worse
mean/tail TTFT than random. There is one completed replay per arm, so these are
observed differences, not confidence intervals or a universal placement-policy ranking.

## 5. Tiered cache reuse and utilization

The following percentages use pooled token-weighted server counter deltas from
four endpoints over approximately the client window. The denominator is all
accounted input tokens, not GET calls, object count or bytes. The four columns
partition reuse/recompute attribution; tier hits are not conditional hit probabilities.

| Arm | HBM % | Framework host % | Backend % | Recomputed % | Backend-hit tokens |
| --- | --- | --- | --- | --- | --- |
| KVCR | 90.795351 | 0.174963 | 0.000277 | 9.029409 | 1152 |
| Mooncake random | 91.260385 | 0.141264 | 0.001611 | 8.596740 | 6720 |
| Mooncake local-first | 90.845948 | 0.141179 | 0.001653 | 9.011220 | 6848 |

Backend storage credit combines local and remote reuse; it does not identify
wire bytes. KVCR has1152 storage-hit tokens, versus6720/6848 for Mooncake, but
all are tiny fractions of the total. KVCR has a greater framework-host fraction
than Mooncake. The approximately0.47 percentage-point HBM attribution gap
versus Mooncake random is much larger than the backend attribution gap.

Per-pool occupancy is available; per-pool **query/hit denominators** for Mamba,
indexer and KV are not. Do not call occupancy a Mamba hit rate or assume a storage
API success means an entire request avoided prefill. A usable hybrid prefix
requires aligned KV/indexer data and the right complete Mamba checkpoint.

The table below reports ranges across four workers of time-weighted allocated
host bytes, in GiB; it is not a simultaneous global occupancy sum. Sampling uses
a15s cap and each row's own covered window, which can differ between arms.

| Arm | Framework host pool | Worker mean allocation range, GiB |
| --- | --- | --- |
| KVCR | kv | 19.922–19.931 |
| KVCR | indexer | 0.623–0.623 |
| KVCR | mamba | 3.360–3.897 |
| Mooncake random | kv | 19.925–19.955 |
| Mooncake random | indexer | 0.623–0.624 |
| Mooncake random | mamba | 3.444–4.049 |
| Mooncake local-first | kv | 19.925–19.953 |
| Mooncake local-first | indexer | 0.623–0.624 |
| Mooncake local-first | mamba | 3.337–4.136 |

Allocated KVCR backend slots can include FILLING/READY/claimed state; the metric
is not globally deduplicated READY payload. Capacity, allocated bytes, utilization,
RSS and usable prefix hits must be kept separate.

## 6. Backend API size, duration and logical bandwidth

These API populations include the integration's lookup/admission/descriptor and
completion behavior. Positive complete operations form the distributions below;
partial/missing cache results remain in outcome counts. API completion bytes are
**logical accepted/completed bytes**, not guaranteed copied traffic: duplicate
KVCR PUTs can be no-ops and Mooncake SDK calls can filter unavailable objects.
Mooncake GET bytes combine local and remote portions, so the rates below are not
remote-only RDMA bandwidth and must not be compared as equivalent to NIXL native bandwidth.

### Completed logical bytes

| Dataset | n | Mean | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| KVCR PUT | 1122 | 113955621.462 | 72650768.000 | 218103808.000 | 218103808.000 |
| KVCR GET | 6 | 24944645.333 | 15335424.000 | 59019280.000 | 59019280.000 |
| Mooncake random PUT | 1051 | 97684063.680 | 63864848.000 | 218103808.000 | 218103808.000 |
| Mooncake random GET | 88 | 47810374.545 | 6815744.000 | 192374374.400 | 218103808.000 |
| Mooncake local-first GET | 143 | 34034838.713 | 6815744.000 | 107007180.800 | 218103808.000 |
| Mooncake local-first PUT | 987 | 102663108.993 | 70094864.000 | 218103808.000 | 218103808.000 |

### Reported logical blocks / typed units

| Dataset | n | Mean | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| KVCR PUT | 1122 | 52.308 | 10.000 | 128.000 | 128.000 |
| KVCR GET | 6 | 3.000 | 0.000 | 9.000 | 11.700 |
| Mooncake random PUT | 1051 | 87.890 | 61.000 | 256.000 | 256.000 |
| Mooncake random GET | 88 | 51.409 | 30.500 | 128.000 | 128.000 |
| Mooncake local-first GET | 143 | 37.350 | 16.000 | 128.000 | 128.000 |
| Mooncake local-first PUT | 987 | 90.214 | 69.000 | 256.000 | 256.000 |

### Input descriptors / Mooncake objects

| Dataset | n | Mean | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| KVCR PUT | 1122 | 158.419 | 256.000 | 260.000 | 260.000 |
| KVCR GET | 6 | 10.333 | 9.000 | 18.000 | 23.400 |
| Mooncake random PUT | 1051 | 133.169 | 102.000 | 256.000 | 260.000 |
| Mooncake random GET | 88 | 77.227 | 45.500 | 225.800 | 256.000 |
| Mooncake local-first GET | 143 | 56.077 | 24.000 | 128.000 | 256.000 |
| Mooncake local-first PUT | 987 | 136.844 | 108.000 | 260.000 | 260.000 |

### API wall duration, ms

| Dataset | n | Mean | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| KVCR PUT | 1122 | 9.054 | 9.672 | 11.662 | 14.035 |
| KVCR GET | 6 | 4.848 | 5.062 | 6.201 | 6.267 |
| Mooncake random PUT | 1051 | 4.014 | 3.116 | 7.406 | 9.611 |
| Mooncake random GET | 88 | 2.047 | 1.352 | 5.615 | 7.786 |
| Mooncake local-first GET | 143 | 1.659 | 0.942 | 4.134 | 8.959 |
| Mooncake local-first PUT | 987 | 4.535 | 3.693 | 9.505 | 10.203 |

### Logical completion bandwidth, GB/s

| Dataset | n | Mean | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| KVCR PUT | 1122 | 14.285 | 12.949 | 21.209 | 93.394 |
| KVCR GET | 6 | 4.686 | 2.661 | 11.253 | 12.712 |
| Mooncake random PUT | 1051 | 22.164 | 24.771 | 30.829 | 32.633 |
| Mooncake random GET | 88 | 14.487 | 7.395 | 33.371 | 38.673 |
| Mooncake local-first GET | 143 | 11.823 | 8.000 | 29.897 | 34.828 |
| Mooncake local-first PUT | 987 | 21.476 | 22.581 | 29.889 | 32.353 |

Reported block units are backend-specific adapter units. KVCR checkpoint-typed
keys and Mooncake objects are not interchangeable with one fixed64-token page;
a zero KV logical-block count with positive checkpoint bytes is not a zero-byte
transfer. Fractional percentile counts are interpolated quantiles.

### Operation outcomes and bytes / summed wall-time ratio

| Dataset | Recorded calls | Eligible distributions | Outcome counts | Logical bytes / summed wall, GB/s |
| --- | --- | --- | --- | --- |
| KVCR PUT | 1122 | 1122 | complete_cache_operation=1122 | 12.586 |
| KVCR GET | 6 | 6 | complete_cache_operation=6 | 5.145 |
| Mooncake random PUT | 1051 | 1051 | complete_cache_operation=1051 | 24.334 |
| Mooncake random GET | 119 | 88 | complete_cache_operation=88, incomplete_cache_operation=31 | 23.351 |
| Mooncake local-first GET | 202 | 143 | complete_cache_operation=143, incomplete_cache_operation=59 | 20.519 |
| Mooncake local-first PUT | 987 | 987 | complete_cache_operation=987 | 22.640 |

The ratio-of-sums is not the arithmetic mean of per-call bandwidth. Overlapping
operation wall times are not elapsed campaign time, so neither number describes
whole-node link utilization. Incomplete Mooncake GETs are cache-operation
outcomes, with no recorded backend exceptions here; they are not automatically
network failures or failed client requests.

### API PUT duration conditioned on completed logical size

| Dataset | Size bin | n | Mean ms | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| KVCR PUT | <1MiB | 7 | 2.502 | 2.319 | 3.164 | 4.049 |
| KVCR PUT | 1-8MiB | 54 | 3.026 | 3.013 | 5.068 | 6.355 |
| KVCR PUT | 8-64MiB | 385 | 6.418 | 4.646 | 7.386 | 11.536 |
| KVCR PUT | 64-256MiB | 676 | 11.105 | 10.548 | 12.245 | 14.561 |
| Mooncake random PUT | <1MiB | 10 | 0.600 | 0.529 | 0.893 | 1.181 |
| Mooncake random PUT | 1-8MiB | 87 | 1.048 | 0.972 | 1.938 | 2.924 |
| Mooncake random PUT | 8-64MiB | 449 | 2.163 | 2.166 | 3.300 | 4.562 |
| Mooncake random PUT | 64-256MiB | 505 | 6.239 | 6.820 | 9.086 | 9.706 |
| Mooncake local-first PUT | <1MiB | 10 | 0.727 | 0.553 | 1.083 | 2.132 |
| Mooncake local-first PUT | 1-8MiB | 42 | 0.786 | 0.575 | 1.587 | 2.483 |
| Mooncake local-first PUT | 8-64MiB | 426 | 2.189 | 2.211 | 3.225 | 4.824 |
| Mooncake local-first PUT | 64-256MiB | 509 | 6.882 | 7.007 | 9.699 | 10.389 |

KVCR remains slower in these broad PUT-size bins, which makes integration-side
PUT work a useful optimization target. Broad-bin distributions still differ in
exact bytes, state type and duplicate content; this is not a matched per-key
causal experiment or proof that PUT wall time lies on the TTFT critical path.

## 7. KVCR eligible native transfers and source lifetimes

There are55 positive, completed, different-agent source WRITEs in the eligible
native/source populations, versus1873 recorded source lifetimes overall.
The1818 excluded source records are not automatically errors. Another1096 native
WRITE records lack eligible source-write attribution. Preserve those records;
do not reinterpret their exclusion as RDMA failure.

Different-agent does not imply different-node. All workers share one node, and
these data do not by themselves prove wire-only RoCE traffic or cross-node scaling.
The source lifetime here includes construction through cleanup; older linker
source timers can stop before cleanup and must not be mixed into this table.

| Metric | n | Mean | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| Native bytes | 55 | 55302047.127 | 59019280.000 | 59019280.000 | 59019280.000 |
| Native typed blocks | 55 | 1.582 | 1.000 | 1.000 | 12.000 |
| Input descriptors | 55 | 4.691 | 4.000 | 4.000 | 17.520 |
| Native descriptor count | 55 | 4.400 | 4.000 | 4.000 | 17.520 |
| Native duration, ms | 55 | 2.040 | 2.303 | 2.477 | 2.518 |
| NIXL wrapper duration, ms | 55 | 2.214 | 2.415 | 2.644 | 2.975 |
| Source through cleanup duration, ms | 55 | 2.343 | 2.552 | 2.776 | 3.105 |
| Native bandwidth, GB/s | 55 | 27.408 | 25.494 | 38.550 | 42.320 |
| NIXL wrapper bandwidth, GB/s | 55 | 24.879 | 24.248 | 33.715 | 38.832 |
| Source bandwidth, GB/s | 55 | 23.337 | 22.997 | 31.084 | 35.796 |

The payload population totals about3.042GB. Native and wrapper/source timers
have different boundaries. The same-payload native/source mean durations differ
by about0.303ms; this is an average inclusive lifetime difference, not a measured
exclusive stage decomposition. Ratio-of-sums throughput is24.978GB/s for the
NIXL wrapper and23.602GB/s through source cleanup. Native per-transfer arithmetic
mean27.408GB/s is a different statistic.

Only two source transfers fall below1MiB;53 fall in8–64MiB, dominated by the
59019280-byte complete checkpoint. This is a narrow population, not a broad
KV-page size sweep. Native descriptor count is an instrumented count, not proof
of post-merge wire SGL layout or memory-fragmentation cost.

| Timer | Payload size bin | n | Mean GB/s | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| Native | <1MiB | 2 | 12.270 | 12.270 | 13.567 | 13.858 |
| Native | 8-64MiB | 53 | 27.979 | 25.538 | 38.575 | 42.330 |
| Source through cleanup | <1MiB | 2 | 2.673 | 2.673 | 3.172 | 3.285 |
| Source through cleanup | 8-64MiB | 53 | 24.117 | 23.010 | 31.235 | 35.820 |

There is no equivalent validated Mooncake remote-only per-source native
population in this published summary. Mooncake API GET rates cannot fill that
gap. The JSON preserves available aggregates and scope flags; exact new CDFs,
remote/local decomposition or parallel-source batch throughput require the
corresponding primitive rows and valid joins, not quantile reconstruction.

## 8. Bottleneck evidence and next investigations

### Observed evidence

| Arm | Whole-pod mean CPU cores | Covered seconds | Profile-window seconds |
| --- | --- | --- | --- |
| KVCR | 9.229 | 1825.030 | 1830.001 |
| Mooncake random | 5.181 | 1825.029 | 1830.002 |
| Mooncake local-first | 5.167 | 1805.028 | 1806.715 |

1. **PUT integration cost is a candidate.** KVCR API PUT mean9.054ms exceeds
   Mooncake4.014/4.535ms, including within broad size bins. Split admission,
   duplicate detection, metadata/descriptor construction, native submit,
   completion and cleanup before selecting a change. Confirm whether the work
   delays scheduler progress or competes for CPU on the request critical path.
2. **CPU work is a candidate, with attribution missing.** KVCR uses about9.23
   whole-pod cores versus5.18/5.17. This includes model/framework workers,
   backends, native threads, observers and collectors. It does not establish
   which thread, GIL behavior, polling loop or library causes the difference.
   Obtain CPU profiles and progress-loop gaps with identical instrumentation.
3. **Usable hybrid-state reuse needs direct measurement.** Storage-hit token
   credit is exceptionally small in all arms. Join request hints, required
   groups/checkpoint boundary, backend ownership/READY state and actual fetch
   results to distinguish absent data, unusable checkpoint alignment, selection
   misses and backend eviction. Do not optimize a synthetic raw hit counter at
   the expense of complete, usable prefixes.
4. **The55 eligible transfers do not show a large source-only collapse.** Their
   mean native duration2.040ms and source duration2.343ms are close compared with
   the multi-second TTFT tail. Old vLLM caller-handoff/pin-wait diagnoses are not
   evidence for this native SGLang campaign. A faster native transfer may help
   some requests, but these summaries cannot quantify its TTFT benefit.
5. **Matched repeats are needed before ranking policies or optimizations.**
   Closed-loop arms have different request counts and cache/batch histories.
   Repeat controls with the same pins, budgets, allocation and observers, then
   retain each behavioral change as a separate measured arm. No such new test
   is launched by this documentation update.

### Why these data do not imply monotonic TTFT versus cache size

These three HiCache arms have the same host/Mamba/KV/indexer budgets; they are
not a cache-size sweep. The linker M2048/M2261/M1536 results cannot be relabeled
as this campaign. More generally, changing checkpoint reservation can trade
space against KV/indexer pools, alter eviction/admission and prefix alignment,
and change closed-loop queueing/batching history. More reserved bytes alone
need not increase complete usable reuse or reduce TTFT. To test monotonicity,
measure all pool capacities and occupancy, semantic hit opportunities, request
history and critical-path timings while changing only one controlled budget.

## 9. Coverage, interpretation and reproduction

All three observer streams lack final close/heartbeat confirmation; lossless
coverage is unknown, even where individual recorded dropped counters are zero.
Tables above describe observed eligible data, not guaranteed complete campaign
transfer populations. Cgroup and occupancy sampling windows/coverage differ;
retain their per-row coverage when extending the analysis.

There are no measured exclusive router/framework/recompute/storage components
for each TTFT, no per-type Mamba/indexer/KV query-hit denominators, and no
validated Mooncake remote-only native distribution in this summary. Mark these
unavailable rather than zero or deriving them from unrelated counters.

Use the separate [KVCR Google Repo guide](sglang-hicache-qwen3.8-kvcr.md) or
[Mooncake Google Repo guide](sglang-hicache-qwen3.8-mooncake.md) to sync immutable
sources and this package. The [KVCR manual](manual-running-kvcr.md) and
[Mooncake manual](manual-running-mooncake.md) provide complete Git-only and
individual-component execution paths. Preserve recorded commands, source/wheel
identity, allocation/native qualification, complete-state checks, observers and
raw captures. Download and verify closed raw data for analysis on the local
server; retain pods and historical results.
