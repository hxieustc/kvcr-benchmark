# Qwen3.8 on four GB300 GPUs: KVCR linker versus Mooncake linker

## 1. Findings and which results to use

The original KVCR linker has two demonstrated problems: expensive global eviction
candidate scanning and a fixed pool partition that gives nearly all128GiB to
Mamba checkpoints. Pool indexing and M1536 redistribution recover much of the
throughput. Subsequent confirmation/GIL/native-poll experiments improve some
outcomes but do not establish general TTFT parity with Mooncake.

Use **M1536 + pool index** as the capacity-tuned GB300 comparison, and
**KVCR native-poll1ms** as the strongest completed mean-TTFT/output-throughput
combination in this optimization sequence. Label the latter with every inherited
change. Keep **Mooncake R3/default GIL** as the original strong control and
**Mooncake GIL1ms** as the scheduling-matched control for the last two KVCR arms.
Report both rather than selecting whichever denominator gives the best ratio.
The fresh random/local-first Mooncake pair answers a separate placement question.

The B200 M2048 result is best within that B200224GiB configuration; it must not
be mixed into a GB300128GiB Mamba-capacity sweep. No global hardware-independent
“best” is established. A single sequential replay per setting gives observed
results, not confidence intervals or a proof of a universal optimum.

## 2. What was held constant and what was not

| Dimension | Common GB300 linker contract |
|---|---|
| Hardware |4GB300 on one ARM64 node;4independent one-GPU workers;TP/DP/EP/PP/CP1 |
| Model |Qwen3.8-Flash-Next-NVFP4, revfc694b54fb0174e0913e6adf86691ef85a4ead47 |
| Source baseline |SGLang5b65e088969b + archived integration; KVCR3d8c9b831e8f; Mooncake0.3.13 |
| Router |Dynamo PR17655 head3f0ef62c22ce; identical recorded wheel hashes; two-tier0.5/4/1.1 |
| Framework cache |No independent HiCache host controller; this is the linker experiment family |
| Backend DRAM |128GiB/worker,512GiB total; PLE64GiB/worker is separate model data |
| GPU/cache geometry |memory fraction.88; BF16KV/state; page64; context262144; MambaGPU480/track256 |
| Model kernels |trtllm_mha; full decode graph; flashinfer linear attention; NEXTN3/topk1/draft4 |
| Synthetic acceptance |2.32, match-expected, real-draft-token; performance control, not quality eval |
| Client |AgentX1.0.12,48lanes,5extra warmups/lane +46primers=286warmup rows |
| Trace |semianalysis_cc_traces_weka_062126_256k, revision8fecd2fc56694469f758f0afbbb6335ad3043740,seed42 |
| Duration |1800s sending plus30sgrace; warmup elapsed time varies |
| Fabric |one assigned RoCE RNIC, same-node peer traffic; KVCR SGL1 and UCXrails1 |

Actual pod/node/GPU identities and startup histories differ across the complete
historical series. The placement pair explicitly checks matching hardware among
its own two arms. The original baseline reused already-loaded services; later
arms generally start fresh. Closed-loop performance changes request progression,
which changes cache history, batching, placement and successful-request cohort.
The same corpus/seed/configuration does not imply identical arrivals or GPU work.
Exact settings and argv are bundled per stage; common instructions are in
[manual reproduction](manual-running.md).

## 3. Memory and complete-state semantics

Both linkers use the Qwen hybrid device-pool assembly. Target/draft KV and compressed
indexer data must accompany the required Mamba/PLE/history checkpoint. Neither
backend can restore a complete prefix using only a checkpoint whose required KV
pages have disappeared. Existing HBM prefix coverage changes which external pages
are actually required.

| Property | KVCR in these experiments | Mooncake in these experiments |
|---|---|---|
| Stored-state requirements |KV/indexer + complete checkpoint at usable boundary |Same required Qwen states |
| Backend memory allocation |Fixed typed physical regions and slot capacities |Object allocations from contributed registered segments |
| Main capacity knobs |TotalDRAM + optional M checkpoint count; derived KVpage count |Total segment bytes; no equivalent fixed semantic M reservation |
| Local/remote discovery |Local READY plus Dynamo-selected hinted source; physical confirmation |Object existence/location through global store metadata |
| Metadata service |No Mooncake-style central object metadata server |Mooncake master |
| Data plane |NIXL/UCX |Mooncake Transfer Engine |
| Model integration |Linker owner, confirmation, claims and per-layer delivery |Lookup plus load/offload threads and session/range GET |
| Remote path distinction |Confirmation can fetch to target DRAM, then local layer delivery |Range GET destinations are registered GPU pointers |
| Repeated PUT |Local READY/writer checks; worker-owned replicas can still exist |Global object existence can avoid repeated storage of existing objects |

Shared object allocation does not guarantee perfect fungibility: allocator
classes, fragmentation, placement and eviction still constrain Mooncake. Its
advantage here is avoiding the captured KVCR allocation's hard semantic partition.
A central metadata server is not logically required for a shared-arena allocator;
they are separate architectural choices. Conversely, adopting a shared arena
alone would not provide Mooncake's global existence/placement behavior.

### 3.1 Exact partition comparison

| Platform/configuration | Total GiB/worker | Checkpoints | Checkpoint reservation GiB | KV/indexer reservation GiB | KVpages |
|---|---:|---:|---:|---:|---:|
| GB300 default / indexedM2261 |128|2261|124.278|3.700|2261|
| GB300 indexedM2176 |128|2176|119.606|8.394|5129|
| GB300 indexedM1536 |128|1536|84.428|43.572|26625|
| GB300 Mooncake |128|not a fixed configured count|shared byte budget|shared byte budget|workload-dependent |
| B200 indexedM2048, context only |224|2048|112.570|111.430|68090|
| B200 Mooncake, context only |224|not a fixed configured count|shared byte budget|shared byte budget|workload-dependent |

Reservation is not resident usage/RSS. Each checkpoint is about56.285MiB, versus
about1.676MiB per full KV/indexer page. The legacy equal-count allocation therefore
reserves97% of the GB300 budget for checkpoints. Reducing M to1536 frees about40.7GiB
for KV/indexer, expanding its page capacity11.8×. This explains why “more Mamba
capacity should always lower TTFT” is the wrong expectation.

Observed pressure supports a real tradeoff: M2261's small KV pools churn heavily;
M1536's Mamba pools reach their cap during23.6–39.2% of the window across workers.
Reducing M further may lose useful checkpoints. B200 M2048 has2.56× as many KVpages
as GB300 M1536 because of its larger total budget, despite its larger checkpoint
count. Hardware/platform also differ. These three results are not a one-parameter
same-platform sweep.

[Historical metric data](results.csv)

## 4. Client performance: complete measured sequence

All latencies below are milliseconds. “ITL” is official `inter_token_latency`;
we do not substitute `full_response_inter_token_latency` or exclusive GPU decode
time. Source exports preserve all percentiles and official error summaries.

| Test | TTFT mean | p50 | p90 | p99 | ITL mean | Output tok/s | Complete/cancel/error |
|---|---:|---:|---:|---:|---:|---:|---|
| Original KVCR, no pool index |4260.858|2359.869|9628.467|29121.284|82.282|559.683|1488/31/0|
| KVCR pool index, M2261 |1117.536|554.360|2091.300|10042.415|13.337|1626.866|3437/3/0|
| KVCR M2176 |1114.331|501.721|2150.147|10765.209|12.769|1780.551|3599/2/0|
| KVCR M1536 |810.548|448.634|1496.135|6517.362|7.698|1985.749|4013/1/0|
| KVCR first-page-negative |825.270|405.011|1621.360|7434.079|7.838|1976.034|3994/1/0|
| KVCR encoded/incremental confirmation |802.134|374.205|1471.851|7797.815|7.402|1984.976|4024/1/0|
| KVCR + GIL1ms |791.982|390.258|1575.908|7014.984|7.516|2012.987|4021/0/0|
| KVCR + GIL/native poll1ms |710.722|397.165|1332.670|5751.875|7.524|2022.970|4030/0/0|
| Mooncake original R3 |543.465|344.607|838.837|4023.220|6.466|2085.028|4131/0/0|
| Mooncake GIL1ms |587.289|340.879|861.616|5750.635|6.790|2061.374|4122/0/0|
| Mooncake fresh random |557.918|331.210|855.276|4853.818|6.200|2061.042|4125/0/0|
| Mooncake fresh local-first |543.374|340.655|833.262|3991.320|6.474|2074.556|4139/0/0|

[Historical metric data](results.csv)

The first two fixes dominate the observed gain. Pool indexing removes unrelated
victim scanning without changing capacities. M1536 then improves retained useful
KV and lowers re-admission pressure. Later CPU/scheduling changes are separate
hypotheses, with mixed means/tails and no monotonic guarantee.

Relative to the appropriate controls:

| KVCR / Mooncake comparison | Mean TTFT ratio | p90 ratio | p99 ratio | Output throughput fraction |
|---|---:|---:|---:|---:|
| GB300 M1536 / original MoonR3 |1.491|1.784|1.620|95.24%|
| GB300 native-poll1ms / original MoonR3 |1.308|1.589|1.430|97.02%|
| GB300 native-poll1ms / MoonGIL1ms |1.210|1.547|1.0002|98.14%|
| B200 M2048+index / matched B200 Moon |1.511|1.861|2.016|97.76%|

Near-throughput parity and similar p99 to one control do not establish mean/p90
TTFT parity. Keep the original stronger MoonR3 tail result visible. These are
ratios of recorded separate runs, not paired causal effects or statistical bounds.

## 5. Cache hit rates and what they mean

These are pooled server **prefill-token outcome fractions**. HBM is device_hit;
DRAM is host_hit from this direct linker, combining local and remote. The separate
storage-prefetch accounting path contributes0 here. Do not relabel DRAM as an
independent framework host cache or add client cache-read percentages to it.

| Test | HBM % | Store DRAM % | Total cached % | Recompute % |
|---|---:|---:|---:|---:|
| Original KVCR |90.455|0.249|90.703|9.297|
| Pool-indexM2261 |91.235|0.300|91.535|8.465|
| M2176 |90.851|1.223|92.074|7.926|
| M1536 |90.412|3.614|94.026|5.974|
| First-page-negative |90.603|3.234|93.837|6.163|
| Encoded/incremental confirmation |90.789|3.097|93.886|6.114|
| GIL1ms |90.457|3.576|94.033|5.967|
| Native poll1ms |91.281|3.051|94.332|5.668|
| MoonR3 |91.334|3.145|94.479|5.521|
| MoonGIL1ms |90.250|4.060|94.310|5.690|
| Moon fresh random |90.391|4.075|94.467|5.533|
| Moon fresh local-first |90.268|4.175|94.443|5.557|

[Historical metric data](results.csv)

M1536 has a larger DRAM hit fraction than MoonR3, but lower HBM reuse and more
recomputation. The final KVCR arm and MoonGIL1ms have almost equal aggregate cache
fractions, yet different mean/p90TTFT. Aggregate hit rate cannot alone explain
the remaining latency gap; state readiness, lookup/restore overhead and scheduling
still matter. Similar totals also do not guarantee the same request-level hits,
state sizes, remote locality or GPU batching.

## 6. Cache churn, duplication and CPU work

The indexedM2261 arm still generates14.009TB of successful local-fill descriptor
payload and123656operations;93.06% of sampled KV additions follow an observed
same-worker removal. This is temporal re-admission, distinct from concurrently
resident duplicate copies. Ready duplicate excess is about1.20% KV/1.44%indexer.

M2176 reduces local-fill payload to9.607TB and80180operations. M1536 reduces it
further to2.212TB and18171operations; sampled owner CPU falls.861→.322summed cores
and inclusive progress CPU1306.735→352.393s versus M2176. These are supporting
mechanism observations; overlapping CPU/wall sums cannot be added to requestTTFT.

M1536's sampled cross-worker resident duplicate excess rises to about17.8%KV/
17.9%indexer while KV re-admission remains about54%. Those percentages measure
different stock/flow populations. They do not prove17.8% of all physical DRAM is
safely reclaimable, nor that a naive drop-on-PUT rule preserves usable prefixes.

Mooncake globally checks object existence and has shared object allocation,
which are plausible advantages over worker-owned fixed typed pools. Its accepted
PUT payload has a different API/population scope; dividing it into KVCR local-fill
bytes is not an RDMA wire-amplification measurement. A code change that removes
an apparent duplicate must preserve active readers, claims, published ownership
and the complete-state restoration contract.

## 7. TTFT, queueing and scheduling

Conceptually client TTFT includes request transport/router work, worker queueing,
cache lookup/confirmation, dependency/transfer waits, scheduling/batching, prefill,
first-output work and response transport. These components can overlap across
requests and across transfer/model pipelines.

| Recorded observation | KVCR M1536 | First-negative | MoonR3 |
|---|---:|---:|---:|
| Mean clientTTFT ms |810.548|825.270|543.465|
| Server queue mean ms |131.640|148.063|48.764|
| First-forward-to-prefill mean ms |533.281|532.349|403.039|

The server timing boundaries/populations are not identical to all client requests.
In particular, do not sum these rows and label the remainder “RDMA overhead.”
Confirmation can block admission, local layer delivery can delay consumption,
and scheduler/native observation gaps can inflate inclusive lifetimes without
changing actual link speed. The existing measurements do not provide a complete,
exclusive per-request TTFT decomposition for every stage.

Configured48lanes is a client replay limit; trace idle gaps and differing response
speed produce different effective concurrency. For M1536/indexM2261, mean effective
concurrency is about15.44/19.65, despite identical configured48. Faster completion
changes how many turns enter the1800s window. Completed-request CDFs omit the
separately reported deadline cancellations, so tail comparisons retain survivor
and population effects.

## 8. KVCR transfer sizes and bandwidth

The following is the successful source-operation population. Native/source GB/s
is **sum successful payload bytes / sum corresponding observed durations**,
not the arithmetic mean of individual transfer bandwidths. Counts do not include
failed zero-byte outcomes. Keys are typed cache keys, not descriptors/token blocks.

| KVCR stage | Successful operations | Mean bytes | Mean cache keys | Native GB/s | Source-operation GB/s |
|---|---:|---:|---:|---:|---:|
| Baseline |47|301648009|306.77|22.652|6.769|
| Pool-indexM2261 |109|387538649|419.63|24.452|14.323|
| M2176 |313|486091340|531.16|27.117|13.885|
| M1536 |693|584947460|644.29|25.211|15.502|
| First-negative |562|647140026|715.37|25.588|14.502|
| Confirmation |523|670325661|743.39|26.858|13.174|
| GIL1ms |526|760046328|845.27|29.316|14.987|
| Native poll1ms |496|673327357|745.02|29.112|16.385|

The positive payloads are hundreds of MB on average, so a blanket explanation
that every low effective-bandwidth observation is a tiny transfer is unsupported.
The remaining source/native gap is compatible with control/claim/scheduling and
completion-observation lifetime outside native transfer, but these aggregates
cannot exclusively attribute it. A failed source operation may have no eligible
READY pieces and submit no native transfer at all. Success-only native telemetry
cannot prove the absence of RDMA errors.

These captures' count/sum/max interval telemetry cannot reconstruct KVCR
per-transfer p50/p90/p99, size-conditioned bandwidth CDFs or exact descriptor
fragmentation distributions. Those are **unavailable**, not zero. The historical
vLLM AgentX transfer CDFs are another experiment family and must not be pasted
into this GB300 SGLang report.

## 9. Mooncake GET distributions and comparison boundaries

Mooncake operation records provide inclusive range_get API distributions. They
combine local and remote source work and layer-range calls; completed payloads
are not verified remote-only wire bytes. GET object/piece counts are not KVCR
logical pages or NIXL descriptor counts.

| Mooncake case | GET n | Mean bytes | Mean physical objects | Mean pieces | Duration mean/p50/p90/p99 ms | Effective GB/s mean/p50/p90/p99 |
|---|---:|---:|---:|---:|---|---|
| OriginalR3 |5319|70707544|1013.58|1523.09|3.429/2.320/7.839/17.407|14.960/18.484/29.739/39.601|
| GIL1ms |6237|75014679|1077.50|1618.95|3.761/2.341/8.725/18.489|14.535/15.831/29.424/39.276|
| Fresh random |6210|75100430|1078.77|1620.85|3.967/2.594/8.428/20.352|14.049/16.704/28.425/38.099|
| Fresh local-first |5940|80624519|1160.50|1743.45|4.296/2.636/9.934/21.735|13.659/14.052/27.323/37.664|

For the fresh pair, size-conditioned mean GET bandwidth is:

| Completed payload | Random GB/s / n | Local-first GB/s / n |
|---|---|---|
|0–1MiB|1.219 /377|1.133 /520|
|1–16MiB|3.460 /2873|3.840 /2691|
|16–64MiB|27.412 /801|27.217 /726|
|64–256MiB|25.541 /1665|25.391 /1470|
|256MiB+|25.019 /494|24.630 /533|

The bin boundaries are binaryMiB; bandwidth uses decimalGB/s. Locality, object
count and fanout can differ within a size bin. These observations show small
inclusive calls can be overhead-bound and large calls attain higher payload/time;
they do not isolate remote RDMA or explain client TTFT causally.

The full fresh-policy source report and its p50/p90/p99 size/bandwidth tables are
copied as [evidence/reports/mooncake-placement.md](evidence/reports/mooncake-placement.md).
Remote-only bytes, per-source overlap/session fanout and physical per-pod NIC
utilization are not recovered by renaming API bytes. Consequently no valid
KVCR-native-versus-Mooncake-remote CDF speedup is claimed here.

## 10. Placement policy and why local-first is not a universal winner

The fresh local-first arm has2.61% lower meanTTFT,2.57% lowerp90 and17.77% lowerp99
than random. However, meanITL is4.42% higher, inclusive meanGET bandwidth is lower
and output throughput is only0.66% higher. Total cached fractions are94.443%/
94.467%, essentially similar at this aggregate level. Both have zero cancellation/
errors and about90% peak store occupancy.

This illustrates why physical object placement, aggregate cache hit rate, API
bandwidth and client latency must remain separate axes. One replay per policy,
sequential order and unobserved remote fractions do not establish that local-first
is a stable best policy or that its TTFT gain was caused by local reads.

## 11. Source identity, correctness and remaining optimization opportunities

The three capacity/index configurations use identical captured SGLang/KVCR/Dynamo
source content after normalizing Git index abbreviation width. The GB300 M changes
are configuration-only. The original baseline lacks the eviction index. Later
confirmation/GIL/native changes have separate code deltas and full patches.
Mooncake shares the Qwen builder but has distinct storage/session behavior; it
is not an identical implementation with a different cache size.

Promising follow-ups, **not implemented or performance-qualified by this bundle**:

1. **Reduce fixed-pool stranding while preserving typed completeness.** M1536 is
   the measured safe configuration alternative; a dynamic shared arena would
   require stable live addresses, extent reuse after reader/DMA release, revised
   descriptor preparation and complete-checkpoint admission. It need not require
   a central metadata service, but it is more than changing allocation counters.
2. **Measure and reduce confirmation/ownership work per useful READY payload.**
   Keep request/candidate, HBM anchor, source incarnation, typed READY coverage,
   failure reason and receive/claim/submit/DONE/cleanup timestamps joined. Current
   aggregate success bandwidth cannot tell which unsuccessful candidate costs
   dominate. Preserve final claim validation and reader lifetime semantics.
3. **Investigate replica retention and eviction/re-admission together.** M1536's
   resident duplication is materially larger than the tiny-pool baseline, but
   global elimination may harm useful locality/availability. Do not infer safe
   reclaimable capacity from sampled duplicate byte-seconds alone.
4. **Keep control/Python/native scheduling experiments separate.** The tested
   GIL/native delay changes help some metrics; they do not make every delay useful
   or prove the hardware link was slow. Use a matched MoonGIL control and stronger
   original control, plus exclusive request-stage measurements before promotion.

Both model paths require full state and real peer correctness; relaxed cached-
token checks or dropped checkpoint siblings would change the benchmark contract.
Keep any follow-up as a fresh named attempt with its own code/config/raw evidence.

## 12. Reproduction and evidence completeness

The three main source documents are [KVCR](architecture-kvcr.md),
[Mooncake](architecture-mooncake.md) and [manual reproduction](manual-running.md).
`stages.json` maps all twelve measured arms to exact historical run directories,
source heads, settings, metrics and source patches. Each `stages/<id>/recorded/`
contains the official export and executed component argv/environment; copied
provenance includes model cards, router build and runtime-setting proofs.

`evidence/patch-validation.json` verifies134tracked changed-file postimages against
recorded Git blob prefixes after isolated file reconstruction. Native helper
extras and supplemental ignored tests/version metadata are preserved separately.
This is not a claim that an entire new checkout, native environment or new GPU
run was qualified. The manual workflow requires those checks on actual hardware.
No model weights, large trace payloads, native wheels or raw archives are bundled.

The external InferenceX result uses oneGB300,3600s,tenextra warmups/lane and direct
SGLang without this router setup. Its meanTTFT1180.793ms/ITL27.891ms/output1293.89tok/s
is historical context, not a matched speedup denominator. Similarly, B200224GiB,
HiCache and older vLLM Guard results are separate populations.

All documentation, patch reconstruction and comparative plots in this folder
were produced locally. Existing experiments, pods, model files and historical
raw/results were left unchanged. New actual benchmark outputs must remain on
workspace; scientific postprocessing runs locally after verified download.
