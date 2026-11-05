# Mooncake random versus local-first placement

Two fresh sequential runs on the same four-GB300 pod and single allocated RoCE RNIC. The second frozen analysis verified identical pod/GPU UUIDs/HCA, model, trace and router. Actual master PID/start/argv policy proofs were captured before and after validation cache clear. Both stores/services were freshly started. Only master allocation strategy differs as an experimental setting.

Setup: original MoonR3/default Python GIL source; Mooncake 0.3.13, replicas=1, 128 GiB store per worker (512 GiB total), requested PLE 64 GiB per worker (model data), independent framework host-cache controller disabled. Qwen3.8-Flash-Next-NVFP4; four independent TP1/DP1/EP1 workers. Dynamo PR17655 head `3f0ef62c22ce2e212fd7a6394ef98d84f5f9a23b`. AgentX pinned dataset revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, seed42, 48 lanes, five extra warmups/lane plus46 primers, 1800-second sending plus normal drain. `WITH_NVIDIA_PEERMEM=0`; DMA-BUF/RDMA qualified. NEXTN synthetic acceptance2.32 is a performance control, not a model-quality evaluation. Full configs, source/hardware proofs, manifests and reproduction scripts are in the sibling task directory and each raw capture.

| Official metric | Random | Local-first |
|---|---:|---:|
| time_to_first_token/avg (ms) | 557.9179 | 543.3736 |
| time_to_first_token/p50 (ms) | 331.2100 | 340.6545 |
| time_to_first_token/p90 (ms) | 855.2760 | 833.2623 |
| time_to_first_token/p99 (ms) | 4,853.8184 | 3,991.3202 |
| inter_token_latency/avg (ms) | 6.2000 | 6.4739 |
| inter_token_latency/p50 (ms) | 5.6655 | 5.6349 |
| inter_token_latency/p90 (ms) | 8.5639 | 8.8443 |
| inter_token_latency/p99 (ms) | 16.7469 | 17.1928 |
| request_latency/avg (ms) | 5,880.3633 | 5,861.7148 |
| request_latency/p50 (ms) | 2,539.3845 | 2,585.1063 |
| request_latency/p90 (ms) | 13,210.3130 | 13,020.1432 |
| request_latency/p99 (ms) | 44,751.5359 | 45,451.6349 |
| output_token_throughput | 2,061.0421 | 2,074.5559 |
| request_throughput | 2.2541 | 2.2617 |

| Phase outcome / cache tier | Random | Local-first |
|---|---:|---:|
| Completed | 4125 | 4139 |
| Cancelled | 0 | 0 |
| Errors | 0 | 0 |
| HBM hit % | 90.3913 | 90.2684 |
| Store DRAM hit % | 4.0754 | 4.1748 |
| Other storage hit % | 0.0000 | 0.0000 |
| Recompute % | 5.5332 | 5.5568 |
| Total cached % | 94.4668 | 94.4432 |

Local-first relative to random (descriptive, one replay each):
- time_to_first_token/avg: -2.61%.
- time_to_first_token/p90: -2.57%.
- time_to_first_token/p99: -17.77%.
- inter_token_latency/avg: +4.42%.
- output_token_throughput/avg: +0.66%.

Server fractions are token-weighted effective-prefill outcomes; store DRAM combines local and remote. ITL is the official per-request inter-token-latency metric. Client cache-read and server-tier populations differ. Latency CDFs use available successful request rows; cancelled/omitted records are not imputed.

![Client latency CDFs](client-latency-cdfs.png)

## API transfer distributions

These are successful complete Python/native API calls combining local and remote work. Completed bytes are not remote-only wire bytes; objects/pieces are not logical token blocks or NIXL descriptors. PUT may accept existing objects without transferring all accepted bytes. Native counter byte/summed-time ratios and arithmetic means of individual API bandwidth are distinct populations/estimands.

| Policy | API | Metric | n | Mean | p50 | p90 | p99 |
|---|---|---|---:|---:|---:|---:|---:|
| random | range_get | completed_bytes | 6210 | 75,100,429.7043 | 14,139,392.0000 | 223,215,616.0000 | 465,698,816.0000 |
| random | range_get | objects | 6210 | 1,078.7692 | 763.0000 | 2,534.0000 | 3,713.0000 |
| random | range_get | pieces | 6210 | 1,620.8507 | 1,108.0000 | 3,549.0000 | 7,106.0000 |
| random | range_get | duration_ms | 6210 | 3.9674 | 2.5945 | 8.4276 | 20.3523 |
| random | range_get | gbps | 6210 | 14.0485 | 16.7039 | 28.4251 | 38.0985 |
| random | batch_put | completed_bytes | 7183 | 145,508,302.1378 | 94,162,960.0000 | 354,226,192.0000 | 508,858,384.0000 |
| random | batch_put | objects | 7183 | 99.4405 | 41.0000 | 337.0000 | 513.0000 |
| random | batch_put | pieces | 7183 | 1,993.5894 | 854.0000 | 6,626.0000 | 10,058.0000 |
| random | batch_put | duration_ms | 7183 | 7.1706 | 3.7671 | 16.8246 | 36.6249 |
| random | batch_put | gbps | 7183 | 24.3571 | 24.5868 | 31.0718 | 33.9031 |
| local_first | range_get | completed_bytes | 5940 | 80,624,518.5428 | 13,733,888.0000 | 252,575,744.0000 | 493,748,224.0000 |
| local_first | range_get | objects | 5940 | 1,160.4995 | 895.0000 | 2,995.0000 | 3,875.0000 |
| local_first | range_get | pieces | 5940 | 1,743.4466 | 1,260.0000 | 3,926.0000 | 7,534.0000 |
| local_first | range_get | duration_ms | 5940 | 4.2956 | 2.6362 | 9.9342 | 21.7354 |
| local_first | range_get | gbps | 5940 | 13.6590 | 14.0519 | 27.3232 | 37.6637 |
| local_first | batch_put | completed_bytes | 7273 | 145,378,340.4152 | 92,405,776.0000 | 352,117,571.2000 | 508,858,384.0000 |
| local_first | batch_put | objects | 7273 | 99.2964 | 39.0000 | 334.6000 | 513.0000 |
| local_first | batch_put | pieces | 7273 | 1,990.7537 | 815.0000 | 6,579.2000 | 10,058.0000 |
| local_first | batch_put | duration_ms | 7273 | 7.6009 | 4.0110 | 18.9068 | 37.6934 |
| local_first | batch_put | gbps | 7273 | 23.0323 | 24.0295 | 27.7518 | 30.6197 |

![range_get API CDFs](range_get-cdfs.png)

![batch_put API CDFs](batch_put-cdfs.png)

## API bandwidth conditioned on completed payload size

Bins use MiB=2^20 bytes. Bandwidth is completed bytes / inclusive API duration, in decimal GB/s. The same size bin does not guarantee identical object count, cache locality or source fanout.

| Policy | API | Payload bin | n | Mean GB/s | p50 | p90 | p99 |
|---|---|---|---:|---:|---:|---:|---:|
| random | range_get | [0, 1) MiB | 377 | 1.2190 | 1.2254 | 1.8992 | 2.6608 |
| random | range_get | [1, 16) MiB | 2873 | 3.4597 | 2.0053 | 2.7691 | 22.4349 |
| random | range_get | [16, 64) MiB | 801 | 27.4118 | 25.3431 | 37.6328 | 40.9617 |
| random | range_get | [64, 256) MiB | 1665 | 25.5411 | 24.8907 | 29.2743 | 30.0872 |
| random | range_get | [256, inf) MiB | 494 | 25.0191 | 24.1781 | 29.5768 | 30.0861 |
| random | batch_put | [0, 1) MiB | 0 | unavailable | unavailable | unavailable | unavailable |
| random | batch_put | [1, 16) MiB | 0 | unavailable | unavailable | unavailable | unavailable |
| random | batch_put | [16, 64) MiB | 1445 | 27.6425 | 27.5783 | 31.2018 | 34.0206 |
| random | batch_put | [64, 256) MiB | 4851 | 24.4479 | 23.6825 | 31.1966 | 33.9885 |
| random | batch_put | [256, inf) MiB | 887 | 18.5086 | 15.3472 | 30.2566 | 31.6951 |
| local_first | range_get | [0, 1) MiB | 520 | 1.1330 | 1.0749 | 1.7449 | 3.8324 |
| local_first | range_get | [1, 16) MiB | 2691 | 3.8403 | 2.0541 | 12.9374 | 22.6055 |
| local_first | range_get | [16, 64) MiB | 726 | 27.2165 | 24.6243 | 37.4019 | 39.5856 |
| local_first | range_get | [64, 256) MiB | 1470 | 25.3907 | 25.1499 | 27.8853 | 28.8696 |
| local_first | range_get | [256, inf) MiB | 533 | 24.6296 | 24.2559 | 27.6047 | 28.3200 |
| local_first | batch_put | [0, 1) MiB | 0 | unavailable | unavailable | unavailable | unavailable |
| local_first | batch_put | [1, 16) MiB | 0 | unavailable | unavailable | unavailable | unavailable |
| local_first | batch_put | [16, 64) MiB | 1469 | 26.6955 | 27.0692 | 28.9921 | 30.9285 |
| local_first | batch_put | [64, 256) MiB | 4904 | 23.0676 | 23.4716 | 26.8374 | 30.8348 |
| local_first | batch_put | [256, inf) MiB | 900 | 16.8605 | 14.7523 | 25.4747 | 28.6820 |

## Store utilization and native API context

| Metric | Random | Local-first |
|---|---:|---:|
| Peak store occupancy % | 90.0045 | 89.9980 |
| Peak container memory GiB | 830.4410 | 847.3063 |

| Policy | Native operation | n | Bytes | Mean ms | Bytes / summed seconds, GB/s |
|---|---|---:|---:|---:|---:|
| random | write:batch_put_from_multi_buffers | 7,235.0000 | 1,055,825,085,488.0000 | 6.8465 | 21.3149 |
| local_first | write:batch_put_from_multi_buffers | 7,300.0000 | 1,063,859,091,520.0000 | 7.2643 | 20.0617 |

Native values combine local and remote API work. They do not supply remote-only GET CDFs or physical wire bandwidth. Per-policy worker-load and store-occupancy plots remain in each `work/analysis` report.

## Scope and reproducibility

One run per policy, sequential order, and differing closed-loop completion populations do not establish a stable causal speedup. Allocation policy is verified; per-object placement and remote-only GET fractions were not instrumented. API bandwidth CDFs cannot prove physical RNIC utilization. The completed historical MoonR3 run is context on another allocation and is not substituted as the matched local-first control. Native master/client endpoint audits passed independently for both runs. See per-policy analysis for worker load, occupancy, native aggregate counters, API exclusions, raw/member hashes and captured sources.

Reproduce LOCAL analysis with the frozen `sources/gb300/mooncake-placement-r1/analyze.py`, then `audit-native-endpoints.py` for each mirror, then this `compare-policies.py`. For local-first analysis, supply `--reference-mirror` pointing to the completed random mirror. All inputs and this report script are hash-bound in `comparison-inputs.json`. Preserve the pod and canonical raw archives.

## Post-capture bookkeeping recovery (local_first)

The replay and all four capture exits were zero, and services had closed normally. The original controller then failed because its mmap-input preservation helper recognized a full path but AIPerf logged only a cache-HIT key. The pinned client source/cache resolver and the exact cache bytes were verified against the random run, the original helper copying/hash checks were reused with only path discovery resolved, and the unchanged frozen finalizer revalidated closure and all packed members. No inference was rerun and no serving configuration changed. The original failed controller state/logs and every pre-existing replay-file hash remain preserved. See `sources/postcapture-cache-hit-recovery/` in this case; the report generator independently rechecks its recorded original raw hashes. This was a collection-helper failure, not a failed replay.
