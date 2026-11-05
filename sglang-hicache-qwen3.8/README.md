# Qwen3.8 native HiCache on GB300

Three completed `hicache-1800-r5` arms: KVCR, Mooncake random placement and
Mooncake local-first placement. The package preserves shared source compatibility
fixes, clean KVCR core, runtime observers and the exact 20/80 host/backend budget.

| Document | Purpose |
|---|---|
| [KVCR Google Repo workflow](sglang-hicache-qwen3.8-kvcr.md) | Immutable manifests, source/package pins and matched KVCR setup. |
| [Mooncake Google Repo workflow](sglang-hicache-qwen3.8-mooncake.md) | Immutable manifests and random/local-first policy selection. |
| [KVCR manual-running guide](manual-running-kvcr.md) | Git-only checkout, installation, qualification, individual services, replay and collection. |
| [Mooncake manual-running guide](manual-running-mooncake.md) | Complete manual procedure for either Mooncake allocation policy. |
| [KVCR versus Mooncake comparison](comparison.md) | Setup and memory geometry, cache tiers, latency/throughput, transfer distributions, bottlenecks and limitations. |

[Sources](sources.lock.json), [runtime client lock](client-package.lock.json),
[historical report](results/historical-report.md) and
[full historical summary](results/historical-summary.json) accompany the guides.
The [former shared manual](manual-running.md) is a compatibility index.

The manuals use fresh workspace roots and qualified allocations, retain pods
and historical results, and download closed data for local analysis. This is a
documentation publication; it does not launch or claim a new benchmark.
