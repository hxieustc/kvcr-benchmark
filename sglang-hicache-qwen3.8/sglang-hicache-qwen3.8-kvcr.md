# Reproduce Qwen3.8 SGLang HiCache + KVCR on GB300

These guides reproduce the **three completed `hicache-1800-r5` experiments**:
KVCR, Mooncake random placement, and Mooncake local-first placement. They use
native SGLang HiCache, with no external linker and no vLLM.

The source setup is common across the three arms. The compatibility fixes below
were necessary before measurement; there is **no separately measured HiCache
optimization sequence** to publish. Earlier failed registration/preflight attempts
are not alternative performance baselines. Source branch names all start with
`sglang-hicache`; manifests pin full immutable commit SHAs, not moving branch tips.

## 1. Exact source and package identities

| Project | Published commit | Feature branch / purpose |
|---|---|---|
| sglang | `081c4c5da234deb73078acfc09174286d5b3af05` | `sglang-hicache-qwen3.8-complete-checkpoints-budget-router-keys` |
| kvcr | `5ce23b1527725cf623a3fc7feb94dc6d3d5fafaa` | `sglang-hicache-qwen3.8-clean-native-base` |
| dynamo | `f28d277edaf403ba5cae65bece687f7b29c0eba3` | `sglang-hicache-qwen3.8-cache-key-handshake` |
| aiperf | `e4bb2f835ffedbbe6b76252ea5c93caec68c5faf` | `agentx-v1.0.12` |

Original SGLang base: `658755604ace9f476b84f98511f76b815606072a`, HiCache PR #45647.
Original Dynamo base: PR #17655, `3f0ef62c22ce2e212fd7a6394ef98d84f5f9a23b`.
The published SGLang/Dynamo commits reconstruct the exact recorded modified file
contents. KVCR is the **unchanged** recorded `5ce23b15…` checkout; its new branch
is a baseline reference, not a performance patch. All Mooncake arms reuse these
same source pins. There is no Mooncake C++/Transfer Engine fork.

AgentX 1.0.12 provides the `aiperf` package/export schema. The manifest's
`software/aiperf` is `SemiAnalysisAI/agentx-harness` at its release tag commit;
runtime installation uses the exact SHA256-locked released wheel in
[client-package.lock.json](client-package.lock.json), followed by the historical
trace-revision fixture. Do not substitute current upstream NVIDIA AIPerf.

Native runtime pins include Torch 2.14.1+cu130, Transformers 5.19.0, NIXL 1.5.0,
SGLang-kernel 0.4.10, FlashInfer 0.7.2, CuTe 4.8.0, Triton 3.8.0,
Mooncake transfer engine 0.3.13 and Dynamo runtime 1.7.0 built from the pinned
PR source. Full effective package pins are in
[reproduction/bootstrap/original-effective-packages.json](reproduction/bootstrap/original-effective-packages.json).
The raw deployment copies can retain old inherited `pins` fields; the actual
source receipts and this source lock are authoritative for serving source.

## 2. Required compatibility patches, not new measured optimizations

The [full SGLang patch](reference/hicache-integration.patch) changes six files:

1. `arg_groups/fields/memory.py`: accept fractional `hicache_size` so the exact
   20/80 byte budget can be represented, rather than rounding to integer GB.
2. `pool_host/base.py` and `pool_host/mamba.py`: floor explicit budgets to complete
   pages instead of adding an extra page that can overrun the assigned bytes.
3. `pool_host/mamba.py`: `get_storage_page_buffer_meta()` exposes every registered
   complete checkpoint component, including PLE/slot state with its own dtype,
   size and stride. It rejects incomplete/noncontiguous unsupported layouts.
4. `storage/kvcr/host_layout.py`: use the complete storage metadata accessor when
   describing indexed regions, and implement `encode_to_router()` to convert
   storage hashes to the router's unsigned hash representation.
5. `storage/mooncake_store/mooncake_store.py`: generate object suffixes for all
   nonempty temporal/conv/slot-state components in matching descriptor order,
   and use the complete metadata accessor for GET/PUT buffers.
6. `managers/scheduler.py`: expose actual `cache_key_is_eagle` to Dynamo. Mamba's
   cache-key mode must not be inferred solely from the NEXTN decode algorithm.

The [Dynamo Python registration patch](reference/dynamo-register-baseline.patch)
consumes that actual scheduler mode for `enable_eagle` and Mooncake metadata.
Its native router code remains the same pinned PR17655 build; `kv_hints.py` and
`state_agent.py` match their original pinned files. The Python handshake source
is identical to the already published linker handshake commit, so its new
HiCache branch points to that same immutable commit.

KVCR has **no local source patch**. Its recorded native progress defaults are
`num_threads=4`, listener enabled and telemetry capture enabled. The linker
experiments' `num_threads=0`, pool-index/capacity tuning, GIL interval or native
poll-delay overlays are not added to this reproduction.

### Runtime instrumentation is part of the measurement setup

[observer/](observer/) contains the recorded `sitecustomize`/import wrappers and
bounded per-process sink. Enable them with `PYTHONPATH` and
`HICACHE_BENCH_OUTPUT` as the renderer does. They observe backend APIs, native
transfers/source lifetimes and actual physical cache geometry. The observer
checks module hashes against its manifest, never reads payload contents or
issues metadata queries just to count hits. It introduces logging/getter work;
a clean Git checkout does not mean an uninstrumented runtime. Removing it is
a different experiment. Lossless coverage was not established for all historical
observer data; retain dropped/error/terminal evidence in new runs too.

## 3. Hardware, model, trace and cache settings

Four one-GPU GB300 workers on one ARM64 node, each TP/DP/EP/PP/CP=1, one assigned
400 Gb/s RoCE port. The historical pod requested 4 GPUs, 40 CPUs (48 maximum)
and 850 GiB RAM. Its [pod/claim receipts](reference/) are allocation examples,
not instructions to reuse a stale node selector, GPU UUID or RDMA claim. Use an
eligible new/idle GB300 allocation through `tsh kubectl`; retain the pod.

Image:
```
lmsysorg/sglang:v0.5.21-cu130-runtime@sha256:417fc90dfa9392d0e62f3c0a586a297b6d75082eccc90e63905832ac58ae85ff
```

Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`, 132680249378 weight bytes.
Use an existing verified read-only snapshot; file/checksum receipts are included.
Trace: `semianalysisai/cc-traces-weka-062126-256k`, revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`, 568864747-byte `traces.jsonl`.
The dataset loader is `semianalysis_cc_traces_weka_062126_256k`.

| Per-worker cache setting | Value |
|---|---:|
| Combined host + backend budget | 137438953472 B = 128 GiB |
| Framework HiCache host request, 20% | 27487790694 B |
| `--hicache-size` (decimal GB) | 27.487790694 |
| Backend-owned DRAM request, 80% | 109951162778 B |
| Historical actual host allocation after complete-page rounding | 27389719952 B |
| Historical actual KVCR backend region allocation | 109934192208 B |
| Historical Moon global segment per worker | 109951162778 B |
| PLE model host storage, separate from cache | requested 64 GiB |

Actual host pools per worker: KV 12598 slots ×1703936 B; indexer 12598 slots
×53248 B; complete Mamba 89 slots ×59019280 B. KV/indexer include 12 target
full-attention layers plus the packed draft layer. Mamba includes 36 target
linear-attention layers; there is no draft-Mamba requirement for this model.
Check actual allocations on replacement hardware instead of assuming these
counts. Backend region capacity/reservation is not READY occupancy or RSS.

Native HiCache flags are:
```
--enable-hierarchical-cache
--hicache-host-memory-mode cache
--hicache-size 27.487790694
--hicache-write-policy write_through
--hicache-storage-prefetch-policy wait_complete
--hicache-mem-layout page_first_direct
--hicache-io-backend direct
--hicache-storage-backend kvcr   # or mooncake
```

All external-linker flags are absent. Both HiCache storage backends GET into
framework **host DRAM**, followed by the framework host→HBM path. Mooncake's
direct-to-GPU *linker* route is not the route being reproduced here.

Shared worker settings: context262144, page64, max-running96, prefill/chunk16384,
memory fraction0.88, BF16 KV/Mamba, GPU Mamba max480, tracking256, target/draft
same model revision, trtllm_mha, full decode graph, flashinfer linear attention.
NEXTN uses 3 steps/top-k1/4 draft tokens with synthetic acceptance2.32,
match-expected/real-draft-token; this is a serving control, not quality evaluation.

Router: `dynamo-two-tier-cost-fn`, cache threshold0.5, absolute balance4, relative
balance1.1, TCP requests, ZMQ KV events, file discovery, four state-agent slots.
Every arm uses the same router source and native binaries. Historical wheel
SHA256s are `51e08fde9d7e5dbabccb9a1e2e49d3bd133cd67c37b98deee00ab7a1054403ed`
for ai_dynamo and `4abfe02c7e66b471d709b75b2cc197229a1969eb5bdf1b5e18af07cba91a75a3`
for ARM runtime. A source rebuild can have different wheel bytes; use its same
recorded build in both backends, and do not claim historical binary-byte identity.

Single-RNIC env: `NIXL_UCX_SGL_ENABLE=1`, `UCX_TLS=rc,self,cuda_copy`,
`UCX_NET_DEVICES=ASSIGNED_RNIC:1`, `UCX_MAX_RNDV_RAILS=1`,
`UCX_MAX_EAGER_RAILS=1`, `UCX_MAX_RMA_RAILS=1`, `UCX_MEMTYPE_CACHE=n`.
Mooncake also requires `WITH_NVIDIA_PEERMEM=0` for the tested DMA-BUF path;
do not force TCP. The recorded HCA name `rocep190s0` is not portable.

AgentX: 48 total closed-loop lanes, seed42, five extra warmup requests/lane plus
mandatory primers (286 observed warmups), 1800-second sending and 30-second
grace. Trajectory ratios0.25–0.75, idle caps300s/system10s. Use the renderer's
complete captured argv/environment, not a shortened alternative command.

## 4. Get the pinned workspace using Google Repo

Install Google's Repo launcher, Git and Git LFS. In a fresh workspace directory:

```bash
mkdir -p /workspace/qwen38-hicache-code
cd /workspace/qwen38-hicache-code
repo init -u https://github.com/hxieustc/kvcr-manifests.git \
  -b develop -m manifests/sglang-hicache-qwen3.8-kvcr.xml
repo sync -j4 --no-clone-bundle --no-tags
repo manifest -r > resolved-manifest.xml
export CODE=$PWD
export B="$CODE/kvcr-benchmark/sglang-hicache-qwen3.8"
export CASE=kvcr
python3 "$B/scripts/verify-sources.py" --code "$CODE"
```

The synced layout is `software/{sglang,kvcr,dynamo,aiperf}`, `kvcr-benchmark/`
and `manifests/`. Software and benchmark projects are SHA-pinned. The manifest
repository's convenience project follows `develop` to avoid a self-referential
commit; save `repo manifest -r` with every experiment. Repo sync does not allocate
a pod, install native packages, download model weights or start a benchmark.
Never sync or switch the source tree of an actively running experiment.

Follow [manual-running.md](manual-running.md) for both a Repo-based checkout and
the manual Git-clone alternative, dependency installation, individual component
startup, real peer qualification, cache reset, client/capture and collection.
All archived commands are in `recorded/<case>/run/`; configs/scripts in
`recorded/<case>/config/` and `scripts/` are historical evidence. The portable
renderer is the supported path-rebasing interface; do not execute an archived
global controller or its old success/stop receipts against another campaign.

## 5. KVCR-specific setup

The native `KVCRStore` runs in process, with `local_dram_bytes=109951162778`,
`enable_remote_hint=true`, `publish_inventory=true`, reachable
`control_advertise_host`, control port24100+32×rank and NIXL port24500+32×rank.
There is no separate vLLM Guard or external linker service. Host buffer descriptors
come from the patched complete-state accessor; KVCR core source is unchanged.

Use manifest `sglang-hicache-qwen3.8-kvcr.xml` and case `kvcr`.
No extra optimized KVCR HiCache manifests are invented: only the completed
baseline fixture was measured. The previous linker pool/GIL/native-delay
optimizations belong to another campaign and must not be layered onto this one.

## 6. Historical results and metric boundaries

| Completed r5 arm | TTFT mean / p50 / p90 / p99 ms | ITL mean ms | Output tok/s |
|---|---|---:|---:|
| KVCR | 850.890 / 358.952 / 1760.283 / 8420.250 | 6.778 | 2030.728 |
| Mooncake random | 764.742 / 341.286 / 1575.787 / 6913.960 | 6.735 | 2034.157 |
| Mooncake local-first | 819.767 / 339.594 / 1740.064 / 8355.629 | 6.775 | 2042.795 |

Each had zero recorded profiling errors/cancellations and286 exported warmups.
These are recorded separate replays, not expected exact outputs or causal
allocation-policy gains. ITL is not GPU decode time. Source/config equivalence
does not reproduce identical arrival, RNG, cache or batch history.

For these native HiCache tests, server tiers mean **HBM / persistent framework
host cache / backend storage-prefetch / recompute**. Old linker `host=backend`
labels do not apply. API bytes are logical completion, not necessarily actual
copied bytes. NIXL/native/source operations and Moon API calls have distinct
timer/population boundaries. Preserve primitive raw rows, failed/partial results,
observer loss flags and cache geometry; analyze downloaded closed data locally.

[Historical report](results/historical-report.md) and
[full historical summary](results/historical-summary.json) retain the measurements.
No new benchmark was run to publish these manifests. Native/GPU qualification
and capture validation must be repeated on the selected allocation.
