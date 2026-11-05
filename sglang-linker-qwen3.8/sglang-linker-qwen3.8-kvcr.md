# Reproduce the GB300 Qwen3.8 KVCR linker experiments

The manifests pin reconstructed commits containing the exact recorded source postimages, not just the old clean upstream heads. Feature branches are for review; `revision` values are immutable commit SHAs.

The shared compatibility patches were already present in both baselines. Capacity and placement policies are config variants, so they reuse the same source commits. The pool-index algorithm changes KVCR, not SGLang; the later lookup/GIL/native polling optimizations change SGLang.

## Select the exact experiment

| Stage | Manifest | SGLang commit | KVCR commit |
|---|---|---|---|
| `kvcr-baseline` | [`sglang-linker-qwen3.8-kvcr.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr.xml) | `76b29b634082` | `3b01cb20b453` |
| `kvcr-pool-index-m2261` | [`sglang-linker-qwen3.8-kvcr-pool-index-m2261.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr-pool-index-m2261.xml) | `76b29b634082` | `013955b73526` |
| `kvcr-m2176` | [`sglang-linker-qwen3.8-kvcr-m2176.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr-m2176.xml) | `76b29b634082` | `013955b73526` |
| `kvcr-m1536` | [`sglang-linker-qwen3.8-kvcr-m1536.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr-m1536.xml) | `76b29b634082` | `013955b73526` |
| `kvcr-first-page-negative` | [`sglang-linker-qwen3.8-kvcr-first-page-negative.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr-first-page-negative.xml) | `bb6795e14ec9` | `013955b73526` |
| `kvcr-incremental-confirmation` | [`sglang-linker-qwen3.8-kvcr-incremental-confirmation.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr-incremental-confirmation.xml) | `115a0fb18396` | `013955b73526` |
| `kvcr-gil-1ms` | [`sglang-linker-qwen3.8-kvcr-gil-1ms.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr-gil-1ms.xml) | `8cfe084a22a6` | `013955b73526` |
| `kvcr-native-poll-1ms` | [`sglang-linker-qwen3.8-kvcr-native-poll-1ms.xml`](https://github.com/hxieustc/kvcr-manifests/blob/develop/manifests/sglang-linker-qwen3.8-kvcr-native-poll-1ms.xml) | `091b400f267e` | `013955b73526` |

## Synchronize using Google Repo

Install Google's Repo launcher from its official distribution and Git/Git LFS.
Public HTTPS remotes in these manifests need no GitHub write credential. In an empty
directory on the workspace PVC:

```bash
mkdir -p /workspace/qwen38-linker-code
cd /workspace/qwen38-linker-code
repo init -u https://github.com/hxieustc/kvcr-manifests.git \
  -b develop -m manifests/sglang-linker-qwen3.8-kvcr.xml
repo sync -j4 --no-clone-bundle --no-tags
repo manifest -r > resolved-manifest.xml
export CODE=$PWD
export B="$CODE/kvcr-benchmark/sglang-linker-qwen3.8"
export STAGE=kvcr-baseline
python3 "$B/scripts/verify-sources.py" --code "$CODE" --stage "$STAGE"
```

To select another row, use its manifest at `repo init -m` in a separate new code
directory and set its stage ID. Each manifest also pins this benchmark repository,
so settings, scripts, dependency locks and source all move together. The manifest
repository's own project follows `develop` to avoid a self-referential SHA; save
`resolved-manifest.xml` with each run. Its source/benchmark projects are SHA-pinned.
Never `repo sync` an active serving checkout. Review one optimization's small
`patches/<stage>/*-from-parent.patch`, and inspect the named feature branches in
[source-publication.json](source-publication.json).

The synced layout is `software/{sglang,kvcr,dynamo,aiperf}`, `kvcr-benchmark/`,
and `manifests/`. Repo synchronizes source only; it does not download model
weights, install Python packages, allocate GPUs, or start services.

## Hardware, model, transport and workload

The historical platform was one ARM64 GB300 node, four GPUs, four independent
workers (TP=DP=EP=PP=CP=1), one allocated 400 Gb/s RoCE port, and the image
`lmsysorg/sglang:v0.5.21-cu130-runtime@sha256:417fc90dfa9392d0e62f3c0a586a297b6d75082eccc90e63905832ac58ae85ff`.
The recorded allocation requested 40 CPUs (limit 48), 850 GiB RAM and 4 GPUs.
Use a fresh eligible allocation; historical node selectors and RDMA claim IDs are receipts.
Keep pods and model weights; outputs belong on the `kv-p2p-workspace` PVC at `/workspace`.

Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47` (132,680,249,378 weight bytes).
Trace: `semianalysisai/cc-traces-weka-062126-256k`, revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`, `traces.jsonl` 568,864,747 bytes.
The client is AgentX 1.0.12, whose import package/export schema is `aiperf`;
current NVIDIA AIPerf is not a drop-in replacement. The source release and exact
wheel SHA256 are both locked in [client-package.lock.json](client-package.lock.json).
The trace revision fixture is the sole historical client loader overlay.

Use 48 total closed-loop lanes, seed 42, five extra warmup requests per lane,
mandatory snapshot primers (286 warmup requests observed: 240 extras + 46 primers),
1,800 seconds of profiling and 30 seconds grace. Dataset trajectory ratios are
0.25–0.75; idle caps are 300 seconds / 10 seconds for system gaps.
Warmup is count based, not half an elapsed-time budget.

SGLang uses 64-token pages, BF16 KV and Mamba state, context 262,144,
max-running-requests 96, max-prefill-tokens 16,384, mem-fraction-static 0.88,
GPU Mamba-state cache size 480 and tracking interval 256. NEXTN uses three steps,
top-k 1, four draft tokens, and the CI synthetic acceptance setting 2.32 with
`match-expected`/real-draft-token behavior. This evaluates serving behavior, not model quality.
Each worker has 128 GiB backend-owned DRAM plus separate 64 GiB PLE model host storage.
No framework HiCache host controller is enabled in these **linker** experiments.

Dynamo uses PR #17655's recorded source plus the same scheduler cache-key handshake
for both backends. The policy is `dynamo-two-tier-cost-fn`: cache threshold 0.5,
absolute balance threshold 4 and relative threshold 1.1, with TCP request plane,
ZMQ KV events, file discovery and four state-agent slots. There is no vLLM component.

For the actual assigned RNIC: `UCX_TLS=rc,self,cuda_copy`, `UCX_NET_DEVICES=RNIC:1`,
`UCX_MAX_RNDV_RAILS=1`, `UCX_MAX_EAGER_RAILS=1`, `UCX_MAX_RMA_RAILS=1`,
`UCX_MEMTYPE_CACHE=n`, `NIXL_UCX_SGL_ENABLE=1`. The historical device name
`rocep172s0:1` is not portable; use the currently allocated device.

Runtime locks and wheel receipts are in [reproduction/bootstrap/](reproduction/bootstrap/).
Key versions: Torch 2.14.1+cu130, Transformers 5.19.0, NIXL 1.5.0,
SGLang-kernel 0.4.10, FlashInfer 0.7.2, CuTe 4.8.0, Triton 3.8.0,
Mooncake 0.3.13 and Dynamo 1.7.0 built from the pinned source.

## Reproduction procedure

Follow [manual-running.md](manual-running.md) after source synchronization. It
contains explicit build/install steps, individual master/state-agent/frontend/
worker/client commands, qualification and cache clearing, raw collection and local
postprocessing. [sources.lock.json](sources.lock.json) maps every stage to all
source SHAs, original upstream bases, historical run IDs and exact settings.
`stages/<stage>/recorded/` contains historical evidence, never fresh success receipts.
The original baseline reused already-loaded preflight services; fresh startup
reproduces its settings and code, not that exact process/cache history.

All stage configs and captured argv/environment are retained. The renderer rebases
operational addresses/paths/namespaces while preserving scientific parameters.
No benchmark was launched to publish these manifests. Source/config verification
cannot establish that a new replay's performance will equal a historical quantile.

## Recorded results for reproduction checks

These are archived results, not new measurements from this publication. ITL is the reported inter-token latency metric; do not relabel it as a differently defined TPOT estimator.

| Stage | TTFT mean / p90 / p99 (ms) | Mean ITL (ms) | Output tokens/s |
|---|---:|---:|---:|
| `kvcr-baseline` | 4260.858 / 9628.467 / 29121.284 | 82.282 | 559.683 |
| `kvcr-pool-index-m2261` | 1117.536 / 2091.300 / 10042.415 | 13.337 | 1626.866 |
| `kvcr-m2176` | 1114.331 / 2150.147 / 10765.209 | 12.769 | 1780.551 |
| `kvcr-m1536` | 810.548 / 1496.135 / 6517.362 | 7.698 | 1985.749 |
| `kvcr-first-page-negative` | 825.270 / 1621.360 / 7434.079 | 7.838 | 1976.034 |
| `kvcr-incremental-confirmation` | 802.134 / 1471.851 / 7797.815 | 7.402 | 1984.976 |
| `kvcr-gil-1ms` | 791.982 / 1575.908 / 7014.984 | 7.516 | 2012.987 |
| `kvcr-native-poll-1ms` | 710.722 / 1332.670 / 5751.875 | 7.524 | 2022.970 |

## Implementation and interpretation

[KVCR implementation, memory layout and optimization details](architecture-kvcr.md); [comparative evidence and metric boundaries](comparison.md).
