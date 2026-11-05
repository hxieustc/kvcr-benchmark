# Manual reproduction of the GB300 linker experiments

This is the common operating procedure for [KVCR](sglang-linker-qwen3.8-kvcr.md) and
[Mooncake](sglang-linker-qwen3.8-mooncake.md). It starts individual components explicitly. The
bundle itself does not allocate a pod, start a service or rerun a benchmark.

## 1. What constitutes a reproduction

Use a new named run, the pinned model/trace/source versions, the selected stage's
published source commits and settings, real four-worker peer validation, the original
count-based warmup, 1800 seconds of sending and normal drain. Preserve raw data,
commands and actual allocation. A fresh replay will not reproduce every request
arrival, cache history or quantile exactly: the client is closed-loop and the
synthetic real-draft acceptance consumes worker RNG state.

The historical baseline also reused already-loaded services from its preceding
preflight attempt; the retained `service-pids.json` records an r1 discovery path
while the scored export is r2. This bundle preserves that fact. The manual
procedure below uses fresh services and discloses the operational difference.
Do not describe it as a byte-identical restoration of that process history.

The completed source stack is SGLang `5b65e088969b872b924355d20a329e9816a1b094`
plus the archived Qwen/instrumentation patches, KVCR
`3d8c9b831e8fde161d510ac233550314724880e9` plus each stage's patch, and Dynamo
PR17655 `3f0ef62c22ce2e212fd7a6394ef98d84f5f9a23b` plus the cache-mode Python
handshake patch. **These are not clean SGLang PR45650/current KVCR main.** That
later clean-source repeat is a different experiment and is excluded from this historical set.

## 2. Allocate or select the correct hardware

The recorded platform is four GB300 GPUs on one aarch64 node, one independent
TP1 worker/GPU, one assigned 400Gb/s RoCE port. The original image is:

```
lmsysorg/sglang:v0.5.21-cu130-runtime@sha256:417fc90dfa9392d0e62f3c0a586a297b6d75082eccc90e63905832ac58ae85ff
```

Original allocation files are in `reproduction/allocation/`: four GPUs,
40 requested/48 maximum CPUs, 850GiB RAM, IPC/RDMA capabilities and an allocated
RoCE claim; workspace at `/workspace`, model mount read-only at `/models-shared`,
shared-memory mount at `/dev/shm`. The EBS PVC file is a historical allocation
receipt, not an instruction to create a second default workspace. Use the
`kv-p2p-workspace` PVC and the cluster's current eligible allocation mechanism.
Do not reuse the historical node selector, reservation or claim identity blindly.

Access the intended GB300 context with `tsh kubectl`:

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-aws-gb300
NS=harryx-kv-egm
# Set POD to an already authorized, eligible four-GB300 allocation.
tsh kubectl --context "$CTX" -n "$NS" get pod "$POD" -o json
tsh kubectl --context "$CTX" -n "$NS" exec "$POD" -- nvidia-smi
tsh kubectl --context "$CTX" -n "$NS" exec "$POD" -- df -h /workspace
```

The user's current default retained pod `kvcr-guard-repeat-20261102-r1` is a
B200 allocation. Running this recipe there would create a separate B200 test,
not reproduce GB300. Arrange a GB300 allocation through your normal cluster workflow; this
package does not allocate hardware. Retain pods after
serving/capture cleanup. Do not move an active experiment.

Before any launch, verify at least100GiB workspace headroom, no active benchmark
lock, idle GPUs, assigned RNIC identity/GUID, port ACTIVE, usable verbs device,
and actual pod CPU/RAM/GPU limits. Free RAM is not interchangeable with a PVC
quota. Never fall back to a shared-model PVC for benchmark outputs.

## 3. Choose a stage and prepare immutable inputs

All twelve stage IDs, full settings and historical run identities are in
`stages.json`. The code/config lineage is summarized in the two backend guides.
Use a fresh workspace prefix for each backend/stage and a distinct Dynamo
namespace/discovery directory. For example, inside the selected pod:

```bash
CODE=/workspace/qwen38-linker-code
B="$CODE/kvcr-benchmark/sglang-linker-qwen3.8"
STAGE=kvcr-m1536
R=/workspace/qwen-linker-kvcr-m1536-repeat-r1
MODEL=/models-shared/verified-Qwen3.8-Flash-Next-NVFP4
RNIC=rocep190s0:1
POD_IP=100.65.62.87
ATTEMPT=repeat-r1
DYN_NAMESPACE=qwenlinkerkvcrm1536r1
```

`RNIC` and `POD_IP` above illustrate syntax; obtain the actual allocated values.
Historical runs used `rocep172s0:1` on their original nodes. This is single-RNIC
semantics, not a requirement that every allocation have that device name.

Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`,132,680,249,378 weight bytes.
The captured file/blob/checksum receipt is `reproduction/model-validation.json`.
Validate it before loading. The tokenizer and target/
draft checkpoints must be from this same revision. Mount existing verified model
weights read-only. This bundle contains no model weights.

Trace: `semianalysisai/cc-traces-weka-062126-256k`, revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`; original `traces.jsonl`568,864,747bytes.
`reproduction/client/trace-receipt.json` preserves repository blob metadata.
Materialize the pinned dataset, four referenced plots and serialized mmap inputs
before the measured phase. Original warmed client cache paths are historical
inputs, not a license to write new data into a closed campaign. A new model/
tokenizer pathname may change the mmap cache key; verify dataset/index bytes and
manifest semantics rather than assuming a filename is sufficient.

## 4. Recreate source and runtime

Use the manifest selected in the backend guide, or manually clone the same exact
commits into `CODE/software/`. The following manual alternative reads the checked-in
lock; it does not substitute current branch heads:

```bash
export CODE=/workspace/qwen38-linker-code
export B="$CODE/kvcr-benchmark/sglang-linker-qwen3.8"
export STAGE=kvcr-m1536
python3 - "$B/sources.lock.json" "$STAGE" "$CODE" <<'PY'
import json,pathlib,subprocess,sys
lock,stage,code=sys.argv[1:];row=json.load(open(lock))[stage]
for name,pin in row['sources'].items():
    dst=pathlib.Path(code)/'software'/name
    if dst.exists():
        actual=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()
        assert actual==pin['sha'],(name,actual,pin['sha'])
    else:
        subprocess.run(['git','clone','https://github.com/'+pin['repository']+'.git',str(dst)],check=True)
        subprocess.run(['git','-C',str(dst),'checkout','--detach',pin['sha']],check=True)
PY
python3 "$B/scripts/verify-sources.py" --stage "$STAGE" --code "$CODE"
mkdir -p "$R/software"
for COMPONENT in sglang kvcr dynamo aiperf; do
  ln -s "$CODE/software/$COMPONENT" "$R/software/$COMPONENT"
done
```

For a fully manual first checkout, clone `hxieustc/kvcr-benchmark` at the exact
benchmark SHA in your selected XML, then set `B` to its `sglang-linker-qwen3.8`
directory. Do not apply the historical full patches to the published feature
commits: those commits already contain the changes. Patches remain review evidence.

The Python/CUDA stack was Torch2.14.1+cu130, Transformers5.19.0, NIXL/nixl-cu13
1.5.0, SGLang-kernel0.4.10, FlashInfer0.7.2, CuTe4.8.0, Triton3.8.0 and Mooncake
0.3.13. `reproduction/bootstrap/original-effective-packages.json` and the original
bootstrap receipt are the complete dependency authority, not this short list.
The original ARM online bootstrap and its pin/receipt helper are included:

```bash
bash "$B/reproduction/bootstrap/bootstrap-online.sh" \
  "$R" "$R/software/dynamo" \
  "$B/reproduction/bootstrap/original-effective-packages.json"
export SGLANG_BUILD_RUST_EXTS=none
export SGLANG_RUST_BUILD_MODE=never
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export PYTHONPATH="$R/software/sglang/python:$R/software/kvcr/src"
"$R/env/bin/python" -m pip install --no-deps --no-build-isolation \
  -e "$R/software/kvcr" -e "$R/software/sglang/python"
```

The bootstrap requires aarch64 Rust1.96.1, gcc/g++, protoc, OpenSSL/hwloc/udev
headers and access to the pinned wheels. It builds Dynamo runtime with
`select-service,kv-indexer` and `--locked`. Reusing the recorded verified wheels
is preferable when available. Their exact SHA256 values are:

| Wheel | SHA256 |
|---|---|
| `ai_dynamo-1.7.0-py3-none-any.whl` | `51e08fde9d7e5dbabccb9a1e2e49d3bd133cd67c37b98deee00ab7a1054403ed` |
| `ai_dynamo_runtime-1.7.0-cp311-abi3-manylinux_2_39_aarch64.whl` | `4abfe02c7e66b471d709b75b2cc197229a1969eb5bdf1b5e18af07cba91a75a3` |

A source rebuild can produce different wheel bytes. Record its new hashes and
use that identical build for both backends; do not claim historical wheel-byte
identity without checking it. The native PR head and the Python cache-mode
patch are separate provenance requirements.

Preserve `nvidia_cutlass_dsl_packages.pth` when composing a private environment;
missing its nested DSL search path caused a previous pre-model import failure.
Verify actual imported Torch/FlashInfer/NIXL/SGLang/KVCR/Dynamo paths. The
native-poll stage additionally requires the bundled helper/SDK overlay and its
exact native binding; changing a Python option alone is insufficient.

Create a separate experiment-only client environment:

```bash
python3 "$B/scripts/install-client.py" --root "$R"
```

The original requirements, installed package list, actual HF loader and trace-pin
patch are preserved in `reproduction/client/`. The installer downloads and SHA256-checks the exact released wheel before pip installation.
The derived constraints keep
version-pinned package rows and exclude image-specific local URLs/editable
projects; they constrain requested client dependencies without installing the
old image's unrelated GPU/framework packages. Verify the resolved client freeze
and loader revision. The original broad `datasets>=4.7.0` alone is not a lock.

## 5. Materialize and inspect component commands

The renderer uses the actual captured `service-pids.json` argv/environment,
rebases only operational paths/addresses/namespace, and retains the chosen
stage's model/cache/kernel/transport settings. It never starts processes.

```bash
python3 "$B/scripts/render-commands.py" --stage "$STAGE" \
  --root "$R" --model "$MODEL" --pod-ip "$POD_IP" --rnic "$RNIC" \
  --attempt "$ATTEMPT" --namespace "$DYN_NAMESPACE" --materialize \
  --role worker-0 > "$R/worker-0.preview.sh"
```

Inspect the preview and `R/config/storage-options.json` before running anything.
The renderer creates a short workspace TMPDIR to stay under ZeroMQ's IPC path
limit. It creates `R/model` as a read-only-input symlink, not a model copy.
The emitted source PYTHONPATH is `R/scripts:R/software/sglang/python:R/software/kvcr/src`.
Dynamo Python is installed from the patched pinned build.

Generate each role's shell file after materialization:

```bash
for ROLE in state-agent frontend worker-0 worker-1 worker-2 worker-3 client collector qualify clear cache-mode; do
  python3 "$B/scripts/render-commands.py" --stage "$STAGE" \
    --root "$R" --model "$MODEL" --pod-ip "$POD_IP" --rnic "$RNIC" \
    --attempt "$ATTEMPT" --namespace "$DYN_NAMESPACE" --role "$ROLE" > "$R/$ROLE.sh"
done
```

For Mooncake, also render `--role mooncake-master`. Start it first and verify
RPC25001/metrics25002. Its executable is the recorded Mooncake0.3.13 binary path;
if the package location differs, resolve the same-version installed binary and
record that path/hash change. Do not change allocation policy accidentally.

## 6. Start components separately and qualify them

In the pod shell, start one recorded session per service. Repeat the following
for state-agent, frontend, and each worker, in that order; for Mooncake start its
master before them:

```bash
RUN="$R/runs/$ATTEMPT"
ROLE=state-agent
setsid bash "$R/$ROLE.sh" > "$RUN/$ROLE.log" 2>&1 &
PID=$!
python3 "$B/scripts/record-process.py" --run "$RUN" --name "$ROLE" --pid "$PID"
```

The six KVCR services are state-agent, frontend, worker0–3. Mooncake has a seventh
master service. `CUDA_VISIBLE_DEVICES=0..3` selects one GPU per worker. Each worker
has its own metrics directory, control/NIXL endpoints and Dynamo system port.
Recorded ports: router24000/system24900, state-agent24901, workers' system/metrics
24910–24913, HTTP24010–24013, KV events24020–24023, KVCR control24100+32r and NIXL
24500+32r. Mooncake worker endpoints are25100+r.

Wait for all four logs to report successful runtime registration and query:

```bash
curl --fail http://127.0.0.1:24000/v1/models/nvidia/Qwen3.8-Flash-Next-NVFP4/ready
bash "$R/cache-mode.sh"
bash "$R/qualify.sh" > "$RUN/preflight.log" 2>&1
bash "$R/clear.sh" > "$RUN/cache-clear.log" 2>&1
```

Require a true ready response, four model cards with normal cache-key mode, and
passed semantic reads on source/local/cold peers1/2/3. The captured validation
uses a4301-token vault-code prompt and expects4288reused tokens; new prompt
length can differ, but all state components must be complete. Exact answer
semantics and real peer data movement are required. Stochastic synthetic draft
acceptance makes token-ID equality alone an inappropriate oracle.

Validate complete native bytes before model timing on a replacement pod, not
just a cache counter. Preserve source pool geometry and all checkpoint siblings.
The original qualification fixture is included in `reproduction/collection/`;
its historical absolute paths require review/rebasing before independent use.
GPU buffers, NIC, native binding and actual import origins must match the case.

Clear all four framework caches through the native request-plane RPC; do not
substitute an HTTP clear route or assume a framework flush necessarily deletes
all backend inventory. Use the recorded backend-specific reset semantics.
Check GIL/native-delay logs after clear as well as first startup: the native
agent can be reconstructed, so qualifying generation1 alone is insufficient.

## 7. Capture and run the client manually

Start the raw collector before warmup. It records worker metrics, GPU load,
shared HCA counters, cgroup CPU/RAM and workspace capacity. Shared port counters
are not exclusively attributable to this pod.

```bash
setsid bash "$R/collector.sh" > "$RUN/collector.log" 2>&1 &
COLLECTOR_PID=$!
set +e
bash "$R/client.sh" > "$RUN/benchmark.log" 2>&1
CLIENT_RC=$?
printf '%s\n' "$CLIENT_RC" > "$RUN/client.exit"
set -e
set +e
kill -TERM "$COLLECTOR_PID"
wait "$COLLECTOR_PID"
COLLECTOR_RC=$?
set -e
printf '%s\n' "$COLLECTOR_RC" > "$RUN/collector.exit"
```

Always close the collector on a failed client too; use a shell trap if scripting
this sequence. Verify its PID/start-time before signaling in any delayed cleanup.
For the historical CPU evidence, also run the included `thread-cpu-sampler.py`
against the actual `service-pids.json`, capture its exit and retain its script hash.

Prefetch the pinned dataset into the same new run cache used by the rendered
client before its timed phase. This reads model weights from their existing
verified path and never writes to the shared model PVC:

```bash
HF_HOME="$R/cache/huggingface" "$R/clientenv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="semianalysisai/cc-traces-weka-062126-256k",
                  repo_type="dataset", revision="8fecd2fc56694469f758f0afbbb6335ad3043740")
PY
```

The actual client command is preserved under each stage's
`recorded/benchmark-command.json`. Core arguments are:

```bash
"$R/clientenv/bin/agentx" profile --url http://127.0.0.1:24000 \
  --model nvidia/Qwen3.8-Flash-Next-NVFP4 --tokenizer "$MODEL" \
  --concurrency 48 --benchmark-duration 1800 --warmup-requests-per-lane 5 \
  --tokenizer-trust-remote-code \
  --server-metrics http://127.0.0.1:24910/metrics http://127.0.0.1:24911/metrics \
                   http://127.0.0.1:24912/metrics http://127.0.0.1:24913/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --output-artifact-dir "$RUN/aiperf_artifacts"
```

Use the renderer rather than accidentally omitting a stage's recorded arguments.
Expected actual warmup records are286:240extra requests plus46mandatory primers.
Warmup is count-based, not a promised fixed elapsed duration. Grace period is30s;
the end-to-end benchmark wall time includes preparation, warmup and drain.
Configured48lanes is not48simultaneously active GPU requests throughout the run.

## 8. Close, download and analyze

Require client/collector zero exits and an official export with agentx scenario,
1800s,48lanes,5extra warmups, the pinned dataset and286warmup rows. Record completed,
cancelled and failed outcomes separately. Preserve all service logs, configuration,
source patches/extras, source heads/imports, registered model cards, preflight,
cache clear, native/setting proofs and metric capture coverage. Do not invent
controller/driver success receipts for the manual procedure: those are produced
only by the corresponding archived controller.

Stop only the recorded owned process families after serving/capture completes,
using PID plus `/proc` start-time and group/session/descendant checks. Wait for GPU
contexts to disappear, retain the pod, and make it available for subsequent tests.
Never use a broad process-name kill. Original stop helpers are supplied as audit
references; they contain historical root/receipt assumptions and must not be run
unreviewed against another campaign.

Archive closed raw with a manifest of every filename, byte size and SHA256; verify
all members before claiming collection complete. Download that closed archive to
the local server and verify its archive and member hashes again. Run all scientific
postprocessing and plots **locally**, not on the GPU pod. Canonical new raw/configs/
logs/results remain on `kv-p2p-workspace`; models remain read-only. Historical raw
may have been pruned remotely after verified download, so consult its removal
ledger and protected local archive backup rather than labeling it lost.

The `recorded/profile_export_aiperf.json` files in this bundle contain the original
results. They are evidence, not new-run outputs. Scientific comparisons must state
actual hardware, source settings, cache partitions, phase bounds, error/cancellation
cohorts and which bandwidth boundary is being measured.

## 9. Published source commits versus original base pins

`sources.lock.json` contains both. The feature commits were reconstructed from
archived source postimages and have new commit identities; they are not falsely
presented as the hashes of the old dirty working trees. Branch parents show the
incremental optimization lineage. Capacity-only M2261/M2176/M1536 share source.
The original setup JSON can contain a stale preparatory `not run` label; completion
evidence is the matching archived export/exit and run receipts, not that draft label.

The native-poll stage deliberately checks the historical image's NIXL wrapper,
native binding and utilities SHA256 at actor construction. Its helper is committed
under `sglang/.../storage/kvcr/_native_progress`, not an untracked hidden dependency.
If a package update changes those bytes, stop rather than silently removing the
check. That would require a separately qualified native-poll experiment.

The archived `run-benchmark.py`/stop/analyze helpers can contain old controller
paths and hardcoded B200/224 GiB report prose; they are retained as evidence.
Use the rendered manual roles and these GB300 docs. Never execute an old global
stop helper or copy its success files. Analyze new closed exports locally and
report the actual 128 GiB/worker GB300 settings.
