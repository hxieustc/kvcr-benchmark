# Manual component execution: SGLang HiCache + Mooncake on GB300

This is the complete manual-running guide for **Mooncake**. It covers Git checkout,
native installation, allocation qualification, separate service launches, replay,
collection and local postprocessing. See the [Mooncake Google Repo workflow](sglang-hicache-qwen3.8-mooncake.md)
for source identities and the Repo alternative; see [comparison.md](comparison.md)
for the three completed arms and metric boundaries.

These instructions start services individually using complete archived commands
rebased by a non-executing renderer. They also work after Google Repo sync.
The examples name a **new** run root. All outputs use the `kv-p2p-workspace` PVC
at `/workspace`; verified shared model weights remain read-only. Keep pods.

## 0. Get the reproduction package without Google Repo

Run this in a new code directory, inside the intended serving environment. Git,
Python3 and network access must already be available. This resolves the exact
benchmark SHA from the selected manifest, records the manifest revision, then
checks out the benchmark repository at that SHA. It does not start any service.
If the Google Repo guide already populated CODE, skip these clone commands and
use that CODE instead.

```bash
export CODE=/workspace/qwen38-hicache-code
export MANIFEST_REF=develop
mkdir -p "$CODE"
git clone --no-checkout https://github.com/hxieustc/kvcr-manifests.git "$CODE/manifests"
git -C "$CODE/manifests" fetch origin "$MANIFEST_REF"
git -C "$CODE/manifests" checkout --detach FETCH_HEAD
git -C "$CODE/manifests" rev-parse HEAD > "$CODE/manifest-revision.txt"
export MANIFEST_XML="$CODE/manifests/manifests/sglang-hicache-qwen3.8-mooncake-random.xml"
BENCHMARK_SHA=$(python3 - "$MANIFEST_XML" <<'PYXML'
import sys,xml.etree.ElementTree as ET
projects=ET.parse(sys.argv[1]).getroot().findall('project')
print(next(p.attrib['revision'] for p in projects if p.attrib.get('path')=='kvcr-benchmark'))
PYXML
)
git clone --no-checkout https://github.com/hxieustc/kvcr-benchmark.git "$CODE/kvcr-benchmark"
git -C "$CODE/kvcr-benchmark" fetch origin "$BENCHMARK_SHA"
git -C "$CODE/kvcr-benchmark" checkout --detach "$BENCHMARK_SHA"
export B="$CODE/kvcr-benchmark/sglang-hicache-qwen3.8"
```

For an exact later repeat, set MANIFEST_REF to the recorded manifest commit,
rather than a moving branch. Keep the resolved XML with the new run. The source
checkout loop in section2 uses this package's immutable source lock. Do not run
these clone commands over an existing active code directory.

## 1. Select allocation and immutable inputs

Select an eligible idle four-GB300 ARM64 pod with the pinned image, at least the
recorded 850 GiB RAM limit, four assigned GPUs and one assigned 400 Gb/s RoCE port.
Record its actual namespace/pod/node/UID/GPU UUIDs/RDMA claim. Check workspace
headroom >=100GiB and no competing benchmark lock before launch. Original pod
and claim JSONs in `reference/` are historical receipts, not current authority.

Use the main guide's Repo command or manually clone the benchmark commit pinned
by the chosen XML. Then inside the allocated environment set:

```bash
export CODE=/workspace/qwen38-hicache-code
export B="$CODE/kvcr-benchmark/sglang-hicache-qwen3.8"
export R=/workspace/qwen38-hicache-mooncake-random-repeat-r1
export CASE=mooncake-random
export ATTEMPT=repeat-r1
export MODEL=/models-shared/verified-Qwen3.8-Flash-Next-NVFP4
export POD_IP=YOUR_CURRENT_POD_IP
export RNIC=YOUR_ASSIGNED_RNIC:1
export DYN_NAMESPACE=hicachemooncakerepeatr1
```

This example uses random placement. To reproduce local-first, select
`sglang-hicache-qwen3.8-mooncake-local-first.xml` in section0, set
`CASE=mooncake-local-first`, and choose a fresh R, ATTEMPT and DYN_NAMESPACE.
The renderer selects `local_first` from that case. Start a fresh master and four
fresh workers for each policy after closing the preceding services; do not reuse
its warmed store. All source pins, cache budgets and workload controls stay the same.

MODEL must be the exact revision in the guides. Compare its files/checksums
against `reproduction/model-validation.json`; do not redownload into or alter
shared model storage. Use distinct roots/namespaces and fresh services for arms.

## 2. Manual Git alternative to Repo

If using Repo, the sources already exist and this checks the selected SHAs.
Otherwise the same code clones each repository at the immutable lock:

```bash
python3 - "$B/sources.lock.json" "$CODE" <<'PY'
import json,pathlib,subprocess,sys
pins=json.load(open(sys.argv[1]));root=pathlib.Path(sys.argv[2])
for name,v in pins.items():
    dst=root/'software'/name
    if not dst.exists():
        subprocess.run(['git','clone','https://github.com/'+v['repository']+'.git',str(dst)],check=True)
        subprocess.run(['git','-C',str(dst),'checkout','--detach',v['sha']],check=True)
    actual=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()
    assert actual==v['sha'],(name,actual,v['sha'])
PY
python3 "$B/scripts/verify-sources.py" --code "$CODE"
mkdir -p "$R/software"
for COMPONENT in sglang kvcr dynamo aiperf; do
  ln -s "$CODE/software/$COMPONENT" "$R/software/$COMPONENT"
done
```

Do not apply the archived dirty patches to these feature commits: the patches
are already incorporated. Historical receipts preserve original bases/diffs;
the new source lock records the reconstructed immutable commits.

## 3. Install the native runtime and client

On the pinned ARM image, supply build tools gcc/g++, pkg-config, protoc and
OpenSSL/hwloc/udev headers. The included bootstrap builds Dynamo with Rust1.96.1,
Maturin1.9.6 and `select-service,kv-indexer`, preserving Cargo.lock:

```bash
bash "$B/reproduction/bootstrap/bootstrap-online.sh" \
  "$R" "$R/software/dynamo" \
  "$B/reproduction/bootstrap/original-effective-packages.json"
export SGLANG_BUILD_RUST_EXTS=none SGLANG_RUST_BUILD_MODE=never
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export PYTHONPATH="$R/software/sglang/python:$R/software/kvcr/src"
"$R/env/bin/python" -m pip install --no-deps --no-build-isolation \
  -e "$R/software/kvcr" -e "$R/software/sglang/python"
python3 "$B/scripts/install-client.py" --root "$R"
```

The client installer verifies the exact AgentX wheel hash, uses captured
dependency constraints/datasets5.0.1 and applies only the pinned HF revision.
Record actual imports/versions, package freeze, wheel hashes and `.so` hashes.
Retain CuTe's `nvidia_cutlass_dsl_packages.pth` resolution; do not fall back to
another image's unrelated native packages. New feature-commit SCM version
strings differ from the old dirty checkout's version; source file postimages
and actual compiled dependencies are the identity requirements.

## 4. Materialize and inspect the component commands

```bash
python3 "$B/scripts/render-commands.py" --case "$CASE" --root "$R" \
  --model "$MODEL" --pod-ip "$POD_IP" --rnic "$RNIC" \
  --attempt "$ATTEMPT" --namespace "$DYN_NAMESPACE" --materialize \
  --role worker-0 > "$R/worker-0.preview.sh"
export C="$R/cases/$CASE"
export RUN="$C/runs/$ATTEMPT"
ROLES="state-agent frontend worker-0 worker-1 worker-2 worker-3 client collector thread-cpu-collector clear cache-mode peer post-clear"
ROLES="mooncake-master $ROLES"
for ROLE in $ROLES; do
  python3 "$B/scripts/render-commands.py" --case "$CASE" --root "$R" \
    --model "$MODEL" --pod-ip "$POD_IP" --rnic "$RNIC" \
    --attempt "$ATTEMPT" --namespace "$DYN_NAMESPACE" --role "$ROLE" > "$R/$ROLE.sh"
done
```

The renderer writes only new-run inputs, copies the original observer and
collector, creates model symlinks and uses a short IPC TMPDIR. It does not start
services or mark qualification passed. Inspect storage JSON embedded in all
four worker commands: the exact host/backend budgets, control endpoints, current
RNIC/IP and desired Moon policy must be correct. A printed command without
`--materialize` is non-mutating. The original archived controller is retained for
audit, not the execution entry point of this manual procedure.

## 5. Bind and run actual RDMA/component qualification before models

On the cluster-access host, save the **current** pod and its assigned RDMA claim,
then copy those JSONs into the new root (replace CTX/NS/POD/R):

```bash
tsh kubectl --context "$CTX" -n "$NS" get pod "$POD" -o json > /tmp/hicache-pod.json
CLAIM=$(tsh kubectl --context "$CTX" -n "$NS" get pod "$POD" \
  -o 'jsonpath={.status.resourceClaimStatuses[?(@.name=="roce")].resourceClaimName}')
tsh kubectl --context "$CTX" -n "$NS" get resourceclaim "$CLAIM" -o json > /tmp/hicache-claim.json
tsh kubectl --context "$CTX" -n "$NS" exec "$POD" -- mkdir -p "$R/setup"
tsh kubectl --context "$CTX" -n "$NS" exec -i "$POD" -- tee "$R/setup/pod-r2-live.json" < /tmp/hicache-pod.json > /dev/null
tsh kubectl --context "$CTX" -n "$NS" exec -i "$POD" -- tee "$R/setup/rdma-claim.json" < /tmp/hicache-claim.json > /dev/null
```

Inside the pod, obtain the assigned GPU UUID list and HCA GUID, then bind the
archived check logic to those explicit inputs:

```bash
GPU_UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
HCA=${RNIC%:*}
NODE_GUID=$(cat "/sys/class/infiniband/$HCA/node_guid")
python3 "$B/scripts/prepare-native.py" --root "$R" \
  --pod-json "$R/setup/pod-r2-live.json" --claim-json "$R/setup/rdma-claim.json" \
  --gpu-uuids "$GPU_UUIDS" --node-guid "$NODE_GUID" --rnic "$RNIC"
export UCX_TLS=rc,self,cuda_copy UCX_NET_DEVICES="$RNIC" UCX_MEMTYPE_CACHE=n UCX_LOG_LEVEL=info
export UCX_MAX_RNDV_RAILS=1 UCX_MAX_EAGER_RAILS=1 UCX_MAX_RMA_RAILS=1
export NIXL_UCX_SGL_ENABLE=1 WITH_NVIDIA_PEERMEM=0
unset MC_FORCE_TCP
"$R/env/bin/python" "$R/qualification/qualify_native_rdma.py" \
  --kind all --attempt "$ATTEMPT" --execute
"$R/env/bin/python" "$R/qualification/qualify-kvcr-hicache-peer-r2.py"
"$R/env/bin/python" "$R/qualification/qualify-kvcr-complete-state-peer.py"
```

`prepare-native.py` writes source/snapshot/hash inputs, not passed receipts. It
changes only operational ROOT/POD/UID/NODE/CLAIM/GPU/RNIC/GUID bindings. Original
hardware, native-SDK byte checks, peer identity, full-byte checks, deadlines and
idle/closure checks remain. Historical `roce` claim naming may differ by cluster;
resolve the actual allocated RDMA claim instead of inventing one. The supplied
driver requires a single port1/400Gb/s GB300 allocation. Adaptation to another
device class or source/binary version is a separately qualified experiment.

The complete-state fixture includes real mixed BF16/int64 Mamba/PLE/history
state, with no framework pin requests. A native test's own agent thread settings
are fixture settings, not changes to the serving KVCR default. Check both native
children pass, all adapter bytes pass and GPUs are idle before starting models.
Do not use the historical passed receipts as new allocation proof.

## 6. Start each serving component manually

Start the Mooncake master first, then state-agent, frontend and four SGLang workers.
The helper below performs one owned launch and records its actual identity;
invoke it separately for every component as shown.

```bash
start_component() {
  local ROLE="$1"
  setsid bash "$R/$ROLE.sh" > "$RUN/$ROLE.log" 2>&1 &
  local PID=$!
  python3 "$B/scripts/record-process.py" --run "$RUN" --name "$ROLE" --pid "$PID"
}
start_component mooncake-master
start_component state-agent
start_component frontend
start_component worker-0
start_component worker-1
start_component worker-2
start_component worker-3
```

The PID must be the shell's exec-replaced service. The recorder reads real
`/proc` argv and start ticks; it never signals processes. A separate PID/start
record is required for the Moon master. Ensure recorded worker order0–3 and
wait for four successful runtime registrations.

Ports: frontend24000/system24900, state-agent24901, worker system24910–24913,
worker HTTP24010–24013, KV events24020–24023, KVCR control24100+32r/NIXL24500+32r,
Moon master25001/metrics25002, Moon worker25100+r/HTTP-metrics25010+r.
The frontend exec wrapper retains the original NUMA-node0 CPU-affinity rule,
intersected with the pod's allowed CPUs; it does not change service PID identity.

```bash
curl --fail http://127.0.0.1:24000/v1/models/nvidia/Qwen3.8-Flash-Next-NVFP4/ready
bash "$R/cache-mode.sh"
bash "$R/peer.sh" > "$RUN/preflight.log" 2>&1
bash "$R/clear.sh" > "$RUN/validation-cache-clear.log" 2>&1
bash "$R/post-clear.sh" > "$RUN/post-clear.log" 2>&1
```

Require exact vault-code semantics, positive complete-state backend GETs on
peers1/2/3, four actual host/backend allocation proofs within budget, packed
draft/indexer geometry and no linker. The live qualifier reads **fresh** native
receipts and the new service/observer records. Its operational path adapter uses
`setup/manual-inputs.json`, not a fabricated historical harness-ready pass file.

Reset is **framework-only on every arm**. Do not call Moon master remove_all:
KVCR's backend clear is a no-op, so that would make backend histories unequal.
Both may retain a small unrelated preflight canary; its UUID occurs before the
first64-token boundary. Do not describe either backend as perfectly empty after
this reset. Each policy arm nevertheless starts with fresh services/store before
its own preflight, never the previous measured workload's warmed backend.

## 7. Prefetch the pinned trace; capture and replay

Before timing, fetch the dataset into the same new client cache:

```bash
HF_HOME="$C/cache/huggingface" "$R/clientenv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="semianalysisai/cc-traces-weka-062126-256k",
                  repo_type="dataset", revision="8fecd2fc56694469f758f0afbbb6335ad3043740")
PY
setsid bash "$R/collector.sh" > "$RUN/collector.log" 2>&1 &
COLLECTOR_PID=$!
setsid bash "$R/thread-cpu-collector.sh" > "$RUN/thread-cpu.log" 2>&1 &
CPU_COLLECTOR_PID=$!
# Require a real first-snapshot readiness marker; preserve failed coverage evidence.
until [ -f "$RUN/first-capture-ready.json" ]; do
  kill -0 "$CPU_COLLECTOR_PID" || exit 1
  sleep 1
done
set +e
bash "$R/client.sh" > "$RUN/benchmark.log" 2>&1
CLIENT_RC=$?
printf '%s\n' "$CLIENT_RC" > "$RUN/client.exit"
set -e
# Close only this verified collector PID/start/session; handle client failure too.
kill -TERM "$COLLECTOR_PID"
kill -TERM "$CPU_COLLECTOR_PID"
set +e
wait "$COLLECTOR_PID"
COLLECTOR_RC=$?
set -e
printf '%s\n' "$COLLECTOR_RC" > "$RUN/collector.exit"
wait "$CPU_COLLECTOR_PID"
```

Use a trap for owned collector cleanup if automating this block. The original
recorder captured four worker metrics, GPU samples, shared HCA counters, cgroup
CPU/memory and capacity every5s; preserve observers as well. The client uses
`R/assets/client-input-maps`. A new MODEL pathname changes tokenizer cache identity,
so generate/verify its mmap input once before timing rather than claiming an old
cache HIT proves byte-identical inputs. Keep cache content and the resolved
model/tokenizer/dataset revisions in the new run's evidence.

## 8. Close, verify, download and analyze locally

Verify official scenario AgentX, 48 lanes,1800s,five extras,286 actual warmups,
client/capture exits, phase completed/cancel/error outcomes and full capture
coverage. Stop only recorded owned service identities after verifying PID/start
and their descendants/groups; preserve uncertain native ownership for inspection.
Wait until all owned GPU contexts are gone. Retain the pod and existing results.
No old global stop helper or broad process-name kill is appropriate.

Archive the closed case plus its observer files, exact rendered commands,
source lock, resolved Repo manifest, configs, native/source/wheel/import receipts,
model/trace/mmap identities, peer/cache/allocation proofs and process closure.
Generate an inventory of every member's filename/bytes/SHA256 and verify the
archive against it, then download and verify again on the local analysis server.
Keep canonical raw outputs on `/workspace`; run scientific postprocessing
**locally**, not on GPU pods. The included historical summaries are reference
results, not new-run validation receipts.

Tier semantics are HBM / persistent framework host / backend storage / recompute.
Preserve native/source versus API timer definitions, self/peer routing, failed
and partial operations, descriptor units, observer loss flags and coverage.
Never claim remote-only RDMA bytes from Moon API completion bytes or infer exact
transfer CDFs from aggregate count/sum counters. The recorded observer does expose
individual native/source records in this HiCache family, unlike older linker
captures; eligibility/join/coverage checks are still necessary.

## 9. Scope of this publication

This is a source/config/fixture reconstruction with manual execution paths.
The archived controller is not silently repinned or re-certified. Original
functional check logic is retained, while fresh operational allocation bindings
and explicit input materialization are documented. Offline validation checks
published source bytes, XML pins, command flags/budgets, Python/shell syntax and
doc links. It does not run a new native, GPU or AgentX benchmark.
