# Manually run Direct Guard AgentX, one service at a time

This guide runs the same single-RNIC direct Guard r18/r19 setup as `run.sh all`, while letting you inspect each config, start each component, decide when to replay, and finalize the capture yourself. It uses the common preparation helper for private package installation and correctness probes; **it does not invoke the automatic benchmark supervisor**. Keep this shell open, preferably inside `tmux`, until final collection. For a detached, supervised test use [the automated guide](reproducing.md).

## 1. Sync sources and benchmark scripts

On the host, start outside any existing `repo` client. Install Google `repo`, Git, Python 3.11+, and Teleport `tsh`, and configure GitHub SSH access.

```bash
mkdir -p guard-direct-code
cd guard-direct-code
repo init -u git@github.com:hxieustc/kvcr-manifests.git \
  -b develop -m manifests/kvcr-guard-direct-vllm.xml
repo sync -j4 --no-clone-bundle --no-tags
repo manifest -r > resolved-manifest.xml
```

The workspace layout is:

```text
guard-direct-code/
├── manifests/                  # manifest repo, no benchmark implementation
├── kvcr-benchmark/guard-direct/ # configs, guides, scripts, analysis, canaries
├── kvcr/                       # pinned harryx/guard-direct code
├── vllm/                       # pinned harryx/guard-direct code
├── aiperf/                     # pinned AgentX implementation
└── dynamo/                     # archived router source reference
```

The manifest keeps exact software source revisions; consult [pins.md](pins.md). Dynamo runtime parity uses the two separately checksum-pinned `1.6.0.dev20261014` wheels, not a build claimed to correspond to the archived source reference. Save the resolved manifest and do not update repositories during a run.

## 2. Allocate or adopt an eight-GPU pod

Inspect availability and PVCs with `tsh kubectl`. See the [allocation/bootstrap guide](reproducing.md) for required tools and image setup. This manual workflow uses exactly the same pod template and requirements:

| Resource | Configuration |
|---|---|
| GPU | Eight B200 GPUs on one node |
| CPU | Request 120, limit 192; frontend uses CPUs 0–7 |
| Container DRAM | Request 1800 GiB, limit 2000 GiB; actual available memory is checked |
| Shared memory | `/dev/shm` memory volume, maximum 1750 GiB |
| Writable storage | Workspace PVC at `/workspace`; allow at least 100 GiB free for raw/derived artifacts |
| Model | Populated shared-model PVC at `/models-shared`, read-only |
| RDMA | ACTIVE `mlx5_0:1`, `/dev/infiniband` character devices, sufficient locking permissions |
| Container lifecycle | Idle retained container, `restartPolicy: Never`; no automatic pod deletion |

Render a pod for review; substitute your node, namespace, PVCs and actual allocatable RDMA resource:

```bash
./kvcr-benchmark/guard-direct/run.sh pod \
  --pod guard-direct-agentx --namespace kvcc-mbench \
  --node YOUR_B200_NODE --rdma-resource YOUR_CLUSTER_RDMA_RESOURCE \
  --workspace-pvc kv-p2p-workspace --model-pvc shared-model-cache \
  --render-only > guard-direct-pod.json
```

Run the same command without `--render-only` to allocate, adding `--receipt guard-direct-pod-receipt.json`. Where the cluster explicitly permits host-device exposure without an extended resource, use `--hostpath-rdma` instead of `--rdma-resource`; native RC/RDMA probes are still mandatory. No dedicated KVCR taint or node-label overwrite is needed.

To reuse a retained pod, record it with `run.sh pod --pod NAME --namespace NS --adopt --receipt FILE`; compare its image, devices, memory limits, mounts and privileges with the template. Adoption does not reconfigure it.

```bash
tsh kubectl -n kvcc-mbench wait --for=condition=Ready pod/guard-direct-agentx --timeout=30m
tsh kubectl -n kvcc-mbench exec -it guard-direct-agentx -- bash
```

**All remaining commands run inside this pod.** Ensure the source workspace is present at `/workspace/guard-direct-code`. If not shared from the host, repeat step 1 there using your normal SSH credential mount/agent mechanism. The image/bootstrap-tool instructions are in the automated guide. Do not put credentials into a run plan or result directory.

## 3. Define the experiment and inspect every config

Use a fresh run ID. The runner never overwrites a started or partially prepared attempt.

```bash
export GD_WORKSPACE=/workspace/guard-direct-code
export GD_GUIDE="$GD_WORKSPACE/kvcr-benchmark/guard-direct"
export GD_RUN_ID=manual-direct-agentx-r1
export GD_RUN="$GD_WORKSPACE/guard-direct-runs/$GD_RUN_ID"
export GD_PLAN="$GD_RUN/plan.json"
mkdir -p "$GD_RUN"
"$GD_GUIDE/run.sh" plan --workspace "$GD_WORKSPACE" --run-id "$GD_RUN_ID" > "$GD_RUN/preview.json"
```

The plan is generated from these checked-in files:

| File under `guard-direct/config/` | What to inspect |
|---|---|
| `pins.json` | Source/image/wheel hashes; model/tokenizer/dataset revisions |
| `experiment.json` | AgentX scenario, corpus, concurrency, seed, warmup, timing, routing policy |
| `kv-transfer.json` | Connector, zero host payload, Guard size, hint-only source selection, control ports, metadata limits |
| `worker-args.json` | Model serving, parallelism, KV formats, graph capture, scheduling and KV-event settings |
| `compilation-config.json` | Complete graph capture sizes; same JSON embedded in worker arguments |
| `pod-template.json` | Image, GPU/CPU/DRAM, PVCs, device exposure and pod lifecycle |
| `runtime-extra.txt`, `aiperf-lock.txt` | Serving additions and archived AIPerf package versions |

```bash
cat "$GD_GUIDE/config/kv-transfer.json"
cat "$GD_GUIDE/config/compilation-config.json"
cat "$GD_GUIDE/config/worker-args.json"
cat "$GD_GUIDE/config/experiment.json"
```

The workload and cache settings must remain:

| Setting | Required value |
|---|---|
| Model/tokenizer | `deepseek-ai/DeepSeek-V4-Pro-DSpark`, snapshot `7c09739fd136abfb70a49ec334157f65f45b52cd` |
| Served model name | `deepseek-ai/DeepSeek-V4-Pro` |
| Worker topology | DP8, TP1, expert parallel; eight workers total |
| KV format | FP8 KV, MXFP4 indexer, block size 256 tokens |
| Model scheduling | Max length 1048576; GPU utilization limit 0.92; max sequences 72; batched tokens 8696; long-prefill threshold 512; prefill interval 16 |
| Speculation | DSpark, six speculative tokens, probabilistic draft sampling |
| Host payload | **0 bytes per worker**; direct GPU↔Guard path |
| Guard payload | **124 GiB per worker**, total **992 GiB**, plus journals and metadata |
| Host journals/headroom | Eight 100 MiB journals, plus at least 768 GiB additional actual startup headroom |
| Source selection | Guard-only READY claims; `opportunistic_query=false`; router hint block size 256; inventory gate enabled |
| Metadata/control | Batch 32768 keys, max 524288 keys/request; localhost control ports 17771–17778 |
| Operation timeouts | Operation 5000 ms, abandon 10000 ms |
| Router | KV mode; soft session affinity TTL 3600 s; decode active-request weight 64 |
| AgentX | `inferencex-agentx-mvp`; all 393 entries from pinned Weka corpus |
| Replay | C72, seed 42, trajectory start ratio 0.25–0.75; trace idle cap 300 s, system idle cap 10 s |
| Timing | 10 warmup requests/lane; warmup grace 1800 s; measured send 3600 s + grace 300 s |
| Session | `X-Dynamo-Session-ID` from correlation ID; session timeout 14400 s |

This is not the older 1 GiB host + 124 GiB host-staged design or the Kimi/WCE JSONL replay. Increasing host bytes, changing pool sizes, source policy or concurrency creates a different experiment.

The model must already exist at the configured snapshot path in the model PVC. If only its mount location differs, pass `--model-path /actual/snapshot/path` consistently to `plan`, `preflight`, and `prepare`. Do not replace the snapshot contents with another model revision.

## 4. Prepare the runtime and data, without starting the model

```bash
"$GD_GUIDE/run.sh" preflight --workspace "$GD_WORKSPACE" --run-id "$GD_RUN_ID"
"$GD_GUIDE/run.sh" prepare --workspace "$GD_WORKSPACE" --run-id "$GD_RUN_ID"
```

Preflight refuses occupied GPUs/ports, wrong GPU count/type, unavailable model/RDMA devices, source drift and insufficient finite cgroup memory headroom. It does not kill existing jobs or reduce cache budgets.

Preparation performs these explicit steps, logged separately under `raw/kvcr/phases/prepare/logs/`:

1. Verify Python 3.12, image vLLM `0.31.1rc1.dev23+g43b4aaea3`, NIXL `1.5.0`, all four source HEADs and tracked-file cleanliness. Save source/version receipts and the resolved manifest.
2. Create `runtime/serving-env` with image system packages. Install the two exact Dynamo wheel files with SHA256 verification and private installation. Install `runtime-extra.txt` and analysis dependencies.
3. Copy KVCR into `runtime/kvcr`, install that private copy, and create `runtime/python/vllm`: compiled extensions link to the pinned image while Python files come from the pinned vLLM checkout. Image-global packages and synced source trees are unchanged.
4. Save KVCR source archive/base patch/hashes; apply the historical startup-only overlay to the private copy (Guard startup 900 s, initial claim socket 1200 s). Transfer operation timeouts stay unchanged.
5. Create `assets/aiperf-env` and a private AIPerf checkout. Pin the existing HF dataset revision hook. Reconstruct the complete dataset and build/adopt the official native mmap cache; save corpus/cache hashes and scenario resolution. Network is allowed in this phase.
6. Run the noncontiguous DRAM RDMA probe, direct Guard DRAM→GPU probe and installed vLLM adapter probe. They verify payload correctness/RC lanes, not only environment strings. Tiny isolated probe pools do not certify serving memory sizes; eight serving-rank attestations do that later.

If exact Dynamo wheels cannot be downloaded, supply the files named in `pins.json` using `prepare --wheelhouse /workspace/wheels`. Checksums must match. Do not fall back to another version or rebuild an unverified source revision.

Inspect preparation receipts before proceeding:

```bash
cat "$GD_RUN/receipts/prepared.json"
cat "$GD_RUN/receipts/runtime-overlay.json"
cat "$GD_RUN/raw/kvcr/phases/prepare/logs/rdma-verification.json"
cat "$GD_RUN/raw/kvcr/phases/prepare/logs/direct-gpu-rdma.json"
cat "$GD_RUN/raw/kvcr/phases/prepare/logs/direct-adapter-smoke.json"
```

A failed/partial preparation is retained; diagnose it and use a fresh run ID. Do not bypass a failed native correctness gate. A successful `prepare` leaves `plan.json` at `$GD_PLAN`. The helper code is `scripts/prepare.py`; each package installation and probe has a separate log for manual debugging.

## 5. Inspect service argv/environment and initialize manual artifacts

The following prints the exact effective commands, including all serialized JSON arguments. It also exposes the full environment whitelist. Measured HF/tokenizer use is offline.

```bash
python3 - "$GD_PLAN" <<'PY'
import json,shlex,sys
p=json.load(open(sys.argv[1]))
for role,argv in p['commands'].items():
    print(role+':\n  '+shlex.join(argv))
print(json.dumps(p['environment'],indent=2))
PY
```

In particular, the UCX/native settings are:

```text
UCX_TLS=rc,cuda_copy
UCX_NET_DEVICES=mlx5_0:1
UCX_MAX_RMA_RAILS=1
NIXL_UCX_SGL_ENABLE=1
KVCR_NVTX_LEVEL=off
UCX_PROTO_ENABLE=y
UCX_PROTO_INFO=y
UCX_MEMTYPE_CACHE=n
UCX_RNDV_THRESH=4096
UCX_ZCOPY_THRESH=4096
UCX_RNDV_SCHEME=get_zcopy
```

Control uses Dynamo TCP request plane, ZMQ event plane, file discovery and run-local IPC. The plan also sets native registration options, unique socket/pool/metadata paths, source-contract provenance, offline HF cache paths and telemetry destinations. Preserve these rather than inheriting arbitrary settings from an older run.

Initialize the manual measured phase. This mirrors directory/provenance creation in the supervisor, without launching anything:

```bash
python3 - "$GD_GUIDE/scripts" "$GD_PLAN" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from workflow import atomic,process_identity,read
p=read(sys.argv[2]);r=Path(p['root'])
assert not (r/'benchmark-started.json').exists(), 'Already started; use a fresh run ID'
ready=read(r/'receipts/prepared.json',{})
assert ready.get('verified') and ready['plan_sha256']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
Path(p['result']).mkdir(parents=True,exist_ok=False)
i=Path(p['ipc']);(i/'rpc').mkdir(parents=True,exist_ok=False);(i/'pools').mkdir()
(r/'discovery').mkdir();(r/'aiperf-mmap').mkdir(exist_ok=True)
atomic(r/'raw/manifest.json',dict(experiment=p['experiment']))
atomic(r/'benchmark-started.json',dict(mode='manual',wall_ns=time.time_ns(),supervisor=process_identity(__import__('os').getpid())))
atomic(r/'status.json',dict(state='manual_starting',run_id=p['run_id'],pod_retained=True))
PY
```

Define this small launcher once in the current shell. It starts **only the requested component**, using the plan's exact argv/env and standard PID/start-tick receipts. Its Python parent waits and reaps the child; keep background launchers alive until shutdown. Only the replay's exit code controls benchmark success.

```bash
gd_start() {
  python3 - "$GD_GUIDE/scripts" "$GD_PLAN" "$1" <<'PY'
import json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from workflow import atomic,read,start_process
p=read(sys.argv[2]);role=sys.argv[3]
child,record=start_process(p,role)
print(json.dumps(record),flush=True)
code=child.wait()
atomic(Path(p['root'])/'receipts'/('manual-'+role+'-exit.json'),dict(exit_code=code,wall_ns=time.time_ns()))
if role=='replay':raise SystemExit(code)
PY
}
```

## 6. Start Guard, router, then vLLM

Start Guard first:

```bash
gd_start guard &
```

The effective command is `numactl --interleave=all SERVING_PYTHON -m kvcr.kvcr_service` with the run-local socket/pool directory, `--guard-count 8`, `--pool-sizes-gb 124`, and the compatibility digest from `kv-transfer.json`. Wait for `memory.sock` before starting workers:

```bash
python3 - "$GD_PLAN" <<'PY'
import json,time,sys
from pathlib import Path
p=json.load(open(sys.argv[1]));s=Path(p['ipc'])/'memory.sock';end=time.monotonic()+1800
while not s.exists():
    if time.monotonic()>end:raise TimeoutError('Inspect guard.log; do not continue')
    time.sleep(1)
print('Guard service socket ready:',s)
PY
gd_start frontend &
gd_start worker &
```

The frontend runs `taskset -c 0-7 SERVING_PYTHON -m dynamo.frontend`, serves HTTP on 8888, and uses the configured model, KV routing, soft affinity and decode weight. Metrics are exposed at 9090. The worker command is `SERVING_PYTHON -m dynamo.vllm` with `worker-args.json` and the effective `kv-transfer.json`. This one deployment command creates all eight DP workers; do not start eight copies of it.

Wait for model readiness and strict eight-rank attestation:

```bash
python3 - "$GD_GUIDE/scripts" "$GD_PLAN" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from workflow import read,wait_ready
wait_ready(read(sys.argv[2]),[])
PY
```

The manual readiness call has no child-process supervisor; inspect logs if startup stops progressing. It checks the HTTP model endpoint and every rank's actual Guard size, zero host payload, VRAM regions, disabled opportunistic queries, zero source framework-pin callbacks, and Python/libc transport environment after native initialization. Its timeout is 7200 seconds, with five-minute progress lines. Model load/registration/graph capture can have low GPU utilization; replay has not started yet.

Logs and rank proofs are here:

```bash
tail -n 30 "$GD_RUN/raw/kvcr/phases/measured/logs/worker.log"
ls "$GD_RUN/raw/kvcr/phases/measured/results/guard/"option-a-rank-*.json
cat "$GD_RUN/receipts/serving-ready.json"
```

## 7. Start observations, then run AIPerf in the foreground

```bash
gd_start monitor &
gd_start rnic &
gd_start replay
```

Do not continue to finalization while replay is active. The monitor samples GPU use, worker/cache metrics and original IB counters every five seconds. The RNIC observer records byte-normalized whole-node port counters for the frozen RNIC analyzer. These rates include other node traffic; they are not exclusively KVCR payload throughput.

Replay invokes the prepared AIPerf binary with the exact `inferencex-agentx-mvp` command in `plan.json`: chat streaming at `http://127.0.0.1:8888/v1/chat/completions`, C72, all 393 entries, pinned tokenizer, seed 42, native AgentX warmup/session scheduling, 3600-second measured duration, 300-second grace, server token counts and server metrics. It is not a fixed-arrival/open-loop JSONL workload. The complete flags are printed in step 5.

AIPerf output is captured in measured `logs/replay.log` and `results/guard/replay-measured/`. Inspect progress from another terminal; the manual path has no five-minute supervisor status loop. The foreground launcher writes `receipts/manual-replay-exit.json` when it reaps AIPerf. A nonzero exit is a failed replay even if some artifacts were exported.

## 8. Quiesce serving and validate the closed capture

After foreground replay returns, run this finalization block. It refuses a missing/failed replay receipt. It stops owned process groups in safe order, waits up to 1800 seconds per step, validates complete worker/Guard telemetry, and only then records completion. A worker shutdown failure stops the sequence before Guard is torn down.

```bash
python3 - "$GD_GUIDE/scripts" "$GD_PLAN" <<'PY'
import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from workflow import atomic,read,same_process,stop_owned,validate_closed
p=read(sys.argv[2]);r=Path(p['root'])
try:
    replay=read(r/'receipts/manual-replay-exit.json')
    assert replay is not None and replay['exit_code']==0, 'Replay missing/failed; retain diagnostics'
    for role in ('replay','worker','frontend','guard','monitor','rnic'):
        record=read(r/'processes'/(role+'.json'))
        if record and same_process(record):stop_owned(record,timeout=1800)
    validate_closed(p)
    atomic(r/'status.json',dict(state='completed',mode='manual',run_id=p['run_id'],aiperf_exit_code=0,pod_retained=True,wall_ns=time.time_ns()))
except BaseException as error:
    atomic(r/'status.json',dict(state='failed',mode='manual',error=str(error),pod_retained=True))
    raise
PY
```

A finalization timeout, open journal, missing rank/Guard identity, dropped record, unmatched operation lifecycle or invalid key journal cannot be labelled a valid completed run. Retain the pod and all evidence. Do not force success with SIGKILL.

If replay/startup fails, use the owned-process stop command below before diagnostic collection; it does not delete the pod. A timeout intentionally leaves remaining owners for inspection. If needed, terminate the still-running foreground replay from a second shell using this command, rather than killing arbitrary model processes.

```bash
"$GD_GUIDE/run.sh" stop --workspace "$GD_WORKSPACE" --run-id "$GD_RUN_ID"
```

For interrupted manual runs, record `state=failed` with the reason in `status.json`; do not run the successful-finalization block against a missing/nonzero replay receipt. Keep the service launcher shell alive so it can reap terminated children.

## 9. Collect raw data, generate plots, and review validity

```bash
"$GD_GUIDE/run.sh" collect --workspace "$GD_WORKSPACE" --run-id "$GD_RUN_ID"
"$GD_GUIDE/run.sh" analyze --workspace "$GD_WORKSPACE" --run-id "$GD_RUN_ID"
cat "$GD_RUN/status.json"
cat "$GD_RUN/collection.json"
cat "$GD_RUN/report.md"
```

Before collection, copy the actual pod manifest/allocation receipt and resolved manifest into `$GD_RUN/receipts/` if not already present. Collection refuses live recorded services, hashes each raw/receipt file, verifies archive member contents and gzip CRC, and preserves originals. Failed-run collection remains useful diagnostics; archive integrity does not turn failed performance data into a valid run.

Review the base, operation, direct and RNIC reports and CSV/PNG outputs for:

- Measured TTFT and TPOT distributions, throughput, successful/failed requests and scenario quality.
- GPU prefix reuse, own/remote Guard cache hit/yield, recomputation and cache occupancy over time, with explicit denominators.
- Worker load and GPU utilization over time.
- Positive-byte native and source-operation transfer size/block-count/duration/bandwidth distributions, CDFs, and size-conditioned mean/p50/p90/p99 tables; separate failure/zero-byte accounting.
- Receive/claim/submit/DONE/cleanup and progress-loop timing where emitted, and closed-capture/reconciliation validity.
- Whole-node directional RNIC usage over the measured window, without calling Tx+Rx application throughput.

See [metrics.md](metrics.md) for exact artifact paths and interpretation. Missing evidence remains unavailable. All workers are on one node using its HCA; these results do not prove cross-node RDMA bandwidth. The pod remains allocated after all service processes exit.
