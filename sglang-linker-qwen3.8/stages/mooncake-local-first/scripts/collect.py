#!/usr/bin/env python3
"""Losslessly collect per-worker metrics, GPU load and raw HCA counters."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--attempt',required=True)
a=p.parse_args()
root=Path('/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-mooncake-placement-local-first')
run=root/'runs'/a.attempt
assert run.is_dir()
stopped=False
def stop(*args):
    global stopped
    stopped=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)

def metrics(rank):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{24910+rank}/metrics',timeout=3) as response:
            return {'rank':rank,'text':response.read().decode()}
    except Exception as error:
        return {'rank':rank,'error':str(error)}

def extra_metrics(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=3) as response:
            return {'port':port,'text':response.read().decode()}
    except Exception as error:return {'port':port,'error':str(error)}

with gzip.open(run/'telemetry.jsonl.gz','at',encoding='utf8') as output,ThreadPoolExecutor(max_workers=8) as pool:
    while not stopped:
        start=time.monotonic()
        sample={'utc':time.time(),'workers':list(pool.map(metrics,range(4))),'mooncake_metrics':list(pool.map(extra_metrics,[25002,25010,25011,25012,25013]))}
        result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        sample['gpu_csv']=result.stdout
        sample['gpu_error']=result.stderr if result.returncode else None
        counters={}
        for f in Path('/sys/class/infiniband/rocep172s0/ports/1/counters').glob('*'):
            try:counters[f.name]=int(f.read_text().strip())
            except (OSError,ValueError):pass
        sample['hca_port_counters']=counters
        sample['hca_counter_path']='/sys/class/infiniband/rocep172s0/ports/1/counters'
        sample['hca_counter_status']='available' if counters else 'missing-or-unreadable'
        sample['hca_counter_scope']='shared physical HCA; not attributable exclusively to this pod'
        sample['workspace_free_bytes']=__import__('shutil').disk_usage(root).free
        sample['cgroup_cpu_stat']=Path('/sys/fs/cgroup/cpu.stat').read_text()
        sample['cgroup_memory_current_bytes']=int(Path('/sys/fs/cgroup/memory.current').read_text())
        sample['cgroup_memory_events']=Path('/sys/fs/cgroup/memory.events').read_text()
        output.write(json.dumps(sample)+'\n');output.flush()
        time.sleep(max(0,5-(time.monotonic()-start)))
