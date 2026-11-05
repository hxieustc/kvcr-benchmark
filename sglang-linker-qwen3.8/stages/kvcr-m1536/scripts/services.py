#!/usr/bin/env python3
"""Start/inspect/stop only this experiment's recorded service processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path('/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-kvcr-pool-index-capacity-followup')

def read_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split()
        return fields[21] if fields[2] != 'Z' else None
    except FileNotFoundError:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['start','status','wait-ready','stop'])
    p.add_argument('--attempt', required=True)
    a = p.parse_args()
    assert a.attempt.replace('-','').isalnum()
    run = ROOT / 'runs' / a.attempt
    run.mkdir(parents=True,exist_ok=True)
    receipt = run / 'service-pids.json'
    if a.action == 'start':
        assert not receipt.exists(), 'Use a fresh attempt; never silently reload existing workers'
        assert json.loads((ROOT/'model-validation.json').read_text())['verified']
        assert json.loads((ROOT/'bootstrap-receipt.json').read_text())['status']=='pins-and-imports-verified'
        assert (ROOT/'runtime-validation.exit').read_text().strip()=='0'
        assert __import__('shutil').disk_usage(ROOT).free > 100*1024**3,'Workspace headroom below maintenance threshold; do not start duplicate maintenance'
        setup_path = ROOT/'config/setup.deployment.json'
        if not setup_path.exists():setup_path=ROOT/'config/setup.approved.json'
        setup = json.loads(setup_path.read_text())
        (run/'setup.resolved.json').write_text(json.dumps(setup,indent=2)+'\n')
        args = json.loads((ROOT/'config/worker-args.proposed.json').read_text())
        env = dict(os.environ)
        env.pop('NATS_SERVER',None)
        env.pop('KVCR_TRANSFER_DIAGNOSTICS',None)
        env.update(setup['worker_env'])
        env.update({k:v for k,v in setup['transport'].items() if k.startswith(('UCX_','NIXL_'))})
        discovery = run/'discovery'
        discovery.mkdir(exist_ok=True)
        tracked = dict(env={
            'DYN_REQUEST_PLANE':'tcp','DYN_EVENT_PLANE':'zmq','DYN_DISCOVERY_BACKEND':'file',
            'DYN_FILE_KV':str(discovery),'DYN_SYSTEM_HOST':'127.0.0.1','DYN_NAMESPACE':setup['router']['namespace'],
            'HF_HOME':str(ROOT/'cache/huggingface'),'HF_XET_CACHE':str(ROOT/'cache/xet'),
            'TMPDIR':str(ROOT/'tmp'),'RUST_LOG':'info','PYTHONNOUSERSITE':'1',
            'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'cache/torchinductor'),
            'TRITON_CACHE_DIR':str(ROOT/'cache/triton'),
            'FLASHINFER_WORKSPACE_BASE':str(ROOT/'cache/flashinfer'),
            'CUDA_CACHE_PATH':str(ROOT/'cache/cuda'),
            'XDG_CACHE_HOME':str(ROOT/'cache/xdg'),
            'SGLANG_CACHE_DIR':str(ROOT/'cache/sglang'),
        })['env']
        env.update(tracked)
        env['PYTHONPATH']='/workspace/sglang-kvcr-parity-20261031-r1/scripts:/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-kvcr-pool-index-capacity-followup/scripts:/workspace/sglang-kvcr-parity-20261031-r1/software/sglang/python:/workspace/sglang-kvcr-parity-20261031-r1/software/kvcr-pool-index/src'
        tracked['PYTHONPATH']=env['PYTHONPATH']
        python = '/workspace/sglang-kvcr-parity-20261031-r1/env/bin/python'
        env['PATH'] = '/workspace/sglang-kvcr-parity-20261031-r1/env/bin'+':'+env['PATH']
        records = []
        def start(name,argv,extra):
            local = dict(env,**{k:str(v) for k,v in extra.items()})
            affinity=None
            if name=='frontend':
                affinity=set()
                for chunk in Path('/sys/devices/system/node/node0/cpulist').read_text().strip().split(','):
                    lo,*hi=map(int,chunk.split('-'));affinity.update(range(lo,(hi[0] if hi else lo)+1))
                affinity &= os.sched_getaffinity(0)
                assert affinity
            with (run/(name+'.log')).open('wb') as log:
                proc = subprocess.Popen(argv,env=local,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=(lambda:os.sched_setaffinity(0,affinity)) if affinity else None)
            record_env={**tracked,**setup['worker_env'],**{k:v for k,v in env.items() if k.startswith(('UCX_','NIXL_'))},**extra}
            rec = {'name':name,'pid':proc.pid,'start':read_start(proc.pid),'argv':argv,'env':{k:str(v) for k,v in record_env.items()}}
            records.append(rec)
            receipt.write_text(json.dumps(records,indent=2)+'\n')
            print('STARTED',name,proc.pid,flush=True)
        ports = setup['ports']
        start('state-agent',[python,'-m','dynamo.kv_state_agent','--max-slots','4'],{'DYN_SYSTEM_PORT':ports['state_agent_system']})
        start('frontend',[python,'-m','dynamo.frontend','--http-host','127.0.0.1','--http-port',str(ports['router_http']),'--namespace',setup['router']['namespace'],'--router-mode','kv','--router-policy-config',str(ROOT/'config/router-policy.json')],{'DYN_SYSTEM_PORT':ports['router_system']})
        for rank in range(4):
            kv = dict(setup['kvcr_options'],advertise_host=os.environ['POD_IP'],control_port=ports['worker_control'][rank],nixl_port=ports['worker_nixl'][rank])
            kv['residency_log_path']=str(run/f'residency-{rank}.jsonl')
            events = {'publisher':'zmq','format':'dynamo','topic':'','endpoint':f"tcp://*:{ports['worker_kv_events'][rank]}"}
            values = dict(args,**{'model-path':'/workspace/sglang-kvcr-parity-20261031-r1/assets-gb300/model','port':str(ports['worker_http'][rank]),'nccl-port':str(17500+rank),'hicache-storage-backend-extra-config':json.dumps(kv),'kv-events-config':json.dumps(events)})
            argv = [python,'-m','dynamo.sglang','--endpoint',f"dyn://{setup['router']['namespace']}.backend.generate"]
            for key,value in values.items():
                if value is False or value is None: continue
                argv.append('--'+key)
                if value is not True: argv.extend(map(str,value if isinstance(value,list) else [value]))
            metrics = run/'prometheus'/f'worker-{rank}'
            metrics.mkdir(parents=True,exist_ok=True)
            start(f'worker-{rank}',argv,{'CUDA_VISIBLE_DEVICES':str(rank),'DYN_SYSTEM_PORT':ports['worker_system'][rank],'PROMETHEUS_MULTIPROC_DIR':str(metrics)})
        (run/'phase.json').write_text(json.dumps({'state':'loading','utc':time.time()})+'\n')
        return
    records = json.loads(receipt.read_text())
    if a.action == 'stop':
        for rec in records:
            if read_start(rec['pid']) == rec['start']:
                os.killpg(rec['pid'],signal.SIGTERM)
        print('Sent SIGTERM only to this attempt’s recorded process groups; pod retained.')
        return
    deadline = time.monotonic()+3600 if a.action == 'wait-ready' else time.monotonic()
    while True:
        alive = {r['name']:read_start(r['pid'])==r['start'] for r in records}
        if not all(alive.values()):
            raise RuntimeError(f'Service exited; inspect this attempt’s logs: {alive}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:24000/v1/models/nvidia/Qwen3.8-Flash-Next-NVFP4/ready',timeout=5) as response:
                ready = json.load(response).get('ready') is True
        except Exception:
            ready = False
        registered = sum('Successfully registered LLM with runtime config' in (run/f'worker-{i}.log').read_text(errors='replace') for i in range(4))
        if ready and registered == 4:
            (run/'phase.json').write_text(json.dumps({'state':'all-workers-ready','utc':time.time()})+'\n')
            print('ALL_FOUR_WORKERS_READY',flush=True)
            return
        if time.monotonic() >= deadline:
            status={'ready':ready,'registered_workers':registered,'alive':alive}
            if a.action=='wait-ready':raise TimeoutError(status)
            print(json.dumps(status))
            return
        time.sleep(10)

if __name__ == '__main__': main()
