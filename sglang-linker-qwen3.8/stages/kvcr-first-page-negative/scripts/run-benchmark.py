#!/usr/bin/env python3
"""Run validated AgentX replay, telemetry collection and analysis in the pod."""
import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-kvcr-first-page-negative')

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--reuse-preflight-from');a=p.parse_args()
    assert a.attempt.replace('-','').isalnum()
    run=ROOT/'runs'/a.attempt
    run.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'benchmark.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'driver.exit').exists(),'Never overwrite a closed replay; select a new attempt'
    env=dict(os.environ)
    client_tmp=Path('/workspace/t/gb3k');client_tmp.mkdir(parents=True,exist_ok=True)
    env.update({'HF_HOME':str(ROOT/'cache/huggingface'),'HF_XET_CACHE':str(ROOT/'cache/xet'),'TMPDIR':str(client_tmp),'MPLCONFIGDIR':str(ROOT/'cache/matplotlib'),'TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':'1'})
    (run/'client-runtime-env.json').write_text(json.dumps({k:env[k] for k in ('HF_HOME','HF_XET_CACHE','TMPDIR','MPLCONFIGDIR','TOKENIZERS_PARALLELISM','PYTHONNOUSERSITE')},indent=2)+'\n')
    env['PATH']=str(ROOT/'clientenv/bin')+':'+env['PATH']
    python='/workspace/sglang-kvcr-parity-20261031-r1/env/bin/python'
    def phase(name,**fields):
        (run/'phase.json').write_text(json.dumps({'state':name,'utc':time.time(),**fields})+'\n')
        print(name,flush=True)
    rc=1
    collector=None
    try:
        phase('waiting-for-four-workers')
        subprocess.run([python,str(ROOT/'scripts/services.py'),'wait-ready','--attempt',a.attempt],env=env,check=True)
        if a.reuse_preflight_from:
            assert a.reuse_preflight_from.replace('-','').isalnum()
            old=ROOT/'runs'/a.reuse_preflight_from
            assert not (old/'benchmark-command.json').exists()
            assert (old/'service-pids.json').read_bytes()==(run/'service-pids.json').read_bytes()
            receipt=json.loads((old/'reuse-receipt.json').read_text())
            assert receipt['preflight_sha256']==__import__('hashlib').sha256((ROOT/'scripts/validate-reuse.py').read_bytes()).hexdigest()
            assert json.loads((old/'preflight/validation.json').read_text())['passed']
            shutil.copytree(old/'preflight',run/'preflight')
            (run/'preflight-reuse-receipt.json').write_text(json.dumps({'from_attempt':a.reuse_preflight_from,'same_live_services':True,'validation_sha256':__import__('hashlib').sha256((old/'preflight/validation.json').read_bytes()).hexdigest(),'reason':'Prior attempt passed all peer checks and stopped before AgentX on an incorrect cache-clear HTTP route.'},indent=2)+'\n')
            phase('reusing-successful-peer-validation',from_attempt=a.reuse_preflight_from)
        else:
            phase('validating-peer-reuse')
            with (run/'preflight.log').open('wb') as log:
                subprocess.run([python,str(ROOT/'scripts/validate-reuse.py'),'--attempt',a.attempt],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((run/'preflight/validation.json').read_text())['passed']
        subprocess.run([python,str(ROOT/'scripts/verify-cache-mode.py'),'--attempt',a.attempt],env=env,check=True)
        phase('clearing-validation-caches')
        with (run/'validation-cache-clear.log').open('wb') as log:
            subprocess.run([python,str(ROOT/'scripts/clear-validation-cache.py'),'--attempt',a.attempt],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert __import__('shutil').disk_usage(ROOT).free > 8*1024**3,'Insufficient result headroom'
        with (run/'collector.log').open('wb') as log:
            collector=subprocess.Popen([python,str(ROOT/'scripts/collect.py'),'--attempt',a.attempt],env=env,stdout=log,stderr=subprocess.STDOUT)
        argv=json.loads((ROOT/'config/client-argv.approved.json').read_text())
        argv[0]=str(ROOT/'clientenv/bin/agentx')
        argv=[x.replace('${VERIFIED_MODEL_PATH}',str(ROOT/'model')) for x in argv]
        argv[argv.index('--output-artifact-dir')+1]=str(run/'aiperf_artifacts')
        (run/'benchmark-command.json').write_text(json.dumps(argv,indent=2)+'\n')
        phase('agentx-warmup-and-profiling',concurrency=48,warmup_extra_requests_per_lane=5,profiling_seconds=1800)
        with (run/'benchmark.log').open('wb') as log:
            client=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
        (run/'client.exit').write_text(str(client.returncode)+'\n')
        assert client.returncode==0,f'AgentX exited {client.returncode}; raw logs preserved'
        export=json.loads((run/'aiperf_artifacts/profile_export_aiperf.json').read_text())
        config=export['input_config'];profile=config['phases'][0]
        assert config['scenario']=='agentx'
        assert profile['duration']==1800 and profile['concurrency']==48
        assert profile['warmup_requests_per_lane']==5
        assert config['datasets'][0]['dataset']=='semianalysis_cc_traces_weka_062126_256k'
        phase('closing-raw-capture')
        collector.terminate();collector_rc=collector.wait(timeout=20);collector=None
        (run/'collector.exit').write_text(str(collector_rc)+'\n')
        assert collector_rc==0, f'Raw telemetry collector exited {collector_rc}; raw logs preserved'
        rc=0
        phase('capture-complete-awaiting-local-analysis',client_exit=0)
    except Exception as error:
        phase('failed-preserving-evidence',error=str(error))
        raise
    finally:
        if collector is not None:
            collector.terminate();collector_rc=collector.wait(timeout=20)
            (run/'collector.exit').write_text(str(collector_rc)+'\n')
        (run/'driver.exit').write_text(str(rc)+'\n')
        # Service processes remain available for inspection/repeat; never delete the pod.

if __name__=='__main__':main()
