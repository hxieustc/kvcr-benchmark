"""Functional gates for four live native HiCache workers, before AgentX timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.request
import uuid

ROOT=Path('/workspace/sglang-hicache-parity-20261103-r1')
HOST=(128*2**30)//5
BACKEND=128*2**30-HOST

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(case,rank):
    for path in (ROOT/'cases'/case/'raw'/f'backend-worker-{rank}').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            try:yield json.loads(line)
            except json.JSONDecodeError:continue  # An active last partial write is not a completed record.
def answer(text,code):
    assert re.match(r'^\s*'+re.escape(code)+r'\b',text),(code,text)
    assert all(n in (code,'7') for n in re.findall(r'[0-9]+',text)),text

def runtime(case,run):
    backend='kvcr' if case=='kvcr' else 'mooncake'
    workers=[]
    for rank,record in enumerate([r for r in read(run/'service-pids.json') if r['name'].startswith('worker-')]):
        assert record['name']==f'worker-{rank}'
        argv=record['argv'];assert '--enable-hierarchical-cache' in argv and not any('linker' in a for a in argv)
        for flag,value in [('hicache-storage-backend',backend),('hicache-host-memory-mode','cache'),('hicache-write-policy','write_through'),('hicache-storage-prefetch-policy','wait_complete'),('hicache-mem-layout','page_first_direct')]:
            assert argv[argv.index('--'+flag)+1]==value,(rank,flag)
        snapshots=[r for r in rows(case,rank) if r['event']=='cache_geometry']
        assert snapshots,(rank,'Missing actual allocator geometry')
        snapshot=max(snapshots,key=lambda x:len(x['pools']))
        pools=snapshot['pools'];names={p['name'] for p in pools}
        assert {'kv','indexer','mamba'}<=names,(rank,names)
        config=read(ROOT/'setup/model-config.json');config=config.get('text_config',config)
        target_full=config['layer_types'].count('full_attention');draft_full=config['mtp']['layer_types'].count('full_attention')
        expected_layers=target_full+draft_full
        for required in ('kv','indexer'):
            state=next(p for p in pools if p['name']==required)
            assert state['target_layers']==target_full and state['total_layers']==expected_layers,(rank,required,state,'Missing target/draft component layers')
        mamba=next(p for p in pools if p['name']=='mamba')
        assert mamba['target_layers']==config['layer_types'].count('linear_attention'),(rank,mamba)
        components=sorted(names|{'draft_kv'})
        host=snapshot['host_bytes'];be=snapshot['backend_bytes']
        assert 0<host<=HOST and 0<be<=BACKEND and host+be<=128*2**30,(rank,host,be)
        workers.append(dict(rank=rank,components=components,host_allocated_bytes=host,backend_allocated_bytes=be,pools=pools))
    assert len(workers)==4
    write(run/'allocation-proof.json',dict(passed=True,workers=workers,requested_host_bytes=HOST,requested_backend_bytes=BACKEND,rounding='Explicit host budgets rounded down to complete physical pages; KVCR backend pools also floor to pages.'))
    proof=dict(passed=True,backend=backend,linker_enabled=False,host_mode='cache',write_policy='write_through',prefetch_policy='wait_complete',layout='page_first_direct',service_pids_sha256=sha(run/'service-pids.json'),backend_preflight_residue='Checked independently; framework reset does not imply backend deletion.')
    write(run/'native-hicache-proof.json',proof)
    return proof

def peer(case,run):
    from transformers import AutoTokenizer
    backend='kvcr' if case=='kvcr' else 'mooncake'
    out=run/'preflight';out.mkdir(exist_ok=True)
    worker_ids=[]
    for rank in range(4):
        text=re.sub(r'\x1b\[[0-9;]*m','',(run/f'worker-{rank}.log').read_text(errors='replace'))
        ids=set(re.findall(r'Backend admission gate created[^\n]*worker_id=([0-9a-f]+)',text))
        assert len(ids)==1 and 'Successfully registered LLM with runtime config' in text,(rank,ids)
        worker_ids.append(int(ids.pop(),16))
    assert len(set(worker_ids))==4
    tok=AutoTokenizer.from_pretrained(ROOT/'cases'/case/'model',trust_remote_code=True)
    code=str(random.randrange(100000,1000000));filler='The capital of France is Paris. '*300
    prompt=f'Document {uuid.uuid4().hex}. {filler}The access code for vault 7 is {code}. {filler}Question: what is the access code for vault 7? Reply with its digits only.'
    tokens=tok.apply_chat_template([{'role':'system','content':'Copy the requested access code exactly. Reply with the digits only, without explanation, tools or other text.'},{'role':'user','content':prompt}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
    minimum=(len(tokens)-1)//256*256;assert minimum>=4096
    write(out/'input.json',dict(expected_code=code,prompt=prompt,input_token_ids=tokens,minimum_reused_tokens=minimum))
    started=time.time_ns();results=[]
    for index,destination in enumerate([0,0,1,2,3,0]):
        if index:time.sleep(15)
        worker=worker_ids[destination]
        payload=dict(model='nvidia/Qwen3.8-Flash-Next-NVFP4',prompt=tokens,temperature=0,max_tokens=64,stream=True,stream_options={'include_usage':True},nvext=dict(backend_instance_id=worker,dp_rank=0,extra_fields=['worker_id','engine_data','completion_token_ids']))
        request=urllib.request.Request('http://127.0.0.1:24000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=1200) as response:raw=response.read().decode()
        (out/f'response-{index}.sse').write_text(raw)
        events=[json.loads(line[5:]) for line in raw.splitlines() if line.startswith('data:') and line[5:].strip()!='[DONE]'];assert not any('error' in e for e in events)
        ext=[e.get('nvext',{}) for e in events]
        reported=[e['worker_id'] for e in ext if 'worker_id' in e];assert reported and all(x['prefill_worker_id']==x['decode_worker_id']==worker for x in reported)
        hits=[e.get('engine_data',{}).get('kv_cache_hit',{}).get('reused_tokens') for e in ext];hits=[x for x in hits if x is not None];assert hits
        hit=max(hits);assert (hit==0 if index==0 else minimum<=hit<len(tokens)),(index,hit,minimum)
        text=tok.decode([t for e in ext for t in e.get('completion_token_ids',[]) ]);answer(text,code)
        result=dict(request=index,destination_rank=destination,cached_tokens=hit,input_tokens=len(tokens),output=text)
        results.append(result);print(json.dumps(result),flush=True)
        if index==0:
            deadline=time.monotonic()+180
            while True:
                puts=[r for r in rows(case,0) if r['event']=='backend_api' and r['operation']=='PUT' and r['wall_ns']>=started and r['success']]
                complete={p['name'] for r in puts for p in r.get('pools',[])}
                if {'kv','indexer','mamba'}<=complete:break
                assert time.monotonic()<deadline,('No complete source export before peer read',complete)
                time.sleep(.2)
    time.sleep(2)
    for rank in range(1,4):
        reads=[r for r in rows(case,rank) if r['event']=='backend_api' and r['operation']=='GET' and r['wall_ns']>=started]
        assert any(r['success'] and r['completed_bytes']>1_000_000 for r in reads),(rank,'No successful real backend GET')
    if backend=='kvcr':
        source=[r for r in rows(case,0) if r['event']=='nixl_transfer' and r['wall_ns']>=started and r['remote_agent']!=r['local_agent'] and r['success']]
        assert len({r['remote_agent'] for r in source})>=3,'No native WRITE to all three peer agents'
        native=read(ROOT/'qualification/kvcr-hicache-peer.json');assert native['passed'] is True
        native_scope='Independent native hybrid adapter fixture full-byte correctness + live 4-worker semantic peer reads; not bitwise GPU/logit equivalence'
    else:
        ready=read(ROOT/'setup/harness-ready.json');native=read(ROOT/ready['native_rdma_qualification'])['children']['mooncake'];assert native['exit']==0 and native['result']['status']=='passed'
        native_scope='Independent native RDMA GPU fixture full-byte correctness + live HiCache semantic peer reads; native GET API byte counts include local+remote portions'
    runtime(case,run)
    proof=dict(passed=True,backend=backend,linker_enabled=False,inventory_protocol='sglang_hicache_v1',remote_peer_ranks=[1,2,3],full_bytes_correct=True,full_bytes_correct_scope=native_scope,components=['kv','indexer','mamba','draft_kv'],requests=results,started_wall_ns=started,note='Synthetic accept length is stochastic; exact leading vault code and peer cache reuse are required, token-ID equality is not used.')
    write(out/'validation.json',proof);return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['kvcr','mooncake-random','mooncake-local-first'],required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['peer','post-clear'],required=True);a=p.parse_args()
    assert a.run.resolve().is_relative_to(ROOT/'cases'/a.case/'runs')
    print(json.dumps(peer(a.case,a.run) if a.phase=='peer' else runtime(a.case,a.run)),flush=True)
