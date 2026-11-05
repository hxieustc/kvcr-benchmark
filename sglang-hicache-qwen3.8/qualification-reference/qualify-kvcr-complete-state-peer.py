"""Native separate-agent restore of real Mamba/PLE checkpoint components."""
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import torch

ROOT=Path('/workspace/sglang-hicache-parity-20261103-r1')

def worker(rank,commands,results):
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig,HiCacheStorageExtraInfo,PoolName,PoolHitPolicy,PoolTransfer
    from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore
    source=ROOT/'software/sglang/test/registered/unit/mem_cache/test_hicache_mamba_slot_side_states.py'
    spec=importlib.util.spec_from_file_location('real_mamba_fixture',source);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    mamba,_=fixture._make_pools(True);mamba.lock=threading.RLock();mamba.clear()
    class Primary:
        page_size=1;size=4;layout='page_first_direct';dtype=torch.uint8
        def __init__(self):self.kv_buffer=torch.zeros((4,8192),dtype=torch.uint8)
        def get_page_buffer_meta(self,indices):return [self.kv_buffer[i].data_ptr() for i in indices.tolist()],[8192]*len(indices)
    primary=Primary()
    config=HiCacheStorageConfig(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,is_mla_model=False,enable_storage_metrics=True,is_page_first_layout=True,model_name='real-complete-mamba-qualification',dp_rank=rank,extra_config={'local_dram_bytes':1<<20,'control_port':35600,'nixl_listen_port':36600,'control_advertise_host':'127.0.0.1','operation_timeout_ms':30000,'startup_timeout_seconds':120})
    backend=KVCRStore(config,primary);backend.register_mem_host_pool_v2(mamba,PoolName.MAMBA)
    pins=[];old_pin=backend._pinning.request_pin
    def pin(keys):pins.append(len(keys));return old_pin(keys)
    backend._pinning.request_pin=pin
    key=hashlib.sha256(b'real-complete-checkpoint').hexdigest()
    def transfer(slot):return [PoolTransfer(PoolName.KV,host_indices=torch.tensor([slot]),keys=[key]),PoolTransfer(PoolName.MAMBA,host_indices=torch.tensor([slot]),keys=[key],hit_policy=PoolHitPolicy.TRAILING_PAGES)]
    try:
        if rank==0:
            primary.kv_buffer[0].fill_(31)
            for i,tensor in enumerate(mamba._iter_page_tensors(0)):tensor.fill_((1<<40)+i if tensor.dtype==torch.int64 else i+1.25)
            written=backend.batch_set_v2(transfer(0));assert all(all(x) for x in written.values()),written
            results.put(dict(event='source-ready',endpoint=backend._control_endpoint))
            assert commands.get(timeout=180)=='stop'
        else:
            endpoint=commands.get(timeout=120)
            info=HiCacheStorageExtraInfo(extra_info={'kv_hints':{'protocol_version':'0.1','message_id':'complete-state','actions':[{'action_id':'copy','action_type':'kv.fetch','action_version':'1.0','payload':{'source_control_endpoint':endpoint,'block_hashes':[int(key[:16],16)],'mode':'copy'}}]}})
            restored=backend.batch_get_v2(transfer(2),extra_info=info);assert all(all(x) for x in restored.values()),restored
            assert torch.all(primary.kv_buffer[2]==31).item()
            checks=[]
            for i,tensor in enumerate(mamba._iter_page_tensors(2)):
                expected=(1<<40)+i if tensor.dtype==torch.int64 else i+1.25
                assert torch.all(tensor==expected).item(),(i,tensor.dtype)
                raw=tensor.contiguous().view(torch.uint8).numpy().tobytes()
                checks.append(dict(component=i,dtype=str(tensor.dtype),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            assert sum(c['bytes'] for c in checks)==mamba.size_per_token and len(checks)==4
            backend.get_stats()
            results.put(dict(event='destination-verified',components=checks,primary_bytes=8192,complete_checkpoint_bytes=mamba.size_per_token))
        assert not pins,pins
        backend.close();results.put(dict(event='closed',rank=rank,framework_pin_requests=len(pins)))
    except BaseException as error:
        results.put(dict(event='error',rank=rank,type=type(error).__name__,error=str(error)));raise

if __name__=='__main__':
    assert os.environ.get('UCX_NET_DEVICES')=='rocep190s0:1' and os.environ.get('NIXL_UCX_SGL_ENABLE')=='1'
    ctx=mp.get_context('spawn');source=ctx.Queue();target=ctx.Queue();results=ctx.Queue();events=[]
    ps=[ctx.Process(target=worker,args=(0,source,results)),ctx.Process(target=worker,args=(1,target,results))]
    ps[0].start();first=results.get(timeout=150);events.append(first);assert first['event']=='source-ready',first
    ps[1].start();target.put(first['endpoint']);checked=results.get(timeout=150);events.append(checked);assert checked['event']=='destination-verified',checked
    source.put('stop')
    for process in ps:process.join(90);assert process.exitcode==0,(process.pid,process.exitcode)
    while True:
        try:events.append(results.get(timeout=1))
        except queue.Empty:break
    receipt=dict(passed=True,events=events,scope='Actual HiCache MambaPoolHost mixed BF16 PLE/int64 history; separate native agents, complete indexed component payloads, same-node assigned UCX RNIC; no GPU model/logit equivalence claim')
    (ROOT/'qualification/kvcr-complete-state-peer.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
