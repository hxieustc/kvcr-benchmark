"""Native two-process KVCR HiCache peer restore, including hybrid side pools."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time

ROOT=Path('/workspace/sglang-hicache-parity-20261103-r1')

def worker(rank,commands,results):
    import torch
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig,HiCacheStorageExtraInfo,PoolName,PoolHitPolicy,PoolTransfer
    from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore
    class Pool:
        def __init__(self,sizes,page_size):
            self.page_size=page_size;self.size=8*page_size;self.layout='page_first_direct';self.dtype=torch.uint8
            self.buffers=[torch.zeros((8,n),dtype=torch.uint8) for n in sizes];self.kv_buffer=self.buffers[0]
        def get_page_buffer_meta(self,indices):
            ptrs=[];sizes=[]
            for start in indices.tolist()[::self.page_size]:
                for buf in self.buffers:
                    ptrs.append(buf[start//self.page_size].data_ptr());sizes.append(buf.shape[1])
            return ptrs,sizes
    pools={PoolName.KV:Pool((8192,3072),2),PoolName.INDEXER:Pool((1024,),2),PoolName.DRAFT:Pool((2048,512),2),PoolName.MAMBA:Pool((9216,3328),1)}
    config=HiCacheStorageConfig(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,is_mla_model=False,enable_storage_metrics=True,is_page_first_layout=True,model_name='native-hybrid-peer-fixture',dp_rank=rank,extra_config={'local_dram_bytes':64*2**20,'control_port':35600,'nixl_listen_port':36600,'control_advertise_host':'127.0.0.1','operation_timeout_ms':30000,'startup_timeout_seconds':120})
    backend=KVCRStore(config,pools[PoolName.KV])
    for name,pool in pools.items():
        if name!=PoolName.KV:backend.register_mem_host_pool_v2(pool,name)
    pins=[]
    old=backend._pinning.request_pin
    def forbidden_pin(keys):pins.append(len(keys));return old(keys)
    backend._pinning.request_pin=forbidden_pin
    keys=[hashlib.sha256(f'qualified-page-{n}'.encode()).hexdigest() for n in range(4)]
    def transfers(destination):
        out=[]
        for name,pool in pools.items():
            count=1 if name==PoolName.MAMBA else 4
            start=4*pool.page_size if destination else 0
            out.append(PoolTransfer(name,host_indices=torch.arange(start,start+count*pool.page_size),keys=keys[-1:] if name==PoolName.MAMBA else keys,hit_policy=PoolHitPolicy.TRAILING_PAGES if name==PoolName.MAMBA else PoolHitPolicy.ALL_PAGES))
        return out
    try:
        if rank==0:
            for p,(name,pool) in enumerate(pools.items()):
                for c,buf in enumerate(pool.buffers):
                    for n in range(1 if name==PoolName.MAMBA else 4):buf[n].fill_(31+17*p+3*c+n)
            written=backend.batch_set_v2(transfers(False))
            assert all(all(v) for v in written.values()),written
            backend.get_stats()
            results.put({'event':'source-ready','endpoint':backend._control_endpoint,'native_agent':backend._agent_name})
            assert commands.get(timeout=180)=='stop'
        else:
            endpoint=commands.get(timeout=120)
            hint={'protocol_version':'0.1','message_id':'native-peer-fixture','actions':[{'action_id':'restore','action_type':'kv.fetch','action_version':'1.0','payload':{'source_control_endpoint':endpoint,'block_hashes':[int(k[:16],16) for k in keys],'mode':'copy'}}]}
            info=HiCacheStorageExtraInfo(extra_info={'kv_hints':hint})
            restored=backend.batch_get_v2(transfers(True),extra_info=info)
            assert all(all(v) for v in restored.values()),restored
            backend.get_stats()
            checks=[]
            for p,(name,pool) in enumerate(pools.items()):
                for c,buf in enumerate(pool.buffers):
                    for n in range(1 if name==PoolName.MAMBA else 4):
                        actual=buf[4+n];assert torch.all(actual==31+17*p+3*c+n).item(),(name,c,n)
                        checks.append({'pool':str(name),'component':c,'page':n,'bytes':actual.numel(),'sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
            results.put({'event':'destination-verified','checks':checks,'bytes_verified':sum(c['bytes'] for c in checks),'framework_pin_requests':len(pins)})
        assert not pins,pins
        backend.close()
        results.put({'event':'closed','rank':rank,'framework_pin_requests':len(pins)})
    except BaseException as error:
        results.put({'event':'error','rank':rank,'type':type(error).__name__,'error':str(error)})
        raise

if __name__=='__main__':
    assert os.environ.get('UCX_NET_DEVICES')=='rocep190s0:1'
    assert os.environ.get('NIXL_UCX_SGL_ENABLE')=='1'
    ctx=mp.get_context('spawn');source=ctx.Queue();target=ctx.Queue();results=ctx.Queue()
    ps=[ctx.Process(target=worker,args=(0,source,results)),ctx.Process(target=worker,args=(1,target,results))]
    events=[];ps[0].start()
    first=results.get(timeout=150);events.append(first);assert first['event']=='source-ready',first
    ps[1].start();target.put(first['endpoint'])
    verified=results.get(timeout=150);events.append(verified);assert verified['event']=='destination-verified',verified
    source.put('stop')
    for p in ps:p.join(90);assert p.exitcode==0,{'pid':p.pid,'exit':p.exitcode}
    while True:
        try:events.append(results.get(timeout=1))
        except queue.Empty:break
    receipt={'passed':True,'events':events,'scope':'Two separate native KVCR agents; explicit remote hint; HiCache host->backend->remote host, full component bytes verified. Same-node RNIC path, not cross-node speedup or GPU restore proof.'}
    (ROOT/'qualification/kvcr-hicache-peer-r2.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
