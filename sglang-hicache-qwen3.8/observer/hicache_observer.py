"""Opt-in HiCache API/native timing capture; never reads payloads or queries metadata."""
import atexit
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import re
from pathlib import Path
import sys
import threading
import time
from bounded_sink import ProcessSinks

TARGETS={
 'sglang.srt.mem_cache.storage.kvcr.kvcr_store':'kvcr',
 'sglang.srt.mem_cache.storage.mooncake_store.mooncake_store':'mooncake',
 'kvcr.runtime.progress':'progress',
 'kvcr.backends.remote_ops':'source',
 'kvcr.backends.dangling_ops':'cleanup',
}

def wrap(cls,name,callback):
    original=getattr(cls,name)
    @functools.wraps(original)
    def call(self,*args,**kwargs):return callback(original,self,args,kwargs)
    setattr(cls,name,call)

def install():
    output=Path(os.environ['HICACHE_BENCH_OUTPUT'])
    manifest=json.loads((Path(__file__).parent/'manifest.json').read_text())
    sinks=ProcessSinks(output,allowed_root=Path('/workspace'),probes={})
    emit=sinks.emit
    context=threading.local()
    atexit.register(sinks.close)

    def physical_pools(obj):
        pools=dict(getattr(obj,'registered_pools',{}))
        primary=getattr(obj,'mem_pool_host',None)
        if primary is not None and getattr(primary,'kv_buffer',None) is not None:
            pools.setdefault('kv',primary)
        return pools

    def pools_snapshot(obj,backend):
        pools=[];seen=set()
        for name,pool in physical_pools(obj).items():
            if id(pool) in seen or getattr(pool,'kv_buffer',None) is None:continue
            seen.add(id(pool))
            from sglang.srt.mem_cache.storage.kvcr.host_layout import describe_pool
            layout=describe_pool(str(name),pool)
            pools.append(dict(name=str(name),slots=int(pool.size//pool.page_size),page_tokens=int(pool.page_size),
                              bytes_per_slot=int(layout.bytes_per_page),
                              allocated_bytes=int((pool.size//pool.page_size)*layout.bytes_per_page),
                              target_layers=int(getattr(pool,'target_layer_num',len(getattr(getattr(pool,'device_pool',None),'qsa_compressed_k_buffer_pool',())) if str(name)=='indexer' else getattr(pool,'num_mamba_layers',0)-len(getattr(pool,'mtp_draft_buffer_infos',())))),
                              total_layers=int(getattr(pool,'layer_num',getattr(pool,'num_mamba_layers',0))),pool_class=type(pool).__module__+'.'+type(pool).__name__,
                              draft_mamba_pools=len(getattr(pool,'mtp_draft_buffer_infos',()))))
        emit('cache_geometry',backend=backend,pools=pools,host_bytes=sum(p['allocated_bytes'] for p in pools),
             backend_bytes=sum(len(x) for x in getattr(obj,'_maps',[])) if backend=='kvcr' else int(obj.config.global_segment_size))

    def patch(module,kind):
        if kind=='kvcr':
            cls=module.KVCRStore
            def start(original,obj,args,kwargs):
                cold=obj._runner is None
                result=original(obj,*args,**kwargs)
                if cold:pools_snapshot(obj,'kvcr')
                return result
            wrap(cls,'_start',start)
            def io(original,obj,args,kwargs):
                transfers=args[0];start=time.perf_counter_ns();wall=time.time_ns();result=None;error=None
                try:
                    result=original(obj,*args,**kwargs);return result
                except BaseException as exc:error=type(exc).__name__;raise
                finally:
                    elapsed=time.perf_counter_ns()-start
                    requested=completed=pieces=logical=0;per_pool=[]
                    for transfer in transfers:
                        name=str(transfer.name);layout=obj._layouts.get(name)
                        if layout is None:continue
                        n=len(transfer.keys or []);b=int(layout.bytes_per_page)
                        flags=(result or {}).get(transfer.name,[])
                        good=sum(bool(v) for v in flags)
                        requested+=n*b;completed+=good*b;pieces+=n*len(layout.components)
                        if name=='kv':logical=n
                        per_pool.append(dict(name=name,pages=n,completed_pages=good,bytes_per_page=b))
                    emit('backend_api',backend='kvcr',operation='PUT' if kwargs['is_set'] else 'GET',wall_ns=wall,
                         duration_ns=elapsed,requested_bytes=requested,completed_bytes=completed,logical_blocks=logical,
                         descriptors=pieces,pools=per_pool,success=error is None and completed==requested,error=error,
                         scope='HiCache adapter call including lock, preparation, KVCR completion; local+remote combined')
            wrap(cls,'_io',io)
        elif kind=='mooncake':
            cls=module.MooncakeStore
            def hybrid(original,obj,args,kwargs):
                previous=getattr(context,'moon_pools',None)
                context.moon_pools=[{'name':str(t.name)} for t in args[0]]
                try:return original(obj,*args,**kwargs)
                finally:context.moon_pools=previous
            wrap(cls,'_batch_io_v2',hybrid)
            def register(original,obj,args,kwargs):
                result=original(obj,*args,**kwargs);pools_snapshot(obj,'mooncake');return result
            wrap(cls,'register_mem_host_pool_v2',register)
            def native(original,obj,args,kwargs):
                keys,ptrs,sizes=args[:3]
                sizes=[sum(s) if isinstance(s,(tuple,list)) else int(s) for s in sizes]
                start=time.perf_counter_ns();wall=time.time_ns();result=None;error=None
                try:
                    result=original(obj,*args,**kwargs);return result
                except BaseException as exc:error=type(exc).__name__;raise
                finally:
                    elapsed=time.perf_counter_ns()-start
                    put=original.__name__.startswith('_put')
                    values=list(result) if result is not None else []
                    completed=sum(s for s,v in zip(sizes,values) if v==0) if put else sum(max(int(v),0) for v in values)
                    page_ids=[re.search(r'(?:^|_)([0-9a-f]{64})(?:_|$)',key) for key in keys]
                    logical=len({m.group(1) for m in page_ids}) if all(page_ids) else None
                    emit('backend_api',backend='mooncake',pools=getattr(context,'moon_pools',None) or [{'name':'kv'}],logical_blocks=logical,operation='PUT' if put else 'GET',wall_ns=wall,
                         duration_ns=elapsed,requested_bytes=sum(sizes),completed_bytes=completed,objects=len(keys),
                         descriptors=sum(len(p) if isinstance(p,(tuple,list)) else 1 for p in ptrs),
                         success=error is None and len(values)==len(sizes) and (values==[0]*len(sizes) if put else values==sizes),
                         error=error,scope='Batched Mooncake native API; local+remote combined, metadata included; physical objects are not logical blocks')
            wrap(cls,'_put_batch_zero_copy_impl',native);wrap(cls,'_get_batch_zero_copy_impl',native)
        elif kind=='progress':
            cls=module._KVCRProgress
            def submit(original,obj,args,kwargs):
                # This only requests the native SDK's existing telemetry after DONE.
                kwargs=dict(kwargs,capture_telemetry=True)
                started=time.perf_counter_ns();wall=time.time_ns()
                result=original(obj,*args,**kwargs)
                ident,submitted=result
                records=getattr(obj,'_bench_transfers',None)
                if records is None:obj._bench_transfers=records={}
                sizes=[int(ref.size) if isinstance(ref,module._MemDescriptor) else int(module._resolve_region(ref,obj._memory_regions,obj.nixl_agent_name).size) for ref in args[1]]
                source=getattr(context,'source',None)
                if source is not None:source._bench_native_id=ident
                records[ident]=dict(transfer_id=ident,source_operation=int(source.op_id[1]) if source is not None else None,
                                    physical_blocks=len(source.source_keys) if source is not None else None,
                                    logical_blocks=len({key.split(b'#',1)[0] for key in source.source_keys}) if source is not None else None,
                                    wall_ns=wall,start_ns=started,submit_ns=time.perf_counter_ns()-started,
                                    requested_bytes=sum(sizes),descriptors=len(sizes),remote_agent=kwargs['remote_side_agent'],
                                    local_agent=obj.nixl_agent_name,operation=args[0],submitted=bool(submitted))
                return result
            def poll(original,obj,args,kwargs):
                result=original(obj,*args,**kwargs)
                if result is not None:
                    record=getattr(obj,'_bench_transfers',{}).pop(args[0],None)
                    if record:
                        record['duration_ns']=time.perf_counter_ns()-record.pop('start_ns')
                        success,telemetry=result
                        native={name:float(getattr(telemetry,name)) for name in ('totalBytes','postDuration','xferDuration','descCount') if telemetry is not None and hasattr(telemetry,name)}
                        emit('nixl_transfer',**record,success=bool(success),native=native,
                             scope='Submit entry to observed DONE plus native handle cleanup; SDK native durations in microseconds when present')
                return result
            wrap(cls,'submit_transfer',submit);wrap(cls,'poll_transfer',poll)
        elif kind=='source':
            def created(original,obj,args,kwargs):
                result=original(obj,*args,**kwargs)
                obj._bench_created_ns=time.perf_counter_ns();obj._bench_created_wall_ns=time.time_ns()
                return result
            wrap(module._SourceWriteOp,'__init__',created)
            def advance(original,obj,args,kwargs):
                context.source=obj
                try:return original(obj,*args,**kwargs)
                finally:context.source=None
            wrap(module._SourceWriteOp,'progress',advance)
        elif kind=='cleanup':
            def cleanup(original,obj,args,kwargs):
                op=args[0];started=time.perf_counter_ns()
                result=original(obj,*args,**kwargs)
                if hasattr(op,'_bench_created_ns'):
                    emit('kvcr_source_operation',wall_ns=op._bench_created_wall_ns,
                         duration_ns=time.perf_counter_ns()-op._bench_created_ns,cleanup_ns=time.perf_counter_ns()-started,
                         source_operation=int(op.op_id[1]),target_operation=op.op_handle,remote_agent=op.remote_agent,
                         transfer_id=getattr(op,'_bench_native_id',None),physical_blocks=len(op.source_keys),
                         logical_blocks=len({key.split(b'#',1)[0] for key in op.source_keys}),completed_bytes=op._backend._kvcr._descriptor_bytes(ref for refs in op.src_descriptors for ref in refs) if op.success else 0,
                         descriptors=sum(len(refs) for refs in op.src_descriptors),success=bool(op.success),state=str(op.state),
                         scope='Source-write operation construction through cleanup; excludes earlier request/control work')
                return result
            wrap(module._DanglingOps,'finish_source',cleanup)
        if kind in ('kvcr','mooncake'):
            def state(original,obj,args,kwargs):
                result=original(obj,*args,**kwargs)
                now=time.monotonic()
                if now-getattr(obj,'_bench_last_state',0)<5:return result
                obj._bench_last_state=now
                host=[]
                for name,pool in physical_pools(obj).items():
                    available=getattr(pool,'available_size',None)
                    if not callable(available) or getattr(pool,'kv_buffer',None) is None:continue
                    host.append(dict(name=str(name),capacity_tokens=int(pool.size),available_tokens=int(available())))
                backend=[]
                runner=getattr(obj,'_runner',None);core=getattr(runner,'_core',None)
                dram=getattr(core,'_local_dram',None)
                if dram is not None:
                    pools=dram._pools
                    for label,(_,length,slot_bytes) in pools._regions.items():
                        free=len(pools._free_slots[label]);capacity=length//slot_bytes
                        backend.append(dict(label=label,capacity_slots=capacity,free_slots=free,slot_bytes=slot_bytes,allocated_bytes=(capacity-free)*slot_bytes))
                emit('cache_occupancy',backend=kind,host_pools=host,backend_pools=backend,
                     scope='Allocator slot accounting, not RSS; indexer host ownership follows primary KV slots; periodic approximate snapshot')
                return result
            wrap(cls,'get_stats',state)
        emit('module_installed',module=module.__name__,source_sha256=manifest[module.__name__])

    class Loader(importlib.abc.Loader):
        def __init__(self,original,kind):self.original,self.kind=original,kind
        def create_module(self,spec):
            method=getattr(self.original,'create_module',None)
            return method(spec) if method else None
        def exec_module(self,module):self.original.exec_module(module);patch(module,self.kind)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname not in TARGETS:return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
            if spec is None or not spec.origin or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()!=manifest[fullname]:
                raise ImportError('HiCache observer source mismatch: '+fullname)
            spec.loader=Loader(spec.loader,TARGETS[fullname]);return spec
    sys.meta_path.insert(0,Finder())
