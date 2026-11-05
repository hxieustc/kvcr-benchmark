"""Observe Python/native call boundaries without adding metadata RPCs."""
import itertools
import json
import threading
import time

class ObservedStore:
    watched={'batch_get_session_start','batch_get_session_end',
             'batch_get_into_multi_buffer_ranges','batch_get_into_multi_buffers',
             'batch_put_from_multi_buffers','batch_put_from'}

    def __init__(self,native,path):
        self.native=native
        self.output=open(path,'a',buffering=65536)
        self.lock=threading.Lock()
        self.ids=itertools.count(1)
        self.sessions={}

    def __getattr__(self,name):
        call=getattr(self.native,name)
        if name not in self.watched:return call
        def observed(*args,**kwargs):
            keys=args[0];key=tuple(keys)
            sizes=args[2] if name.startswith(('batch_get_into','batch_put_from')) else []
            sizes=[sum(x) if isinstance(x,(list,tuple)) else x for x in sizes]
            pieces=sum(len(x) if isinstance(x,(list,tuple)) else 1 for x in args[1]) if sizes else 0
            with self.lock:
                session=self.sessions.get(key)
                if name=='batch_get_session_start':
                    session=next(self.ids);self.sessions[key]=session
            start=time.perf_counter_ns();utc=time.time();result=None;error=None
            try:
                result=call(*args,**kwargs)
                return result
            except BaseException as exc:
                error=type(exc).__name__+': '+str(exc)
                raise
            finally:
                elapsed=time.perf_counter_ns()-start
                values=list(result) if isinstance(result,(list,tuple)) else []
                get=name.startswith('batch_get_into');put=name.startswith('batch_put_from')
                completed=sum(max(v,0) for v in values) if get else sum(size for size,v in zip(sizes,values) if v==0) if put else 0
                success=error is None and (values==sizes if get else values==[0]*len(keys) if put or name=='batch_get_session_start' else True)
                row=dict(utc=utc,method=name,session_id=session,objects=len(keys),pieces=pieces,
                         requested_bytes=sum(sizes),completed_bytes=completed,duration_ns=elapsed,
                         success=success,error=error,first_key=keys[0] if keys else None,
                         scope='Python/native API, local and remote combined; no additional metadata queries')
                with self.lock:
                    self.output.write(json.dumps(row,separators=(',',':'))+'\n')
                    if put:self.output.flush()
                    if name=='batch_get_session_end':
                        self.sessions.pop(key,None);self.output.flush()
        return observed

    def close(self):
        try:return self.native.close()
        finally:
            with self.lock:self.output.flush();self.output.close()
