"""Opt-in bounded benchmark observations. No serving APIs or policy changes."""
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time


def primitive(value, depth=0):
    if depth > 6:
        return False
    if value is None or type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is str:
        return len(value) <= 8192
    if type(value) in (list, tuple):
        return len(value) <= 256 and all(primitive(v, depth+1) for v in value)
    if type(value) is dict:
        return len(value) <= 128 and all(type(k) is str and primitive(v, depth+1) for k,v in value.items())
    return False


class Sink:
    def __init__(self, directory, *, allowed_root=Path('/workspace'), max_queue=4096, max_bytes=256*2**20, external_drops=lambda:0):
        directory=Path(directory).absolute(); root=Path(allowed_root).resolve()
        if directory.resolve()!=directory or not directory.is_relative_to(root):
            raise ValueError('Diagnostic output must remain inside the workspace root')
        if max_queue<=0 or max_bytes<512:
            raise ValueError('Invalid diagnostic bound')
        directory.mkdir(parents=True,exist_ok=True)
        self.path=directory/f'observer-{os.getpid()}-{time.time_ns()}.jsonl'
        self._file=self.path.open('xb',buffering=0)
        self.owner_pid=os.getpid()
        self._queue=queue.Queue(maxsize=max_queue); self._stop=threading.Event()
        self.max_bytes=max_bytes; self.bytes_written=0
        self.dropped_records=0; self.invalid_records=0; self.writer_error=None
        self.external_drops=external_drops
        self._thread=threading.Thread(target=self._write,name='kvcr-observer',daemon=True)
        self._thread.start()

    def emit(self, event, **fields):
        # Validate without repr(), tensor conversion or keeping SDK object references.
        if type(event) is not str or not primitive(fields):
            self.invalid_records+=1; return
        if self._stop.is_set() or self.writer_error is not None:
            self.dropped_records+=1; return
        record=dict(fields,event=event,emit_wall_ns=time.time_ns(),pid=os.getpid(),tid=threading.get_native_id())
        record.setdefault('wall_ns',record['emit_wall_ns'])
        record.setdefault('monotonic_ns',time.monotonic_ns())
        record.setdefault('thread_cpu_ns',time.thread_time_ns())
        # Detach mutable primitive containers: callers may otherwise mutate queued metadata.
        # Serialize on the writer; recursively snapshot here without touching live objects.
        def snapshot(v):
            if type(v) is dict:return {k:snapshot(x) for k,x in v.items()}
            if type(v) in (list,tuple):return tuple(snapshot(x) for x in v)
            return v
        try:self._queue.put_nowait(snapshot(record))
        except queue.Full:self.dropped_records+=1

    def _write(self):
        flushed=time.monotonic()
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:record=self._queue.get(timeout=.1)
                except queue.Empty:record=None
                if record is not None:
                    data=json.dumps(record,separators=(',',':'),allow_nan=False)+'\n'
                    size=len(data.encode())
                    if self.bytes_written+size<=self.max_bytes:
                        self._file.write(data.encode());self.bytes_written+=size
                    else:self.dropped_records+=1
                if time.monotonic()-flushed>=1:
                    heartbeat=(json.dumps(dict(event='observer_heartbeat',wall_ns=time.time_ns(),
                        pid=os.getpid(),dropped_records=self.dropped_records+self.external_drops(),
                        invalid_records=self.invalid_records,queued_records=self._queue.qsize(),
                        bytes_written=self.bytes_written))+'\n').encode()
                    if self.bytes_written+len(heartbeat)<=self.max_bytes:
                        self._file.write(heartbeat);self.bytes_written+=len(heartbeat)
                    self._file.flush();flushed=time.monotonic()
            self._file.write(json.dumps(dict(event='observer_closed',dropped_records=self.dropped_records+self.external_drops(),
                invalid_records=self.invalid_records,bytes_written=self.bytes_written,pid=os.getpid())).encode()+b'\n')
            self._file.flush()
        except Exception as error:
            self.writer_error=type(error).__name__
        finally:self._file.close()

    def close(self):
        if os.getpid()!=self.owner_pid:return False
        self._stop.set();self._thread.join(timeout=5)
        return not self._thread.is_alive() and self.writer_error is None


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class ProcessSinks:
    """One lazy writer per process; contention drops rather than waits.

    Fork resets only runtime bookkeeping, never flushing/joining the parent's
    writer. Source probe tracking is not reset here; child summaries say so.
    """
    def __init__(self, output, *, allowed_root, probes):
        self.output,self.allowed_root,self.probes=output,allowed_root,probes
        self.installed={};self.inherited_modules={};self.forked=False
        self._reset()
        if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=self._after_fork)

    def _reset(self):
        self.owner_pid=os.getpid();self.init_lock=threading.Lock()
        self.sink=None;self.initialization_drops=0;self.closed=False

    def _after_fork(self):
        self.inherited_modules=dict(self.installed);self.forked=True
        self._reset()  # Do not touch inherited queue/Event/Thread/file locks.

    def emit(self, event, **fields):
        if self.closed:
            self.initialization_drops+=1;return
        try:
            sink=self.sink
            if sink is None:
                if not self.init_lock.acquire(blocking=False):
                    self.initialization_drops+=1;return
                try:
                    sink=self.sink
                    if sink is None:
                        sink=Sink(self.output,allowed_root=self.allowed_root,
                                  external_drops=lambda:self.initialization_drops)
                        sink.emit('observer_startup',inherited_modules=dict(self.inherited_modules))
                        self.sink=sink
                finally:self.init_lock.release()
            sink.emit(event,**fields)
        except Exception:
            # A later successful initialization reports losses. If it never
            # succeeds, missing startup/heartbeat evidence means unknown coverage.
            self.initialization_drops+=1

    def close(self):
        if not self.init_lock.acquire(blocking=False):return False
        try:
            sink=self.sink
            if sink is None:
                self.closed=True;return True
            if not self.closed:
                self.closed=True
                summary=getattr(self.probes.get('source'),'summary',None)
                if callable(summary):
                    try:
                        fields=dict(summary())
                        fields.update(runtime_close=True,state_inherited_after_fork=self.forked)
                        sink.emit('source_probe.summary',**fields)
                    except Exception as error:
                        sink.emit('source_probe.summary_error',exception_type=type(error).__name__)
        finally:self.init_lock.release()
        return sink.close()


class ProbeLoader(importlib.abc.Loader):
    def __init__(self, original, callback):self.original,self.callback=original,callback
    def create_module(self,spec):
        create=getattr(self.original,'create_module',None)
        return create(spec) if create else None
    def exec_module(self,module):
        self.original.exec_module(module)
        self.callback(module)


class ProbeFinder(importlib.abc.MetaPathFinder):
    def __init__(self,modules,probes,emit,installed=None):
        self.modules,self.probes,self.emit=modules,probes,emit
        self.installed={} if installed is None else installed
    def find_spec(self,fullname,path=None,target=None):
        expected=self.modules.get(fullname)
        if expected is None:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None or not spec.origin or digest(spec.origin)!=expected['sha256']:
            raise ImportError('Diagnostic source hash mismatch: '+fullname)
        original=spec.loader
        probe=self.probes[expected['probe']]
        def observe(module):
            probe.install({fullname:module},self.emit)
            self.installed[fullname]=expected['sha256']
            self.emit('module_installed',module=fullname,source_sha256=expected['sha256'])
        spec.loader=ProbeLoader(original,observe)
        return spec


def install(env=None, *, allowed_root=Path('/workspace')):
    env=os.environ if env is None else env
    if env.get('KVCR_BENCH_OBSERVER')!='1':return None
    output=Path(env['KVCR_BENCH_OBSERVER_DIR']).absolute()
    if output.resolve()!=output or not output.is_relative_to(Path(allowed_root).resolve()):
        raise ValueError('Observer output escapes workspace')
    manifest=json.loads(Path(env['KVCR_BENCH_OBSERVER_MANIFEST']).read_text())
    probes={};here=Path(__file__).resolve().parent
    for name in ('request','source'):
        file=here/(name+'_probe.py')
        if digest(file)!=manifest['probe_sha256'][file.name]:raise ValueError('Probe hash mismatch: '+name)
        spec=importlib.util.spec_from_file_location('_benchmark_'+name+'_probe',file)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        spec.loader.exec_module(module);probes[name]=module
    if any(name in sys.modules for name in manifest['modules']):
        raise RuntimeError('Observer must be installed before instrumented source imports')
    sinks=ProcessSinks(output,allowed_root=allowed_root,probes=probes)
    finder=ProbeFinder(manifest['modules'],probes,sinks.emit,sinks.installed)
    finder.sinks=sinks;finder.close=sinks.close
    atexit.register(sinks.close)
    sys.meta_path.insert(0,finder)
    return finder
