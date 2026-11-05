"""Bounded /proc counters only: no cmdlines, GPU APIs or serving synchronization."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time


def parse_stat(text):
    pid,rest=text.split(' (',1);comm,tail=rest.rsplit(') ',1);fields=tail.split()
    return {'id':int(pid),'comm':comm,'state':fields[0],'ppid':int(fields[1]),
            'utime_ticks':int(fields[11]),'stime_ticks':int(fields[12]),'start_ticks':int(fields[19])}


def collect_snapshot(proc_root=Path('/proc'), max_processes=256, max_threads=4096,
                     max_process_scan=4096, max_task_scan=16384):
    """Skip only completely observed all-Z task groups; uncertainty is coverage loss.

    Directory scans and recorded rows have separate finite budgets. Recheck task
    membership before skipping; validate leader identity before publishing any group.
    /proc is never an atomic view.
    """
    result=dict(processes=[], threads=[], missing_raced_reads=0, truncated=False,
                dead_zombie_processes_skipped=0, processes_scanned=0,
                task_entries_scanned=0, scan_budget_exhausted=False,
                zombie_checks_incomplete=0)
    errors=(OSError, ValueError, IndexError)

    def uncertain():
        result['missing_raced_reads']+=1

    def entries(path, budget):
        rows=[];complete=True
        try:
            for entry in path.iterdir():
                if not entry.name.isdigit():continue
                if len(rows)>=budget:
                    result['scan_budget_exhausted']=True
                    result['truncated']=True;complete=False;break
                rows.append(entry)
        except OSError:
            uncertain();complete=False
        return sorted(rows,key=lambda entry:int(entry.name)),complete

    def read(entry):
        row=parse_stat((entry/'stat').read_text(errors='replace'))
        if row['id']!=int(entry.name):raise ValueError('raced identity')
        return row

    processes, _=entries(proc_root,max_process_scan)
    for entry in processes:
        if len(result['processes'])>=max_processes or len(result['threads'])>=max_threads:
            result['truncated']=True;break
        result['processes_scanned']+=1
        try:process=read(entry)
        except errors:
            uncertain();continue
        tasks, complete=entries(entry/'task',max_task_scan-result['task_entries_scanned'])
        result['task_entries_scanned']+=len(tasks)
        rows=[];known=complete and bool(tasks)
        for task in tasks:
            if len(rows)+len(result['threads'])>=max_threads:
                result['truncated']=True;known=False;break
            try:rows.append(read(task))
            except errors:
                uncertain();known=False
        dead_candidate=False
        if process['state']=='Z':
            if known and all(row['state']=='Z' for row in rows):
                again, complete_again=entries(entry/'task',max_task_scan-result['task_entries_scanned'])
                result['task_entries_scanned']+=len(again)
                known=(complete_again and [t.name for t in tasks]==[t.name for t in again])
                dead_candidate=known
        # Task rows must never be attached to a reused PID's initial identity.
        # Validate every group after its task reads, including ordinary live PIDs.
        try:final=read(entry)
        except errors:
            uncertain();continue
        if final['start_ticks']!=process['start_ticks']:
            uncertain();continue
        if process['state']=='Z':
            if dead_candidate and final['state']=='Z':
                result['dead_zombie_processes_skipped']+=1
                continue
            if not known or (dead_candidate and final['state']!='Z'):
                result['zombie_checks_incomplete']+=1;uncertain()
        result['processes'].append(process)
        for row in rows:
            row['pid']=process['id'];row['process_start_ticks']=process['start_ticks']
            result['threads'].append(row)
        if not tasks:uncertain()
    result['coverage_incomplete']=bool(result['missing_raced_reads'] or result['truncated'])
    return result


def publish_first_capture_ready(run, snapshot, services, sampler, script_sha256):
    """First-snapshot preflight only; a marker is not ongoing coverage proof."""
    if any(snapshot.get(flag,False) for flag in
           ('truncated','scan_budget_exhausted','coverage_incomplete','missing_raced_reads')):
        raise RuntimeError('first CPU capture has incomplete coverage')
    if not isinstance(services,list) or not services or len(services)>256:
        raise RuntimeError('missing or invalid service-pids.json roots')
    try:
        roots=[]
        for service in services:
            if (not service['name'] or not str(service['pid']).isdigit()
                    or not str(service['start']).isdigit()):
                raise ValueError('invalid service identity')
            root=dict(name=service['name'],pid=int(service['pid']),start_ticks=int(service['start']))
            if root['pid']<=0 or root['start_ticks']<=0:raise ValueError('invalid service identity')
            roots.append(root)
    except (KeyError,TypeError,ValueError) as error:
        raise RuntimeError('invalid service identity proof') from error
    identities={(row['id'],row['start_ticks']) for row in snapshot['processes']}
    for root in [*roots,dict(pid=sampler['id'],start_ticks=sampler['start_ticks'])]:
        if (root['pid'],root['start_ticks']) not in identities:
            raise RuntimeError(f"first CPU capture missing identity {root['pid']}:{root['start_ticks']}")
    marker=run/'first-capture-ready.json'
    if marker.exists():raise FileExistsError(marker)
    temporary=run/'first-capture-ready.json.tmp'
    proof=dict(status='ready',snapshot_index=1,utc=time.time(),service_roots=roots,
               sampler=sampler,script_sha256=script_sha256,
               scope='First snapshot only; later sampling gaps/exited threads remain possible.')
    with temporary.open('x') as out:
        out.write(json.dumps(proof,indent=2)+'\n')
    temporary.replace(marker)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    assert a.run.resolve().is_relative_to(Path(__file__).resolve().parents[1]/'cases')
    assert not (a.run/'thread-cpu-meta.json').exists()
    assert not (a.run/'thread-cpu.jsonl.gz').exists()
    assert not (a.run/'first-capture-ready.json').exists()
    services=json.loads((a.run/'service-pids.json').read_text())
    sampler_stat=parse_stat(Path(f'/proc/{os.getpid()}/stat').read_text())
    sampler=dict(id=os.getpid(),start_ticks=sampler_stat['start_ticks'])
    stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    meta={'clock_ticks_per_second':os.sysconf('SC_CLK_TCK'),'interval_seconds':5,'max_processes':256,'max_threads':4096,'max_process_scan':4096,'max_task_scan':16384,'zombie_filter':'all-tasks-Z-with-membership-and-leader-recheck','sampler_pid':os.getpid(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Visible pod process/thread counters; identities include start_ticks. Samples miss exited threads and are not exclusive operation CPU or TTFT components. No process cmdlines/environments captured.'}
    (a.run/'thread-cpu-meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    count=0
    with gzip.open(a.run/'thread-cpu.jsonl.gz','xt') as out:
        while not stop.is_set():
            began=time.monotonic_ns();cpu=time.thread_time_ns();utc=time.time()
            snapshot=collect_snapshot(**{key:meta[key] for key in
                ('max_processes','max_threads','max_process_scan','max_task_scan')})
            out.write(json.dumps({'utc':utc,'monotonic_ns':began,**snapshot,'sample_wall_ns':time.monotonic_ns()-began,'sample_thread_cpu_ns':time.thread_time_ns()-cpu},separators=(',',':'))+'\n');out.flush();count+=1
            if count==1:
                publish_first_capture_ready(a.run,snapshot,services,sampler,meta['script_sha256'])
            stop.wait(max(0,5-(time.monotonic_ns()-began)/1e9))
    (a.run/'thread-cpu-collector.exit').write_text('0\n')
    print('THREAD_CPU_CAPTURE_CLOSED',count,flush=True)


if __name__=='__main__':main()
