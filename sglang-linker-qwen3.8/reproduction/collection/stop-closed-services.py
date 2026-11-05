"""Stop a closed run's verified process family, including live zombie-leader threads."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
root=Path('/workspace/sglang-kvcr-parity-20261031-r1')
assert a.case.resolve().is_relative_to(root/'cases')
run=a.case/'runs'/a.attempt
assert (run/'driver.exit').exists(),'Do not stop an active benchmark'
records=json.loads((run/'service-pids.json').read_text())
def proc_fields(path):
    try:return path.read_text().rsplit(') ',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
def state(pid):
    fields=proc_fields(Path('/proc')/str(pid)/'stat')
    return (fields[19],int(fields[1])) if fields else None
states={int(p.name):state(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdecimal()}
family={r['pid'] for r in records if (s:=states.get(r['pid'])) and s[0]==r['start']}
while True:
    more={pid for pid,s in states.items() if s and s[1] in family}
    if more<=family:break
    family|=more
expected={pid:states[pid][0] for pid in family}
def live():
    result=[]
    for pid,start in expected.items():
        s=state(pid)
        if s is None or s[0]!=start:continue
        try:
            tasks=list((Path('/proc')/str(pid)/'task').iterdir())
        except (FileNotFoundError,ProcessLookupError):continue
        tids=[int(t.name) for t in tasks if (fields:=proc_fields(t/'stat')) and fields[0]!='Z']
        if tids:result.append((pid,tids))
    return result
receipt={'utc':time.time(),'run':str(run),'expected_process_starts':expected,'pod_retained':True,'initial_live':live()}
for pid,_ in receipt['initial_live']:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
deadline=time.monotonic()+10
while live() and time.monotonic()<deadline:time.sleep(1)
receipt['remaining_after_grace']=live()
for pid,tids in receipt['remaining_after_grace']:
    for tid in tids:
        try:os.kill(tid,signal.SIGKILL)
        except ProcessLookupError:pass
(run/'completed-service-stop.json').write_text(json.dumps(receipt,indent=2)+'\n')
# Driver waits separately for CUDA memory release before any next model load.
print('VERIFIED_CLOSED_RUN_SERVICES_STOPPED',len(receipt['initial_live']),flush=True)
