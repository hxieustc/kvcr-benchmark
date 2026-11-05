"""Stop only verified experiment service families; retain Kubernetes allocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

r=Path('/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-kvcr-pool-index')
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
records=json.loads((r/'runs'/a.attempt/'service-pids.json').read_text())
def stat(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().split();return (s[21],int(s[3]),s[2])
    except FileNotFoundError:return None
states={int(p.name):stat(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdecimal()}
family={x['pid'] for x in records if (s:=states.get(x['pid'])) and s[0]==x['start'] and s[2]!='Z'}
while True:
    more={pid for pid,s in states.items() if s and s[1] in family}
    if more<=family:break
    family|=more
receipt=r/'logs'/f'service-stop-{time.time_ns()}.json'
receipt.write_text(json.dumps({'attempt':a.attempt,'processes':{pid:states[pid] for pid in sorted(family)},'pod_retained':True},indent=2))
for pid in sorted(family,reverse=True):
    if (s:=stat(pid)) and s[0]==states[pid][0] and s[2]!='Z':os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+60
while any((s:=stat(pid)) and s[0]==states[pid][0] and s[2]!='Z' for pid in family):
    if time.monotonic()>deadline:raise TimeoutError('Recorded service teardown incomplete; inspect receipt, do not start duplicates')
    time.sleep(1)
print('Verified stopped service processes:',len(family),'receipt:',receipt,'pod retained',flush=True)
