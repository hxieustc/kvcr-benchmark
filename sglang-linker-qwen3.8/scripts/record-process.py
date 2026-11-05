"""Record a manually started service PID/start-time, without signaling it."""
import argparse,fcntl,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args()
assert a.run.is_relative_to('/workspace') and a.pid>0
fields=Path(f'/proc/{a.pid}/stat').read_text().rsplit(') ',1)[1].split();assert fields[0]!='Z'
a.run.mkdir(parents=True,exist_ok=True)
with (a.run/'process-record.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=a.run/'service-pids.json';rows=json.loads(path.read_text()) if path.exists() else [];assert not any(r['name']==a.name for r in rows)
 rows.append({'name':a.name,'pid':a.pid,'start':fields[19],'argv':Path(f'/proc/{a.pid}/cmdline').read_bytes().decode().strip('\0').split('\0'),'origin':'manual reproduction'})
 path.write_text(json.dumps(rows,indent=2)+'\n')
print(a.name,a.pid,fields[19])
