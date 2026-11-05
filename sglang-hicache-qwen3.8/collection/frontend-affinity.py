"""Apply the recorded frontend NUMA-node0 affinity, then exec its real argv."""
import os,sys
from pathlib import Path
affinity=set()
for chunk in Path('/sys/devices/system/node/node0/cpulist').read_text().strip().split(','):
    lo,*hi=map(int,chunk.split('-'));affinity.update(range(lo,(hi[0] if hi else lo)+1))
affinity &= os.sched_getaffinity(0)
assert affinity
os.sched_setaffinity(0,affinity)
os.execvpe(sys.argv[1],sys.argv[1:],os.environ)
