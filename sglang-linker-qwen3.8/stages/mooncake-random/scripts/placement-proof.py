import argparse,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('placement','/workspace/sglang-kvcr-parity-20261031-r1/sources/gb300/mooncake-placement-r1/common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
out=a.run/'provenance/placement-after-clear.json';assert not out.exists();c.write(out,c.policy_proof(a.run,phase='after-clear'))
