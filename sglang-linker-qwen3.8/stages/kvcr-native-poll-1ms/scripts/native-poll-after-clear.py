import argparse,importlib.util,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('native_model','/workspace/sglang-kvcr-parity-20261031-r1/sources/gb300/native-poll-delay-model-r1/common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
out=a.run/'provenance/native-poll-setting.json';assert not out.exists(),'Never overwrite post-clear native proof'
c.write(out,c.after_clear_proof(a.run));print(json.dumps({'passed':True,'phase':'after-validation-cache-clear'}),flush=True)
