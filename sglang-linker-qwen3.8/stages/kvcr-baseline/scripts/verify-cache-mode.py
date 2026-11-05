"""Archive and validate actual worker registration metadata for this Mamba run."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
r=Path('/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-kvcr')/'runs'/a.attempt
def modes(value):
    if isinstance(value,str):
        try:return modes(json.loads(value))
        except (ValueError,TypeError):return []
    if isinstance(value,list):return [mode for item in value for mode in modes(item)]
    if isinstance(value,dict):
        return ([value['enable_eagle']] if 'enable_eagle' in value else [])+[mode for key,item in value.items() if key!='enable_eagle' for mode in modes(item)]
    return []
cards=[]
for path in sorted((r/'discovery/v1/mdc').glob('*')):
    if not path.is_file():continue
    raw=path.read_bytes();document=json.loads(raw)
    found=modes(document)
    if not found:continue
    assert found==[False],(path,found)
    cards.append({'path':str(path.relative_to(r)),'sha256':hashlib.sha256(raw).hexdigest(),'enable_eagle_values':found,'document':document})
assert len(cards)==4,len(cards)
out=r/'provenance';out.mkdir(exist_ok=True)
(out/'registered-model-cards.json').write_text(json.dumps({'passed':True,'expected_cache_mode':'normal 64-token blocks; NEXTN decoding unchanged','cards':cards},indent=2)+'\n')
print('VERIFIED_FOUR_REGISTERED_CACHE_MODES_FALSE')
