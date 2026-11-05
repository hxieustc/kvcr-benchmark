"""Verify the immutable HiCache source lock without importing GPU/runtime modules."""
from pathlib import Path
import argparse,json,subprocess
p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);a=p.parse_args()
b=Path(__file__).resolve().parents[1];pins=json.loads((b/'sources.lock.json').read_text())
for name,v in pins.items():
 r=a.code/'software'/name
 sha=subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip()
 assert sha==v['sha'],(name,sha,v['sha'])
 assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain','--untracked-files=no'],text=True),(name,'Tracked changes')
 print(name,sha)
print('Source revisions match; this does not qualify native transfer or model execution.')
