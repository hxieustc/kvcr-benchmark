"""Check a synced source tree against one historical experiment's immutable lock."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--code',type=Path,required=True);a=p.parse_args()
b=Path(__file__).resolve().parents[1];row=json.loads((b/'sources.lock.json').read_text())[a.stage]
for name,lock in row['sources'].items():
    repo=a.code/'software'/name
    got=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if got!=lock['sha']:raise SystemExit(f"{name}: expected {lock['sha']}, got {got}")
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True)
    if dirty:raise SystemExit(f'{name}: tracked source changes present')
    print(name,got)
print('Source revisions match',a.stage,'; binary/import/data/GPU qualification remains separate.')
