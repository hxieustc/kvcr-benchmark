"""Install only into an explicitly supplied NEW experiment clientenv."""
import argparse,hashlib,json,subprocess,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert str(r).startswith('/workspace/') and not (r/'clientenv').exists()
b=Path(__file__).resolve().parents[1];lock=json.loads((b/'client-package.lock.json').read_text())
(r/'build/wheels').mkdir(parents=True,exist_ok=True);wheel=r/'build/wheels'/lock['wheel']
data=urllib.request.urlopen(lock['url']).read();assert hashlib.sha256(data).hexdigest()==lock['sha256'];wheel.write_bytes(data)
subprocess.run(['python3','-m','venv',str(r/'clientenv')],check=True);py=str(r/'clientenv/bin/python')
subprocess.run([py,'-m','pip','install','-c',str(b/'reproduction/client/client-constraints.txt'),str(wheel),'datasets==5.0.1'],check=True)
subprocess.run([py,str(b/'scripts/pin-client-trace.py'),'--clientenv',str(r/'clientenv')],check=True)
(r/'client-packages.txt').write_bytes(subprocess.check_output([py,'-m','pip','freeze','--all']))
