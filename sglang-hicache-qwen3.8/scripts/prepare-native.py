"""Bind archived qualification checks to an explicitly supplied NEW allocation.

Writes scripts/snapshots/byte manifests only; does not query GPUs or run qualification.
"""
from pathlib import Path
import argparse,ast,hashlib,json,re,shutil
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--pod-json',type=Path,required=True)
p.add_argument('--claim-json',type=Path,required=True);p.add_argument('--gpu-uuids',required=True);p.add_argument('--node-guid',required=True);p.add_argument('--rnic',required=True)
a=p.parse_args();R=a.root.resolve();assert R.is_relative_to('/workspace') and R!=Path('/workspace');assert re.fullmatch('[a-zA-Z0-9_-]+:1',a.rnic)
pod=json.loads(a.pod_json.read_text());claim=json.loads(a.claim_json.read_text());gpu=tuple(a.gpu_uuids.split(','));assert len(gpu)==len(set(gpu))==4
assert pod['status']['phase']=='Running';hca=a.rnic.split(':')[0]
devs=claim['status']['allocation']['devices']['results'];assert len(devs)==1 and devs[0]['device']==hca and devs[0]['pool']==pod['spec']['nodeName']
assert any(r['kind']=='Pod' and r['uid']==pod['metadata']['uid'] for r in claim['metadata']['ownerReferences'])
Q=R/'qualification';Q.mkdir(parents=True,exist_ok=True);(R/'setup').mkdir(exist_ok=True)
assert not list(Q.glob('native-rdma-*')),'Never rebind a qualified or active allocation'
for src,dst in [(a.pod_json,R/'setup/pod-r2-live.json'),(a.claim_json,R/'setup/rdma-claim.json')]:
 if src.resolve()!=dst.resolve():shutil.copyfile(src,dst)
driver=(B/'qualification-reference/qualify_native_rdma.py').read_text();tree=ast.parse(driver)
replace={'ROOT':'Path('+repr(str(R))+')','POD':repr(pod['metadata']['name']),'UID':repr(pod['metadata']['uid']),
 'NODE':repr(pod['spec']['nodeName']),'CLAIM':repr(claim['metadata']['name']),'GPU_UUIDS':repr(gpu),'HCA':repr(hca),'GUID':repr(a.node_guid)}
# AST column offsets permit HCA and GUID assignments on the same original line.
lines=driver.splitlines(keepends=True);edits=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in replace:
  start=sum(map(len,lines[:n.lineno-1]))+n.col_offset;end=sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
  edits.append((start,end,n.targets[0].id+'='+replace[n.targets[0].id]))
assert len(edits)==len(replace)
for start,end,value in sorted(edits,reverse=True):driver=driver[:start]+value+driver[end:]
(Q/'qualify_native_rdma.py').write_text(driver)
for name in ['nixl_peer.py','mooncake_peer.py','qualify-kvcr-hicache-peer-r2.py','qualify-kvcr-complete-state-peer.py','test_hicache_mamba_slot_side_states.py']:
 s=(B/'qualification-reference'/name).read_text().replace('/workspace/sglang-hicache-parity-20261103-r1',str(R)).replace('rocep190s0',hca)
 s=s.replace('kvcr-hicache-peer-r2.json','kvcr-hicache-peer.json')
 s=s.replace("ROOT/'software/sglang/test/registered/unit/mem_cache/test_hicache_mamba_slot_side_states.py'","ROOT/'qualification/test_hicache_mamba_slot_side_states.py'")
 (Q/name).write_text(s)
names=['qualify_native_rdma.py','nixl_peer.py','mooncake_peer.py']
digest={name:hashlib.sha256((Q/name).read_bytes()).hexdigest() for name in names}
(Q/'freeze.json').write_text(json.dumps({'files_sha256':digest,'scope':'New operational allocation binding; original functional/native SDK checks preserved; not an execution receipt.'},indent=2)+'\n')
print(json.dumps({'prepared':str(Q),'gpu_uuids':gpu,'rnic':a.rnic,'qualification_executed':False},indent=2))
