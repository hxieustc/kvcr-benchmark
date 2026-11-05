"""Exact HiCache r2 hardware qualification. Default is a non-executing plan."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/sglang-hicache-parity-20261103-r1')
POD='sglang-hicache-gb300-4gpu-20261103-r2'
UID='c7285a40-d1b6-4dae-8f97-d624e1894ec8'
NODE='ip-100-64-187-192.us-east-2.compute.internal'
CLAIM=POD+'-roce-8v4t7'
GPU_UUIDS=('GPU-31fb8ff7-0284-ef4c-4c55-9526c514da62','GPU-74d2625e-92f0-0b3f-a4f4-e80430b9b7b7',
           'GPU-d3f03387-9b1d-5af0-cf9f-8e19acd54515','GPU-edcb8e40-ad1d-74e8-4a54-981b5c94ef65')
HCA='rocep190s0';GUID='0201:36ff:fe2f:9d67';N=8*2**20
ENV={'UCX_TLS':'rc,self,cuda_copy','UCX_NET_DEVICES':HCA+':1','UCX_MEMTYPE_CACHE':'n',
     'UCX_LOG_LEVEL':'info','UCX_MAX_RNDV_RAILS':'1','UCX_MAX_EAGER_RAILS':'1',
     'UCX_MAX_RMA_RAILS':'1','NIXL_UCX_SGL_ENABLE':'1','WITH_NVIDIA_PEERMEM':'0'}
NATIVE_SHA='16471607d803d4878c64a57b6de36fe90de6807c07c364b95da98f91b3dd20b9'
SDK=Path('/opt/sglang/lib/python3.12/site-packages/nixl_cu13')
SDK_PINS={'_api.py':'79ac70eed826672f09a21df7e71c8646654e761961c9783ecd8e145c0a7149a7',
 '_bindings.cpython-312-aarch64-linux-gnu.so':'8cc0e0abd677a087023e6cc2295d09a19c8d2086722fa5d9415cfe69e137d1e2',
 '_utils.cpython-312-aarch64-linux-gnu.so':'489c88ee233ee22cbcb75c8a36e68b67ce124c284254f9bc367a9b616898e70e'}
BYTE_TESTS={'GPU-to-peer-host','host-to-peer-GPU','GPU-to-peer-GPU','GET-peer-host-to-GPU','GET-peer-GPU-to-GPU'}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def check_env(env):
    for k,v in ENV.items():assert env.get(k)==v,(k,env.get(k),v)
    assert env.get('MC_FORCE_TCP','').lower() not in ('1','true')

def check_hardware(api,claim,gpus,hca):
    assert api['metadata']['name']==POD and api['metadata']['uid']==UID
    assert api['status']['phase']=='Running' and api['spec']['nodeName']==NODE
    assert claim['metadata']['name']==CLAIM
    owners=[r for r in claim['metadata']['ownerReferences'] if r['kind']=='Pod']
    assert len(owners)==1 and owners[0]['uid']==UID and owners[0]['name']==POD
    claims=[r for r in api['status']['resourceClaimStatuses'] if r['name']=='roce']
    assert len(claims)==1 and claims[0]['resourceClaimName']==CLAIM
    rows=claim['status']['allocation']['devices']['results']
    assert len(rows)==1 and rows[0]['device']==HCA and rows[0]['pool']==NODE
    expected=set(json.loads(api['metadata']['annotations']['dgxc.nvidia.com/devices'])['devices']['nvidia.com/gpu'])
    assert expected==set(GPU_UUIDS) and len(gpus)==4 and {r['uuid'] for r in gpus}==expected
    assert {r['index'] for r in gpus}=={0,1,2,3} and all('GB300' in r['name'] for r in gpus)
    assert hca['device']==HCA and hca['port']==1 and hca['node_guid']==GUID
    assert 'ACTIVE' in hca['state'] and 'LinkUp' in hca['phys_state']
    assert hca['rate'].split()[0]=='400' and hca['link_layer']=='Ethernet'

def check_idle(apps,mem):
    assert not apps.strip(),'GPU compute process remains'
    assert len(mem)==4 and all(0<=int(x)<2000 for x in mem),'GPU allocations >=2000 MiB'

def smi(*args):return subprocess.check_output(['nvidia-smi',*args],text=True)
def idle():
    apps=smi('--query-compute-apps=pid','--format=csv,noheader')
    mem=smi('--query-gpu=memory.used','--format=csv,noheader,nounits').splitlines()
    check_idle(apps,mem);return dict(compute_apps=apps,memory_used_mib=list(map(int,mem)))

def check_moon(r,exit):
    assert exit==0 and r['status']=='passed' and r['with_nvidia_peermem']=='0'
    assert r['protocol']=='rdma' and r['hca']==HCA and r['buffer_bytes']==N
    assert r['mooncake']=='0.3.13' and r['torch'].startswith('2.14.1') and r['native_sha256']==NATIVE_SHA
    assert re.fullmatch('[0-9a-f]{64}',r['expected_sha256'])
    assert r['cuda_visible_devices']==','.join(GPU_UUIDS) and r['gpu_uuid']==GPU_UUIDS[0]
    ops=r['operations'];checks=[x for x in ops if x['operation'] in BYTE_TESTS]
    assert len(checks)==5 and {x['operation'] for x in checks}==BYTE_TESTS
    assert all(x['return_code']==0 and x['full_bytes_correct'] is True and x['bytes']==N for x in checks)
    regs=[x for x in ops if x['operation']=='register'];inits=[x for x in ops if x['operation']=='initialize']
    assert len(regs)==4 and {x['buffer'] for x in regs}=={'GPU-source','GPU-destination','host-store','peer-GPU'}
    assert len(inits)==2 and all(x['return_code']==0 for x in regs+inits)

def check_nixl(r,exit):
    assert exit==0 and r['status']=='passed' and r['registrations_released'] is True
    assert r['peer_exits']==[0,0] and len(r['peers'])==2
    assert len({p['pid'] for p in r['peers']})==len({p['name'] for p in r['peers']})==2
    assert all(int(p['start'])>0 for p in r['peers'])
    closed=r['close_identities'];assert len(closed)==2
    assert [(p['pid'],p['start'],p['name']) for p in closed]==[(p['pid'],p['start'],p['name']) for p in r['peers']]
    assert all(p['registrations_released'] is True for p in closed)
    ops=r['operations'];assert len(ops)==4
    assert {(x['operation'],x['source'],x['target']) for x in ops}=={(k,s,1-s) for k in ('READ','WRITE') for s in (0,1)}
    assert all(x['bytes']==N and x['full_bytes_correct'] is True and x['terminal']=='DONE' for x in ops)

def sdk_gate():
    for name,digest in SDK_PINS.items():assert sha(SDK/name)==digest,(SDK/name,'SDK changed')
    native=Path('/opt/sglang/lib/python3.12/site-packages/mooncake/engine.so')
    assert sha(native)==NATIVE_SHA
    return {str(p):sha(p) for p in [native,*[SDK/n for n in SDK_PINS]]}

def execute(kind,attempt):
    assert re.fullmatch('[a-zA-Z0-9_-]+',attempt)
    assert str(HERE)==str(ROOT/'qualification'),'Stage private files into NEW R/qualification'
    freeze=read(HERE/'freeze.json')
    for name,digest in freeze['files_sha256'].items():assert Path(name).name==name and sha(HERE/name)==digest
    api=read(ROOT/'setup/pod-r2-live.json');claim=read(ROOT/'setup/rdma-claim.json')
    assert os.environ.get('POD_NAME',os.environ.get('HOSTNAME'))==POD
    assert os.environ.get('POD_IP')==api['status']['podIP']
    check_env(os.environ);pins=sdk_gate()
    raw=smi('--query-gpu=index,uuid,name','--format=csv,noheader')
    gpus=sorted([dict(index=int(a),uuid=b.strip(),name=c.strip()) for a,b,c in (l.split(',',2) for l in raw.splitlines())],key=lambda g:g['index'])
    port=Path('/sys/class/infiniband')/HCA/'ports/1'
    hca=dict(device=HCA,port=1,node_guid=(port.parents[1]/'node_guid').read_text().strip(),**{n:(port/n).read_text().strip() for n in ('state','phys_state','rate','link_layer')})
    check_hardware(api,claim,gpus,hca)
    assert shutil.disk_usage(ROOT).free>100*2**30
    out=ROOT/'qualification'/('native-rdma-'+attempt)
    with (ROOT/'benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=idle();assert not out.exists() and not out.is_symlink();out.mkdir()
        for name in [*freeze['files_sha256'],'freeze.json']:shutil.copyfile(HERE/name,out/name)
        write(out/'pod.json',api);write(out/'claim.json',claim)
        proof=dict(status='incomplete',pod=dict(name=POD,uid=UID,node=NODE),hca=hca,gpu_inventory=gpus,
                   idle_before=before,env={k:os.environ[k] for k in ENV},native_sha256=pins,freeze_sha256=sha(HERE/'freeze.json'),
                   scope='Same-node single assigned RNIC functional byte qualification; not inference, cross-node, bandwidth or HiCache hit qualification.',children={})
        write(out/'qualification.json',proof)
        try:
            for name in (['mooncake','nixl'] if kind=='all' else [kind]):
                env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in gpus))
                cmd=[str(ROOT/'env/bin/python'),'-B',str(HERE/(name+'_peer.py')),'--out',str(out/(name+'.json'))]
                if name=='mooncake':cmd+=['--host',os.environ['POD_IP']]
                with (out/(name+'.log')).open('xb') as log:
                    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    proof['children'][name]=dict(pid=proc.pid,command=cmd,cuda_visible_devices=env['CUDA_VISIBLE_DEVICES']);write(out/'qualification.json',proof)
                    # wait(timeout) never kills a child with uncertain native ownership.
                    rc=proc.wait(timeout=180)
                result=read(out/(name+'.json'));(check_moon if name=='mooncake' else check_nixl)(result,rc)
                proof['children'][name].update(exit=rc,result=result)
            deadline=time.monotonic()+30
            while True:
                try:proof['idle_after']=idle();break
                except AssertionError:
                    if time.monotonic()>=deadline:raise
                    time.sleep(.2)
            proof['status']='passed'
        except BaseException as error:
            proof.update(status='unqualified',error=repr(error),cleanup='No automatic kill. Inspect recorded parent/peer PIDs before manual owned cleanup.');raise
        finally:
            proof['files_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='qualification.json'}
            write(out/'qualification.json',proof)
        print(json.dumps(dict(status=proof['status'],output=str(out))),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--kind',choices=['all','mooncake','nixl'],default='all');p.add_argument('--attempt',default='r1');a=p.parse_args()
    if a.execute:execute(a.kind,a.attempt)
    else:print(json.dumps(dict(execute=False,kind=a.kind,output=str(ROOT/'qualification'/('native-rdma-'+a.attempt)),env=ENV,lock=str(ROOT/'benchmark.lock'))))
if __name__=='__main__':main()
