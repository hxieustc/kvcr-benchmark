"""Small real RDMA control/DMABUF comparison, with byte checks. No inference analysis."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+chr(10))


def child(out,host):
    import torch
    import mooncake.engine as native
    n=8*2**20;engines=[];buffers=[];registered=[]
    result=dict(with_nvidia_peermem=os.environ.get('WITH_NVIDIA_PEERMEM'),host=host,hca='rocep190s0',
        protocol='rdma',buffer_bytes=n,torch=torch.__version__,mooncake=importlib.metadata.version('mooncake-transfer-engine-cuda13'),
        native_path=native.__file__,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),operations=[])
    assert result['mooncake']=='0.3.13' and result['torch'].startswith('2.14.1')
    assert os.environ.get('MC_FORCE_TCP') not in ('1','true'),'No TCP fallback permitted'
    try:
        torch.cuda.set_device(0)
        props=torch.cuda.get_device_properties(0)
        gpu_uuid=str(props.uuid)
        if not gpu_uuid.startswith('GPU-'):gpu_uuid='GPU-'+gpu_uuid
        result.update(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu_uuid=gpu_uuid,gpu_name=props.name)
        from qualify_native_rdma import GPU_UUIDS
        assert gpu_uuid==GPU_UUIDS[0] and os.environ['CUDA_VISIBLE_DEVICES']==','.join(GPU_UUIDS)
        a=torch.arange(n,dtype=torch.int64,device='cuda').remainder_(251).to(torch.uint8)
        c=torch.zeros(n,dtype=torch.uint8,device='cuda');d=torch.zeros_like(c)
        b=torch.zeros(n,dtype=torch.uint8,pin_memory=True);buffers.extend((a,b,c,d))
        torch.cuda.synchronize();expected=a.cpu()
        for _ in range(2):
            e=native.TransferEngine();engines.append(e)
            rc=e.initialize(host,'P2PHANDSHAKE','rdma','rocep190s0')
            result['operations'].append(dict(operation='initialize',return_code=rc));assert rc==0
        for index,tensor,label in ((0,a,'GPU-source'),(0,c,'GPU-destination'),(1,b,'host-store'),(1,d,'peer-GPU')):
            rc=engines[index].register_memory(tensor.data_ptr(),n)
            result['operations'].append(dict(operation='register',buffer=label,return_code=rc));write(out,result)
            if rc!=0:
                result['status']='registration-failed';write(out,result);return 1
            registered.append((index,tensor))
        peers=[host+':'+str(e.get_rpc_port()) for e in engines]
        tests=[(0,'write',a,b,1,'GPU-to-peer-host'),(1,'write',b,c,0,'host-to-peer-GPU'),
               (0,'write',a,d,1,'GPU-to-peer-GPU'),(0,'read',c,b,1,'GET-peer-host-to-GPU'),
               (1,'read',d,a,0,'GET-peer-GPU-to-GPU')]
        for actor,kind,local,remote,peer,label in tests:
            destination=remote if kind=='write' else local
            destination.zero_();torch.cuda.synchronize()
            method=getattr(engines[actor],'transfer_sync_'+kind)
            rc=method(peers[peer],local.data_ptr(),remote.data_ptr(),n,'rdma');torch.cuda.synchronize()
            correct=rc==0 and torch.equal(destination.cpu(),expected)
            result['operations'].append(dict(operation=label,return_code=rc,full_bytes_correct=correct,bytes=n));write(out,result)
            assert correct,label
        for index,tensor in reversed(registered):assert engines[index].unregister_memory(tensor.data_ptr())==0
        result['status']='passed';result['expected_sha256']=hashlib.sha256(expected.numpy().tobytes()).hexdigest();write(out,result);return 0
    except BaseException as error:
        result.update(status='failed',error=repr(error));write(out,result);raise


def main():
    from qualify_native_rdma import ROOT, check_env, sdk_gate
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--host',required=True);a=p.parse_args()
    assert a.out.resolve().is_relative_to(ROOT/'qualification')
    check_env(os.environ);sdk_gate()
    assert not a.out.exists() and not a.out.is_symlink()
    return child(a.out,a.host)
if __name__=='__main__':raise SystemExit(main())
