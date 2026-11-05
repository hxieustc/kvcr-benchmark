"""Isolated NIXL self-WRITE diagnostic; host timings are not pure GPU/DMA times."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pages',type=int,nargs='+',default=[16,128,512,1024]);p.add_argument('--repetitions',type=int,default=20);a=p.parse_args()
assert str(a.out).startswith('/workspace/sglang-kvcr-parity-20261031-r1/microbench/')
assert 1<=a.repetitions<=100 and all(1<=n<=1024 for n in a.pages)
assert __import__('shutil').disk_usage('/workspace').free>100*2**30
for name,expected in {'NIXL_UCX_SGL_ENABLE':'1','UCX_TLS':'rc,self,cuda_copy','UCX_NET_DEVICES':'rocep172s0:1','UCX_MAX_EAGER_RAILS':'1','UCX_MAX_RNDV_RAILS':'1','UCX_MAX_RMA_RAILS':'1'}.items():assert os.environ.get(name)==expected,(name,os.environ.get(name))
import numpy as np
import torch
from nixl import nixl_agent,nixl_agent_config

a.out.mkdir(parents=True,exist_ok=False)
block=65536;layers=26;maximum=max(a.pages)
host=torch.empty((maximum*layers,block),dtype=torch.uint8,pin_memory=True)
host.fill_(83);host.view(torch.int64)[:,0]=torch.arange(maximum*layers,dtype=torch.int64)
gpu=torch.empty((maximum,block),dtype=torch.uint8,device='cuda:0');gpu.fill_(0);torch.cuda.synchronize()
name=f'kvcr-layout-diagnostic-{os.getpid()}'
agent=nixl_agent(name,nixl_agent_config(backends=['UCX'],num_threads=0,enable_prog_thread=True))
registrations=[agent.register_memory([(host.data_ptr(),host.numel(),0,'')],mem_type='DRAM'),agent.register_memory([(gpu.data_ptr(),gpu.numel(),0,'')],mem_type='VRAM')]
source=agent.prep_xfer_dlist('',np.array([(host.data_ptr(),block,0,block,maximum*layers)],dtype=np.uint64),mem_type='DRAM',backends=['UCX'])
destination=agent.prep_xfer_dlist(name,np.array([(gpu.data_ptr(),block,0,block,maximum)],dtype=np.uint64),mem_type='VRAM',backends=['UCX'])
metadata={'utc':time.time(),'pages':a.pages,'repetitions':a.repetitions,'block_bytes':block,'host_layer_count':layers,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True),'cpu_affinity':sorted(os.sched_getaffinity(0)),'env':{k:v for k,v in os.environ.items() if k.startswith(('UCX_','NIXL_'))},'agent_config':{'num_threads':0,'enable_prog_thread':True,'backend':'UCX'},'nixl_wrapper':inspect.getfile(nixl_agent),'nixl_wrapper_sha256':hashlib.sha256(Path(inspect.getfile(nixl_agent)).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Idle self-agent WRITE DRAM to GPU0. Models stopped. Synthetic ideal/counterfactual geometry, not sampled production addresses or inference TTFT. Host prepare/post/completion observation and fence times are separate; no pure DMA claim.'}
(a.out/'setup.json').write_text(json.dumps(metadata,indent=2)+'\n')
checks=[]
with (a.out/'samples.jsonl').open('x',buffering=1) as output:
    for pages in a.pages:
        permutation=np.random.default_rng(42).permutation(pages).tolist()
        for layout,stride,scattered in [('layer_first',1,False),('page_first',layers,False),('layer_first_scattered_gpu',1,True),('page_first_scattered_gpu',layers,True)]:
            src=[i*stride for i in range(pages)];dst=permutation if scattered else list(range(pages))
            gpu.fill_(0);torch.cuda.synchronize()
            for repetition in range(-2,a.repetitions):
                torch.cuda.nvtx.range_push(f'{layout}/pages={pages}/rep={repetition}')
                cpu=time.thread_time_ns();start=time.perf_counter_ns()
                handle=agent.make_prepped_xfer('WRITE',source,src,destination,dst,backends=['UCX'])
                prepared=time.perf_counter_ns();state=agent.transfer(handle);posted=time.perf_counter_ns();polls=0
                while state!='DONE':
                    assert state in ('PROC','IN_PROG','SUCCESS','NEW'),state
                    assert time.perf_counter_ns()-start<60*10**9,'Native transfer deadline; raw evidence retained'
                    state=agent.check_xfer_state(handle);polls+=1
                    if state!='DONE':time.sleep(0)
                done=time.perf_counter_ns();cpu_done=time.thread_time_ns();torch.cuda.synchronize();fenced=time.perf_counter_ns()
                agent.release_xfer_handle(handle);torch.cuda.nvtx.range_pop()
                output.write(json.dumps({'layout':layout,'pages':pages,'bytes':pages*block,'input_descriptors':pages,'source_stride_bytes':stride*block,'scattered_gpu':scattered,'repetition':repetition,'warmup':repetition<0,'prepare_ms':(prepared-start)/1e6,'post_ms':(posted-prepared)/1e6,'completion_observation_ms':(done-posted)/1e6,'total_host_ms':(done-start)/1e6,'host_thread_cpu_ms':(cpu_done-cpu)/1e6,'post_done_cuda_fence_ms':(fenced-done)/1e6,'polls':polls})+'\n')
            expected=torch.empty((pages,block),dtype=torch.uint8);expected[dst]=host[src]
            actual=gpu[:pages].cpu().numpy().tobytes();wanted=expected.numpy().tobytes()
            assert actual==wanted,(layout,pages,'payload mismatch')
            checks.append({'layout':layout,'pages':pages,'bytes':len(actual),'sha256':hashlib.sha256(actual).hexdigest(),'verified':True})
            print('PAYLOAD_VERIFIED',layout,pages,flush=True)
agent.release_dlist_handle(source);agent.release_dlist_handle(destination)
for registration in registrations:agent.deregister_memory(registration)
(a.out/'payload-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(a.out/'COMPLETE.json').write_text(json.dumps({'utc':time.time(),'samples':len(a.pages)*4*(a.repetitions+2),'payload_checks':len(checks),'all_payloads_verified':True})+'\n')
print('SELF_COPY_DIAGNOSTIC_COMPLETE',a.out,flush=True)
