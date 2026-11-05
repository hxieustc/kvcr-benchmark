"""Print one component's recorded command, rebased to a NEW workspace root.

--materialize explicitly creates only configs/scripts/directories at the new root;
this tool NEVER starts processes, contacts Kubernetes, or alters source checkouts.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shlex
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True);p.add_argument('--role',default='worker-0',choices=['state-agent','frontend','worker-0','worker-1','worker-2','worker-3','mooncake-master','client','collector','qualify','clear','cache-mode']);p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--pod-ip',required=True);p.add_argument('--rnic',required=True,help='assigned RNIC:port, e.g. rocep190s0:1');p.add_argument('--namespace',default='qwenlinkerrepro');p.add_argument('--attempt',default='repeat-r1');p.add_argument('--materialize',action='store_true');a=p.parse_args()
assert a.root.is_absolute() and a.root.is_relative_to('/workspace') and a.root!=Path('/workspace')
assert a.model.is_absolute() and re.fullmatch(r'[a-zA-Z0-9_-]+:[0-9]+',a.rnic)
assert re.fullmatch(r'[a-zA-Z0-9-]+',a.attempt) and re.fullmatch(r'[a-zA-Z0-9-]+',a.namespace)
s={s['id']:s for s in json.loads((B/'stages.json').read_text())}[a.stage];R=str(a.root);run=a.root/'runs'/a.attempt
records=json.loads((B/'stages'/a.stage/'recorded/service-pids.json').read_text());record=next((r for r in records if r['name']==a.role),None)
old_root='/workspace/sglang-kvcr-parity-20261031-r1';old_case=old_root+'/cases/'+s['case'];old_model=s['setup']['model']['path'];short='/workspace/t/l'+hashlib.sha256(R.encode()).hexdigest()[:8]
def rewrite(t):
 t=re.sub(re.escape(old_case)+r'/runs/[^/\s"\']+',str(run),t)
 for comp,path in s['source_directories'].items():t=t.replace(path,R+'/software/'+comp)
 for source in [old_model,'/workspace/sglang-linker-qwen38-agentx-20261031-r1/model',old_root+'/assets-gb300/model']:t=t.replace(source,str(a.model))
 t=t.replace(old_root+'/sources/gb300/native-poll-delay-model-r1',str(B/'reproduction/native-poll-model'))
 t=t.replace(old_case,R).replace(old_root,R).replace('sglkvcrreview',a.namespace)
 t=t.replace('/workspace/sglang-linker-qwen38-agentx-20261031-r1/cache',R+'/cache')
 return t
if a.materialize:
 assert not (run/'benchmark.log').exists(),'Never re-materialize an active or previously scored attempt'
 for name in ['config','scripts','cache','tmp','assets/client-input-maps']:(a.root/name).mkdir(parents=True,exist_ok=True)
 for name in ['discovery','provenance','preflight','prometheus']: (run/name).mkdir(parents=True,exist_ok=True)
 for rank in range(4):(run/'prometheus'/f'worker-{rank}').mkdir(exist_ok=True)
 Path(short).mkdir(parents=True,exist_ok=True)
 for folder in ['config','scripts']:
  for src in (B/'stages'/a.stage/folder).rglob('*'):
   if src.is_file():
    out=a.root/folder/src.relative_to(B/'stages'/a.stage/folder);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(rewrite(src.read_text()).replace('/infiniband/rocep172s0/','/infiniband/'+a.rnic.split(':')[0]+'/'))
 # Metadata, not fake bootstrap/validation/controller success receipts.
 (run/'manual-reproduction.json').write_text(json.dumps({'stage':a.stage,'historical_attempt':s['attempt'],'new_attempt':a.attempt,'model':str(a.model),'rnic':a.rnic,'pod_ip':a.pod_ip,'namespace':a.namespace,'changed_operational_paths':True,'scientific_settings_source':'stages/'+a.stage+'/stage.json'},indent=2)+'\n')
 # Original validators use ROOT/model for tokenizer access. Existing weights stay read-only.
 link=a.root/'model'
 if not link.exists():link.symlink_to(a.model)
 if s['backend']=='mooncake':
  d=s['setup']['mooncake'];extra={k:v for k,v in d.items() if k in ['master_server_address','metadata_server','global_segment_size','protocol','device_name','standalone_storage','extra_backend_tag','python_gil_switch_interval_seconds']};extra['device_name']=a.rnic.split(':')[0]
 else:extra=s['setup']['kvcr_options']
 (a.root/'config/storage-options.json').write_text(json.dumps(extra,indent=2)+'\n')
base=next(r for r in records if r['name']=='worker-0')['env'];env={k:rewrite(str(v)) for k,v in base.items() if k not in ['CUDA_VISIBLE_DEVICES','PROMETHEUS_MULTIPROC_DIR']}
env.update(DYN_FILE_KV=str(run/'discovery'),DYN_NAMESPACE=a.namespace,UCX_NET_DEVICES=a.rnic,POD_IP=a.pod_ip,PYTHONNOUSERSITE='1',TMPDIR=short)
env['PYTHONPATH']=':'.join([R+'/scripts',R+'/software/sglang/python',R+'/software/kvcr/src'])
env['PATH']=R+'/env/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
if record:
 env.update({k:rewrite(str(v)) for k,v in record['env'].items()})
 env.update(DYN_FILE_KV=str(run/'discovery'),DYN_NAMESPACE=a.namespace,UCX_NET_DEVICES=a.rnic,POD_IP=a.pod_ip,PYTHONPATH=':'.join([R+'/scripts',R+'/software/sglang/python',R+'/software/kvcr/src']),TMPDIR=short,PATH=R+'/env/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
 if a.role.startswith('worker-'):env['PROMETHEUS_MULTIPROC_DIR']=str(run/'prometheus'/a.role)
 argv=[rewrite(x) for x in record['argv']]
 if a.role!='mooncake-master':argv[0]=R+'/env/bin/python'
 if '--router-policy-config' in argv:argv[argv.index('--router-policy-config')+1]=R+'/config/router-policy.json'
 if '--hicache-storage-backend-extra-config' in argv:
  pos=argv.index('--hicache-storage-backend-extra-config')+1;opts=json.loads(argv[pos])
  if s['backend']=='kvcr':opts['advertise_host']=a.pod_ip
  else:
   opts['device_name']=a.rnic.split(':')[0];opts['local_hostname']=a.pod_ip+':'+str(opts['local_hostname']).rsplit(':',1)[-1] if 'local_hostname' in opts else None
   if opts.get('local_hostname') is None:opts.pop('local_hostname',None)
  argv[pos]=json.dumps(opts,separators=(',',':'))
 # Original baseline reused r1 services; its discovery/log paths are deliberately new here.
elif a.role=='client':
 argv=json.loads((B/'stages'/a.stage/'recorded/benchmark-command.json').read_text());argv=[rewrite(x) for x in argv];argv[0]=R+'/clientenv/bin/agentx';argv[argv.index('--output-artifact-dir')+1]=str(run/'aiperf_artifacts');argv[argv.index('--tokenizer')+1]=str(a.model)
 env.update(AIPERF_DATASET_MMAP_CACHE_DIR=R+'/assets/client-input-maps',MPLCONFIGDIR=R+'/cache/matplotlib',TOKENIZERS_PARALLELISM='false')
else:
 names={'collector':'collect.py','qualify':'validate-reuse.py','clear':'clear-validation-cache.py','cache-mode':'verify-cache-mode.py'}
 argv=[R+'/env/bin/python',R+'/scripts/'+names[a.role],'--attempt',a.attempt];env['DYN_SYSTEM_PORT']='0'
print('# '+a.stage+' / '+a.role+'; no process has been started by this renderer')
print('exec '+shlex.join(['env','-u','NATS_SERVER','-u','KVCR_TRANSFER_DIAGNOSTICS',*[k+'='+v for k,v in sorted(env.items())],*argv]))
