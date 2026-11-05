"""Render recorded HiCache component commands; --materialize creates new run inputs only."""
from pathlib import Path
import argparse,hashlib,json,re,shlex,shutil

B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',required=True,choices=['kvcr','mooncake-random','mooncake-local-first'])
p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--pod-ip',required=True);p.add_argument('--rnic',required=True)
p.add_argument('--attempt',default='repeat-r1');p.add_argument('--namespace',default='hicacherepro')
p.add_argument('--role',choices=['state-agent','frontend',*[f'worker-{n}' for n in range(4)],'mooncake-master','client','collector','thread-cpu-collector','clear','cache-mode','peer','post-clear'],default='worker-0')
p.add_argument('--materialize',action='store_true');a=p.parse_args()
assert a.root.is_absolute() and a.root.is_relative_to('/workspace') and a.root!=Path('/workspace')
assert a.model.is_absolute() and re.fullmatch(r'[a-zA-Z0-9_-]+:1',a.rnic)
assert re.fullmatch('[a-zA-Z0-9_-]+',a.attempt) and re.fullmatch('[a-zA-Z0-9_-]+',a.namespace)
R=a.root;C=R/'cases'/a.case;run=C/'runs'/a.attempt
old='/workspace/sglang-hicache-parity-20261103-r1';legacy='/workspace/sglang-kvcr-parity-20261031-r1'
source=B/'recorded'/a.case;records=json.loads((source/'run/service-pids.json').read_text())
short='/workspace/t/h'+hashlib.sha256(str(R).encode()).hexdigest()[:8]
def rewrite(t):
 t=t.replace(old+'/cases/'+a.case+'/runs/hicache-1800-r5',str(run))
 t=t.replace(legacy+'/assets-gb300/model',str(a.model))
 t=t.replace('/workspace/sglang-linker-qwen38-agentx-20261031-r1/clientenv',str(R/'clientenv'))
 t=t.replace('/workspace/sglang-linker-qwen38-agentx-20261031-r1/cache',str(R/'cache'))
 t=t.replace(old,str(R)).replace(legacy,str(R)).replace('sglkvcrreview',a.namespace)
 t=t.replace('rocep190s0',a.rnic.split(':')[0])
 return t
if a.materialize:
 assert not (run/'benchmark.log').exists(),'Never change active or completed run inputs'
 for d in [run/'provenance',run/'preflight',run/'discovery',C/'config',C/'scripts',C/'cache',R/'scripts',R/'setup/observer',R/'assets/client-input-maps',R/'tmp',Path(short)]:d.mkdir(parents=True,exist_ok=True)
 for rank in range(4):(run/'prometheus'/f'worker-{rank}').mkdir(parents=True,exist_ok=True)
 for folder in ['config','scripts']:
  for f in (source/folder).rglob('*'):
   if f.is_file():
    out=C/folder/f.relative_to(source/folder);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(rewrite(f.read_text()))
 for f in (B/'observer').glob('*'):
  if f.is_file():shutil.copyfile(f,R/'setup/observer'/f.name)
 shutil.copyfile(B/'reference/model-config.json',R/'setup/model-config.json')
 shutil.copyfile(B/'collection/thread-cpu-sampler.py',R/'scripts/thread-cpu-sampler.py')
 shutil.copyfile(B/'collection/frontend-affinity.py',R/'scripts/frontend-affinity.py')
 link=C/'model'
 if not link.exists():link.symlink_to(a.model)
 # Genuine historical setup is input metadata, not a newly passed validation receipt.
 (run/'setup.resolved.json').write_text(rewrite((source/'run/setup.resolved.json').read_text()))
 live=(B/'qualification-reference/live-hicache-qualifier.py').read_text()
 live=live.replace(old,str(R)).replace("ready=read(ROOT/'setup/harness-ready.json')","ready=read(ROOT/'setup/manual-inputs.json')")
 (R/'setup/live-hicache-qualifier.py').write_text(live)
 (R/'setup/manual-inputs.json').write_text(json.dumps({'native_rdma_qualification':'qualification/native-rdma-'+a.attempt+'/qualification.json'})+'\n')
 (run/'manual-inputs.json').write_text(json.dumps({'case':a.case,'historical_attempt':'hicache-1800-r5','attempt':a.attempt,'root':str(R),'model':str(a.model),'rnic':a.rnic,'namespace':a.namespace,'scope':'Input materialization only; no validation or service started.'},indent=2)+'\n')

base=next(r for r in records if r['name']=='worker-0')['env'];record=next((r for r in records if r['name']==a.role),None)
env={k:rewrite(str(v)) for k,v in (record['env'] if record else base).items()}
env.update(DYN_NAMESPACE=a.namespace,DYN_FILE_KV=str(run/'discovery'),UCX_NET_DEVICES=a.rnic,POD_IP=a.pod_ip,TMPDIR=short,
           PYTHONPATH=':'.join(map(str,[R/'setup/observer',R/'scripts',C/'scripts',R/'software/sglang/python',R/'software/kvcr/src'])),
           PATH=str(R/'env/bin')+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
if record is None:
 for key in ['HICACHE_BENCH_OUTPUT','CUDA_VISIBLE_DEVICES','PROMETHEUS_MULTIPROC_DIR']:env.pop(key,None)
 env['DYN_SYSTEM_PORT']='0'
if a.role.startswith('worker-'):
 env['PROMETHEUS_MULTIPROC_DIR']=str(run/'prometheus'/a.role)
 env['HICACHE_BENCH_OUTPUT']=str(C/('raw/backend-'+a.role))
if record:
 argv=[rewrite(str(t)) for t in record['argv']]
 if a.role=='frontend':argv=[str(R/'env/bin/python'),str(R/'scripts/frontend-affinity.py'),*argv]
 if a.role.startswith('worker-'):
  i=argv.index('--hicache-storage-backend-extra-config')+1;opts=json.loads(argv[i])
  if a.case=='kvcr':opts['control_advertise_host']=a.pod_ip
  else:
   opts['device_name']=a.rnic.split(':')[0];opts['local_hostname']=a.pod_ip+':'+opts['local_hostname'].rsplit(':',1)[-1]
  argv[i]=json.dumps(opts,separators=(',',':'))
elif a.role=='client':
 argv=[rewrite(str(t)) for t in json.loads((source/'run/benchmark-command.json').read_text())]
 argv[0]=str(R/'clientenv/bin/agentx')
 env.update(json.loads(rewrite((source/'run/client-runtime-env.json').read_text())))
 env.update(AIPERF_DATASET_MMAP_CACHE_DIR=str(R/'assets/client-input-maps'),TMPDIR=short)
else:
 env.pop('HICACHE_BENCH_OUTPUT',None);env.pop('CUDA_VISIBLE_DEVICES',None);env['DYN_SYSTEM_PORT']='0'
 if a.role=='thread-cpu-collector':
  argv=[str(R/'env/bin/python'),str(R/'scripts/thread-cpu-sampler.py'),'--run',str(run)]
 elif a.role in ('peer','post-clear'):
  argv=[str(R/'env/bin/python'),str(R/'setup/live-hicache-qualifier.py'),'--case',a.case,'--run',str(run),'--phase',a.role]
 else:
  script={'collector':'collect.py','clear':'clear-validation-cache.py','cache-mode':'verify-cache-mode.py'}[a.role]
  argv=[str(R/'env/bin/python'),str(C/'scripts'/script),'--attempt',a.attempt]
print('# '+a.case+' / '+a.role+'; renderer does not start processes')
print('exec '+shlex.join(['env','-u','NATS_SERVER','-u','KVCR_TRANSFER_DIAGNOSTICS',*[k+'='+v for k,v in sorted(env.items())],*argv]))
