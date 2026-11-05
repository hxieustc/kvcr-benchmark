"""Private full-model native-delay guards; no SDK/serving work at import."""
import ast,hashlib,importlib.util,json,re,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CASE='gb300-kvcr-native-poll-delay';ATTEMPT='gb300-native-poll-1ms-r1'
PARENT='gb300-kvcr-gil-interval';PARENT_ATTEMPT='gb300-gil-1ms-r1'
SOURCE='sglang-native-poll-delay';STAGING='sources/gb300/native-poll-delay-staging-r1'
SG_FILE='python/sglang/srt/mem_cache/storage/kvcr/kvcr_direct_linker.py'
CANDIDATE_PATCH_SHA='d6134ee32e8da437f6f00e6a0b6dda386a932378fb07c32d5eed4cd6baecb6cf'
NEW_FILE_SHA='a5b47d0330655e92e14224bd3cdbd7d060ca4f78b71f351845e47cc8de863d0e'
LEGACY='sources/gb300/gil-interval-model-r1'
def load(path):
 s=importlib.util.spec_from_file_location(str(path),path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+chr(10))
assert sha(ROOT/LEGACY/'common.py')=='199530c0d3143b5e3505626b9146728c4cb65d610bc55c34752d2b5794c590eb'
legacy=load(ROOT/LEGACY/'common.py');base=legacy.base;validation=legacy.validation
legacy.NEW_FILE_SHA=NEW_FILE_SHA  # private module globals only, preserving original GIL proof algorithm
setting_lines=legacy.setting_lines;gil_proof=legacy.gil_proof;verify_closed_gil=legacy.verify_closed_gil
check_static=legacy.check_static;normalized_patch=legacy.normalized_patch
assert sha(ROOT/STAGING/'stage.py')=='ee9a31cbc09ccdb06403baae6ba64ed4012cf6858b3b029362fcfc9f931e1d1d'
S=load(ROOT/STAGING/'stage.py')

def config_delta(root,control,setup):
 assert control['sglang_source_directory']=='sglang-gil-interval' and control['workers']==4
 assert (control['concurrency_total'],control['profiling_seconds'],control['warmup']['extra_requests_per_lane'])==(48,1800,5)
 assert control['kvcr_options']['mamba_checkpoint_capacity']==1536 and control['kvcr_options']['dram_size_gb']==control['cache']['kvcr_dram_gib_per_worker']==128
 assert control['kvcr_options']['python_gil_switch_interval_seconds']==.001 and control['cache']['ple_host_gib_per_worker']==64
 assert 'nixl_prog_thread_delay_us' not in control['kvcr_options']
 assert type(setup['kvcr_options']['nixl_prog_thread_delay_us']) is int
 expected=json.loads(json.dumps(control).replace(str(root/'cases'/PARENT),str(root/'cases'/CASE)))
 expected['sglang_source_directory']=SOURCE;expected['kvcr_options']['nixl_prog_thread_delay_us']=1000;expected['pod']=setup['pod']
 actual=json.loads(json.dumps(setup))
 for obj in (expected,actual):
  for key in ('state','experiment'):obj.pop(key,None)
 assert actual==expected,'Only new source/option1000, case paths and experiment/pod metadata may differ'

def generate(root,control,pod):
 old=root/'cases'/PARENT;new=root/'cases'/CASE
 files={name:(old/name).read_text().replace(str(old),str(new)).replace(str(root/'software/sglang-gil-interval/python'),str(root/'software'/SOURCE/'python')) for name in base.static_inventory(old)}
 setup=json.loads(json.dumps(control).replace(str(old),str(new)));setup['sglang_source_directory']=SOURCE;setup['kvcr_options']['nixl_prog_thread_delay_us']=1000;setup['pod']=pod
 setup['state']='prepared native background true/poll1000us full1800; not launched';setup['experiment']=dict(parent_case=PARENT,parent_attempt=PARENT_ATTEMPT,single_change='Native background retained; pthrDelay0 to1000us')
 config_delta(root,control,setup);files['config/setup.deployment.json']=json.dumps(setup,indent=2)+chr(10)
 guard_path=root/'sources/gb300/native-poll-delay-model-r1/common.py'
 files['scripts/native-poll-after-clear.py']=("import argparse,importlib.util,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()\n"+"s=importlib.util.spec_from_file_location('native_model',"+repr(str(guard_path))+");c=importlib.util.module_from_spec(s);s.loader.exec_module(c)\n"+"out=a.run/'provenance/native-poll-setting.json';assert not out.exists(),'Never overwrite post-clear native proof'\nc.write(out,c.after_clear_proof(a.run));print(json.dumps({'passed':True,'phase':'after-validation-cache-clear'}),flush=True)\n")
 anchor="        assert __import__('shutil').disk_usage(ROOT).free"
 driver=files['scripts/run-benchmark.py'];assert driver.count(anchor)==1
 hook="        subprocess.run([python,str(ROOT/'scripts/native-poll-after-clear.py'),'--run',str(run)],env=env,check=True)\n"
 files['scripts/run-benchmark.py']=driver.replace(anchor,hook+anchor)
 assert str(root/'software'/SOURCE/'python') in files['scripts/services.py'] and '/software/sglang-gil-interval/python' not in files['scripts/services.py']
 for name,body in files.items():
  if name.endswith('.py'):ast.parse(body,filename=name)
 tree=ast.parse(files['scripts/run-benchmark.py']);tmp=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='client_tmp' for t in n.targets)]
 assert len(tmp)==1 and len((ast.literal_eval(tmp[0].args[0])+'/tmpabcdefgh/aiperf/dataset_manager_proxy_frontend.ipc').encode())+1<=108
 return files,setup

def staged_source(root):
 original,before=S.parent_source(root);repo=root/'software'/SOURCE;proof=read(root/STAGING/'stage-receipt.json')
 assert proof['passed'] is True and proof['original_preserved'] is True and proof['status']=='staged-not-launched'
 assert proof['source']==str(original) and proof['destination']==str(repo) and proof['source_head']==S.HEAD
 assert proof['parent_full_patch_sha256']==S.BASE_PATCH and proof['base_file_sha256']==S.BASE and proof['candidate_file_sha256']==NEW_FILE_SHA
 assert proof['fixture_freeze_sha256']==S.FIXTURE_FREEZE and proof['original_source_inventory']==before
 after=S.inventory(repo);S.check_inventory(before,after,root);assert after==proof['candidate_source_inventory']
 assert proof['changed_source_files']==[SG_FILE] and proof['added_source_files']==sorted(set(after)-set(before))
 patch=S.git(repo,'diff','--binary','HEAD');assert S.git(repo,'rev-parse','HEAD',text=True).strip()==S.HEAD
 assert patch==(root/STAGING/'candidate-sglang-abbrev8.patch').read_bytes() and hashlib.sha256(patch).hexdigest()==proof['full_patch_sha256']==CANDIDATE_PATCH_SHA
 blocks=lambda d:[b for b in normalized_patch(d).split(b'diff --git ')[1:] if not b.startswith(('a/'+SG_FILE+' ').encode())]
 assert blocks(patch)==blocks(S.git(original,'diff','--binary','HEAD')),'Unrelated tracked patch changed'
 return after

def runtime_preflight(root,setup,head,inventory):
 proof=read(root/STAGING/'runtime-preflight.json')
 assert proof['passed'] is True and proof['serving_launched'] is False and proof['native_agent_constructed'] is False
 assert proof['python']==str(root/'env/bin/python')
 names={'sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker':NEW_FILE_SHA,'sglang.srt.mem_cache.storage.kvcr.native_progress_delay':S.fixture_gate(root)[S.PREFIX+'/native_progress_delay.py'],
  'sglang.srt.mem_cache.unified_cache.unified_cache_linker':'e4ac59ef4c80ba55e2e037d6d0d37f819d4478ef1a2c1a8f425ac6e07387ff5c',
  'kvcr.runtime.core':'367142d2aaada4a2f347a7ee727133b9516bd99d780270a473ef681950fc6488','kvcr.runtime.progress':'0ce4de88aaf68836644c4f892468b9e43e287aad25f844749411775d0549382b','kvcr.backends.local_dram':'a14c555ee06eb4b68e6fe2a04cd0c7207d96ba2356e1bc987725ce72a595f293'}
 for name,wanted in names.items():
  path=root/'software'/(SOURCE+'/python' if name.startswith('sglang.') else 'kvcr-pool-index/src')/(name.replace('.','/')+'.py')
  assert proof['modules'][name]==dict(path=str(path),sha256=wanted) and sha(path)==wanted
 for name,wanted in [('nixl_wrapper','79ac70eed826672f09a21df7e71c8646654e761961c9783ecd8e145c0a7149a7'),('nixl_bindings','8cc0e0abd677a087023e6cc2295d09a19c8d2086722fa5d9415cfe69e137d1e2')]:
  row=proof['modules'][name];assert row['sha256']==wanted and '/nixl_cu13/' in row['path'] and sha(Path(row['path']))==wanted
 for rel,wanted in S.fixture_gate(root).items():
  path=root/'software'/SOURCE/rel;assert proof['source_files'][rel]==dict(path=str(path),sha256=wanted) and inventory[rel]==wanted
 for key,value in {**setup['worker_env'],**setup['transport']}.items():
  if key.startswith(('UCX_','NIXL_')):assert proof['env'][key]==value
 return proof

def parent_gate(root):
 evidence=S.native_gate(root);evidence.update(base.baseline_gate(root));base.pair.gate(root,root/'cases/gb300-kvcr')
 case=root/'cases'/PARENT;run=case/'runs'/PARENT_ATTEMPT;collected=read(case/'COLLECTION_COMPLETE.json');archive=root/f'{PARENT}-{PARENT_ATTEMPT}.tar.gz'
 assert collected['case']==PARENT and collected['attempt']==PARENT_ATTEMPT and collected['all_member_hashes_verified'] is True and collected['archive']==str(archive) and sha(archive)==collected['sha256'] and archive.stat().st_size==collected['bytes']
 manifest=case/f'{PARENT_ATTEMPT}-snapshot-manifest.json';files=read(manifest)['files'];assert len(files)==collected['verified_files']
 prep=read(case/'sources/prepare-receipt.json');check_static(case,prep['generated_sha256'])
 paths=[case/'sources/prepare-receipt.json',*[case/n for n in prep['generated_sha256']],*[run/n for n in ('setup.resolved.json','runner-complete.json','service-pids.json','service-closure.json','collection-integrity.json','driver.exit','client.exit','collector.exit','thread-cpu-collector.exit','provenance/component-sources.json','provenance/sglang-dirty.patch','provenance/kvcr-dirty.patch','provenance/source-bindings.json','provenance/candidate-source-inventory.json','provenance/gil-setting.json','provenance/gil-setting-closed.json')]]
 for path in paths:assert files[str(path.relative_to(root))]['sha256']==sha(path),'Closed GIL1ms parent proof changed'
 validation.validate_run(run,write_integrity=False);assert read(run/'runner-complete.json')['services_closed'] is True
 closure=read(run/'service-closure.json');assert closure['passed'] is True and closure['remaining']==[] and closure['empty_observations']>=2
 roots=read(run/'service-pids.json');assert len(roots)==6 and {(r['pid'],int(r['start'])) for r in roots}<={tuple(r) for r in closure['identities']}
 bindings=read(run/'provenance/source-bindings.json');assert bindings['sglang']==dict(head=S.HEAD,patch_sha256=S.BASE_PATCH)
 assert bindings['kvcr']==dict(head=base.KVCR_HEAD,patch_sha256=base.KVCR_PATCH_SHA,tree_sha256=base.KVCR_TREE_SHA)
 commit=root/'analysis'/PARENT/PARENT_ATTEMPT/'local-analysis-complete.json';a=read(commit)
 assert a['case']==PARENT and a['attempt']==PARENT_ATTEMPT and a['analysis_executed_on_local_server'] is True and a['canonical_copies_uploaded'] is True and a['raw_verification']['manifest_sha256']==sha(manifest)
 evidence.update(read(run/'provenance/baseline-evidence.json')['sha256'])
 # Parent proof files are retained explicitly; no inferred partial A+B source clone.
 paths += [case/'COLLECTION_COMPLETE.json',manifest,commit,root/S.RESULTS]
 paths += [root/STAGING/n for n in ('stage.py','freeze-receipt.json','ready-receipt.json','README.md','stage-receipt.json','candidate-sglang-abbrev8.patch','runtime-preflight.py','runtime-preflight.json','current-pod.json')]
 paths += [root/S.FIXTURE/n for n in (*read(root/S.FIXTURE/'freeze-receipt.json')['files_sha256'],'freeze-receipt.json','ready-receipt.json')]
 paths += [root/LEGACY/n for n in (*read(root/LEGACY/'freeze-receipt.json')['files_sha256'],'freeze-receipt.json')]
 launcher=root/'sources/gb300/native-poll-delay-launch-r1';launcher_freeze=launcher/'freeze-receipt.json';launcher_review=read(launcher/'review.json')
 assert launcher_review['passed'] is True and launcher_review['freeze_sha256']==sha(HERE/'freeze-receipt.json'),'Launcher review must bind this final harness'
 for n,d in read(launcher_freeze)['files_sha256'].items():
  path=launcher/n;assert path.resolve().is_relative_to(launcher) and not path.is_symlink() and sha(path)==d;paths.append(path)
 paths.append(launcher_freeze)
 evidence.update({str(p.relative_to(root)):sha(p) for p in paths})
 for n,d in evidence.items():assert sha(root/n)==d
 return prep,read(run/'setup.resolved.json'),evidence

def source_gate(root,setup,prep,provenance):
 if prep.get('case')==CASE:
  check_static(root/'cases'/CASE,prep['generated_sha256']);assert {p.name:sha(p) for p in HERE.glob('*.py')}==prep['harness_sha256']
 assert setup['sglang_source_directory']==SOURCE
 inventory=staged_source(root);runtime_preflight(root,setup,S.HEAD,inventory)
 result={};parent=root/'cases'/PARENT/'runs'/PARENT_ATTEMPT/'provenance'
 for component,directory in [('sglang',SOURCE),('kvcr','kvcr-pool-index')]:
  repo=root/'software'/directory;head=S.git(repo,'rev-parse','HEAD',text=True).strip();patch=S.git(repo,'diff','--binary','HEAD');digest=hashlib.sha256(patch).hexdigest()
  if component=='sglang':assert head==S.HEAD and patch==(root/STAGING/'candidate-sglang-abbrev8.patch').read_bytes()
  else:
   assert prep['source_head']==head==base.KVCR_HEAD and digest==base.KVCR_PATCH_SHA and base.source_tree(repo)==base.KVCR_TREE_SHA
   for n,d in prep['source_files_sha256'].items():assert (repo/n).resolve().is_relative_to(repo.resolve()) and sha(repo/n)==d
  metadata=provenance/'component-sources.json'
  if metadata.exists():assert read(metadata)[component]==dict(directory=str(repo),commit=head)
  path=provenance/(component+'-dirty.patch')
  if path.exists():assert sha(path)==digest
  path.write_bytes(patch);result[component]=dict(head=head,patch_sha256=digest)
 result['kvcr']['tree_sha256']=base.KVCR_TREE_SHA
 for name,value in [('source-bindings.json',result),('candidate-source-inventory.json',dict(source_directory=SOURCE,files=inventory,candidate_sha256=NEW_FILE_SHA))]:
  path=provenance/name
  if path.exists():assert read(path)==value,'Captured source proof changed'
  write(path,value)
 for rel in S.PAYLOAD:
  source=root/'software'/SOURCE/rel;dest=provenance/'native-poll-runtime'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(dest)==sha(source)
  else:shutil.copyfile(source,dest)
 return result

def case_gate(root):
 _,control,evidence=parent_gate(root);case=root/'cases'/CASE;prep=read(case/'sources/prepare-receipt.json')
 assert prep['status']=='prepared-not-launched' and prep['case']==CASE and prep['baseline_evidence_sha256']==evidence
 check_static(case,prep['generated_sha256']);setup=read(case/'config/setup.deployment.json');config_delta(root,control,setup)
 assert setup['pod']==prep['pod'];expected,_=generate(root,control,prep['pod'])
 for n,text in expected.items():assert (case/n).read_text()==text
 assert {p.name:sha(p) for p in HERE.glob('*.py')}==prep['harness_sha256']
 return setup,evidence

NATIVE_MARKER='KVCR native progress constructor '
STOCK_SHA='79ac70eed826672f09a21df7e71c8646654e761961c9783ecd8e145c0a7149a7'
OVERLAY_SHA='472e81be7133e73d86efa2e3e71fa5a9dda46377d7886e9ecae8346713e2f150'
BINDING_SHA='8cc0e0abd677a087023e6cc2295d09a19c8d2086722fa5d9415cfe69e137d1e2'

def native_lines(run,expected_generations=1):
 setup=read(run/'setup.resolved.json');canonical=Path(setup['canonical_root']).parents[1]
 inventory=read(run/'provenance/candidate-source-inventory.json');assert inventory['source_directory']==SOURCE and inventory['candidate_sha256']==NEW_FILE_SHA
 helper=S.PREFIX+'/native_progress_delay.py';assert inventory['files'][helper]==S.fixture_gate()[helper]
 roots=read(run/'service-pids.json');results=[]
 strict=read(run.parents[3]/S.NATIVE/'capture-gpu/bg_true_delay_1000us/result.json')['actors'][0]['receipt']['native_config_handoff']['native_config']['syncMode']
 for worker in range(4):
  path=run/f'worker-{worker}.log';raw=path.read_bytes();found=[];names=[]
  for number,line in enumerate(raw.splitlines(keepends=True),1):
   text=re.sub(r'\x1b\[[0-9;]*m','',line.decode(errors='strict'))
   if 'Initialized NIXL agent: ' in text:names.append((number,text.split('Initialized NIXL agent: ',1)[1].strip()))
   if NATIVE_MARKER not in text:continue
   record=json.loads(text.split(NATIVE_MARKER,1)[1]);h=record['handoff'];cfg=h['native_config'];source=record['source']
   assert record['event']=='kvcr_native_progress_constructor' and type(record['pid']) is int and record['pid']>0 and type(record['rank']) is int and record['rank']==0
   assert type(record['requested_delay_us']) is int and record['requested_delay_us']==1000 and record['pid']==h['pid']
   assert h['constructor_returned'] is True and type(h['native_tid']) is int and h['native_tid']>0 and re.fullmatch(r'sglang-\S+-0-[0-9a-f]{32}',h['agent_name'])
   for name,value in [('pthrDelay',1000),('lthrDelay',100000),('syncMode',strict)]:assert type(cfg[name]) is int and cfg[name]==value
   for name in ('useProgThread','useListenThread','captureTelemetry'):assert cfg[name] is True
   root=[r for r in roots if r['name']==f'worker-{worker}'];assert len(root)==1;argv=root[0]['argv'];flag='--hicache-storage-backend-extra-config';assert argv.count(flag)==1
   options=json.loads(argv[argv.index(flag)+1]);assert type(options['nixl_prog_thread_delay_us']) is int and options['nixl_prog_thread_delay_us']==1000 and options['python_gil_switch_interval_seconds']==.001
   assert type(cfg['listenPort']) is int and cfg['listenPort']==options['nixl_port']==setup['ports']['worker_nixl'][worker]
   assert source==dict(requested_delay_us=1000,stock_wrapper='/opt/sglang/lib/python3.12/site-packages/nixl_cu13/_api.py',stock_wrapper_sha256=STOCK_SHA,overlay_module='nixl_cu13._api_poll_delay_r1',overlay_sha256=OVERLAY_SHA,native_binding_sha256=BINDING_SHA)
   assert record['helper_path']==str(canonical/'software'/SOURCE/helper)
   previous_line=found[-1]['line'] if found else 0
   initializers=[n for n,name in names if name==h['agent_name'] and previous_line<n<number]
   assert initializers,'Native constructor record lacks a new real SDK initializer'
   found.append(dict(worker=f'worker-{worker}',generation=len(found)+1,sdk_init_line=initializers[-1],record=record,line=number,line_sha256=hashlib.sha256(line).hexdigest()))
  assert len(found)==expected_generations,'Unexpected native constructor generation count'
  assert len({r['record']['handoff']['agent_name'] for r in found})==1,'Engine reset changed linker layout endpoint'
  results.append(dict(**found[-1],generations=found,log_prefix_bytes=len(raw),log_prefix_sha256=hashlib.sha256(raw).hexdigest()))
 assert len({r['record']['pid'] for r in results})==4,'Native owning PID reused across workers'
 return results

def native_proof(run,proc=Path('/proc'),*,expected_generations=1,phase='before-preflight'):
 gil=gil_proof(run,proc);stored=read(run/'provenance/gil-setting.json');assert gil['service_pids_sha256']==stored['service_pids_sha256']
 for current,old in zip(gil['workers'],stored['workers']):
  for key in ('worker','pid','rank','actual','start','root_pid','root_start'):assert current[key]==old[key]
 closure=base.load('service-closure.py');rows=[]
 for item,worker in zip(native_lines(run,expected_generations),gil['workers']):
  record=item['record'];assert item['worker']==worker['worker'] and all(g['record']['pid']==worker['pid'] for g in item['generations'])
  pid=record['pid'];tid=record['handoff']['native_tid'];path=proc/str(pid)/'task'/str(tid)/'stat';first=closure.stat(path);second=closure.stat(path)
  assert first['pid']==tid and first['start']>=worker['start']
  for row in (first,second):assert row['state'] not in ('Z','X')
  assert all(first[k]==second[k] for k in ('pid','start','ppid','group','session')),'Constructor task identity raced'
  # Re-read process ancestry too; S/R changes are allowed, PID/start changes are not.
  for old in worker['ancestry']:
   current=closure.stat(proc/str(old['pid'])/'stat');assert current['state'] not in ('Z','X') and all(old[k]==current[k] for k in ('pid','start','ppid','group','session'))
  rows.append(dict(**item,pid=pid,start=worker['start'],root_pid=worker['root_pid'],root_start=worker['root_start'],native_tid=tid,native_tid_start=first['start']))
 return dict(passed=True,phase=phase,expected_generations=expected_generations,requested_delay_us=1000,workers=rows,service_pids_sha256=sha(run/'service-pids.json'),candidate_source_inventory_sha256=sha(run/'provenance/candidate-source-inventory.json'),candidate_sha256=NEW_FILE_SHA,scope='Actual constructor handoff/return and live PID/TID observation; not live native-delay getter, continuous measurement or background-helper thread identity')

def after_clear_proof(run,proc=Path('/proc')):
 prepath=run/'provenance/native-poll-setting-preclear.json';pre=read(prepath)
 assert pre['passed'] is True and pre['phase']=='before-preflight' and pre['expected_generations']==1
 clearpath=run/'validation-cache-clear.json';clear=read(clearpath)
 assert read(run/'preflight/validation.json')['passed'] is True
 assert clear['endpoint']=='sglkvcrreview.backend.clear_kv_blocks' and clear['transport']=='tcp'
 assert len(clear['workers'])==4 and len({r['instance_id'] for r in clear['workers']})==4 and all(r['result']['status']=='success' for r in clear['workers'])
 proof=native_proof(run,proc,expected_generations=2,phase='after-validation-cache-clear')
 for current,old in zip(proof['workers'],pre['workers']):
  assert current['worker']==old['worker'] and current['pid']==old['pid'] and current['start']==old['start']
  assert current['generations'][0]==old['generations'][0],'Startup generation changed'
  raw=(run/(old['worker']+'.log')).read_bytes()
  assert hashlib.sha256(raw[:old['log_prefix_bytes']]).hexdigest()==old['log_prefix_sha256'],'Startup prefix changed before clear proof'
  assert sum(len(line) for line in raw.splitlines(keepends=True)[:current['line']-1])>=old['log_prefix_bytes'],'New constructor predates startup checkpoint'
 proof.update(validation_cache_clear_sha256=sha(clearpath),preclear_proof_sha256=sha(prepath),boundary='After all original clear RPCs return and before collector/AgentX launch; latest live engine, no claim of an SDK live-delay getter')
 return proof


def verify_closed_native(run):
 proof=read(run/'provenance/native-poll-setting.json');assert proof['passed'] is True and proof['phase']=='after-validation-cache-clear' and proof['expected_generations']==2 and proof['requested_delay_us']==1000 and proof['candidate_sha256']==NEW_FILE_SHA
 assert proof['service_pids_sha256']==sha(run/'service-pids.json') and proof['candidate_source_inventory_sha256']==sha(run/'provenance/candidate-source-inventory.json')
 closure=read(run/'service-closure.json');assert closure['passed'] is True and closure['remaining']==[];known={tuple(r) for r in closure['identities']}
 assert proof['validation_cache_clear_sha256']==sha(run/'validation-cache-clear.json') and proof['preclear_proof_sha256']==sha(run/'provenance/native-poll-setting-preclear.json')
 observed=native_lines(run,2);assert len(proof['workers'])==4
 for item,old in zip(observed,proof['workers']):
  for key in ('worker','record','line','line_sha256','generation','sdk_init_line','generations'):assert item[key]==old[key],'Closed native constructor evidence changed'
  raw=(run/(item['worker']+'.log')).read_bytes();assert len(raw)>=old['log_prefix_bytes'] and hashlib.sha256(raw[:old['log_prefix_bytes']]).hexdigest()==old['log_prefix_sha256'],'Native startup log prefix changed'
  assert (old['pid'],old['start']) in known and (old['root_pid'],old['root_start']) in known
 return dict(passed=True,native_setting_proof_sha256=sha(run/'provenance/native-poll-setting.json'),worker_log_sha256={f'worker-{i}':sha(run/f'worker-{i}.log') for i in range(4)},constructor_TID_scope='Archived live observation; enclosing process closure, no /proc reads after stop',parity_result=False)
