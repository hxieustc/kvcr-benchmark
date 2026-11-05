import ast,importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def load(name):
 s=importlib.util.spec_from_file_location(name,HERE/name);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Tests(unittest.TestCase):
 def setUp(self):
  self.assertTrue((HERE/'common.py').is_file(),'New private harness required')
  self.c=load('common.py');self.t=tempfile.TemporaryDirectory(dir=HERE);self.addCleanup(self.t.cleanup);self.root=Path(self.t.name)
  source=Path('/tmp/kvcr-parity-gil-1ms-r1/cases/gb300-kvcr-gil-interval');self.old=self.root/'cases'/self.c.PARENT
  for name in ('scripts','config'):shutil.copytree(source/name,self.old/name)
  for p in self.old.rglob('*'):
   if p.is_file():p.write_text(p.read_text().replace('/workspace/sglang-kvcr-parity-20261031-r1',str(self.root)))
  self.control=self.c.read(self.old/'config/setup.deployment.json')
  shutil.copytree(ROOT/self.c.S.FIXTURE,self.root/self.c.S.FIXTURE,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 def test_generation_only_new_source_option_and_paths(self):
  files,setup=self.c.generate(self.root,self.control,'replacement-r2');self.c.config_delta(self.root,self.control,setup)
  self.assertEqual(setup['kvcr_options'],dict(self.control['kvcr_options'],nixl_prog_thread_delay_us=1000))
  self.assertEqual(setup['profiling_seconds'],1800);self.assertEqual(setup['transport'],self.control['transport']);self.assertEqual(setup['worker_env'],self.control['worker_env'])
  self.assertIn('/software/'+self.c.SOURCE+'/python',files['scripts/services.py']);self.assertNotIn('/software/sglang-gil-interval/python',files['scripts/services.py'])
  for field,value in [('mamba_checkpoint_capacity',2176),('dram_size_gb',224),('python_gil_switch_interval_seconds',.005),('nixl_prog_thread_delay_us',True)]:
   bad=json.loads(json.dumps(setup));bad['kvcr_options'][field]=value
   with self.assertRaises(AssertionError):self.c.config_delta(self.root,self.control,bad)
 def test_full_inventory_rejects_unrelated_or_missing_added_helpers(self):
  before={self.c.SG_FILE:self.c.S.BASE,'python/sglang/_version.py':'old','test-ignored.py':'test'}
  after=self.c.S.expected_inventory(before);self.c.S.check_inventory(before,after)
  for change in (dict(after,extra='bad'),{k:v for k,v in after.items() if not k.endswith('native_progress_delay.py')}):
   with self.assertRaises(AssertionError):self.c.S.check_inventory(before,change)
 def test_dry_prepare_no_case_and_source_failure_no_writes(self):
  p=load('prepare.py');(self.old/'sources').mkdir();self.c.write(self.old/'sources/prepare-receipt.json',{})
  with patch.object(p.c,'parent_gate',return_value=({},self.control,{})),patch.object(p.c,'source_gate',side_effect=AssertionError('source incomplete')):
   with self.assertRaises(AssertionError):p.prepare(self.root,pod='replacement-r2')
   self.assertFalse((self.root/'cases'/self.c.CASE).exists())
  with patch.object(p.c,'parent_gate',return_value=({},self.control,{})),patch.object(p.c,'source_gate',return_value={}),patch.object(p.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(300*2**30,0,300*2**30)):
   self.assertEqual(p.prepare(self.root,pod='replacement-r2')['status'],'preflight-only');self.assertFalse((self.root/'cases'/self.c.CASE).exists())
 def test_archive_all_members_never_overwrites_original_workload_validator(self):
  a=load('archive.py');case=self.root/'cases'/self.c.CASE;case.mkdir();f=case/'raw';f.write_bytes(b'raw')
  r=a.pack(self.root,case,self.c.ATTEMPT,{str(f.relative_to(self.root)):dict(bytes=3,sha256=self.c.sha(f))});self.assertTrue(r['all_member_hashes_verified'])
  with self.assertRaises(AssertionError):a.pack(self.root,case,self.c.ATTEMPT,{})
  tree=ast.parse(Path(a.validation.__file__).read_text());method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_run');self.assertIn('(1800, 48, 5)',ast.unparse(method))
 def native_fixture(self):
  run=self.root/'cases'/self.c.CASE/'runs'/self.c.ATTEMPT;run.mkdir(parents=True);(run/'provenance').mkdir()
  proc=self.root/'proc';proc.mkdir();roots=[]
  setup={'canonical_root':str(run.parents[1]),'pod':'test-pod','ports':{'worker_nixl':[24170+i for i in range(4)]}}
  self.c.write(run/'setup.resolved.json',setup)
  payload=self.c.S.fixture_gate(ROOT)
  self.c.write(run/'provenance/candidate-source-inventory.json',dict(source_directory=self.c.SOURCE,files=payload,candidate_sha256=self.c.NEW_FILE_SHA))
  self.c.write(self.root/self.c.S.NATIVE/'capture-gpu/bg_true_delay_1000us/result.json',{'actors':[{'receipt':{'native_config_handoff':{'native_config':{'syncMode':7}}}}]})
  def stat(path,pid,ppid,start,state='S'):
   path.parent.mkdir(parents=True,exist_ok=True);path.write_text(f'{pid} (python) '+' '.join([state,str(ppid),'100','100']+['0']*15+[str(start)]))
  for i in range(4):
   owner=200+i;tid=300+i;parent=100+i;agent='sglang-test-pod-0-'+f'{i+1:032x}'
   stat(proc/str(parent)/'stat',parent,1,10);stat(proc/str(owner)/'stat',owner,parent,20);stat(proc/str(owner)/'task'/str(tid)/'stat',tid,parent,21)
   kv={'nixl_prog_thread_delay_us':1000,'python_gil_switch_interval_seconds':.001,'nixl_port':24170+i}
   roots.append(dict(name=f'worker-{i}',pid=parent,start='10',argv=['python','--hicache-storage-backend-extra-config',json.dumps(kv)]))
   record=dict(event='kvcr_native_progress_constructor',pid=owner,rank=0,requested_delay_us=1000,helper_path=str(self.root/'software'/self.c.SOURCE/self.c.S.PREFIX/'native_progress_delay.py'),handoff=dict(pid=owner,native_tid=tid,agent_name=agent,constructor_returned=True,native_config=dict(pthrDelay=1000,useProgThread=True,useListenThread=True,lthrDelay=100000,captureTelemetry=True,syncMode=7,listenPort=24170+i)),source=dict(requested_delay_us=1000,stock_wrapper='/opt/sglang/lib/python3.12/site-packages/nixl_cu13/_api.py',stock_wrapper_sha256='79ac70eed826672f09a21df7e71c8646654e761961c9783ecd8e145c0a7149a7',overlay_module='nixl_cu13._api_poll_delay_r1',overlay_sha256='472e81be7133e73d86efa2e3e71fa5a9dda46377d7886e9ecae8346713e2f150',native_binding_sha256='8cc0e0abd677a087023e6cc2295d09a19c8d2086722fa5d9415cfe69e137d1e2'))
   (run/f'worker-{i}.log').write_text('Initialized NIXL agent: '+agent+'\n'+f'KVCR Python GIL switch interval pid={owner} rank=0 before=0.005 actual=0.001\n'+'KVCR native progress constructor '+json.dumps(record)+'\n')
  self.c.write(run/'service-pids.json',roots);return run,proc
 def test_four_actual_constructor_proof_and_closed_prefix_no_live_proc(self):
  self.assertTrue(callable(getattr(self.c,'native_proof',None)),'Preclient native proof required')
  run,proc=self.native_fixture();gil=self.c.gil_proof(run,proc);self.c.write(run/'provenance/gil-setting.json',gil)
  self.c.write(run/'provenance/native-poll-setting-preclear.json',self.c.native_proof(run,proc))
  self.append_clear_generation(run,proc)
  proof=self.c.after_clear_proof(run,proc);self.assertEqual(len(proof['workers']),4)
  self.c.write(run/'provenance/native-poll-setting.json',proof)
  self.c.write(run/'service-closure.json',dict(passed=True,remaining=[],identities=[[r['pid'],r['start']] for w in gil['workers'] for r in w['ancestry']]))
  self.assertTrue(self.c.verify_closed_native(run)['passed'])
  p=run/'worker-0.log';p.write_text(p.read_text()+'profile append\n');self.assertTrue(self.c.verify_closed_native(run)['passed'])
  p.write_text(p.read_text().replace('before=0.005','before=0.006'))
  with self.assertRaises(AssertionError):self.c.verify_closed_native(run)
 def test_native_missing_duplicate_wrong_value_source_or_stale_tid_reject(self):
  self.assertTrue(callable(getattr(self.c,'native_proof',None)),'Preclient native proof required')
  run,proc=self.native_fixture();self.c.write(run/'provenance/gil-setting.json',self.c.gil_proof(run,proc))
  p=run/'worker-0.log';original=p.read_text()
  for change in ('missing','duplicate','wrong-pid','wrong-value','bg-off','bad-source','wrong-agent','TID-reuse'):
   with self.subTest(change=change):
    lines=original.splitlines(True);rec=json.loads(lines[-1].split('KVCR native progress constructor ')[1])
    if change=='missing':lines=lines[:-1]
    elif change=='duplicate':lines.append(lines[-1])
    else:
     if change=='wrong-pid':rec['pid']=999
     if change=='wrong-value':rec['handoff']['native_config']['pthrDelay']=0
     if change=='bg-off':rec['handoff']['native_config']['useProgThread']=False
     if change=='bad-source':rec['source']['overlay_sha256']='0'*64
     if change=='wrong-agent':rec['handoff']['agent_name']='other'
     lines[-1]='KVCR native progress constructor '+json.dumps(rec)+'\n'
    p.write_text(''.join(lines))
    if change=='TID-reuse':(proc/'200/task/300/stat').write_text((proc/'200/task/300/stat').read_text().replace(' 21',' 19'))
    with self.assertRaises((AssertionError,FileNotFoundError)):self.c.native_proof(run,proc)
    p.write_text(original)
 def append_clear_generation(self,run,proc):
  for i in range(4):
   path=run/f'worker-{i}.log';record=json.loads(path.read_text().splitlines()[-1].split('KVCR native progress constructor ')[1]);record['handoff']['native_tid']=400+i
   path.write_text(path.read_text()+'Initialized NIXL agent: '+record['handoff']['agent_name']+'\n'+'KVCR native progress constructor '+json.dumps(record)+'\n')
   old=proc/str(200+i)/'task'/str(300+i)/'stat';data=old.read_text().replace(str(300+i)+' (',str(400+i)+' (').rsplit(' ',1)[0]+' 30';old.unlink()
   target=old.parent.parent/str(400+i)/'stat';target.parent.mkdir();target.write_text(data)
  self.c.write(run/'validation-cache-clear.json',dict(endpoint='sglkvcrreview.backend.clear_kv_blocks',transport='tcp',workers=[dict(instance_id=i,begin_utc=1.,end_utc=2.,result={'status':'success'}) for i in range(4)]))
  self.c.write(run/'preflight/validation.json',dict(passed=True))
 def test_two_native_generations_after_clear_gate_and_final_stability(self):
  self.assertTrue(callable(getattr(self.c,'after_clear_proof',None)),'After-clear actual engine gate required')
  run,proc=self.native_fixture();self.c.write(run/'provenance/gil-setting.json',self.c.gil_proof(run,proc))
  pre=self.c.native_proof(run,proc);self.c.write(run/'provenance/native-poll-setting-preclear.json',pre)
  self.append_clear_generation(run,proc)
  after=self.c.after_clear_proof(run,proc);self.assertEqual([len(w['generations']) for w in after['workers']],[2]*4)
  self.assertEqual([w['native_tid'] for w in after['workers']],[400,401,402,403])
  self.c.write(run/'provenance/native-poll-setting.json',after)
  gil=self.c.read(run/'provenance/gil-setting.json');self.c.write(run/'service-closure.json',dict(passed=True,remaining=[],identities=[[r['pid'],r['start']] for w in gil['workers'] for r in w['ancestry']]))
  self.assertTrue(self.c.verify_closed_native(run)['passed'])
  log=run/'worker-0.log';latest=log.read_text().splitlines()[-1];log.write_text(log.read_text()+latest+'\n')
  with self.assertRaises(AssertionError):self.c.verify_closed_native(run)
 def test_real_capture_wrapper_blocks_original_client_on_native_proof_failure(self):
  from types import SimpleNamespace
  tree=ast.parse((HERE/'run.py').read_text());main=next(n for n in tree.body if getattr(n,'name',None)=='main');capture=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='capture')
  run=self.root/'capture-run';(run/'provenance').mkdir(parents=True);calls=[]
  stub=SimpleNamespace(ROOT=self.root,CASE=self.c.CASE,source_gate=lambda *a:None,read=lambda *a:{},gil_proof=lambda r:{'passed':True},write=lambda *a:None,native_proof=lambda r:(_ for _ in ()).throw(AssertionError('missing actual native owners')))
  env={'c':stub,'setup':{},'original_capture':lambda *a,**kw:calls.append(a)}
  exec(compile(ast.Module(body=[capture],type_ignores=[]),str(HERE/'run.py'),'exec'),env)
  with self.assertRaises(AssertionError):env['capture']('python','case',run,'attempt',{},lambda *a:None)
  self.assertEqual(calls,[])
if __name__=='__main__':unittest.main()


