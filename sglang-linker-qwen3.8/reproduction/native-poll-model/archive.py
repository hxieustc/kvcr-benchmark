"""Validate/retain closed raw only; no benchmark analysis."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time
spec=importlib.util.spec_from_file_location('first_page_common',Path(__file__).with_name('common.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
validation=c.validation


def pack(root,case,attempt,files):
    archive=root/f'{c.CASE}-{attempt}.tar.gz';manifest=case/f'{attempt}-snapshot-manifest.json'
    assert not archive.exists() and not manifest.exists() and not (case/'COLLECTION_COMPLETE.json').exists(),'Never overwrite raw capture'
    c.write(manifest,dict(files=files,utc=time.time()))
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for name in sorted(files):tar.add(root/name,arcname=name,recursive=False)
        tar.add(manifest,arcname='snapshot-manifest.json',recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(files)+1 and {m.name for m in members}==set(files)|{'snapshot-manifest.json'}
        for member in members:
            assert member.isfile();stream=tar.extractfile(member)
            if member.name=='snapshot-manifest.json':assert stream.read()==manifest.read_bytes()
            else:assert member.size==files[member.name]['bytes'] and hashlib.file_digest(stream,'sha256').hexdigest()==files[member.name]['sha256']
    receipt=dict(case=c.CASE,attempt=attempt,baseline_attempt=c.base.BASELINE,parent_case=c.PARENT,parent_attempt=c.PARENT_ATTEMPT,
        archive=str(archive),bytes=archive.stat().st_size,sha256=c.sha(archive),verified_files=len(files),all_member_hashes_verified=True,
        pod_retained=True,analysis_location='local server after verified download',utc=time.time())
    c.write(case/'COLLECTION_COMPLETE.json',receipt);return receipt


def finalize(root,case_name,attempt):
    assert case_name==c.CASE and attempt==c.ATTEMPT
    case=root/'cases'/case_name;run=case/'runs'/attempt
    with (root/'gb300-finalize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        setup,evidence=c.case_gate(root)
        c.source_gate(root,setup,c.read(case/'sources/prepare-receipt.json'),run/'provenance')
        complete=c.read(run/'runner-complete.json');assert complete['attempt']==attempt and complete['services_closed'] is True
        validation.validate_run(run)
        c.base.load('service-closure.py').confirm_closed(run)
        c.base.pair.wait_idle()
        c.write(run/'provenance/gil-setting-closed.json',c.verify_closed_gil(run))
        c.write(run/'provenance/native-poll-setting-closed.json',c.verify_closed_native(run))
        proof=c.read(run/'provenance/baseline-evidence.json');assert proof['sha256']==evidence
        for name,digest in evidence.items():assert c.sha(root/name)==digest
        selected=[root/'scripts',root/'runtime-validation.receipt.json',root/'runtime-validation.exit',root/'sources/dynamo-build.json',
            root/'sources/online-bootstrap-receipt.json',root/'sources/dynamo.patch',root/'bootstrap-online.exit',root/'assets-gb300/model-validation.json',
            c.HERE]
        selected += [case/p for p in ('scripts','config','sources',f'runs/{attempt}',f'logs/{attempt}.log')]+[root/n for n in evidence]
        files={}
        for path in selected:
            assert path.exists(),path
            for p in ([path] if path.is_file() else path.rglob('*')):
                if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
                parts=p.relative_to(root).parts
                if len(parts)>4 and parts[0]=='cases' and parts[2]=='runs' and parts[4]=='analysis':continue
                files[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=c.sha(p))
        assert set(evidence)<=set(files),'Archive must include listed parent/control proofs'
        return pack(root,case,attempt,files)
