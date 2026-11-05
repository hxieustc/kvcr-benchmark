"""Prepare a new full1800s case; default read-only checks, never launch."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
spec=importlib.util.spec_from_file_location('first_page_common',Path(__file__).with_name('common.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def prepare(root,execute=False,*,pod):
    root=Path(root).resolve();new=root/'cases'/c.CASE;old=root/'cases'/c.PARENT
    assert not new.exists() and not new.is_symlink(),'Never overwrite a prepared case'
    original,control,evidence=c.parent_gate(root)
    files,setup=c.generate(root,control,pod=pod)
    with tempfile.TemporaryDirectory(prefix='native-poll-source-') as folder:c.source_gate(root,setup,original,Path(folder))
    assert shutil.disk_usage(root).free>100*2**30,'Workspace headroom below100GiB'
    receipt={**original,'status':'prepared-not-launched' if execute else 'preflight-only','case':c.CASE,
        'pod':pod,'sglang_source_directory':c.SOURCE,'parent_case':c.PARENT,'parent_attempt':c.PARENT_ATTEMPT,'baseline_evidence_sha256':evidence,
        'generated_sha256':{n:c.hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()},
        'harness_sha256':{p.name:c.sha(p) for p in c.HERE.glob('*.py')},'source_change':dict(parent_sha256=c.S.BASE,new_file_sha256=c.NEW_FILE_SHA,payload_sha256=c.S.fixture_gate(root))}
    if not execute:return receipt
    with tempfile.TemporaryDirectory(prefix='.first-page-prepare-',dir=root/'cases') as folder:
        stage=Path(folder)/c.CASE;stage.mkdir()
        for name in ('logs','runs','tmp'):(stage/name).mkdir()
        shutil.copytree(old/'sources',stage/'sources',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','client-input-maps'))
        for path in old.iterdir():
            if path.is_symlink():(stage/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
        for name,text in files.items():
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        shutil.copytree(root/c.S.FIXTURE,stage/'sources/native-poll-delay-approved',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        c.write(stage/'sources/prepare-receipt.json',receipt)
        assert not new.exists() and not new.is_symlink();stage.rename(new)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=c.ROOT);p.add_argument('--execute',action='store_true');p.add_argument('--pod',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.root,a.execute,pod=a.pod),indent=2))
