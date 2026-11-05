"""Scoped entry to original fresh-service controller; no analysis on GPU pod."""
import argparse
import importlib.util
import os
from pathlib import Path
spec=importlib.util.spec_from_file_location('first_page_common',Path(__file__).with_name('common.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',choices=[c.ATTEMPT],default=c.ATTEMPT);a=p.parse_args()
    assert os.environ.get('KVCR_BENCH_OBSERVER') not in ('1','true'),'Observer overlay is not part of this fullrun arm'
    for flag in ('KVCR_LOCAL_COPY_OBSERVER','KVCR_TRANSFER_DIAGNOSTICS'):assert os.environ.get(flag) not in ('1','true'),'No diagnostic overlay in this arm'
    setup=c.read(c.ROOT/'cases'/c.CASE/'config/setup.deployment.json')
    assert os.environ.get('POD_NAME',os.environ.get('HOSTNAME'))==setup['pod'],'Launch from explicitly prepared replacement pod'
    runner=c.base;original_load=runner.load
    def dispatch(name):
        return c.load(c.HERE/'archive.py') if name=='finalize-case.py' else original_load(name)
    # Fresh module instance only: no shared helper files or other controllers change.
    runner.ROOT=c.ROOT;runner.CASE=c.CASE;runner.CASES=(c.CASE,)
    original_capture=runner.capture_client
    def capture(*args,**kwargs):
        run=args[2];proof_path=run/'provenance/gil-setting.json'
        assert not proof_path.exists(),'Never overwrite GIL startup proof'
        c.source_gate(c.ROOT,setup,c.read(c.ROOT/'cases'/c.CASE/'sources/prepare-receipt.json'),run/'provenance')
        c.write(proof_path,c.gil_proof(run))
        native_path=run/'provenance/native-poll-setting.json'
        assert not native_path.exists(),'Never overwrite actual native constructor proof'
        c.write(native_path,c.native_proof(run))
        return original_capture(*args,**kwargs)
    runner.capture_client=capture
    runner.case_gate=c.case_gate;runner.source_gate=c.source_gate;runner.load=dispatch
    runner.main(['--case',c.CASE,'--attempt',a.attempt])

if __name__=='__main__':main()
