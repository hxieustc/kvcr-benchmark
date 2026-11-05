#!/usr/bin/env python3
"""Provisional, offline ARM bootstrap. Default is read-only preflight, not a build."""
import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile

HEAD='3f0ef62c22ce2e212fd7a6394ef98d84f5f9a23b'
TOOLS={'maturin':'1.9.6','patchelf':'0.19.1.0','setuptools-rust':'1.11.1',
       'setuptools-scm':'9.2.2','hatchling':'1.27.0'}
REQUIRED_STACK=('torch','nixl','nixl-cu13','sglang','sglang-kernel','flashinfer-python','triton')
TARGET_STACK={'torch':'2.14.1+cu130','nixl':'1.5.0','nixl-cu13':'1.5.0','sglang':'0.5.21',
              'sglang-kernel':'0.4.10','flashinfer-python':'0.7.2','flashinfer-cubin':'0.7.2',
              'flashinfer-jit-cache':'0.7.2+cu130','triton':'3.8.0',
              'mooncake-transfer-engine-cuda13':'0.3.13'}
IMAGE_QUERY='''import importlib.metadata as m,json,platform,sys,sysconfig
norm=lambda n:n.lower().replace('_','-').replace('.','-')
names={d.metadata['Name'] for d in m.distributions() if d.metadata['Name']}
print(json.dumps(dict(machine=platform.machine(),python=f'{sys.version_info.major}.{sys.version_info.minor}',
 purelib=sysconfig.get_path('purelib'),paths=sys.path,packages={norm(n):m.version(n) for n in names})))'''


def fail(message):raise RuntimeError(message)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def workspace(path):
    path=path.resolve()
    if not path.is_relative_to('/workspace') or path==Path('/workspace'):fail('output must be below /workspace')
    return path


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='fresh output prefix; never reused/deleted')
    p.add_argument('--inputs',type=Path,required=True,help='staged source, locks, wheels, ARM Rust/Cargo cache')
    p.add_argument('--image-python',type=Path,required=True,help='absolute interpreter from inspected ARM image')
    p.add_argument('--rust-bin',type=Path,help='ARM rustc/cargo directory; default INPUTS/rust/bin')
    p.add_argument('--source-diff-sha256',default=hashlib.sha256(b'').hexdigest())
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--min-free-gib',type=int,default=100)
    p.add_argument('--allow-locked-fetch',action='store_true',help='permit Cargo network fetch with --locked; default offline')
    p.add_argument('--execute',action='store_true',help='create a fresh venv and build; otherwise preflight only')
    return p


def build_env(args,parent=None):
    env=dict(os.environ if parent is None else parent)
    for key in list(env):
        if key.startswith(('PYTHON','PIP_','UV_','CARGO_','RUSTUP_')) or key in (
                'VIRTUAL_ENV','RUSTFLAGS','RUSTC','RUSTDOC','CARGO_ENCODED_RUSTFLAGS'):
            env.pop(key)
    rust=args.rust_bin or args.inputs/'rust/bin'
    env.update(PATH=f'{args.root}/env/bin:{rust}:{args.image_python.parent}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
               PYTHONNOUSERSITE='1',PIP_NO_INDEX='1',PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK='1',RUSTUP_TOOLCHAIN='1.96.1',RUSTUP_AUTO_INSTALL='0',
               RUSTUP_HOME=str(args.inputs/'rustup-arm64'),CARGO_HOME=str(args.inputs/'cargo-cache'),
               CARGO_TARGET_DIR=str(args.root/'target-aarch64'),CARGO_BUILD_JOBS=str(args.jobs),
               TMPDIR=str(args.root/'tmp'))
    return env


def validate_image(report,lock):
    if report['machine']!='aarch64':fail('requires inspected aarch64 image; no x86/cross-build fallback')
    if report['python']!=lock['python']:fail('image Python differs from explicit stack lock')
    if not re.search(r'sha256:[0-9a-f]{64}$',lock.get('image_digest','')):fail('full caller-verified image digest required')
    for path in [report['purelib'],*report['paths']]:
        if 'x86_64' in path or Path(path).is_relative_to('/workspace'):fail('old workspace/x86 image package path rejected: '+path)
    for name in REQUIRED_STACK:
        if name not in lock['packages']:fail('missing image stack pin: '+name)
    for name,version in lock['packages'].items():
        if report['packages'].get(name)!=version:fail('image stack version mismatch: '+name)
    if any(lock['packages'][n]!='1.5.0' for n in ('nixl','nixl-cu13')):fail('NIXL 1.5.0 required')


def checked_wheel(path,expected):
    if sha(path)!=expected:fail('wheel SHA mismatch: '+str(path))
    with zipfile.ZipFile(path) as z:
        metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
        wheel=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))))
        tags=wheel.get_all('Tag',[])
        if not tags or any(not (t.endswith('-any') or t.endswith('_aarch64')) for t in tags):fail('non-ARM wheel: '+path.name)
    name=re.sub(r'[-_.]+','-',metadata['Name'].lower());version=metadata['Version']
    if name in TOOLS and version!=TOOLS[name]:fail('build-tool pin mismatch: '+name)
    return name,version


def validate_runtime(image_packages,selected,constraints):
    for name,version in TARGET_STACK.items():
        if constraints.get(name)!=version:fail('exact old-stack target pin required: '+name+'=='+version)
    for name,version in selected.items():
        if constraints.get(name)!=version:fail('unapproved runtime wheel/version: '+name)
    effective={**image_packages,**selected}
    for name,version in constraints.items():
        if effective.get(name)!=version:fail('missing exact ARM runtime artifact: '+name+'=='+version)


def run(command,env,cwd=None):
    return subprocess.check_output([str(x) for x in command],env=env,cwd=cwd,text=True).strip()


def build_commands(args):
    py=str(args.root/'env/bin/python');source=args.root/'dynamo-src';out=str(args.root/'wheels')
    return [[str(args.root/'env/bin/maturin'),'build','--locked',*([] if args.allow_locked_fetch else ['--offline']),'--release',
             '--features','select-service,kv-indexer','--target','aarch64-unknown-linux-gnu',
             '--interpreter',py,'--out',out],
            [py,'-I','-m','pip','wheel','--no-deps','--no-index','--no-build-isolation',str(source),'--wheel-dir',out]]


def preflight(args):
    args.root=workspace(args.root)
    if args.root.exists():fail('output already exists; choose a new attempt prefix')
    if not args.image_python.is_absolute():fail('image-python must be absolute')
    if not 1<=args.jobs<=8 or args.min_free_gib<1:fail('invalid build/disk bounds')
    parent=next(p for p in args.root.parents if p.exists())
    if shutil.disk_usage(parent).free<args.min_free_gib*2**30:fail('workspace headroom below preflight bound')
    env=build_env(args);probe=dict(env);probe.pop('TMPDIR')  # Fresh output does not exist yet.
    lock=json.loads((args.inputs/'image-stack-lock.json').read_text())
    image=json.loads(run([args.image_python,'-I','-c',IMAGE_QUERY],probe));validate_image(image,lock)
    if 'x86_64' in env.get('LD_LIBRARY_PATH','') or '/workspace/' in env.get('LD_LIBRARY_PATH',''):fail('review inherited LD_LIBRARY_PATH before bootstrap')
    rust=run(['rustc','-vV'],probe)
    if 'release: 1.96.1' not in rust or 'host: aarch64-unknown-linux-gnu' not in rust:fail('preinstalled ARM Rust 1.96.1 required')
    if not run(['cargo','--version'],probe).startswith('cargo 1.96.1 '):fail('Cargo 1.96.1 required')
    for tool in ('cc','c++','pkg-config','protoc'):
        if not shutil.which(tool,path=env['PATH']):fail('missing system build prerequisite: '+tool)
    subprocess.run(['pkg-config','--exists','openssl','hwloc','libudev'],env=probe,check=True)
    source=args.inputs/'dynamo'
    if run(['git','-C',source,'rev-parse','HEAD'],probe)!=HEAD:fail('wrong Dynamo head')
    if run(['git','-C',source,'ls-files','--others','--exclude-standard'],probe):fail('untracked source requires explicit separate receipt; not silently copied')
    diff=subprocess.check_output(['git','-C',str(source),'diff','--binary','HEAD'],env=probe)
    if hashlib.sha256(diff).hexdigest()!=args.source_diff_sha256:fail('Dynamo diff does not match approved SHA')
    cargo=tomllib.loads((source/'lib/bindings/python/Cargo.toml').read_text())
    if not {'select-service','kv-indexer'}<=cargo.get('features',{}).keys():fail('expected Cargo features missing; inspect exact source')
    tools=json.loads((args.inputs/'build-tools-lock.json').read_text());seen={}
    if not isinstance(tools,dict) or not 1<=len(tools)<=64:fail('invalid wheel manifest')
    for filename,digest in tools.items():
        if Path(filename).name!=filename or not filename.endswith('.whl'):fail('invalid wheel filename')
        name,version=checked_wheel(args.inputs/'wheels'/filename,digest)
        if name in seen:fail('duplicate wheel distribution')
        seen[name]=version
        if name not in TOOLS and image['packages'].get(name,version)!=version:fail('tool dependency would change image package: '+name)
    if not TOOLS.keys()<=seen.keys():fail('missing pinned build-tool wheels')
    if 'cmake' not in seen and not shutil.which('cmake',path=env['PATH']):fail('CMake must be staged in workspace tools or as a hash-pinned ARM wheel')
    runtime=json.loads((args.inputs/'runtime-wheels-lock.json').read_text())
    constraints=json.loads((args.inputs/'runtime-target-lock.json').read_text())
    selected={}
    if not isinstance(runtime,dict) or len(runtime)>128:fail('invalid runtime wheel manifest')
    for filename,digest in runtime.items():
        if Path(filename).name!=filename or not filename.endswith('.whl'):fail('invalid runtime wheel filename')
        name,version=checked_wheel(args.inputs/'runtime-wheels'/filename,digest)
        if name in selected:fail('duplicate runtime wheel distribution')
        selected[name]=version
    validate_runtime(image['packages'],selected,constraints)
    return dict(status='preflight-only',head=HEAD,source_diff_sha256=args.source_diff_sha256,
                cargo_lock_sha256=sha(source/'Cargo.lock'),image=image,image_lock=lock,tool_wheels=tools,
                rustc=rust,runtime_wheels=runtime,runtime_constraints=constraints,
                runtime_overrides={n:dict(image=image['packages'].get(n),target=v) for n,v in constraints.items() if image['packages'].get(n)!=v},
                commands=build_commands(args),note='Image digest is supplied/verified by main, not independently attested here.')


def execute(args,receipt):
    args.root.mkdir(parents=True,exist_ok=False)
    env=build_env(args);py=args.root/'env/bin/python';receipt['status']='failed'
    for name in ('tmp','wheels','tool-wheels','runtime-wheels','dynamo-src'):(args.root/name).mkdir()
    try:
        # Source snapshot + approved diff: build backends never modify input/history.
        with (args.root/'source.tar').open('wb') as out:
            subprocess.run(['git','-C',str(args.inputs/'dynamo'),'archive',HEAD],env=env,stdout=out,check=True)
        with tarfile.open(args.root/'source.tar') as tar:tar.extractall(args.root/'dynamo-src',filter='data')
        diff=subprocess.check_output(['git','-C',str(args.inputs/'dynamo'),'diff','--binary','HEAD'],env=env)
        if hashlib.sha256(diff).hexdigest()!=receipt['source_diff_sha256']:fail('source diff changed after preflight')
        if diff:subprocess.run(['git','apply','-'],cwd=args.root/'dynamo-src',env=env,input=diff,check=True)
        subprocess.run([str(args.image_python),'-I','-m','venv','--without-pip',str(args.root/'env')],env=env,check=True)
        purelib=Path(run([py,'-I','-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],env))
        (purelib/'arm-image-packages.pth').write_text('import site; site.addsitedir('+repr(receipt['image']['purelib'])+')\n')
        runtime=[]
        for name,digest in receipt['runtime_wheels'].items():
            target=args.root/'runtime-wheels'/name;shutil.copyfile(args.inputs/'runtime-wheels'/name,target)
            checked_wheel(target,digest);runtime.append(str(target))
        if runtime:
            subprocess.run([str(py),'-I','-m','pip','install','--no-deps','--no-index','--ignore-installed',*runtime],env=env,check=True)
        tools=[]
        for name,digest in receipt['tool_wheels'].items():
            target=args.root/'tool-wheels'/name;shutil.copyfile(args.inputs/'wheels'/name,target)
            checked_wheel(target,digest);tools.append(str(target))
        subprocess.run([str(py),'-I','-m','pip','install','--no-deps','--no-index','--ignore-installed',*tools],env=env,check=True)
        for command,cwd in zip(build_commands(args),(args.root/'dynamo-src/lib/bindings/python',args.root)):
            subprocess.run(command,cwd=cwd,env=env,check=True)
        wheels=sorted((args.root/'wheels').glob('*.whl'));names={}
        for wheel in wheels:
            name,version=checked_wheel(wheel,sha(wheel));names[name]=version
        if names!={'ai-dynamo':'1.7.0','ai-dynamo-runtime':'1.7.0'}:fail('unexpected Dynamo wheel set/version')
        subprocess.run([str(py),'-I','-m','pip','install','--no-deps','--no-index','--ignore-installed',*[str(w) for w in wheels]],env=env,check=True)
        after=json.loads(run([py,'-I','-c',IMAGE_QUERY],env))
        for name,version in receipt['image']['packages'].items():
            if name not in {*TOOLS,*receipt['runtime_constraints'],'ai-dynamo','ai-dynamo-runtime'} and after['packages'].get(name)!=version:fail('image stack drift in venv: '+name)
        if any('x86_64' in p or (Path(p).is_relative_to('/workspace') and not Path(p).is_relative_to(args.root/'env')) for p in after['paths']):fail('unexpected workspace/x86 Python path after build')
        validate_runtime(after['packages'],{},receipt['runtime_constraints'])
        subprocess.run([str(py),'-I','-m','pip','check'],env=env,check=True)
        run([py,'-I','-c',"import torch; assert torch.__version__=='2.14.1+cu130'; assert torch.version.cuda=='13.0'"],env)
        receipt['dynamo_runtime_module']=run([py,'-I','-c',"import dynamo.runtime as r,importlib.metadata as m,sys; from pathlib import Path; assert m.version('ai-dynamo')==m.version('ai-dynamo-runtime')=='1.7.0'; assert Path(r.__file__).resolve().is_relative_to(Path(sys.prefix)); print(r.__file__)"],env)
        receipt.update(status='complete',installed=after,wheels_sha256={w.name:sha(w) for w in wheels})
    finally:
        (args.root/'bootstrap-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def main(argv=None):
    args=parser().parse_args(argv);receipt=preflight(args)
    print(json.dumps(receipt,indent=2),flush=True)
    if args.execute:execute(args,receipt)


if __name__=='__main__':main()
