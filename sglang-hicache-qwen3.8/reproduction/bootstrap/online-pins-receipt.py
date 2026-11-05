"""Exact runtime pins and effective/hash receipt for the short online bootstrap."""
import hashlib
import importlib.metadata as m
import json
from pathlib import Path
import platform
import sys

NAMES=('torch','sglang-kernel','flashinfer-python','flashinfer-cubin','flashinfer-jit-cache',
       'triton','nixl','nixl-cu13','mooncake-transfer-engine-cuda13','torchvision',
       'nvidia-cutlass-dsl','nvidia-cutlass-dsl-libs-base','nvidia-cutlass-dsl-libs-core',
       'nvidia-cutlass-dsl-libs-cu13','sgl-deep-gemm','sgl-deep-ep')


def pins(original):
    result={name:original[name] for name in NAMES}
    assert result['sglang-kernel']=='0.4.10' and result['torch'].split('+')[0]=='2.14.1'
    assert result['triton']=='3.8.0' and result['flashinfer-python']=='0.7.2'
    result['torch']='2.14.1+cu130'  # Native CUDA build; original metadata omitted local suffix.
    return result


def requests(original):
    selected={}
    for name,version in pins(original).items():
        try:actual=m.version(name)
        except m.PackageNotFoundError:actual=None
        if actual==version or (name=='torch' and actual==original['torch']):continue
        requirement=name+('[cu13]' if name in ('flashinfer-python','nvidia-cutlass-dsl') else '')
        selected[requirement]=version
    return selected


def receipt(original,root):
    import torch
    import dynamo.runtime
    assert platform.machine()=='aarch64'
    assert torch.__version__=='2.14.1+cu130' and torch.version.cuda=='13.0'
    names={d.metadata['Name'] for d in m.distributions() if d.metadata['Name']}
    effective={name:m.version(name) for name in sorted(names)}  # Resolve precedence, not last distribution wins.
    for name,version in pins(original).items():
        actual=m.version(name)
        assert actual==version or (name=='torch' and actual==original['torch']),(name,actual,version)
    assert m.version('ai-dynamo')==m.version('ai-dynamo-runtime')=='1.7.0'
    assert Path(dynamo.runtime.__file__).resolve().is_relative_to(root/'env')
    hashes={}
    for p in [*sorted((root/'build/wheels').glob('*.whl')),root/'build/rustup-init',root/'sources/Cargo.lock',root/'sources/dynamo.patch']:
        with p.open('rb') as f:hashes[str(p.relative_to(root))]=hashlib.file_digest(f,'sha256').hexdigest()
    data=dict(status='pins-and-imports-verified',machine=platform.machine(),effective_packages=effective,
              native_torch_version=torch.__version__,cuda=torch.version.cuda,sha256=hashes,
              expected_platform_exclusions={'torchcodec':'Upstream SGLang excludes Linux aarch64/arm64/armv7l; not required even if fallback image contains it.'},
              optional_extras_not_requested={'nixl-cu12':'CUDA13 path selected','nvidia-cutlass-dsl-libs-cu12':'CuTe [cu13] extra selected'},
              dynamo_head='f28d277edaf403ba5cae65bece687f7b29c0eba3',
              qualification='Bootstrap only; pip-check outcome is separate. No inference/transport/GPU qualification.')
    (root/'sources/online-bootstrap-receipt.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    mode,source,target=sys.argv[1:];original=json.loads(Path(source).read_text())
    if mode=='pins':Path(target).write_text(''.join(f'{n}=={v}\n' for n,v in pins(original).items()))
    elif mode=='requests':Path(target).write_text(''.join(f'{n}=={v}\n' for n,v in requests(original).items()))
    elif mode=='receipt':receipt(original,Path(target).resolve())
    else:raise SystemExit('mode must be pins, requests or receipt')
