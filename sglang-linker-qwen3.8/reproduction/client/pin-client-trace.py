#!/usr/bin/env python3
"""Bind the existing HF loader revision seam in this experiment-only client."""
import difflib
import json
from pathlib import Path
from aiperf.dataset.loader import base_hf_dataset

root=Path('/workspace/sglang-linker-qwen38-agentx-20261031-r1')
sha=json.loads((root/'trace-receipt.json').read_text())['revision']
path=Path(base_hf_dataset.__file__)
assert path.resolve().is_relative_to(root/'clientenv'),path
before=path.read_text()
needle='    hf_revision: str | None = None'
assert before.count(needle)==1,'Refuse an unexpected or already patched loader'
after=before.replace(needle,f'    hf_revision: str | None = "{sha}"')
path.write_text(after)
(root/'config/client-trace-pin.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/aiperf/dataset/loader/base_hf_dataset.py',tofile='b/aiperf/dataset/loader/base_hf_dataset.py')))
print('Pinned experiment-only client HF loader revision',sha)
