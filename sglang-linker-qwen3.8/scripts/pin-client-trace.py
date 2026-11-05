"""Pin the HF revision inside a newly created, experiment-only clientenv."""
from pathlib import Path
import argparse,hashlib,json
from aiperf.dataset.loader import base_hf_dataset
p=argparse.ArgumentParser();p.add_argument('--clientenv',type=Path,required=True);a=p.parse_args()
path=Path(base_hf_dataset.__file__).resolve();assert a.clientenv.is_absolute() and a.clientenv.is_relative_to('/workspace');assert path.is_relative_to(a.clientenv.resolve()),path
old=path.read_text();needle='    hf_revision: str | None = None';assert old.count(needle)==1,'Unexpected/already patched client'
revision='8fecd2fc56694469f758f0afbbb6335ad3043740';new=old.replace(needle,'    hf_revision: str | None = '+repr(revision));path.write_text(new)
print(json.dumps({'module':str(path),'revision':revision,'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()}))
