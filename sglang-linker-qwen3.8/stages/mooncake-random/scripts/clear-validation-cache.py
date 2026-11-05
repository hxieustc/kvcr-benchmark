"""Clear all four idle workers through SGLang's native request-plane RPC."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
root=Path('/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-mooncake-placement-random')
r=root/'runs'/a.attempt
assert json.loads((r/'preflight/validation.json').read_text())['passed']
os.environ.pop('NATS_SERVER',None)
os.environ.update(DYN_DISCOVERY_BACKEND='file',DYN_FILE_KV=str(r/'discovery'),DYN_REQUEST_PLANE='tcp',DYN_EVENT_PLANE='zmq',DYN_SYSTEM_HOST='127.0.0.1',DYN_SYSTEM_PORT='0')
from dynamo.runtime import DistributedRuntime

async def main():
    runtime=DistributedRuntime(asyncio.get_running_loop(),discovery_backend='file',request_plane='tcp',event_plane='zmq')
    client=await runtime.endpoint('sglkvcrreview.backend.clear_kv_blocks').client()
    await asyncio.wait_for(client.wait_for_instances(),30)
    deadline=time.monotonic()+30
    while len(client.instance_ids())<4 and time.monotonic()<deadline:await asyncio.sleep(.1)
    ids=sorted(client.instance_ids());assert len(ids)==4,ids
    async def clear(instance_id):
        start=time.time()
        try:
            stream=await client.direct({},instance_id,annotated=False)
            replies=[reply async for reply in stream]
            result=replies[0] if len(replies)==1 else {'status':'error','replies':replies}
        except Exception as error:result={'status':'error','message':str(error)}
        row={'instance_id':instance_id,'begin_utc':start,'end_utc':time.time(),'result':result}
        print(json.dumps(row),flush=True)
        return row
    rows=await asyncio.wait_for(asyncio.gather(*(clear(i) for i in ids)),900)
    (r/'validation-cache-clear.json').write_text(json.dumps({'endpoint':'sglkvcrreview.backend.clear_kv_blocks','transport':'tcp','workers':rows},indent=2)+'\n')
    assert all(x['result'].get('status')=='success' for x in rows),rows
    runtime.shutdown()
asyncio.run(main())
import subprocess
subprocess.run(["/workspace/sglang-kvcr-parity-20261031-r1/env/bin/python",str(root/"scripts/clear-store.py")],check=True)
time.sleep(15)
print('All four caches cleared; loaded models retained',flush=True)
