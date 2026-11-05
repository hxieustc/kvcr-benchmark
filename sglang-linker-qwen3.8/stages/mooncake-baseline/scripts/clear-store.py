"""Clear the private benchmark master after all framework caches have flushed."""
from pathlib import Path
import json
import os
import time
from mooncake.store import MooncakeDistributedStore

client=MooncakeDistributedStore()
assert client.setup(f"{os.environ['POD_IP']}:25220",'P2PHANDSHAKE',0,16<<20,'rdma','rocep172s0','127.0.0.1:25001')==0
try:
    result=client.remove_all()
    print(json.dumps({'utc':time.time(),'private_master':'127.0.0.1:25001','remove_all_result':result,'temporary_client_segment_bytes':0}),flush=True)
finally:client.close()
