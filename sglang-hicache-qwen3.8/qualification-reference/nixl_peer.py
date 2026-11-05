"""Two real spawned processes, registered DRAM, UCX peer WRITE and READ.

No self-agent transfer, torch import, fake status or forced cancellation. On a
native/IPC failure peers retain their agent and buffers for owned investigation.
"""
import argparse
import inspect
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from qualify_native_rdma import N, ROOT, SDK, SDK_PINS, check_env, sdk_gate, sha, write


def start_ticks():return Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]


def peer(conn,rank,out):
    # Imported only in the spawned child after the explicit execution gate.
    import numpy as np
    from nixl import nixl_agent,nixl_agent_config
    check_env(os.environ);sdk_gate()
    wrapper=Path(inspect.getfile(nixl_agent)).resolve()
    assert wrapper==SDK/'_api.py' and sha(wrapper)==SDK_PINS['_api.py']
    name=f'hicache-rdma-{os.getpid()}-{rank}'
    agent=nixl_agent(name,nixl_agent_config(backends=['UCX'],num_threads=0,enable_prog_thread=True))
    source=(np.arange(N,dtype=np.uint32)%251).astype(np.uint8)
    target=np.zeros(N,dtype=np.uint8)
    # Distinct initial payloads in the two source regions.
    source[:]=((source.astype(np.uint16)+rank*3)%251).astype(np.uint8)
    regions=[agent.register_memory([(x.ctypes.data,N,0,'')],mem_type='DRAM',backends=['UCX']) for x in (source,target)]
    identity=dict(pid=os.getpid(),start=start_ticks(),name=name,wrapper=str(wrapper),wrapper_sha256=sha(wrapper))
    conn.send(dict(**identity,metadata=agent.get_agent_metadata(),source=source.ctypes.data,target=target.ctypes.data))
    other=None
    try:
        while True:
            command=conn.recv();op=command['op']
            if op=='connect':
                other=command['peer'];assert other['pid']!=os.getpid() and other['name']!=name
                loaded_name=agent.add_remote_agent(other['metadata'])
                # Actual SDK probe: loadRemoteMD returns bytes despite ->str.
                if isinstance(loaded_name,bytes):loaded_name=loaded_name.decode('utf-8',errors='strict')
                assert isinstance(loaded_name,str) and loaded_name==other['name']
                conn.send(dict(connected=True))
            elif op=='zero':target.fill(0);conn.send(dict(zeroed=True))
            elif op=='transfer':
                kind=command['kind'];assert kind in ('WRITE','READ')
                local=source if kind=='WRITE' else target
                remote=other['target'] if kind=='WRITE' else other['source']
                local_descs=agent.get_xfer_descs([(local.ctypes.data,N,0)],mem_type='DRAM')
                remote_descs=agent.get_xfer_descs([(remote,N,0)],mem_type='DRAM')
                handle=agent.initialize_xfer(kind,local_descs,remote_descs,other['name'],backends=['UCX'])
                state=agent.transfer(handle);deadline=time.monotonic()+30;polls=0
                while state!='DONE':
                    assert state in ('PROC','PEND'),state
                    assert time.monotonic()<deadline,'Native deadline; retained handle and buffers'
                    state=agent.check_xfer_state(handle);polls+=1
                    if state!='DONE':time.sleep(0)
                agent.release_xfer_handle(handle)
                conn.send(dict(terminal=state,polls=polls))
            elif op=='verify':
                # Every byte checked, not a sparse checksum or initial pattern.
                expected=(np.arange(N,dtype=np.uint32)%251+command['source_rank']*3)%251
                good=bool(np.array_equal(target,expected.astype(np.uint8)))
                conn.send(dict(full_bytes_correct=good,bytes=N,sha256=__import__('hashlib').sha256(target.tobytes()).hexdigest()))
            elif op=='close':
                agent.remove_remote_agent(other['name'])
                for reg in regions:agent.deregister_memory(reg,backends=['UCX'])
                regions.clear();del agent
                conn.send(dict(registrations_released=True,**identity));return
            else:raise AssertionError(op)
    except BaseException as error:
        marker=out/f'peer-{rank}-retained.json'
        if not marker.exists():write(marker,dict(**identity,error=repr(error),traceback=traceback.format_exc(),status='retained-unknown-native-ownership'))
        try:conn.send(dict(error=repr(error),retained=True,**identity))
        except (OSError,EOFError):pass
        # Never exit/deregister buffers whose native access may still be active.
        while True:time.sleep(1)


def receive(conn):
    assert conn.poll(60),'Peer IPC deadline; do not kill retained native actors'
    result=conn.recv();assert 'error' not in result,result
    return result


def request(conn,op,**fields):conn.send(dict(op=op,**fields));return receive(conn)


def run(out):
    assert out.resolve().is_relative_to(ROOT/'qualification')
    check_env(os.environ);sdk_gate();assert not out.exists() and not out.is_symlink()
    ctx=mp.get_context('spawn');children=[];pipes=[]
    result=dict(status='incomplete',peers=[],operations=[],peer_exits=[],registrations_released=False,
                scope='Distinct processes, same-node UCX RC peer DRAM; not cross-node, GPU or measured bandwidth')
    write(out,result)
    try:
        for rank in (0,1):
            parent,child=ctx.Pipe();proc=ctx.Process(target=peer,args=(child,rank,out.parent))
            proc.start();child.close();children.append(proc);pipes.append(parent)
            result.setdefault('spawned_pids',[]).append(proc.pid);write(out,result)
        descriptions=[receive(c) for c in pipes]
        result['peers']=[{k:v for k,v in p.items() if k not in ('metadata','source','target')} for p in descriptions];write(out,result)
        for rank in (0,1):request(pipes[rank],'connect',peer=descriptions[1-rank])
        for kind in ('WRITE','READ'):
            for initiator in (0,1):
                src=initiator if kind=='WRITE' else 1-initiator;dst=1-src
                request(pipes[dst],'zero')
                terminal=request(pipes[initiator],'transfer',kind=kind)
                checked=request(pipes[dst],'verify',source_rank=src)
                result['operations'].append(dict(operation=kind,source=src,target=dst,initiator=initiator,**terminal,**checked));write(out,result)
                assert checked['full_bytes_correct'],'Payload mismatch'
        # Both peers idle: no native operation outstanding before metadata removal.
        closed=[request(c,'close') for c in pipes]
        assert all(r['registrations_released'] for r in closed)
        for proc in children:proc.join(30);assert proc.exitcode==0
        result.update(status='passed',registrations_released=True,peer_exits=[p.exitcode for p in children],close_identities=closed)
    except BaseException as error:
        result.update(status='unqualified',error=repr(error),cleanup='No terminate/kill; preserve peers and registered buffers for owned cleanup.');raise
    finally:write(out,result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
