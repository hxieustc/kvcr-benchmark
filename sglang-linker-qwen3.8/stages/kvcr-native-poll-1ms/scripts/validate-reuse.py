#!/usr/bin/env python3
"""Require peer reuse, successful transfers and an exact expected vault code.

Pinned linker: wrapper threshold=1; a new leaf is counted on its first insert
(unified_cache_linker.py:274; unified_tree_core.py:1106,1413). The framework
HiCache write_back second-touch rule does not apply without a host controller.

Synthetic match-expected AL=2.32 draws length 2/3 on the worker CPU RNG, then
force-accepts real draft tokens (spec_utils.py:440-519). Temperature zero does
not make request-to-request token equality a valid oracle. This preflight
checks a strict semantic answer and remote reuse, not token/logit equivalence.
"""
import argparse
from collections import Counter
import json
import random
import re
import time
import urllib.request
import uuid
from pathlib import Path
from transformers import AutoTokenizer

def validate_answer(text, code):
    # The prompt already ends with the answer label. Reject a mere occurrence
    # of the expected code after an incorrect answer or in an echoed question.
    assert re.match(r'^\s*' + re.escape(code) + r'\b', text), (code, text)
    numbers = re.findall(r'[0-9]+', text)
    assert all(number in (code, '7') for number in numbers), (code, text)


def encode_canary(tokenizer, document):
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': 'Copy the requested access code exactly. Reply with the digits only, without explanation, tools or other text.'},
         {'role': 'user', 'content': document}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)


def telemetry_totals(path):
    totals = Counter()
    for line in path.read_text(errors='replace').splitlines():
        match = re.search(r'KVCR telemetry rank=\d+ interval_s=\S+ stats=(\{.*\})', line)
        if match:
            totals.update({key: value for key, value in json.loads(match.group(1)).items()
                           if key.endswith(('.counter', '.count'))})
    return totals


def telemetry_delta(current, baseline):
    assert all(current.get(key, 0) >= value for key, value in baseline.items()), 'Telemetry log reset during preflight'
    return Counter({key: value - baseline.get(key, 0) for key, value in current.items()})


def require_peer_telemetry(telemetry, peer_ranks=(1,)):
    for rank, totals in telemetry.items():
        assert not any('failed' in key.lower() and value > 0 for key, value in totals.items()), (rank, totals)
    for rank in peer_ranks:
        values=telemetry[str(rank)]
        direct=(values['kvcr_duration_seconds[remote_deliver,success].count']>0 and values['kvcr_transfer_bytes[remote_deliver].counter']>0)
        staged=(values['kvcr_duration_seconds[remote_fetch,success].count']>0 and values['kvcr_transfer_blocks[remote_fetch].counter']>0 and values['kvcr_duration_seconds[local_deliver,success].count']>0 and values['kvcr_transfer_bytes[local_deliver].counter']>0)
        assert direct or staged,telemetry
    assert telemetry['0']['kvcr_transfer_bytes[source_write].counter'] > 0, telemetry


def wait_for_export(path, baseline, timeout=90):
    # Do not contact the peer until the source has completed its GPU->owned
    # DRAM copy. A cache hit on the warm source proves HBM reuse, not export.
    deadline = time.monotonic() + timeout
    while True:
        delta = telemetry_delta(telemetry_totals(path), baseline)
        assert not any('failed' in key.lower() and value > 0 for key, value in delta.items()), delta
        if (delta['kvcr_duration_seconds[local_fill,success].count'] > 0
                and delta['kvcr_transfer_bytes[local_fill].counter'] > 0):
            return
        assert time.monotonic() < deadline, ('Source export did not complete before peer probe', delta)
        time.sleep(1)


p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
root=Path('/workspace/sglang-kvcr-parity-20261031-r1/cases/gb300-kvcr-native-poll-delay')
run=root/'runs'/a.attempt
out=run/'preflight';out.mkdir(exist_ok=True)
workers=[]
for rank in range(4):
    log=re.sub(r'\x1b\[[0-9;]*m','',(run/f'worker-{rank}.log').read_text(errors='replace'))
    ids=set(re.findall(r'Backend admission gate created[^\n]*worker_id=([0-9a-f]+)',log))
    assert len(ids)==1 and 'Successfully registered LLM with runtime config' in log,(rank,ids)
    workers.append(int(ids.pop(),16))
assert len(set(workers))==4
tokenizer=AutoTokenizer.from_pretrained(root/'model',trust_remote_code=True)
code=str(random.randrange(100000,1000000))
filler='The capital of France is Paris. '*300
prompt=f'Document {uuid.uuid4().hex}. {filler}The access code for vault 7 is {code}. {filler}Question: what is the access code for vault 7? Reply with its digits only.'
tokens=encode_canary(tokenizer,prompt)
minimum=(len(tokens)-1)//256*256
assert minimum>=4096
(out/'input.json').write_text(json.dumps({'expected_code':code,'minimum_reused_tokens':minimum,'prompt':prompt,'input_token_ids':tokens},indent=2)+'\n')
results=[]
# Cold insertion can export immediately (threshold 1). One local warm check
# verifies HBM reuse; the final source probe brackets the peer read. Extra
# warm primes cannot repair a failed local_fill.
baseline={str(rank):telemetry_totals(run/f'worker-{rank}.log') for rank in range(4)}
for index,destination in enumerate([0,0,1,2,3,0]):
    if index:time.sleep(15)
    worker=workers[destination]
    payload={'model':'nvidia/Qwen3.8-Flash-Next-NVFP4','prompt':tokens,'temperature':0,'max_tokens':64,'stream':True,'stream_options':{'include_usage':True},'nvext':{'backend_instance_id':worker,'dp_rank':0,'extra_fields':['worker_id','engine_data','completion_token_ids']}}
    request=urllib.request.Request('http://127.0.0.1:24000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(request,timeout=1200) as response:raw=response.read().decode()
    (out/f'response-{index}.sse').write_text(raw)
    events=[json.loads(line[5:]) for line in raw.splitlines() if line.startswith('data:') and line[5:].strip()!='[DONE]']
    assert not any('error' in e for e in events),events
    ext=[e.get('nvext',{}) for e in events]
    output=[token for item in ext for token in item.get('completion_token_ids',[])]
    reported=[item['worker_id'] for item in ext if 'worker_id' in item]
    assert reported and all(x['prefill_worker_id']==x['decode_worker_id']==worker for x in reported)
    hits=[item.get('engine_data',{}).get('kv_cache_hit',{}).get('reused_tokens') for item in ext]
    hits=[x for x in hits if x is not None]
    assert hits,(index,events)
    hit=max(hits)
    assert hit==0 if index==0 else minimum<=hit<len(tokens),(index,hit,minimum)
    text=tokenizer.decode(output)
    validate_answer(text,code)
    result={'request':index,'destination_rank':destination,'input_tokens':len(tokens),'cached_tokens':hit,'output':text,'token_ids':output,'elapsed_seconds':time.monotonic()-start}
    results.append(result);print(json.dumps(result),flush=True)
    if index==0:wait_for_export(run/'worker-0.log',baseline['0'])
token_ids_equal={str(row['destination_rank']):row['token_ids']==results[-1]['token_ids'] for row in results if row['destination_rank']!=0}
time.sleep(15)
telemetry={str(rank):telemetry_delta(telemetry_totals(run/f'worker-{rank}.log'),baseline[str(rank)]) for rank in range(4)}
require_peer_telemetry(telemetry,peer_ranks=(1,2,3))
(out/'validation.json').write_text(json.dumps({'passed':True,'requests':results,'telemetry_baseline':baseline,'telemetry_deltas':telemetry,'remote_source_token_ids_equal':token_ids_equal,'answer_check':'exact leading vault code, no conflicting numeric answer','note':'Synthetic real-draft acceptance is stochastic; token IDs are diagnostic only. Passed means semantic vault-code readback plus cache/peer-transfer checks, not token/logit equivalence or model quality qualification.'},indent=2)+'\n')
print('PEER_REUSE_PREFLIGHT_PASSED',flush=True)
