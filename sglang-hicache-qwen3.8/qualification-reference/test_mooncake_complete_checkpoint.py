"""Actual Mamba/Mooncake metadata; only native SDK byte-copy calls are faked.

Run with the matched R/env and this campaign's SGLang PYTHONPATH. No native
Mooncake engine, GPU allocation, model or RDMA object is constructed.
"""
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

SOURCE=Path(__file__).resolve().parents[1]/'software/sglang'
spec=importlib.util.spec_from_file_location('moon_complete_mamba_fixture',SOURCE/'test/registered/unit/mem_cache/test_hicache_mamba_slot_side_states.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class NativeByteCopies:
    """Replaces native SDK methods AFTER real keys/pointers/sizes are built."""
    def __init__(self):self.objects={}
    def batch_is_exist(self,keys):return [int(k in self.objects) for k in keys]
    def batch_put_from(self,keys,ptrs,sizes,*config):
        assert len(keys)==len(ptrs)==len(sizes)
        for key,ptr,size in zip(keys,ptrs,sizes):
            assert isinstance(ptr,int) and isinstance(size,int) and ptr>0 and size>0
            self.objects[key]=ctypes.string_at(ptr,size)
        return [0]*len(keys)
    def batch_get_into(self,keys,ptrs,sizes):
        results=[]
        for key,ptr,size in zip(keys,ptrs,sizes):
            value=self.objects.get(key)
            if value is None:results.append(-1);continue
            assert len(value)==size
            ctypes.memmove(ptr,value,size);results.append(size)
        return results


def make_store(host):
    store=MooncakeStore.__new__(MooncakeStore)
    store.registered_pools={PoolName.MAMBA:host}
    store.mem_pool_host=SimpleNamespace(kv_buffer=None)  # KV-independent checkpoint query.
    store.config_prefix='complete-checkpoint-fixture'
    store.mha_suffix='0';store.mla_suffix='0';store._use_group_semantics=False
    store.store=NativeByteCopies()
    return store


def transfer(keys,slots):
    return PoolTransfer(PoolName.MAMBA,host_indices=torch.tensor(slots),keys=keys,
                        hit_policy=PoolHitPolicy.TRAILING_PAGES)


def initialized():
    host,_=fixture._make_pools(True)
    for slot in (0,2):
        for component,tensor in enumerate(host._iter_page_tensors(slot)):
            tensor.fill_((1<<40)+slot if tensor.dtype==torch.int64 else component+slot+1.25)
    return host,make_store(host),['1'*64,'2'*64]


class CompleteMooncakeCheckpoint(unittest.TestCase):
    def test_component_keys_include_both_mixed_dtype_side_states(self):
        host,store,keys=initialized()
        encoded,width=store._get_hybrid_page_component_keys(keys,transfer(keys,[0,2]))
        self.assertEqual(width,4,'temporal + conv + BF16 PLE + int64 N-gram must all be keys')
        self.assertEqual(len(encoded),8)
        self.assertEqual(encoded,[key+suffix for key in keys for suffix in
            ('_0_temporal','_0_conv_0','_0_slot_state_0','_0_slot_state_1')])
        self.assertEqual([t.dtype for t in host.slot_state_buffers],[torch.bfloat16,torch.int64])

    def test_real_metadata_full_bytes_roundtrip_to_different_nonadjacent_slots(self):
        host,store,keys=initialized()
        expected=[host.get_data_page(s).numpy().tobytes() for s in (0,2)]
        self.assertEqual(store.batch_set_v2([transfer(keys,[0,2])])[PoolName.MAMBA],[True,True])
        self.assertEqual(len(store.store.objects),8)
        self.assertEqual(sum(map(len,store.store.objects.values())),2*host.size_per_token)
        for slot in (3,1):
            for tensor in host._iter_page_tensors(slot):tensor.fill_(-7)
        self.assertEqual(store.batch_get_v2([transfer(keys,[3,1])])[PoolName.MAMBA],[True,True])
        self.assertEqual([host.get_data_page(s).numpy().tobytes() for s in (3,1)],expected)
        # Preserve the explicit rejection for backends using the legacy accessor.
        with self.assertRaises(NotImplementedError):host.get_page_buffer_meta(torch.tensor([0]))

    def test_missing_extra_rejects_complete_hit_and_full_restore(self):
        host,store,keys=initialized()
        self.assertEqual(store.batch_set_v2([transfer(keys,[0,2])])[PoolName.MAMBA],[True,True])
        self.assertEqual(len(store.store.objects),8)
        tail=transfer(keys[-1:],[2])  # One-checkpoint trailing window, not ALL_PAGES.
        self.assertIn(2,store.batch_exists_v2(keys,[tail]).restorable_prefix_pages)
        missing=store._tag_keys([keys[1]+'_0_slot_state_1'])[0]
        del store.store.objects[missing]
        hit=store.batch_exists_v2(keys,[tail])
        self.assertEqual(hit.restorable_prefix_pages,[1])
        self.assertNotEqual(hit.kv_hit_pages,2)
        self.assertEqual(store.batch_get_v2([transfer(keys,[3,1])])[PoolName.MAMBA],[True,False])

    def test_empty_components_skipped_without_renumbering_suffixes(self):
        host,store,keys=initialized()
        host.temporal_buffer=torch.empty((host.size,host.num_mamba_layers,1,0),dtype=torch.bfloat16)
        host.temporal_state_elem_size=0
        host.conv_buffer.insert(0,torch.empty((host.size,host.num_mamba_layers,1,0),dtype=torch.bfloat16))
        host.conv_state_shapes.insert(0,(0,));host.conv_state_elem_sizes.insert(0,0)
        host.slot_state_buffers.insert(1,torch.empty((host.size,0),dtype=torch.int64))
        host.slot_state_device_tensors.insert(1,torch.empty((host.size,0),dtype=torch.int64))
        host.size_per_token=host.get_size_per_token()
        encoded,width=store._get_hybrid_page_component_keys(keys,transfer(keys,[0,2]))
        self.assertEqual(width,3)
        self.assertEqual(encoded,[key+suffix for key in keys for suffix in
            ('_0_conv_1','_0_slot_state_0','_0_slot_state_2')])
        ptrs,sizes=store._build_pool_buffer_meta(host,torch.tensor([0,2]),encoded,width,None)
        self.assertEqual(len(ptrs),6)
        self.assertTrue(all(n>0 for n in sizes))
        for offset,slot in ((0,0),(3,2)):
            raw=b''.join(ctypes.string_at(p,n) for p,n in zip(ptrs[offset:offset+3],sizes[offset:offset+3]))
            self.assertEqual(raw,host.get_data_page(slot).numpy().tobytes())

    def test_pool_without_new_accessor_uses_legacy_metadata(self):
        host,_=fixture._make_pools(False)
        legacy=SimpleNamespace(page_size=1,get_page_buffer_meta=host.get_page_buffer_meta)
        store=make_store(host);keys=['a_temporal','a_conv','b_temporal','b_conv']
        actual=store._build_pool_buffer_meta(legacy,torch.tensor([0,2]),keys,2,None)
        self.assertEqual(actual,host.get_page_buffer_meta(torch.tensor([0,2])))

if __name__=='__main__':unittest.main()
