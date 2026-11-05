"""Exercise actual host pool constructors without allocating CUDA buffers."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

class HostBudgetRounding(unittest.TestCase):
    def test_explicit_kv_budget_is_not_rounded_above_requested_bytes(self):
        methods={name:lambda *a,**k:None for name in HostKVCache.__abstractmethods__}
        methods['clear']=lambda self:None
        Dummy=type('GeometryOnly',(HostKVCache,),methods)
        pool=Dummy(dtype=torch.uint8,device_capacity_tokens=1,size_per_token=7,
                   start_layer=0,end_layer=1,host_to_device_ratio=2,host_size=1000/1e9,
                   page_size=64,layout='page_first_direct',pin_memory=False,device='cpu',is_dummy=True)
        self.assertLessEqual(pool.size*pool.size_per_token,1000)
        self.assertEqual(pool.size%64,0)

    def test_explicit_mamba_budget_is_not_rounded_above_requested_bytes(self):
        info=SimpleNamespace(validate=lambda:None,conv=[torch.zeros((2,16,4),dtype=torch.float32)],
                             temporal=torch.zeros((2,16,8),dtype=torch.float32),extra_checkpoint_buffers=())
        with patch.object(MambaPoolHost,'init_kv_buffer',return_value=None), \
             patch.object(MambaPoolHost,'_configure_npu_mamba_io'), \
             patch.object(MambaPoolHost,'_init_write_back_staging_buffers'), \
             patch.object(MambaPoolHost,'clear'):
            pool=MambaPoolHost(info,2,1000/1e9,device_capacity=1,pin_memory=False,layout='page_first_direct')
        self.assertLessEqual(pool.size*pool.size_per_token,1000)

if __name__=='__main__':unittest.main()
