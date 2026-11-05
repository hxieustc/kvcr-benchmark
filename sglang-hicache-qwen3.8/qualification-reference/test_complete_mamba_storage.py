"""Real HiCache checkpoint byte format, including PLE BF16/int64 side rows."""
import ctypes
import importlib.util
from pathlib import Path
import unittest
import torch
from sglang.srt.mem_cache.storage.kvcr.host_layout import describe_pool

SOURCE=Path(__file__).resolve().parents[1]/'software/sglang'
spec=importlib.util.spec_from_file_location('mamba_side_fixture',SOURCE/'test/registered/unit/mem_cache/test_hicache_mamba_slot_side_states.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

class CompleteMambaStorage(unittest.TestCase):
    def test_metadata_contains_entire_flat_checkpoint(self):
        host,_=fixture._make_pools(True)
        for n,tensor in enumerate(host._iter_page_tensors(1)):tensor.fill_(n+3)
        ptrs,sizes=host.get_storage_page_buffer_meta(torch.tensor([1]))
        flat=b''.join(ctypes.string_at(p,n) for p,n in zip(ptrs,sizes))
        self.assertEqual(len(ptrs),4) # temporal, conv, PLE short-conv, int64 N-gram.
        self.assertEqual(flat,host.get_data_page(1).numpy().tobytes())
        self.assertEqual(sum(sizes),host.size_per_token)

    def test_indexed_kvcr_layout_and_copy_cover_all_components(self):
        host,_=fixture._make_pools(True)
        for n,tensor in enumerate(host._iter_page_tensors(0)):tensor.fill_(n+5)
        layout=describe_pool('mamba',host)
        self.assertEqual(len(layout.components),4)
        self.assertEqual(layout.bytes_per_page,host.size_per_token)
        for component in layout.components:
            ctypes.memmove(component.address+2*component.stride,component.address,component.size)
        self.assertEqual(host.get_data_page(0).numpy().tobytes(),host.get_data_page(2).numpy().tobytes())

    def test_legacy_zero_copy_still_rejects_extra_state(self):
        host,_=fixture._make_pools(True)
        with self.assertRaises(NotImplementedError):host.get_page_buffer_meta(torch.tensor([0]))
        plain,_=fixture._make_pools(False)
        self.assertEqual(plain.get_storage_page_buffer_meta(torch.tensor([0])),plain.get_page_buffer_meta(torch.tensor([0])))

if __name__=='__main__':unittest.main()
