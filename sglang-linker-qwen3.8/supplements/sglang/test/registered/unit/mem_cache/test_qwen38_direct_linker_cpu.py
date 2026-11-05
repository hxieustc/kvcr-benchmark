"""Dependency-light tests of the pinned QSA direct-linker source.

Run directly with Python + torch (no SGLang installation or CUDA). Source
selection avoids importing serving dependencies; pool constructor stubs supply
CPU storage, while assembly, transfer expansion and KVCR geometry are real.
"""

import ast
import ctypes
import hashlib
import json
import sys
import types
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def source_definitions(path, names, namespace):
    namespace.setdefault("__name__", __name__)
    tree = ast.parse((ROOT / path).read_text())
    selected = [n for n in tree.body if getattr(n, "name", None) in names]
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected]
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace)
    return namespace


def source_methods(path, class_name, names):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    ns = {"torch": torch}
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name in names:
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), ns)
    return ns


class HybridLinearKVPool:
    pass


class QSATokenToKVPool(HybridLinearKVPool):
    pass


class MHATokenToKVPool:
    def _kv_tokens_per_row(self):
        return self.page_size if self.hnd else 1


class TestQwen38DirectLinkerCPU(unittest.TestCase):
    def setUp(self):
        self.modules = {}
        self.imports = patch.dict(sys.modules)
        self.imports.start()
        self.addCleanup(self.imports.stop)
        def module(name, **attrs):
            result = types.ModuleType(name)
            result.__dict__.update(attrs)
            self.modules[name] = result
            sys.modules[name] = result
            return result
        storage = module("sglang.srt.mem_cache.hicache_storage")
        from enum import Enum
        storage.__dict__.update(Enum=Enum, dataclass=dataclass)
        source_definitions("mem_cache/hicache_storage.py", {"PoolName", "PoolHitPolicy", "PoolTransfer"}, storage.__dict__)
        component = module("sglang.srt.mem_cache.unified_cache.component_type")
        component.__dict__["Enum"] = Enum
        source_definitions("mem_cache/unified_cache/component_type.py", {"ComponentType"}, component.__dict__)
        self.memory = types.SimpleNamespace(enable_unified_memory=False)
        module("sglang.srt.runtime_context", get_memory=lambda: self.memory)
        module("sglang.srt.mem_cache.memory_pool", HybridLinearKVPool=HybridLinearKVPool, MHATokenToKVPool=MHATokenToKVPool)
        module("sglang.srt.mem_cache.qsa_kv_pool", QSATokenToKVPool=QSATokenToKVPool)
        module("kvcr.types", RegionDescriptor=types.SimpleNamespace, MemoryRef=types.SimpleNamespace)
        # Execute the actual small assembler in full, including its imports.
        assembler = module("sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler")
        sys.modules[assembler.__name__] = assembler
        self.addCleanup(sys.modules.pop, assembler.__name__, None)
        exec(compile((ROOT / "mem_cache/hybrid_cache/linker_pool_assembler.py").read_text(), "linker_pool_assembler.py", "exec"), assembler.__dict__)
        self.assembler = assembler
        strategy = module("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler", Any=object, ComponentType=component.ComponentType)
        source_definitions("mem_cache/hybrid_cache/hybrid_pool_assembler.py", {"StackStrategy", "_MambaStrategy", "_stage_local_layer_mapping"}, strategy.__dict__)
        sys.modules[strategy.__name__] = strategy
        self.addCleanup(sys.modules.pop, strategy.__name__, None)
        strategy._select_strategy = lambda pool, components: strategy._MambaStrategy()
        self.strategy = strategy
        self.PoolName = storage.PoolName
        self.Policy = storage.PoolHitPolicy
        self.Transfer = storage.PoolTransfer
        self.Component = component.ComponentType
        ns = dict(hashlib=hashlib, json=json, Counter=Counter, dataclass=dataclass, PoolName=self.PoolName)
        self.Layout = source_definitions("mem_cache/storage/kvcr/device_layout.py", {"PreparedTransfer", "KVCRLayout"}, ns)["KVCRLayout"]

    def pools(self, *, hnd=False):
        # A late PLE window belongs to full-attention layer 5, after the last
        # linear layer (4). It must be ready before CoW of the whole slot.
        state = types.SimpleNamespace(
            temporal=torch.arange(4 * 9 * 3, dtype=torch.bfloat16).reshape(4, 9, 3),
            conv=[torch.arange(4 * 9 * 2, dtype=torch.bfloat16).reshape(4, 9, 2)],
            intermediate_ssm=torch.full((4, 9, 4, 3), -99.0),
            replayssm_d=torch.full((4, 9, 4, 3), -98.0),
        )
        ple = source_definitions("mem_cache/ple_state_pool.py", {"ShortConvPool", "NGramPool"}, {"torch": torch, "PLE_NGRAM_STATE_LAYER_ID": (1 << 32) - 1})
        conv = ple["ShortConvPool"].__new__(ple["ShortConvPool"])
        conv.layer_map = {1: 0, 5: 1}
        conv.conv_state = torch.arange(2 * 9 * 6, dtype=torch.bfloat16).reshape(2, 9, 2, 3)
        ngram = ple["NGramPool"].__new__(ple["NGramPool"])
        ngram.context = torch.arange(9 * 3, dtype=torch.int64).reshape(9, 3)
        methods = source_methods("mem_cache/memory_pool.py", "MambaPool", {"_iter_transfer_state_entries"})
        mamba = types.SimpleNamespace(mamba_cache=state, mamba_layer_ids=[0, 1, 3, 4], num_mamba_layers=4, conv_slice_axis=0, _slot_siblings=[conv, ngram])
        tree = ast.parse((ROOT / "mem_cache/memory_pool.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MambaPool")
        assignment = next(n for n in cls.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_NON_TRANSFER_STATE_FIELDS" for t in n.targets))
        fields = {}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "memory_pool.py", "exec"), fields)
        mamba._NON_TRANSFER_STATE_FIELDS = fields["_NON_TRANSFER_STATE_FIELDS"]
        mamba._iter_transfer_state_entries = types.MethodType(methods["_iter_transfer_state_entries"], mamba)
        def qsa(layers):
            pool = QSATokenToKVPool()
            pool.start_layer = 0
            pool.page_size = 4
            pool.dtype = torch.bfloat16
            pool.qsa_compress_ratio = 2
            pool.mamba_pool = mamba
            pool.full_attention_layer_id_mapping = {layer: i for i, layer in enumerate(layers)}
            full = MHATokenToKVPool()
            full.dtype, full.page_size, full.hnd = torch.bfloat16, 4, hnd
            full.is_quantized_kv_cache = False
            full.k_scale_buffer = full.native_k_scale_buffer = None
            full.layer_num = len(layers)
            shape = (5, 2, 4, 3) if hnd else (20, 2, 3)
            full.k_buffer = [torch.arange(120, dtype=torch.bfloat16).reshape(shape) + 10 * i for i in range(len(layers))]
            full.v_buffer = [x + 500 for x in full.k_buffer]
            pool.full_kv_pool = full
            pool.qsa_compressed_k_buffer_pool = [torch.arange(10 * 3, dtype=torch.bfloat16).reshape(10, 1, 3) + i for i in range(len(layers))]
            return pool
        target, draft = qsa([2, 5]), qsa([0])
        req = types.SimpleNamespace(mamba_pool=mamba, mamba_map={0: 0, 1: 1, 3: 2, 4: 3}, mamba_ckpt_pool=None, short_conv_pool=conv, ngram_pool=ngram)
        params = types.SimpleNamespace(req_to_token_pool=req, mtp_draft_device_pools=(draft,), mtp_draft_mamba_pools=(), enable_mamba_extra_buffer=True, pp_size=1, attn_cp_size=1)
        return target, draft, params

    def group(self, target, params):
        return self.assembler.resolve_hybrid_device_pool_group(kvcache=target, page_size=4, params=params, components={self.Component.FULL, self.Component.MAMBA})

    def test_qsa_strategy_restores_every_persistent_piece_at_new_slots(self):
        for hnd in (False, True):
            with self.subTest(hnd=hnd):
                target, draft, params = self.pools(hnd=hnd)
                self.assertTrue(self.strategy._MambaStrategy().matches(target, {self.Component.FULL, self.Component.MAMBA}))
                try:
                    group = self.group(target, params)
                except ValueError as error:
                    self.fail(f"Approved QSA layout was rejected: {error}")
                self.assertEqual(set(group.entry_map), {self.PoolName.KV, self.PoolName.MAMBA, self.PoolName.INDEXER})
                self.assertEqual(group.num_layers, 6)
                layout = self.Layout(group, model_name="pinned-qwen38", endpoint_name="worker")
                def transfers(page, slot):
                    return [self.Transfer(name=self.PoolName.KV, keys=["prefix"], device_indices=torch.arange(page * 4, (page + 1) * 4)), self.Transfer(name=self.PoolName.MAMBA, keys=["prefix"], device_indices=torch.tensor([slot]), hit_policy=self.Policy.TRAILING_PAGES)]
                source = layout.prepare(transfers(1, 2))
                destination = layout.prepare(transfers(3, 6))
                registrations = {r.label: r for r in layout.registrations}
                def address(ref):
                    reg = registrations[ref.label]
                    return reg.addr + ref.element_index * reg.stride, reg.size
                snapshot = {}
                for batch in layout.mappings(source):
                    for key, refs in batch.items():
                        for ref in refs:
                            addr, size = address(ref)
                            snapshot[key, ref.label] = ctypes.string_at(addr, size)
                # Every required tensor is advertised, not just every advertised
                # tensor copied. Enumerate independent expected checkpoint/paged data.
                expected = []
                for pool in (target, draft):
                    for buf in (*pool.full_kv_pool.k_buffer, *pool.full_kv_pool.v_buffer):
                        expected.append((buf, 1 if hnd else 4, 3 if hnd else 12, 1 if hnd else 4))
                    expected.extend((buf, 2, 6, 2) for buf in pool.qsa_compressed_k_buffer_pool)
                mamba = params.req_to_token_pool.mamba_pool
                checkpoint = [s for _, s, _, _ in mamba._iter_transfer_state_entries()]
                expected.extend((buf, 2, 6, 1) for buf in checkpoint)
                self.assertEqual(len(snapshot), len(expected))
                expected_bytes = [buf[src:src + span].contiguous().view(torch.uint8).numpy().tobytes() for buf, src, _, span in expected]
                self.assertCountEqual(list(snapshot.values()), expected_bytes)
                untouched = [buf[0].clone() for buf, _, _, _ in expected]
                for buf, _, dst, span in expected:
                    buf[dst:dst + span].zero_()
                loaded = []
                for layer in range(group.num_layers):
                    for batch in layout.mappings(destination, layer):
                        for key, refs in batch.items():
                            for ref in refs:
                                addr, size = address(ref)
                                ctypes.memmove(addr, snapshot[key, ref.label], size)
                                loaded.append((key, ref.label))
                    if layer == 0:
                        # Request-wide history is consumed before the layer loop;
                        # all checkpoint pieces must land at the first gate.
                        for buf in checkpoint:
                            self.assertTrue(torch.equal(buf[2], buf[6]))
                self.assertCountEqual(loaded, snapshot.keys())
                for index, (buf, src, dst, span) in enumerate(expected):
                    self.assertTrue(torch.equal(buf[src:src + span], buf[dst:dst + span]))
                    self.assertTrue(torch.equal(buf[0], untouched[index]))
                self.assertTrue(torch.all(mamba.mamba_cache.intermediate_ssm == -99))
                self.assertTrue(torch.all(mamba.mamba_cache.replayssm_d == -98))

    def test_missing_checkpoint_cannot_resolve_a_full_hit(self):
        target, _, params = self.pools()
        try:
            group = self.group(target, params)
        except ValueError as error:
            self.fail(f"Approved QSA layout was rejected: {error}")
        self.assertEqual(group.resolve_transfers([self.Transfer(name=self.PoolName.KV, keys=["prefix"])]), [])
        resolved = group.resolve_transfers([self.Transfer(name=self.PoolName.KV, keys=["prefix"]), self.Transfer(name=self.PoolName.MAMBA, keys=["prefix"], hit_policy=self.Policy.TRAILING_PAGES)])
        self.assertEqual([t.hit_policy for t in resolved], [self.Policy.ALL_PAGES, self.Policy.ALL_PAGES, self.Policy.TRAILING_PAGES])

    def test_rejects_layouts_that_would_drop_state(self):
        cases = [
            (lambda t, d, p: setattr(t, "qsa_compress_ratio", 3), "compress"),
            (lambda t, d, p: setattr(self.memory, "enable_unified_memory", True), "unified"),
            (lambda t, d, p: setattr(p, "pp_size", 2), "PP1/CP1"),
            (lambda t, d, p: setattr(p, "attn_cp_size", 2), "PP1/CP1"),
            (lambda t, d, p: setattr(d, "mamba_pool", object()), "shared"),
            (lambda t, d, p: setattr(p, "mtp_draft_device_pools", (d, d)), "one"),
            (lambda t, d, p: setattr(d, "qsa_compress_ratio", 4), "compress"),
            (lambda t, d, p: setattr(p.req_to_token_pool, "mamba_ckpt_pool", object()), "int8"),
            (lambda t, d, p: setattr(p, "mtp_draft_mamba_pools", (object(),)), "draft"),
            (lambda t, d, p: setattr(p, "enable_mamba_extra_buffer", False), "extra.buffer"),
            (lambda t, d, p: setattr(t.full_kv_pool, "is_quantized_kv_cache", True), "scale"),
        ]
        for change, message in cases:
            with self.subTest(message=message):
                self.memory.enable_unified_memory = False
                target, draft, params = self.pools()
                change(target, draft, params)
                with self.assertRaisesRegex(ValueError, message):
                    self.group(target, params)

    def test_qsa_resume_at_page_boundary_overwrites_stale_pending_ring(self):
        # Exercise the source's actual eager decode/extend ring writes and
        # compression plan, with normalization replaced by a deterministic
        # coordinate-sensitive function (no model weights or rotary kernel).
        funcs = source_definitions(
            "layers/attention/qsa/metadata.py",
            {"build_pending_ring_slots", "build_group_ring_slots", "build_rope_position_matrix"},
            {"torch": torch},
        )
        methods = source_methods(
            "layers/attention/qsa/qsa_indexer.py", "QSAIndexer",
            {"update_key_state_and_compress", "_pending_ring_slots", "_group_ring_slots", "_rope_from_matrix"},
        )
        average = source_definitions(
            "layers/attention/qsa/kernel.py", {"average_pool_qsa_keys"}, {"torch": torch}
        )["average_pool_qsa_keys"]
        methods.update(funcs, average_pool_qsa_keys=average)
        for fn in methods.values():
            if isinstance(fn, types.FunctionType):
                fn.__globals__.update(funcs, average_pool_qsa_keys=average)
        for ratio in (2, 4, 8):
            for extend in (False, True):
                with self.subTest(ratio=ratio, extend=extend):
                    results = []
                    for stale in (-1000, 1000):
                        key_ring = torch.full((3 * ratio, 1, 3), float(stale))
                        rope_ring = torch.full((3 * ratio, 3), stale, dtype=torch.int64)
                        pool = types.SimpleNamespace(qsa_rope_position_buffer=rope_ring)
                        pool.get_qsa_key_state_buffer = lambda layer: key_ring
                        pool.set_qsa_key_state_buffer = lambda layer, slots, values: key_ring.index_copy_(0, slots, values)
                        pool.set_qsa_rope_position_buffer = lambda slots, positions: rope_ring.index_copy_(0, slots, positions.expand(3, -1).T)
                        compressed = []
                        pool.set_qsa_compressed_k_buffer = lambda layer, locs, values: compressed.append(values.clone())
                        indexer = types.SimpleNamespace(compress_ratio=ratio, layer_id=0, rotary_emb=types.SimpleNamespace())
                        for name in ("update_key_state_and_compress", "_pending_ring_slots", "_group_ring_slots", "_rope_from_matrix"):
                            setattr(indexer, name, types.MethodType(methods[name], indexer))
                        indexer._use_fused_compress = lambda pool: False
                        indexer.normalize_compressed_keys = lambda keys, positions: keys + positions[:, None, None]
                        # 64 is a full KV page boundary; no member of the next
                        # group precedes this resume point. Include a nonzero
                        # starting RoPE coordinate to catch stale positions.
                        for offset in range(0, ratio, ratio if extend else 1):
                            count = ratio if extend else 1
                            positions = torch.arange(64 + offset, 64 + offset + count)
                            complete = offset + count == ratio
                            metadata = types.SimpleNamespace(
                                token_to_kv_pool=pool,
                                compress_member_rows=torch.tensor([0]) if extend else None,
                                token_to_batch_idx=torch.zeros(count, dtype=torch.int64),
                                req_pool_indices=torch.tensor([2]),
                                sequence_lengths=torch.tensor([64 + offset + count]),
                                is_cuda_graph=False,
                                write_locs=torch.tensor([1]) if complete else torch.empty(0, dtype=torch.int64),
                                compress_group_positions=torch.tensor([64 + ratio - 1]),
                                compress_group_ring_locs=None,
                                compress_sequence_ids=torch.tensor([0]),
                                extend_rope_matrix=None,
                            )
                            keys = positions.float()[:, None, None].expand(-1, 1, 3).contiguous()
                            indexer.update_key_state_and_compress(keys, positions, positions + 700, metadata)
                        self.assertEqual(len(compressed), 1)
                        results.append(compressed[0])
                    self.assertTrue(torch.equal(*results))
                    expected = torch.full((1, 1, 3), 64 + (ratio - 1) / 2 + 764)
                    self.assertTrue(torch.equal(results[0], expected))

    def test_lookup_requires_indexer_pages_and_exact_checkpoint_boundary(self):
        target, _, params = self.pools()
        group = self.group(target, params)
        layout = self.Layout(group, model_name="pinned-qwen38", endpoint_name="worker")
        prepared = group.resolve_transfers([
            self.Transfer(name=self.PoolName.KV, keys=["a", "b", "c"]),
            self.Transfer(name=self.PoolName.MAMBA, keys=["c"], hit_policy=self.Policy.TRAILING_PAGES),
        ])
        lookup = source_methods(
            "mem_cache/storage/kvcr/kvcr_direct_linker.py", "KVCRDirectLinker", {"_lookup"}
        )["_lookup"]
        lookup.__globals__.update(PoolHitPolicy=self.Policy)
        inventory = {self.PoolName.KV: [True, True, True], self.PoolName.INDEXER: [True, True, False], self.PoolName.MAMBA: [False, True, True]}
        def query(keys, request_id):
            name, _ = layout.decode_key(keys[0])
            return [(types.SimpleNamespace(value="HIT" if hit else "MISS"), types.SimpleNamespace(value="DRAM")) for hit in inventory[name]]
        linker = types.SimpleNamespace(layout=layout, _request_scopes={}, _discard_lookup=lambda scope: None, _runtime=types.SimpleNamespace(client=types.SimpleNamespace(query=query)), _shard=False)
        self.assertEqual(lookup(linker, "req", ["a", "b", "c"], prepared), [2])
        inventory[self.PoolName.MAMBA] = [False, False, False]
        self.assertEqual(lookup(linker, "req", ["a", "b", "c"], prepared), [])

    def test_existing_page_layout_regressions(self):
        # Reuse the upstream geometry suite rather than duplicating its cases.
        import io
        from typing import Any
        ns = dict(
            unittest=unittest, torch=torch, patch=patch, Mock=Mock,
            SimpleNamespace=types.SimpleNamespace, CustomTestCase=unittest.TestCase,
            DevicePoolEntry=self.assembler.DevicePoolEntry,
            DevicePoolGroup=self.assembler.DevicePoolGroup,
            KVCRLayout=self.Layout, PoolName=self.PoolName,
            PoolTransfer=self.Transfer, PoolHitPolicy=self.Policy, Any=Any,
        )
        names = {"TestKVCRLayout", "TestDevicePoolEntry", "TestDevicePoolGroup"}
        source_definitions(
            ROOT.parents[2] / "test/registered/unit/mem_cache/test_linker_pool_assembler.py",
            names, ns,
        )
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromTestCase(ns[name])
            for name in sorted(names)
        )
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output).run(suite)
        self.assertTrue(result.wasSuccessful(), output.getvalue())
        print(f"upstream page-layout regressions: {result.testsRun} passed")

    def test_factory_registers_request_state_layer_barrier(self):
        # Execute the real factory with construction/transport replaced, so the
        # regression tests ordering/wiring without importing the server stack.
        components = types.ModuleType("sglang.srt.mem_cache.unified_cache.components")
        components.ComponentType = self.Component
        radix = types.ModuleType("sglang.srt.mem_cache.unified_radix_cache")
        radix.UnifiedRadixCache = object
        kvcr = types.ModuleType("sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker")
        counter = object()
        kvcr.KVCRDirectLinker = Mock(return_value=types.SimpleNamespace(layer_done_counter=counter))
        req_pool, kv_pool, tp_worker = Mock(), Mock(), Mock()
        del req_pool.req_to_c128_sidecar
        params = types.SimpleNamespace(req_to_token_pool=req_pool, token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: kv_pool))
        ctx = types.SimpleNamespace(params=params, server_args=None, is_hybrid_ssm=True, is_hybrid_swa=False, enable_hierarchical_cache=False, tp_worker=tp_worker)
        cache = types.SimpleNamespace(components={self.Component.FULL: None, self.Component.MAMBA: None})
        def attach(linker):
            cache.linker = linker
        cache.init_cache_linker = attach
        ns = dict(get_memory=lambda: types.SimpleNamespace(enable_unified_cache_external_linker=True, unified_cache_external_linker_backend="kvcr"), get_disagg=lambda: types.SimpleNamespace(disaggregation_decode_retraction_backup="none"), use_mlx=lambda: False)
        factory = source_definitions("mem_cache/registry.py", {"create_unified_radix_cache"}, ns)["create_unified_radix_cache"]
        with patch.dict(sys.modules, {components.__name__: components, radix.__name__: radix, kvcr.__name__: kvcr}):
            factory(ctx, cache_class=lambda p: cache)
        req_pool.register_layer_transfer_counter.assert_called_once_with(counter)
        kv_pool.register_layer_transfer_counter.assert_called_once_with(counter)
        tp_worker.register_hicache_layer_transfer_counter.assert_called_once_with(counter)


if __name__ == "__main__":
    unittest.main(verbosity=2)
