"""Audit actual pinned KVCR hint membership with SGLang's multi-pool adapter.

Run with KVCR source and its CPU test utilities (no installs/NIXL/GPU needed):
  PYTHONPATH=kvcr/src:kvcr/tests/unit python \
    sglang/test/registered/unit/mem_cache/test_qwen38_kvcr_hints_cpu.py

Only NIXL and peer control use the pinned upstream fakes. KVCR construction,
submit_hint, parsing, dispatch, query, KeyAdapter and InventorySink are real.
SGLang definitions are source-loaded to avoid importing serving dependencies.
"""

import hashlib
import importlib.util
import json
import sys
from collections import Counter, namedtuple
from dataclasses import dataclass, replace
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest import TestCase, main
from unittest.mock import patch

import msgspec
import torch
from _kvcr_test_utils import FakeBytesControl, FakeNixlAgent, _use_nixl_agent

from kvcr.types import CacheTier, QueryStatus

from test_qwen38_direct_linker_cpu import ROOT, source_definitions, source_methods


def load_runtime():
    path = ROOT / "mem_cache/storage/kvcr/runtime.py"
    spec = importlib.util.spec_from_file_location("qwen38_hint_audit_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KVCRRuntime


class TestQwen38KVCRHintsCPU(TestCase):
    def setUp(self):
        storage = source_definitions(
            "mem_cache/hicache_storage.py",
            {"PoolName", "PoolHitPolicy", "PoolTransfer"},
            dict(Enum=Enum, dataclass=dataclass),
        )
        self.Pool = storage["PoolName"]
        self.Policy = storage["PoolHitPolicy"]
        self.Transfer = storage["PoolTransfer"]
        pools = source_definitions(
            "mem_cache/hybrid_cache/linker_pool_assembler.py",
            {"DevicePoolEntry", "DevicePoolGroup"},
            dict(torch=torch, replace=replace, **storage),
        )
        self.names = (self.Pool.KV, self.Pool.INDEXER, self.Pool.MAMBA)
        entries = [
            pools["DevicePoolEntry"](
                name=name,
                indices_from_pool=(self.Pool.KV if name != self.Pool.MAMBA else name),
                device_pool=None,
                components=[[torch.zeros((8, 4), dtype=torch.bfloat16)]],
                layer_mapping={0: 0},
                page_size=1 if name == self.Pool.MAMBA else 4,
                rows_are_pages=True,
            )
            for name in self.names
        ]
        group = pools["DevicePoolGroup"](entries, num_layers=1, page_size=4)
        layout_type = source_definitions(
            "mem_cache/storage/kvcr/device_layout.py",
            {"PreparedTransfer", "KVCRLayout"},
            storage | dict(hashlib=hashlib, json=json, Counter=Counter, dataclass=dataclass),
        )["KVCRLayout"]
        self.layout = layout_type(group, model_name="pinned-qwen38", endpoint_name="target")
        # Test the constructor's actual KeyAdapter and inventory callback bindings,
        # not a substitute decode function or a mocked client.query.
        with (
            _use_nixl_agent(FakeNixlAgent()),
            patch("kvcr.ZmqPeerControlChannel", return_value=FakeBytesControl()),
        ):
            self.runtime = load_runtime()(
                self.layout,
                {
                    "kvcr_service_socket_path": None,
                    "dram_size_gb": 4096 / (1 << 30),
                    "advertise_host": "127.0.0.1",
                },
                rank=0,
            )
        self.addCleanup(self.runtime.close)
        self.client = self.runtime.client
        self.remote = self.client._core._remote_fw_dram
        self.assertFalse(self.remote._options.opportunistic_query)
        self.assertIs(self.remote._key_adapter, self.runtime)
        hints_module = ModuleType("qwen38_hint_envelope")
        hints_module.__dict__.update(msgspec=msgspec, Any=Any, Dict=Dict, List=List)
        self.enterContext(patch.dict(sys.modules, {hints_module.__name__: hints_module}))
        hints = source_definitions(
            "managers/kv_hints.py",
            {"KvHintAction", "KvHintsEnvelope", "decode_kv_hints_envelope"},
            hints_module.__dict__,
        )
        self.decode_hint = hints["decode_kv_hints_envelope"]
        self.hashes = ("0123456789abcdef" + "0" * 48, "fedcba9876543210" + "1" * 48)

    def keys(self, storage_hash):
        return [self.layout.encode_key(storage_hash, name) for name in self.names]

    def hint(self, storage_hashes, *, request="req", source="tcp://source:1"):
        envelope = self.decode_hint({
            "protocol_version": "0.1",
            "message_id": "membership-audit",
            "actions": [{
                "action_id": "fetch",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": source,
                    "block_hashes": [int(h[:16], 16) for h in storage_hashes],
                },
            }],
        })
        self.client.submit_hint(envelope, request_id=request)

    def test_logical_hash_hint_authorizes_all_three_pools_without_encode_expansion(self):
        keys = [key for value in self.hashes for key in self.keys(value)]
        with patch.object(self.runtime, "encode", wraps=self.runtime.encode) as encode:
            self.hint(self.hashes)
            self.assertEqual(
                self.client.query(keys, request_id="req"),
                [(QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)] * len(keys),
            )
            encode.assert_not_called()
        self.assertEqual(
            self.remote._request_hints["req"].block_hashes,
            frozenset(int(h[:16], 16) for h in self.hashes),
        )
        for value in self.hashes:
            self.assertEqual([self.runtime.decode(k) for k in self.keys(value)], [int(value[:16], 16)] * 3)

    def test_unlisted_hashes_and_wrong_request_are_not_authorized(self):
        self.hint(self.hashes[:1])
        self.assertEqual(self.client.query(self.keys(self.hashes[1]), "req"), [(QueryStatus.MISS, None)] * 3)
        self.assertEqual(self.client.query(self.keys(self.hashes[0]), "other-request"), [(QueryStatus.MISS, None)] * 3)
        self.client.discard_hint("req")
        self.assertEqual(self.client.query(self.keys(self.hashes[0]), "req"), [(QueryStatus.MISS, None)] * 3)

    def test_same_source_replaces_membership_and_different_source_fails_closed(self):
        self.hint(self.hashes[:1])
        self.hint(self.hashes[1:])
        self.assertEqual(self.client.query(self.keys(self.hashes[0]), "req"), [(QueryStatus.MISS, None)] * 3)
        self.assertEqual(self.client.query(self.keys(self.hashes[1]), "req"), [(QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)] * 3)
        self.hint(self.hashes, source="tcp://other-source:2")
        self.assertTrue(self.remote._request_hints["req"].failed)
        self.assertEqual(self.client.query(self.keys(self.hashes[1]), "req"), [(QueryStatus.MISS, None)] * 3)

    def test_real_linker_lookup_uses_native_query_for_all_pool_names(self):
        self.hint(self.hashes)
        transfers = self.layout.group.resolve_transfers([
            self.Transfer(name=self.Pool.KV, keys=list(self.hashes)),
            self.Transfer(name=self.Pool.MAMBA, keys=[self.hashes[-1]], hit_policy=self.Policy.TRAILING_PAGES),
        ])
        lookup = source_methods(
            "mem_cache/storage/kvcr/kvcr_direct_linker.py", "KVCRDirectLinker", {"_lookup"}
        )["_lookup"]
        lookup.__globals__["PoolHitPolicy"] = self.Policy
        linker = SimpleNamespace(
            layout=self.layout, _runtime=self.runtime, _request_scopes={"req": "req"},
            _discard_lookup=lambda scope: None, _shard=False,
        )
        self.assertEqual(lookup(linker, "req", self.hashes, transfers), [1, 2])
        self.hint(self.hashes[:1])
        self.assertEqual(lookup(linker, "req", self.hashes, transfers), [1])

    def test_linker_context_forwards_hash_membership_in_attempt_scope(self):
        set_context = source_methods(
            "mem_cache/storage/kvcr/kvcr_direct_linker.py",
            "KVCRDirectLinker", {"_set_context"},
        )["_set_context"]
        set_context.__globals__["decode_kv_hints_envelope"] = self.decode_hint
        linker = SimpleNamespace(
            _scopes={}, _request_scopes={}, _remote_hints_supported=True,
            _source_rank_offset=0, _runtime=self.runtime,
        )
        handle = namedtuple("Handle", "rid attempt_id")("request", 7)
        envelope = {
            "protocol_version": "0.1", "message_id": "context-audit",
            "actions": [{
                "action_id": "fetch", "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://source:1",
                    "block_hashes": [int(self.hashes[0][:16], 16)],
                },
            }],
        }
        set_context(linker, handle, envelope)
        scope = "7:request:7"
        self.assertEqual(linker._request_scopes[handle.rid], scope)
        self.assertEqual(
            self.client.query(self.keys(self.hashes[0]), scope),
            [(QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)] * 3,
        )
        self.assertEqual(
            self.client.query(self.keys(self.hashes[1]), scope),
            [(QueryStatus.MISS, None)] * 3,
        )

    def test_inventory_callback_preserves_all_physical_keys_and_logical_hashes(self):
        keys = self.keys(self.hashes[0])
        core = self.client._core
        self.assertTrue(core._publish_inventory(keys, CacheTier.LOCAL_G2, removed=True))
        self.assertEqual(self.runtime.removals, {key: self.hashes[0] for key in keys})
        # Re-add INDEXER alone: its exact physical key clears independently.
        core._publish_inventory([keys[1]], CacheTier.LOCAL_G2, removed=False)
        self.assertEqual(set(self.runtime.removals), {keys[0], keys[2]})
        foreign = b"sglang-kvcr-v1:foreign:kv:" + self.hashes[0].encode()
        core._publish_inventory([foreign], CacheTier.LOCAL_G2, removed=True)
        self.assertNotIn(foreign, self.runtime.removals)
        with self.assertRaisesRegex(ValueError, "Unknown KVCR layout key"):
            self.runtime.decode(foreign)


if __name__ == "__main__":
    main(verbosity=2)
