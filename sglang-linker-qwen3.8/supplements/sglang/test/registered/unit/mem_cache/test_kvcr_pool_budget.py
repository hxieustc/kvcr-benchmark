"""CPU-only pool sizing checks; source-load runtime without serving imports.

Run: python -B test/registered/unit/mem_cache/test_kvcr_pool_budget.py
"""

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "kvcr_pool_budget_runtime",
    ROOT / "python/sglang/srt/mem_cache/storage/kvcr/runtime.py",
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def layout(pools, trailing=()):
    families = {name.split("_", 1)[0] for name, _, _ in pools}
    return SimpleNamespace(
        namespace="unchanged",
        endpoint_name="budget-test",
        registrations=[],
        pool_layouts=[(name, size) for name, size, _ in pools],
        pool_counts=Counter({name: count for name, _, count in pools}),
        trailing_counts=Counter(dict(trailing)),
        group=SimpleNamespace(
            entries=[
                SimpleNamespace(
                    name=name,
                    indices_from_pool="kv" if name == "indexer" else name,
                    page_size=1 if name == "mamba" else 64,
                )
                for name in sorted(families)
            ]
        ),
    )


def qsa_layout():
    return layout(
        [
            ("indexer_4096", 4096, 13),
            ("kv_65536", 65536, 26),
            ("mamba_1572864", 1572864, 36),
            ("mamba_16", 16, 1),
            ("mamba_184320", 184320, 1),
            ("mamba_61440", 61440, 36),
        ]
    )


class TestPoolBudget(unittest.TestCase):
    def sizes(self, geometry, budget, divisor=1, checkpoints=None):
        self.assertTrue(hasattr(runtime, "_pool_lengths"), "missing pure pool sizer")
        return runtime._pool_lengths(geometry, budget, divisor, checkpoints)

    def test_real_default_is_exactly_legacy(self):
        geometry = qsa_layout()
        expected = [
            size * geometry.pool_counts[name] * 3957
            for name, size in geometry.pool_layouts
        ]
        self.assertEqual(self.sizes(geometry, 224 << 30), expected)
        self.assertEqual(sum(expected), 240492468048)
        self.assertEqual(
            sum(n // s for n, (_, s) in zip(expected, geometry.pool_layouts)), 447141
        )

    def test_real_independent_capacities_and_rounding(self):
        geometry = qsa_layout()
        for checkpoints, pages, slack in ((1024, 102483, 937984), (2048, 68090, 24576)):
            for divisor in (1, 4, 1024):
                with self.subTest(checkpoints=checkpoints, divisor=divisor):
                    lengths = self.sizes(geometry, 224 << 30, divisor, checkpoints)
                    for (name, size), length in zip(geometry.pool_layouts, lengths):
                        count = checkpoints if name.startswith("mamba_") else pages
                        self.assertEqual(
                            length, size * geometry.pool_counts[name] * count
                        )
                    self.assertEqual((224 << 30) - sum(lengths), slack)
                    self.assertLess(slack, 1757184)

    def test_legacy_and_swa_unchanged(self):
        geometry = layout([("small", 8, 2), ("large", 24, 2)])
        self.assertEqual(self.sizes(geometry, 128), [32, 96])
        geometry.trailing_counts["large"] = 2
        for divisor, budget, expected in (
            (1, 128, [32, 96]),
            (4, 128, [64, 48]),
            (1024, 64, [16, 48]),
        ):
            self.assertEqual(self.sizes(geometry, budget, divisor), expected)
        with self.assertRaises(ValueError):
            self.sizes(geometry, 63, 4)

    def test_tiny_exact_fit_and_remainder(self):
        geometry = layout([("kv_8", 8, 1), ("mamba_16", 16, 2)])
        for budget, expected in ((72, [8, 64]), (79, [8, 64]), (80, [16, 64])):
            self.assertEqual(self.sizes(geometry, budget, checkpoints=2), expected)
        for budget in (0, 63, 64, 71):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.sizes(geometry, budget, checkpoints=2)

    def test_checkpoint_capacity_requires_positive_builtin_integer(self):
        class IntegerLike(int):
            pass

        for value in (True, False, 0, -1, 1.0, "1", IntegerLike(1)):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(
                ValueError, "mamba_checkpoint_capacity"
            ):
                self.sizes(qsa_layout(), 224 << 30, checkpoints=value)
        with self.assertRaises(ValueError):
            self.sizes(qsa_layout(), 224 << 30, checkpoints=10**100)

    def test_invalid_divisors(self):
        for value in (True, 0, -1, 1.0, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "divisor"
            ):
                self.sizes(qsa_layout(), 224 << 30, divisor=value)

    def test_unsupported_topologies_fail_closed(self):
        geometries = [
            layout([("kv_8", 8, 1)]),
            layout([("mamba_16", 16, 1)]),
            layout(
                [("kv_8", 8, 1), ("mamba_16", 16, 1), ("swa_8", 8, 1)],
                [("swa_8", 1)],
            ),
            layout([("kv_8", 8, 1), ("mamba_16", 16, 1), ("other_8", 8, 1)]),
        ]
        bad_source = qsa_layout()
        bad_source.group.entries[0].indices_from_pool = "swa"
        geometries.append(bad_source)
        bad_page = qsa_layout()
        bad_page.group.entries[-1].page_size = 64
        geometries.append(bad_page)
        mixed = qsa_layout()
        mixed.trailing_counts["mamba_16"] = 1
        geometries.append(mixed)
        for geometry in geometries:
            with self.subTest(pools=geometry.pool_layouts), self.assertRaisesRegex(
                ValueError, "topology"
            ):
                self.sizes(geometry, 224 << 30, checkpoints=1)

    def test_runtime_option_reaches_process_local_allocation(self):
        modules = {
            name: ModuleType(name) for name in ("kvcr", "kvcr.config", "kvcr.types")
        }
        modules["kvcr"].KVCR = lambda *args: SimpleNamespace(
            close=lambda: None, backends=args[2]
        )
        modules["kvcr"].KVCRBindings = SimpleNamespace
        modules["kvcr"].KVCRSocketError = type("SocketError", (Exception,), {})
        modules["kvcr"].ZmqPeerControlChannel = lambda *args: SimpleNamespace(
            endpoint="local"
        )
        for name in (
            "KVCRBackendConfigs", "KVCRConfig", "KVCRGuardConfig", "RemoteFWDramOptions"
        ):
            setattr(modules["kvcr.config"], name, SimpleNamespace)
        modules["kvcr.config"].LocalDramOptions = lambda pools: SimpleNamespace(
            pools=pools
        )
        modules["kvcr.types"].CacheTier = SimpleNamespace(LOCAL_G2=object())
        modules["kvcr.types"].KVCRStartupError = type("StartupError", (Exception,), {})
        geometry = layout([("kv_8", 8, 1), ("mamba_16", 16, 2)])
        options = dict(
            dram_size_gb=128 / (1 << 30),
            kvcr_service_socket_path=None,
            advertise_host="127.0.0.1",
            mamba_checkpoint_capacity=2,
        )
        with patch.dict(sys.modules, modules):
            instance = runtime.KVCRRuntime(geometry, options, rank=0)
            try:
                self.assertEqual(
                    [n for _, _, n in instance.client.backends.local_dram.pools],
                    [64, 64],
                )
            finally:
                instance.close()
        with self.assertRaisesRegex(ValueError, "process-local"):
            runtime.KVCRRuntime(
                geometry, options | {"kvcr_service_socket_path": "/guard.sock"}, rank=0
            )
        for value in (True, 0, -1, float("nan"), float("inf"), "1"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "dram_size_gb"
            ):
                runtime.KVCRRuntime(
                    geometry, options | {"dram_size_gb": value}, rank=0
                )


if __name__ == "__main__":
    unittest.main()
