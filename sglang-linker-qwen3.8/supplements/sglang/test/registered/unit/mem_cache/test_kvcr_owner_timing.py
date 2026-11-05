"""CPU-only owner timing regressions; source-load real code without serving imports.

Run: python -B test/registered/unit/mem_cache/test_kvcr_owner_timing.py
"""

import ast
import contextlib
import json
import threading
import time
import unittest
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/mem_cache/storage/kvcr/kvcr_direct_linker.py"
)


def load_source():
    tree = ast.parse(SOURCE.read_text())
    # Keep real definitions/constants, excluding imports and the unrelated GPU
    # gather class. Postponed annotations avoid requiring torch or KVCR.
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign))
        and getattr(node, "name", None) != "_SplitLayerGather"
        and not (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "logger" for t in node.targets)
        )
    ]
    namespace = dict(
        __name__=__name__,
        Counter=Counter,
        deque=deque,
        dataclass=dataclass,
        field=field,
        replace=replace,
        Future=Future,
        Queue=Queue,
        Empty=Empty,
        threading=threading,
        time=time,
        json=json,
        logger=Mock(),
        UnifiedCacheLinker=object,
        StorageMedium=SimpleNamespace(CPU="cpu"),
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    code = compile(
        ast.fix_missing_locations(ast.Module([future, *body], [])), str(SOURCE), "exec"
    )
    exec(code, namespace)
    return namespace


class TestOwnerTiming(unittest.TestCase):
    def setUp(self):
        self.ns = load_source()
        self.Linker = self.ns["KVCRDirectLinker"]

    def timing(self, role="owner"):
        return self.ns["_OwnerTiming"](rank=3, role=role)

    def bare_linker(self, enabled=False):
        linker = self.Linker.__new__(self.Linker)
        linker._owner_timing = self.timing() if enabled else None
        linker._caller_timing = threading.local() if enabled else None
        linker._timing_rank = 3
        linker._commands = Queue()
        linker._command_lock = threading.Lock()
        linker._closed = False
        return linker

    def test_interval_aggregation_rate_limit_reset_and_bounded_labels(self):
        with patch.object(time, "perf_counter", return_value=10.0):
            timing = self.timing()
        timing.observe("call_service_seconds", 0.25, "_lookup")
        timing.observe("call_service_seconds", 0.75, "_lookup")
        with patch.object(time, "perf_counter", return_value=14.99):
            timing.log_stats()
        self.ns["logger"].info.assert_not_called()
        with patch.object(time, "perf_counter", return_value=15.0):
            timing.log_stats()
        record = json.loads(self.ns["logger"].info.call_args.args[1])
        self.assertEqual(
            (record["rank"], record["role"], record["interval_s"]), (3, "owner", 5.0)
        )
        stats = record["stats"]
        self.assertEqual(stats["call_service_seconds[_lookup].count"], 2)
        self.assertEqual(stats["call_service_seconds[_lookup].sum"], 1.0)
        self.assertEqual(stats["call_service_seconds[_lookup].max"], 0.75)
        self.assertFalse(timing.stats)
        for i in range(10000):
            timing.observe("call_service_seconds", 1.0, f"unexpected_{i}")
        self.assertLessEqual(len(timing.stats), 65)
        self.assertEqual(sum(v[0] for v in timing.stats.values()), 10000)
        self.assertLessEqual(max(map(len, timing.stats)), 160)

    def test_logging_failure_cannot_escape_into_serving(self):
        timing = self.timing()
        timing.observe("progress_wall_seconds", 0.1)
        self.ns["logger"].info.side_effect = RuntimeError("broken sink")
        with patch.object(time, "perf_counter", return_value=timing.last_log + 5):
            timing.log_stats()
        for _ in range(100):
            timing.observe("progress_wall_seconds", 0.1)
        self.assertFalse(timing.stats)

    def test_default_off_does_not_read_timing_clocks_or_change_call_result(self):
        linker = self.bare_linker()

        class ImmediateQueue:
            def put(self, command):
                function, args, future = command
                future.set_result(function(*args))

        linker._commands = ImmediateQueue()
        with (
            patch.object(time, "perf_counter", side_effect=AssertionError("clock")),
            patch.object(time, "thread_time", side_effect=AssertionError("CPU clock")),
        ):
            self.assertEqual(linker._call(lambda x: x + 1, 4), 5)
            linker._progress_impl = Mock(return_value="progress")
            self.assertEqual(linker._progress(), "progress")
            linker._prepare_offload_impl = Mock(return_value=[{"key": "refs"}])
            self.assertEqual(linker._prepare_offload("transfers"), [{"key": "refs"}])
        self.ns["logger"].info.assert_not_called()

    def test_call_wall_records_success_and_original_exception(self):
        linker = self.bare_linker(enabled=True)
        error = ValueError("native failure")

        class ImmediateQueue:
            def put(self, command):
                function, args, future = command
                self.enqueued = future._kvcr_enqueued_at
                try:
                    future.set_result(function(*args))
                except BaseException as caught:
                    future.set_exception(caught)

        linker._commands = ImmediateQueue()
        self.assertEqual(linker._call(lambda: 7), 7)

        def fail():
            raise error

        with self.assertRaises(ValueError) as caught:
            linker._call(fail)
        self.assertIs(caught.exception, error)
        stats = linker._caller_timing.stats.stats
        self.assertEqual(stats["call_wall_seconds[<lambda>]"][0], 1)
        self.assertEqual(stats["call_wall_seconds[fail]"][0], 1)
        self.assertTrue(all(v[1] >= 0 for v in stats.values()))

    def test_progress_records_wall_cpu_gaps_and_preserves_exception(self):
        linker = self.bare_linker(enabled=True)
        linker._last_progress_at = None
        error = RuntimeError("poll failed")
        linker._progress_impl = Mock(side_effect=[17, error])
        with (
            patch.object(time, "perf_counter", side_effect=[10.0, 10.5, 12.0, 12.75]),
            patch.object(time, "thread_time", side_effect=[1.0, 1.1, 1.2, 1.4]),
            patch.object(linker._owner_timing, "log_stats"),
        ):
            self.assertEqual(linker._progress(), 17)
            with self.assertRaises(RuntimeError) as caught:
                linker._progress()
        self.assertIs(caught.exception, error)
        stats = linker._owner_timing.stats
        self.assertEqual(stats["progress_wall_seconds[]"], (2, 1.25, 0.75))
        self.assertAlmostEqual(stats["progress_cpu_seconds[]"][1], 0.3)
        self.assertEqual(stats["progress_service_gap_seconds[]"], (1, 1.5, 1.5))

    def test_prepare_and_submit_preserve_returns_and_exceptions(self):
        linker = self.bare_linker(enabled=True)
        linker._prepare_offload_impl = Mock(return_value=[{"k": ["r"]}])
        self.assertEqual(linker._prepare_offload("t"), [{"k": ["r"]}])
        error = ValueError("bad descriptors")
        linker._prepare_offload_impl.side_effect = error
        with self.assertRaises(ValueError) as caught:
            linker._prepare_offload("t")
        self.assertIs(caught.exception, error)
        self.assertEqual(linker._owner_timing.stats["offload_prepare_seconds[]"][0], 2)
        native = Mock(side_effect=[9, error])
        self.assertEqual(linker._timed_submit(native, {"k": []}, request_id="scope"), 9)
        with self.assertRaises(ValueError) as caught:
            linker._timed_submit(native, {"k": []}, request_id="scope")
        self.assertIs(caught.exception, error)
        self.assertEqual(native.call_args.kwargs, {"request_id": "scope"})

    def test_real_owner_loop_records_queue_service_and_preserves_future_errors(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                linker = self.bare_linker(enabled)
                linker.pool_group = SimpleNamespace(
                    entries=[
                        SimpleNamespace(kv_buffer=[SimpleNamespace(is_cuda=False)])
                    ]
                )
                linker._make_runtime = lambda: SimpleNamespace(log_stats=lambda: None)
                linker._started = Future()
                linker._fatal = None
                linker._batches = deque()
                linker._offloads = deque()
                linker._progress = lambda: None
                linker._reset = lambda: None
                linker._close = lambda: setattr(linker, "_closed", True)
                error = ValueError("owner command failed")

                def succeed():
                    return 42

                def fail():
                    raise error

                thread = threading.Thread(target=linker._run, daemon=True)
                thread.start()
                linker._started.result(timeout=2)
                try:
                    self.assertEqual(linker._call(succeed), 42)
                    with self.assertRaises(ValueError) as caught:
                        linker._call(fail)
                    self.assertIs(caught.exception, error)
                    self.assertIsNone(linker._fatal)
                finally:
                    linker._call(linker._close)
                    thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                if enabled:
                    stats = linker._owner_timing.stats
                    for name in ("succeed", "fail"):
                        for metric in ("call_queue_seconds", "call_service_seconds"):
                            self.assertEqual(stats[f"{metric}[{name}]"][0], 1)
                            self.assertGreaterEqual(stats[f"{metric}[{name}]"][1], 0)
                else:
                    self.ns["logger"].info.assert_not_called()

    def test_mapping_timer_covers_real_mapping_and_failure(self):
        linker = self.bare_linker(enabled=True)
        linker._prepare_stream = None
        linker._shard = False
        prepared = object()
        linker.layout = SimpleNamespace(
            prepare=Mock(return_value=prepared),
            registrations=[SimpleNamespace(label="region", size=8)],
            mappings=Mock(return_value=[{"key": [SimpleNamespace(label="region")]}]),
        )
        self.ns["nullcontext"] = contextlib.nullcontext
        result = linker._prepare_offload(["transfer"])
        self.assertEqual(list(result[0]), ["key"])
        linker.layout.mappings.assert_called_once_with(prepared)
        error = ValueError("mapping failed")
        linker.layout.mappings.side_effect = error
        with self.assertRaises(ValueError) as caught:
            linker._prepare_offload(["transfer"])
        self.assertIs(caught.exception, error)
        stats = linker._owner_timing.stats
        for metric in (
            "offload_prepare_seconds",
            "offload_layout_seconds",
            "offload_mapping_seconds",
        ):
            self.assertEqual(stats[f"{metric}[]"][0], 2)

    def test_caller_aggregates_are_thread_local(self):
        linker = self.bare_linker(enabled=True)

        class ImmediateQueue:
            def put(self, command):
                function, args, future = command
                future.set_result(function(*args))

        linker._commands = ImmediateQueue()
        timings = []

        def call():
            linker._call(lambda: None)
            timings.append(linker._caller_timing.stats)

        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(timings), 2)
        self.assertIsNot(timings[0], timings[1])
        self.assertEqual(
            [t.stats["call_wall_seconds[<lambda>]"][0] for t in timings], [1, 1]
        )

    def test_service_timer_stops_before_future_publication(self):
        linker = self.bare_linker(enabled=True)
        linker.pool_group = SimpleNamespace(
            entries=[SimpleNamespace(kv_buffer=[SimpleNamespace(is_cuda=False)])]
        )
        linker._make_runtime = lambda: SimpleNamespace(log_stats=lambda: None)
        linker._started = Future()
        linker._fatal = None
        linker._batches = deque()
        linker._offloads = deque()
        linker._progress = lambda: None
        linker._reset = lambda: None
        linker._close = lambda: None

        class PublishingFuture(Future):
            def set_result(inner, result):
                self.assertIn(
                    "call_service_seconds[command]", linker._owner_timing.stats
                )
                super().set_result(result)

        def command():
            linker._closed = True
            return 23

        future = PublishingFuture()
        future._kvcr_enqueued_at = time.perf_counter()
        linker._commands.put((command, (), future))
        linker._run()
        self.assertEqual(future.result(timeout=1), 23)

    def test_real_progress_submits_identical_native_work_with_profile_on_and_off(self):
        observations = []
        for enabled in (False, True):
            linker = self.bare_linker(enabled)
            linker._last_progress_at = None
            calls = []

            class Client:
                def poll_completed(self):
                    calls.append(("poll",))
                    return []

                def deliver(self, blocks, request_id=None):
                    calls.append(("deliver", blocks, request_id))
                    return 7

                def deposit(self, blocks):
                    calls.append(("deposit", blocks))
                    return 8

            blocks = {b"key": ["ref"]}
            ready = SimpleNamespace(query=lambda: True)
            batch = self.ns["_Batch"](
                0, (), ready, iter([(0, "scope", blocks), (0, None, None)]), [1]
            )
            offload = self.ns["_Offload"]((), ready, (), {b"key"}, work=iter([blocks]))
            linker._runtime = SimpleNamespace(client=Client(), events=deque())
            linker._operations = {}
            linker._hazards = set()
            linker._batches = deque([batch])
            linker._offloads = deque([offload])
            linker._progress()
            observations.append(
                (calls, batch.active, batch.remaining, batch.exhausted, offload.active)
            )
            if enabled:
                for name in ("deliver", "deposit"):
                    self.assertEqual(
                        linker._owner_timing.stats[f"submission_seconds[{name}]"][0], 1
                    )
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(
            observations[0][0],
            [("poll",), ("deliver", blocks, "scope"), ("deposit", blocks)],
        )

    def test_confirmation_counts_candidates_but_not_cleanup(self):
        linker = self.bare_linker(enabled=True)
        linker._request_scopes = {}
        linker._discard_lookup = Mock()
        self.assertFalse(linker._confirm_lookup("request", (), (), 0))
        self.assertTrue(linker._confirm_lookup("request", (), (), 1))
        self.assertEqual(
            linker._owner_timing.stats["confirmation_candidates[]"], (1, 1.0, 1)
        )

    def test_constructor_pops_and_validates_option_before_runtime(self):
        # Execute the real config-selection slice; the rest of construction
        # requires model pools and CUDA. Verify runtime sees the filtered dict.
        tree = ast.parse(SOURCE.read_text())
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "KVCRDirectLinker"
        )
        init = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        start = next(
            i for i, n in enumerate(init.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "options"
        )
        stop = next(
            i for i, n in enumerate(init.body)
            if isinstance(n, ast.Assign)
            and ast.unparse(n.targets[0]) == "self._publish_inventory"
        )
        code = compile(
            ast.fix_missing_locations(ast.Module(init.body[start:stop], [])),
            str(SOURCE), "exec",
        )
        for options, enabled in [
            ({}, False),
            ({"profile_owner": False}, False),
            ({"profile_owner": True, "dram_size_gb": 224}, True),
        ]:
            linker = self.Linker.__new__(self.Linker)
            namespace = dict(self.ns, self=linker, options=options, rank=2)
            exec(code, namespace)
            self.assertNotIn("profile_owner", namespace["options"])
            self.assertEqual(linker._owner_timing is not None, enabled)
        for value in [1, "true", None]:
            with self.assertRaisesRegex(ValueError, "profile_owner"):
                exec(
                    code,
                    dict(
                        self.ns,
                        self=self.Linker.__new__(self.Linker),
                        options={"profile_owner": value},
                        rank=2,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
