"""Ownership protocol regressions; fake transport is not RTL evidence."""
import asyncio
import gc
import importlib.util
from pathlib import Path
import sys
import unittest
import weakref
from unittest.mock import patch

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
sys.argv = sys.argv[:1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools/adoracc' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor():
    return dict(schema='adora.guarded-inplace-large.v1', support_version=3,
                module_identity='a' * 64, entry='overlay', width=353, height=1,
                items=353, source_elements=353, tile_items=256, value_mode='direct',
                range_lo=None, range_hi=None, value_nonzero=False, source_argument=0,
                x_argument=1, y_argument=2, enable_argument=3, destination_argument=4)


class Runtime:
    PRIVATE_STREAM_OWNERSHIP_VERSION = 1

    def __init__(self):
        self.live, self.created, self.released, self.calls = [], [], [], []
        self.fail_on = None
        self.fail_after = False
        self.gate = None

    def create_stream(self):
        class Stream:
            async def synchronize(self):
                pass
        stream = Stream()
        self.live.append(stream)
        self.created.append(stream)
        return stream

    async def retire_completed_stream(self, stream):
        if self.gate is not None:
            await self.gate.wait()
        if len(self.created) == self.fail_on and not self.fail_after:
            raise RuntimeError('retirement failed before effect')
        if stream in self.live:
            self.live.remove(stream)
            self.released.append(stream)
        if len(self.created) == self.fail_on and self.fail_after:
            raise RuntimeError('retirement failed after effect')


class Ownership(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.support = load('prepared_xy')
        self.compiler = load('c_entry')
        self.runtime = Runtime()
        self.src = np.arange(353, dtype=np.uint16)
        self.x = np.arange(353, dtype=np.int16)
        self.y = np.zeros(353, dtype=np.int16)
        self.enabled = np.ones(353, dtype=np.int16)
        self.base = np.full(370, 555, np.uint16)
        self.dst = self.base[:353]

    def bind(self, tile=None):
        async def tile_impl(runtime, packed, idx, enabled, old, *, _adora_stream):
            self.assertIn(_adora_stream, runtime.live)
            runtime.calls.append(_adora_stream)
            old[:] = packed
        return self.support.bind_prepared_entry(tile or tile_impl, descriptor(),
                                                private_stream_ownership=True)

    def make(self, tile=None):
        make, execute, once = self.bind(tile)
        return make(self.x, self.y, self.enabled), once

    async def test_exact_stream_retired_after_publication(self):
        plan, _ = self.make()
        retire = self.runtime.retire_completed_stream
        async def observed(stream):
            n = len(self.runtime.created)
            self.assertEqual(plan._cursor, n)
            self.assertEqual(plan.completed_ranges, ((0, 256), (256, 353))[:n])
            np.testing.assert_array_equal(self.dst[:min(256*n, 353)], self.src[:min(256*n, 353)])
            self.assertIs(plan.pending_stream, stream)
            await retire(stream)
        self.runtime.retire_completed_stream = observed
        await plan.execute(self.runtime, self.src, self.dst)
        self.assertEqual(self.runtime.calls, self.runtime.created)
        self.assertEqual(self.runtime.released, self.runtime.created)
        self.assertFalse(self.runtime.live)
        self.assertFalse(plan.cleanup_pending)
        self.assertIsNone(plan.pending_stream)
        np.testing.assert_array_equal(self.base[353:], 555)
        plan.close()

    async def test_cleanup_first_second_and_after_effect_is_retryable_without_republish(self):
        for fail_on, after in ((1, False), (2, False), (2, True)):
            with self.subTest(block=fail_on, after=after):
                self.runtime = Runtime()
                self.runtime.fail_on, self.runtime.fail_after = fail_on, after
                self.dst.fill(555)
                plan, _ = self.make()
                with self.assertRaises(self.support.PlanCleanupError) as caught:
                    await plan.execute(self.runtime, self.src, self.dst)
                error = caught.exception
                self.assertIs(error.plan, plan)
                self.assertIs(error.cause, error.__cause__)
                self.assertEqual(error.phase, 'retire_completed_stream')
                self.assertEqual(error.block, ((0, 256), (256, 353))[fail_on - 1])
                self.assertEqual(error.cursor, fail_on)
                self.assertFalse(plan.poisoned)
                self.assertTrue(plan.cleanup_pending)
                self.assertIn(plan, self.support._PENDING_CLEANUP_PLANS)
                self.assertEqual(len(self.runtime.calls), fail_on)
                end = 256 if fail_on == 1 else 353
                np.testing.assert_array_equal(self.dst[:end], self.src[:end])
                np.testing.assert_array_equal(self.dst[end:], 555)
                saved = self.dst.copy()
                with self.assertRaises(ValueError):
                    plan.close()
                with self.assertRaises(ValueError):
                    await plan.execute(self.runtime, self.src, self.dst)
                self.runtime.fail_on = None
                await error.retry_cleanup()
                await plan.retry_cleanup()
                self.assertEqual(len(self.runtime.released), fail_on)
                self.assertEqual(len(self.runtime.calls), fail_on)
                np.testing.assert_array_equal(self.dst, saved)
                self.assertFalse(plan.cleanup_pending)
                self.assertNotIn(plan, self.support._PENDING_CLEANUP_PLANS)
                plan.close()

    async def test_once_preserves_cleanup_error_and_plan_handle(self):
        _, once = self.make()
        self.runtime.fail_on = 2
        with self.assertRaises(self.support.PlanCleanupError) as caught:
            await once(self.runtime, self.src, self.x, self.y, self.enabled, self.dst)
        error = caught.exception
        self.assertFalse(error.plan.closed)
        self.assertEqual(error.completed_ranges, ((0, 256), (256, 353)))
        self.runtime.fail_on = None
        await error.retry_cleanup()
        error.plan.close()

    async def test_saved_error_cannot_cleanup_a_later_frame(self):
        plan, _ = self.make()
        retire = self.runtime.retire_completed_stream
        attempts = []
        async def observed(stream):
            attempts.append(stream)
            await retire(stream)
        self.runtime.retire_completed_stream = observed
        self.runtime.fail_on = 1
        with self.assertRaises(self.support.PlanCleanupError) as caught_a:
            await plan.execute(self.runtime, self.src, self.dst)
        error_a = caught_a.exception
        self.runtime.fail_on = None
        await error_a.retry_cleanup()
        self.assertFalse(plan.cleanup_pending)
        self.dst.fill(444)
        self.runtime.fail_on = 2
        with self.assertRaises(self.support.PlanCleanupError) as caught_b:
            await plan.execute(self.runtime, self.src, self.dst)
        error_b = caught_b.exception
        stream_b = plan.pending_stream
        saved = self.dst.copy()
        calls = len(self.runtime.calls)
        self.assertEqual(len(attempts), 3)
        await error_a.retry_cleanup()
        await error_a.retry_cleanup()
        self.assertEqual(len(attempts), 3)
        self.assertTrue(plan.cleanup_pending)
        self.assertIs(plan.pending_stream, stream_b)
        self.assertIn(stream_b, self.runtime.live)
        self.assertEqual(plan.completed_ranges, ((0, 256),))
        self.assertEqual(len(self.runtime.calls), calls)
        np.testing.assert_array_equal(self.dst, saved)
        self.runtime.fail_on = None
        await error_b.retry_cleanup()
        await error_b.retry_cleanup()
        await error_a.retry_cleanup()
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(self.runtime.released), 2)
        self.assertEqual(len(self.runtime.calls), calls)
        np.testing.assert_array_equal(self.dst, saved)
        plan.close()

    async def test_pending_cleanup_process_root_and_concurrent_retry(self):
        plan, _ = self.make()
        self.runtime.fail_on = 1
        with self.assertRaises(self.support.PlanCleanupError):
            await plan.execute(self.runtime, self.src, self.dst)
        key = id(plan)
        ref = weakref.ref(plan._scratch[0])
        del plan
        gc.collect()
        self.assertIsNotNone(ref())
        plan = next(p for p in self.support._PENDING_CLEANUP_PLANS if id(p) == key)
        self.runtime.fail_on = None
        self.runtime.gate = asyncio.Event()
        task = asyncio.create_task(plan.retry_cleanup())
        await asyncio.sleep(0)
        with self.assertRaisesRegex(ValueError, 'cleanup'):
            await plan.retry_cleanup()
        self.runtime.gate.set()
        await task
        self.assertEqual(len(self.runtime.released), 1)
        plan.close()

    async def test_unknown_completion_keeps_exact_stream_and_prefix(self):
        entered = asyncio.Event()
        async def tile(runtime, packed, idx, en, old, *, _adora_stream):
            runtime.calls.append(_adora_stream)
            if len(runtime.calls) == 2:
                entered.set()
                await asyncio.Event().wait()
            old[:] = packed
        plan, _ = self.make(tile)
        task = asyncio.create_task(plan.execute(self.runtime, self.src, self.dst))
        await entered.wait()
        stream = plan.pending_stream
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(plan.poisoned)
        self.assertIs(plan.pending_stream, stream)
        self.assertIs(plan._retained[3], stream)
        self.assertEqual(plan.completed_ranges, ((0, 256),))
        self.assertEqual(len(self.runtime.released), 1)
        np.testing.assert_array_equal(self.dst[256:], 555)
        with self.assertRaises(ValueError):
            await plan.retry_cleanup()
        with self.assertRaises(ValueError):
            plan.close()

    async def test_old_runtime_rejected_before_create_even_zero_work(self):
        for active in (True, False):
            self.enabled[:] = int(active)
            plan, _ = self.make()
            self.runtime.PRIVATE_STREAM_OWNERSHIP_VERSION = 0
            with self.assertRaisesRegex(ValueError, 'ownership'):
                await plan.execute(self.runtime, self.src, self.dst)
            self.assertFalse(self.runtime.created)
            self.assertFalse(plan.poisoned)
            plan.close()

    async def test_once_close_cannot_mask_original_preparation_error(self):
        _, once = self.make()
        with patch.object(self.support.np, 'take', side_effect=MemoryError('prepare failed')):
            with patch.object(self.support._PreparedPlan, 'close', side_effect=RuntimeError('close failed')):
                with self.assertRaises(self.support.PlanExecutionError) as caught:
                    await once(self.runtime, self.src, self.x, self.y, self.enabled, self.dst)
        self.assertIsInstance(caught.exception.__cause__, MemoryError)
        self.assertEqual(str(caught.exception.close_error), 'close failed')
        self.assertFalse(self.runtime.created)
        self.assertFalse(caught.exception.plan.poisoned)
        caught.exception.plan.close()

    async def test_publication_failure_stays_execution_error_with_safe_cleanup(self):
        class RejectWrite(np.ndarray):
            def __setitem__(self, key, value):
                raise RuntimeError('public destination rejected write')
        plan, _ = self.make()
        with self.assertRaises(self.support.PlanExecutionError) as caught:
            await plan.execute(self.runtime, self.src, self.dst.view(RejectWrite))
        self.assertNotIsInstance(caught.exception, self.support.PlanCleanupError)
        self.assertEqual(caught.exception.completed_ranges, ())
        self.assertEqual(caught.exception.failed_range, (0, 256))
        self.assertFalse(plan.poisoned)
        self.assertTrue(plan.cleanup_pending)
        np.testing.assert_array_equal(self.dst, 555)
        await plan.retry_cleanup()
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertEqual(len(self.runtime.released), 1)
        self.assertEqual(plan.completed_ranges, ())
        plan.close()

    async def test_cancellation_during_cleanup_keeps_published_prefix_safe(self):
        plan, _ = self.make()
        self.runtime.gate = asyncio.Event()
        task = asyncio.create_task(plan.execute(self.runtime, self.src, self.dst))
        await asyncio.sleep(0)
        self.assertTrue(plan.cleanup_pending)
        self.assertEqual(plan.completed_ranges, ((0, 256),))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await task
        self.assertEqual(caught.exception.phase, 'retire_completed_stream')
        self.assertIs(caught.exception.plan, plan)
        self.assertFalse(plan.poisoned)
        self.assertTrue(plan.cleanup_pending)
        self.runtime.gate.set()
        await plan.retry_cleanup()
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertEqual(len(self.runtime.released), 1)
        np.testing.assert_array_equal(self.dst[:256], self.src[:256])
        np.testing.assert_array_equal(self.dst[256:], 555)
        plan.close()

    async def test_unknown_exception_and_explicit_cancellation_never_retire(self):
        for failure in (RuntimeError('no completion'), asyncio.CancelledError('explicit cancel')):
            self.runtime = Runtime()
            async def tile(runtime, *buffers, _adora_stream):
                runtime.calls.append(_adora_stream)
                raise failure
            plan, _ = self.make(tile)
            expected = self.support.PlanExecutionError if isinstance(failure, Exception) else type(failure)
            with self.assertRaises(expected):
                await plan.execute(self.runtime, self.src, self.dst)
            self.assertTrue(plan.poisoned)
            self.assertEqual(plan.completed_ranges, ())
            self.assertEqual(self.runtime.released, [])
            self.assertIs(plan.pending_stream, self.runtime.created[0])

    async def test_generated_binding_passes_exact_stream_to_private_entry(self):
        raw = ('import numpy as np\n'
               'def apply_writeback_tasks(tasks):\n'
               '    pass\n'
               'async def _adora_large_tile(runtime,arg_0,arg_1,arg_2,arg_3):\n'
               '    stream = runtime.create_stream()\n'
               '    writeback_tasks = []\n'
               '    runtime.calls.append(stream)\n'
               '    arg_3[:] = arg_0\n'
               '    await stream.synchronize()\n'
               '    try:\n'
               '        apply_writeback_tasks(writeback_tasks)\n'
               '    except NameError:\n'
               '        pass\n')
        source, private = self.compiler.large_checked_module(raw, descriptor())
        namespace = {'__name__': 'ownership_generated', '__file__': '<generated>'}
        exec(source, namespace)
        plan = namespace['make_plan'](self.x, self.y, self.enabled)
        await namespace['execute_plan'](self.runtime, plan, self.src, self.dst)
        self.assertEqual(len(self.runtime.created), 2)
        self.assertEqual(self.runtime.calls, self.runtime.created)
        self.assertEqual(self.runtime.released, self.runtime.created)
        self.assertNotIn('runtime.create_stream()', private)
        plan.close()

    async def test_generated_writeback_name_error_never_publishes_or_retires(self):
        raw = ('import numpy as np\n'
               'def apply_writeback_tasks(tasks):\n'
               '    raise NameError("actual final writeback failed")\n'
               'async def _adora_large_tile(runtime,arg_0,arg_1,arg_2,arg_3):\n'
               '    stream = runtime.create_stream()\n'
               '    writeback_tasks = []\n'
               '    runtime.calls.append(stream)\n'
               '    arg_3[:] = arg_0\n'
               '    await stream.synchronize()\n'
               '    try:\n'
               '        apply_writeback_tasks(writeback_tasks)\n'
               '    except NameError:\n'
               '        pass\n')
        source, _ = self.compiler.large_checked_module(raw, descriptor())
        namespace = {'__name__': 'writeback_generated', '__file__': '<generated>'}
        exec(source, namespace)
        plan = namespace['make_plan'](self.x, self.y, self.enabled)
        with self.assertRaises(RuntimeError) as caught:
            await namespace['execute_plan'](self.runtime, plan, self.src, self.dst)
        self.assertEqual(type(caught.exception).__name__, 'PlanExecutionError')
        self.assertIsInstance(caught.exception.__cause__, NameError)
        self.assertEqual(caught.exception.completed_ranges, ())
        self.assertTrue(plan.poisoned)
        self.assertFalse(plan.cleanup_pending)
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertEqual(self.runtime.released, [])
        self.assertIs(plan.pending_stream, self.runtime.created[0])
        self.assertIs(plan._retained[3], self.runtime.created[0])
        np.testing.assert_array_equal(self.dst, 555)

    async def test_owned_tail_is_required_and_other_profiles_preserve_guard(self):
        raw = ('import numpy as np\n'
               'async def _adora_large_tile(runtime,arg_0,arg_1,arg_2,arg_3):\n'
               '    stream = runtime.create_stream()\n'
               '    writeback_tasks = []\n'
               '    await stream.synchronize()\n'
               '    try:\n'
               '        apply_writeback_tasks(writeback_tasks)\n'
               '    except NameError:\n'
               '        pass\n')
        for malformed in (raw.replace('    await stream.synchronize()\n', ''), raw + '    pass\n'):
            with self.assertRaisesRegex(ValueError, 'synchronize/writeback'):
                self.compiler.large_checked_module(malformed, descriptor())
        # Non-owned profiles preserve the mapper's missing-helper compatibility.
        source, private = self.compiler.checked_module(raw, self.compiler.large_local_contract(),
                                                       allow_partial=True)
        self.assertIn('except NameError:', private)
        namespace = {'__name__': 'legacy_guard', '__file__': '<generated>'}
        exec(source, namespace)
        packed, idx, enabled, old = (np.zeros(256, dtype) for dtype in
                                      (np.uint16, np.int16, np.int16, np.uint16))
        await namespace['_adora_large_tile'](self.runtime, packed, idx, enabled, old)
        self.assertEqual(len(self.runtime.created), 1)
        self.assertEqual(self.runtime.released, [])


if __name__ == '__main__':
    unittest.main()
