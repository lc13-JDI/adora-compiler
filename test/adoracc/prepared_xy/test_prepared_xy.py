"""Host protocol tests: the tile stand-in is not a hardware correctness claim."""
import asyncio
import gc
import weakref
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
sys.argv = sys.argv[:1]
MODULE = ROOT / 'tools/adoracc/prepared_xy.py'


def descriptor(items=4, nonzero=False):
    return dict(schema='adora.guarded-inplace-large.v1', support_version=2,
                module_identity='a' * 64, entry='overlay', width=8193, height=1,
                items=items, source_elements=8193, tile_items=256,
                value_nonzero=nonzero, source_argument=0, x_argument=1,
                y_argument=2, enable_argument=3, destination_argument=4)


class HostProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_range_descriptor_is_explicit_and_fail_closed(self):
        d = dict(descriptor(), support_version=3, value_mode='range', range_lo=32768, range_hi=60000)
        self.module._validate_descriptor(d)
        for changes in [dict(value_mode=None), dict(value_mode='unknown'), dict(value_nonzero=True),
                        dict(range_lo=0), dict(range_lo=True), dict(range_hi=65536),
                        dict(range_lo=60001), dict(range_hi=None), dict(value_mode='direct')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.module._validate_descriptor(dict(d, **changes))
        for key in ('value_mode', 'range_lo', 'range_hi'):
            missing = dict(d)
            del missing[key]
            with self.assertRaises(ValueError):
                self.module._validate_descriptor(missing)

    async def test_range_mode_never_filters_or_skips_pixels_on_host(self):
        calls = []
        async def tile(runtime, packed, idx, enabled, dst):
            calls.append((packed.copy(), enabled.copy(), dst.copy()))
            valid = (idx >= 0) & (enabled != 0)
            values = packed[idx[valid]]
            dst[valid] = np.where((values >= 32768) & (values <= 60000), values, 0)
        d = dict(descriptor(), support_version=3, value_mode='range', range_lo=32768, range_hi=60000)
        make, _, _ = self.bind(tile=tile, desc=d)
        x, y, en = self.arrays()
        plan = make(x, y, en)
        src = np.full(8193, 60001, np.uint16)
        old = np.full(4, 43, np.uint16)
        await plan.execute(object(), src, old)
        np.testing.assert_array_equal(old, [0, 0, 43, 43])
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0][:2], [60001, 60001])
        np.testing.assert_array_equal(calls[0][1][:4], [1, 1, 0, 0])
        src[8192], src[0] = 32768, 60000
        old[:] = 51
        await plan.execute(object(), src, old)
        np.testing.assert_array_equal(old, [32768, 60000, 51, 51])
        self.assertEqual(len(calls), 2)
        plan.close()

    def setUp(self):
        self.assertTrue(MODULE.exists(), 'production prepared P host support is missing')
        spec = importlib.util.spec_from_file_location('prepared_xy', MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.calls = []

    def bind(self, items=4, nonzero=False, tile=None, desc=None):
        async def semantic_tile(runtime, src, idx, en, dst):
            self.calls.append(tuple(b.copy() for b in (src, idx, en, dst)))
            valid = (idx >= 0) & (en != 0)
            if nonzero:
                valid &= src[np.maximum(idx, 0)] != 0
            dst[valid] = src[idx[valid]]
        return self.module.bind_prepared_entry(tile or semantic_tile,
                                               desc or descriptor(items, nonzero))

    def arrays(self, items=4):
        x = np.zeros(items, np.int16)
        x[:min(items, 4)] = [8192, 0, 8192, -1][:min(items, 4)]
        y = np.zeros(items, np.int16)
        en = np.ones(items, np.int16)
        en[:min(items, 4)] = [2, -1, 0, 1][:min(items, 4)]
        return x, y, en

    async def test_owned_snapshot_fresh_pixels_and_tail(self):
        make, _, _ = self.bind()
        x, y, en = self.arrays()
        plan = make(x, y, en)
        x[:] = y[:] = en[:] = 0
        src = np.zeros(8193, np.uint16)
        src[8192], src[0] = 30, 10
        dst = np.array([7, 8, 9, 10], np.uint16)
        report = await plan.execute(None, src, dst)
        np.testing.assert_array_equal(dst, [30, 10, 9, 10])
        packed, idx, mask, old = self.calls[0]
        np.testing.assert_array_equal(idx[:4], [0, 1, -1, -1])
        self.assertTrue(np.all(idx[4:] == -1))
        self.assertTrue(np.all(mask[4:] == 0))
        self.assertTrue(np.all(packed[4:] == 0))
        self.assertTrue(np.all(old[4:] == 0))
        self.assertEqual(report['completed_ranges'], ((0, 4),))
        src[8192], src[0] = 65535, 0
        dst[:] = [70, 80, 90, 100]
        await plan.execute(None, src, dst)
        np.testing.assert_array_equal(dst, [65535, 0, 90, 100])
        plan.close()
        self.assertTrue(plan.closed)
        plan.close()
        with self.assertRaises(ValueError):
            await plan.execute(None, src, dst)

    async def test_readonly_alias_cannot_mutate_owned_plan(self):
        make, _, _ = self.bind()
        x, y, en = self.arrays()
        alias = x.view()
        x.flags.writeable = False
        plan = make(x, y, en)
        alias[:] = 0
        src = np.arange(8193, dtype=np.uint16)
        dst = np.full(4, 42, np.uint16)
        await plan.execute(None, src, dst)
        self.assertEqual(int(dst[0]), 8192)
        changed = make(alias, y, en)
        await changed.execute(None, src, dst)
        self.assertEqual(int(dst[0]), 0)
        plan.close()
        changed.close()

    async def test_skip_uses_read_guard_only_and_z1_stays_device(self):
        make, _, _ = self.bind(nonzero=True)
        x, y, en = self.arrays()
        plan = make(x, y, en)
        src = np.zeros(8193, np.uint16)
        dst = np.full(4, 55, np.uint16)
        await plan.execute(None, src, dst)
        self.assertEqual(len(self.calls), 1)
        np.testing.assert_array_equal(dst, [55] * 4)
        src[8192] = 65535
        await plan.execute(None, src, dst)
        self.assertEqual(int(dst[0]), 65535)
        en[:] = 0
        empty = make(x, y, en)
        result = await empty.execute(None, src, dst)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result['active_tiles'], 0)
        self.assertEqual(result['skipped_tiles'], 1)
        plan.close()
        empty.close()

    async def test_prechecks_have_no_effect_and_reject_foreign_plan(self):
        make, execute, once = self.bind()
        x, y, en = self.arrays()
        plan = make(x, y, en)
        src = np.zeros(8193, np.uint16)
        dst = np.full(4, 77, np.uint16)
        ro = dst.copy()
        ro.flags.writeable = False
        bads = [(src[:-1], dst), (src.astype(np.int16), dst),
                (src, ro), (src, src[:4]), (src.byteswap().view('>u2'), dst),
                (src, np.zeros(8, np.uint16)[::2]), (list(src), dst)]
        for a, b in bads:
            with self.assertRaises(ValueError):
                await plan.execute(None, a, b)
        foreign = self.bind()[0](*self.arrays())
        with self.assertRaises(ValueError):
            await execute(None, foreign, src, dst)
        with self.assertRaises(ValueError):
            await execute(None, object(), src, dst)
        with self.assertRaises(ValueError):
            await once(None, src, x, y, en, x.view(np.uint16))
        self.assertEqual(self.calls, [])
        np.testing.assert_array_equal(dst, [77] * 4)
        plan.close()
        foreign.close()

    async def test_create_prechecks_and_descriptor_limits(self):
        make, _, _ = self.bind()
        x, y, en = self.arrays()
        for args in [(x, x, en), (x.astype(np.uint16), y, en),
                     (x[:-1], y, en), (x.reshape(2, 2), y, en),
                     (x.byteswap().view('>i2'), y, en)]:
            with self.assertRaises(ValueError):
                make(*args)
        for changes in [dict(support_version=1), dict(support_version=3), dict(schema='external'),
                        dict(module_identity='unverified'), dict(width=True),
                        dict(items=2**63), dict(width=2**63),
                        dict(source_elements=0), dict(items=0), dict(width=0),
                        dict(height=0), dict(tile_items=128),
                        dict(x_argument=0), dict(value_nonzero=1)]:
            with self.assertRaises(ValueError):
                self.bind(desc=dict(descriptor(), **changes))

    async def test_once_wrapper_original_parameter_order(self):
        desc = descriptor()
        desc.update(source_argument=4, x_argument=3, y_argument=2,
                    enable_argument=1, destination_argument=0)
        _, _, once = self.bind(desc=desc)
        src = np.arange(8193, dtype=np.uint16)
        x, y, en = self.arrays()
        dst = np.full(4, 13, np.uint16)
        await once(None, dst, en, y, x, src)
        np.testing.assert_array_equal(dst, [8192, 0, 13, 13])

    async def test_bounded_scratch_tail_counts_and_no_host_merge(self):
        async def sentinel(runtime, src, idx, en, dst):
            self.calls.append(tuple(id(b) for b in (src, idx, en, dst)))
            dst[:] = 60000  # Every returned slot must be copied, including invalid.
        make, _, _ = self.bind(items=353, tile=sentinel)
        x, y, en = self.arrays(353)
        plan = make(x, y, en)
        src = np.zeros(8193, np.uint16)
        base = np.full(355, 17, np.uint16)
        report = await plan.execute(None, src, base[1:-1])
        self.assertEqual(report['completed_ranges'], ((0, 256), (256, 353)))
        np.testing.assert_array_equal(base[1:-1], np.full(353, 60000, np.uint16))
        np.testing.assert_array_equal(base[[0, -1]], [17, 17])
        self.assertEqual(len(self.calls), 2)
        self.assertLessEqual(plan.stats['scratch_bytes'], 8192)
        self.assertLessEqual(plan.stats['plan_bytes'], 32 * 353 + 4096)
        plan.close()

    async def test_pending_close_concurrent_reject_and_failure_ranges(self):
        entered, resume = asyncio.Event(), asyncio.Event()
        async def tile(runtime, src, idx, en, dst):
            self.calls.append(dst)
            if len(self.calls) == 2:
                entered.set()
                await resume.wait()
                raise RuntimeError('device completion unknown')
            dst[:] = 91
        make, _, _ = self.bind(items=353, tile=tile)
        plan = make(*self.arrays(353))
        src, dst = np.zeros(8193, np.uint16), np.zeros(353, np.uint16)
        task = asyncio.create_task(plan.execute(None, src, dst))
        await entered.wait()
        with self.assertRaises(ValueError):
            plan.close()
        with self.assertRaises(ValueError):
            await plan.execute(None, src, dst)
        resume.set()
        with self.assertRaises(self.module.PlanExecutionError) as caught:
            await task
        self.assertEqual(caught.exception.completed_ranges, ((0, 256),))
        self.assertEqual(caught.exception.failed_range, (256, 353))
        self.assertTrue(plan.poisoned)
        np.testing.assert_array_equal(dst[:256], np.full(256, 91))
        np.testing.assert_array_equal(dst[256:], np.zeros(97))
        with self.assertRaises(ValueError):
            plan.close()
        with self.assertRaises(ValueError):
            await plan.execute(None, src, dst)

    async def test_generated_module_is_standalone_and_isolates_public_names(self):
        spec = importlib.util.spec_from_file_location('c_entry', ROOT / 'tools/adoracc/c_entry.py')
        compiler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiler)
        raw = ('import numpy as np\n'
               'async def _adora_large_tile(runtime, arg_0, arg_1, arg_2, arg_3):\n'
               '    runtime.append(tuple(b.copy() for b in (arg_0,arg_1,arg_2,arg_3)))\n'
               '    valid = (arg_1 >= 0) & (arg_2 != 0)\n'
               '    arg_3[valid] = arg_0[arg_1[valid]]\n')
        for name in ('overlay', 'globals', 'compile', 'exec', 'len', 'np'):
            source, _ = compiler.large_checked_module(raw, dict(descriptor(), entry=name))
            namespace = {'__name__': 'generated_' + name, '__file__': '<standalone generated>'}
            exec(source, namespace)
            x, y, en = self.arrays()
            plan = namespace['make_plan'](x, y, en)
            src = np.arange(8193, dtype=np.uint16)
            src[8192] = 65535
            dst = np.full(4, 77, np.uint16)
            runtime = []
            await plan.execute(runtime, src, dst)
            np.testing.assert_array_equal(dst, [65535, 0, 77, 77])
            self.assertTrue(all(b.dtype == np.int16 for b in runtime[0]))
            await namespace[name](runtime, src, x, y, en, dst)
            # A separate import is foreign even with identical artifact identity.
            other = {'__name__': 'other_module', '__file__': '<standalone generated>'}
            exec(source, other)
            with self.assertRaises(ValueError):
                await other['execute_plan'](runtime, plan, src, dst)
            plan.close()

    async def test_public_metadata_and_descriptor_inputs_are_detached(self):
        desc = descriptor()
        make, _, _ = self.bind(desc=desc)
        plan = make(*self.arrays())
        desc['items'] = 1
        exposed = plan.descriptor
        exposed['source_elements'] = 1
        exposed['value_nonzero'] = True
        stats = plan.stats
        stats['active_tiles'] = 0
        src = np.zeros(8193, np.uint16)
        dst = np.full(4, 22, np.uint16)
        report = await plan.execute(None, src, dst)
        report['timings_ns']['total'] = -1
        self.assertEqual(plan.descriptor['items'], 4)
        self.assertEqual(plan.descriptor['source_elements'], 8193)
        self.assertEqual(plan.stats['active_tiles'], 1)
        self.assertGreaterEqual(plan.last_report['timings_ns']['total'], 0)
        np.testing.assert_array_equal(dst, [0, 0, 22, 22])
        plan.close()

    async def test_poison_survives_dropping_embedded_module_namespace(self):
        # Generated support executes in a private dict, not a sys.modules entry.
        # A registry only inside that dict would be a collectable reference cycle.
        namespace = {'__name__': 'temporary_embedded_support'}
        exec(MODULE.read_text(), namespace)
        async def broken(runtime, *buffers):
            raise RuntimeError('unknown completion')
        make, execute, once = namespace['bind_prepared_entry'](broken, descriptor())
        plan = make(*self.arrays())
        retained_array = weakref.ref(plan._scratch[0])
        try:
            await plan.execute(object(), np.zeros(8193, np.uint16), np.zeros(4, np.uint16))
        except RuntimeError:
            pass
        del plan, make, execute, once, namespace
        gc.collect()
        self.assertIsNotNone(retained_array(), 'uncertain DMA storage needs a process-rooted registry')

    async def test_scheduler_unschedule_and_close_retains_inflight_plan(self):
        class Suspended:
            def __await__(self):
                yield 'device pending'
        async def tile(runtime, src, idx, en, dst):
            await Suspended()
        make, _, _ = self.bind(tile=tile)
        plan = make(*self.arrays())
        coroutine = plan.execute(object(), np.zeros(8193, np.uint16), np.zeros(4, np.uint16))
        try:
            self.assertEqual(coroutine.send(None), 'device pending')
            # cocotb 1.9 Task.kill unschedules without throwing into a started
            # coroutine. Storage must be rooted even before exception handling.
            self.assertTrue(hasattr(plan, 'pending'), 'pending lifecycle state is required')
            self.assertTrue(plan.pending)
            self.assertIn(plan, self.module._INFLIGHT_PLANS)
            self.assertEqual(plan.completed_ranges, ())
            with self.assertRaises(ValueError):
                plan.close()
        finally:
            coroutine.close()  # Scheduler/GC close injects GeneratorExit.
        self.assertTrue(plan.poisoned)
        self.assertFalse(plan.pending)
        self.assertIn(plan, self.module._POISONED_PLANS)

    async def test_cancellation_retains_inflight_storage_and_reports_ranges(self):
        entered = asyncio.Event()
        async def tile(runtime, src, idx, en, dst):
            entered.set()
            await asyncio.Event().wait()
        make, _, _ = self.bind(tile=tile)
        plan = make(*self.arrays())
        task = asyncio.create_task(plan.execute(object(), np.zeros(8193, np.uint16),
                                                np.zeros(4, np.uint16)))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await task
        self.assertEqual(caught.exception.completed_ranges, ())
        self.assertEqual(caught.exception.failed_range, (0, 4))
        self.assertTrue(plan.poisoned)
        self.assertTrue(any(value is plan for value in self.module._POISONED_PLANS))
        with self.assertRaises(ValueError):
            plan.close()

    async def test_window_modes_typed_buffers_and_bounded_memory(self):
        items = 256 * 67 + 1
        x, y, en = self.arrays(items)
        x[::7] = -1
        en[256:512] = 0
        src = np.arange(8193, dtype=np.uint16)
        expected = np.full(items, 73, np.uint16)
        valid = (en != 0) & (x >= 0)
        expected[valid] = src[x[valid]]
        reference_buffers = None
        for preparation, bookkeeping, probes in (
                ('tile', 'eager', True), ('window', 'eager', True),
                ('window', 'cursor', True), ('window', 'cursor', False)):
            self.calls.clear()
            make, _, _ = self.bind(items=items)
            plan = make(x, y, en, preparation=preparation, window_tiles=3,
                        bookkeeping=bookkeeping, probes=probes)
            dst = np.full(items, 73, np.uint16)
            report = await plan.execute(None, src, dst)
            np.testing.assert_array_equal(dst, expected)
            for call in self.calls:
                for value, dtype in zip(call, (np.uint16, np.int16, np.int16, np.uint16)):
                    self.assertEqual(value.shape, (256,))
                    self.assertEqual(value.dtype, dtype)
                    self.assertTrue(value.flags.c_contiguous)
            if reference_buffers is None:
                reference_buffers = list(self.calls)
            else:
                for actual, reference in zip(self.calls, reference_buffers):
                    for a, b in zip(actual, reference):
                        np.testing.assert_array_equal(a, b)
                self.assertEqual(len(self.calls), len(reference_buffers))
            self.assertEqual(plan.stats['preparation'], preparation)
            self.assertLessEqual(plan.stats['scratch_bytes'], 8192)
            self.assertLessEqual(plan.stats['execution_scratch_bound_bytes'], 3 * 256 * 16)
            self.assertGreaterEqual(report['timings_ns']['total'], 0)
            for phase in ('precheck', 'prepare', 'pack', 'device', 'copyback'):
                if probes:
                    self.assertGreaterEqual(report['timings_ns'][phase], 0)
                else:
                    self.assertIsNone(report['timings_ns'][phase])
            plan.close()

    async def test_769_progress_snapshots_failure_and_cancellation(self):
        for cancel, bookkeeping in ((False, 'cursor'), (True, 'cursor'),
                                    (False, 'eager'), (True, 'eager')):
            entered, resume = asyncio.Event(), asyncio.Event()
            calls = []
            fail = False
            async def tile(runtime, packed, idx, en, old):
                calls.append(old)
                if fail and len(calls) == 2:
                    entered.set()
                    await resume.wait()
                    raise RuntimeError('second active tile failed')
                old[:] = 91
            make, _, _ = self.bind(items=769, tile=tile)
            x, y, en = self.arrays(769)
            en[256:512] = 0
            plan = make(x, y, en, window_tiles=2, bookkeeping=bookkeeping)
            src, dst = np.zeros(8193, np.uint16), np.full(769, 13, np.uint16)
            previous = await plan.execute(object(), src, dst)
            saved = plan.completed_ranges
            self.assertEqual(saved, ((0, 256), (512, 768), (768, 769)))
            dst.fill(13)
            calls.clear()
            fail = True
            runtime = object()
            task = asyncio.create_task(plan.execute(runtime, src, dst))
            await entered.wait()
            progress = plan.completed_ranges
            self.assertEqual(progress, ((0, 256),))
            self.assertEqual(plan.last_report, previous)
            self.assertEqual(saved, ((0, 256), (512, 768), (768, 769)))
            with self.assertRaises(ValueError):
                await plan.execute(runtime, src, dst)
            with self.assertRaises(ValueError):
                plan.close()
            if cancel:
                task.cancel()
                exception = asyncio.CancelledError
            else:
                resume.set()
                exception = self.module.PlanExecutionError
            with self.assertRaises(exception) as caught:
                await task
            self.assertEqual(len(calls), 2)
            self.assertEqual(caught.exception.failed_range, (512, 768))
            self.assertEqual(caught.exception.completed_ranges, progress)
            self.assertEqual(plan.completed_ranges, progress)
            self.assertEqual(plan.last_report, previous)
            self.assertTrue(plan.poisoned)
            self.assertIn(plan, self.module._POISONED_PLANS)
            self.assertIs(plan._retained[0], runtime)
            self.assertIs(plan._retained[1], src)
            self.assertIs(plan._retained[2], dst)
            np.testing.assert_array_equal(dst[:256], np.full(256, 91))
            np.testing.assert_array_equal(dst[256:], np.full(513, 13))

    async def test_window_options_and_templates_are_immutable(self):
        make, _, _ = self.bind(items=769)
        arrays = self.arrays(769)
        for options in (dict(window_tiles=0), dict(window_tiles=257),
                        dict(window_tiles=True), dict(window_tiles=1.5),
                        dict(preparation='native'), dict(bookkeeping='unknown'),
                        dict(probes=1)):
            with self.assertRaises(ValueError):
                make(*arrays, **options)
        plan = make(*arrays, window_tiles=2)
        self.assertEqual(plan.stats['window_tiles'], 2)
        for window in plan._windows:
            for array in window[1:]:
                self.assertFalse(array.flags.writeable)
                with self.assertRaises(ValueError):
                    array.flags.writeable = True
        plan.close()

    async def test_boundary_lengths_last_only_and_rebuilt_enable(self):
        src = np.arange(8193, dtype=np.uint16)
        src[0], src[1], src[8192] = 0, 32768, 65535
        for items in (1, 255, 256, 257, 353):
            for nonzero in (False, True):
                make, _, _ = self.bind(items=items, nonzero=nonzero)
                x = np.resize(np.array([0, 1, 8192, -1], np.int16), items)
                y, en = np.zeros(items, np.int16), np.zeros(items, np.int16)
                empty = make(x, y, en, window_tiles=2)
                en[-1], x[-1] = -2, 8192
                last_only = make(x, y, en, window_tiles=2)
                en.fill(2)
                full = make(x, y, en, window_tiles=2)
                for plan, mode in ((empty, 'empty'), (last_only, 'last'), (full, 'full')):
                    dst = np.full(items, 41, np.uint16)
                    expected = dst.copy()
                    if mode == 'last':
                        expected[-1] = 65535
                    elif mode == 'full':
                        valid = x >= 0
                        if nonzero:
                            valid &= src[np.maximum(x, 0)] != 0
                        expected[valid] = src[x[valid]]
                    await plan.execute(None, src, dst)
                    np.testing.assert_array_equal(dst, expected)
                    plan.close()

    async def test_success_progress_snapshots_and_precheck_preserve_history(self):
        gates = [asyncio.Event() for _ in range(3)]
        entered = [asyncio.Event() for _ in range(3)]
        calls = []
        async def tile(runtime, packed, idx, en, old):
            i = len(calls)
            calls.append(i)
            entered[i].set()
            await gates[i].wait()
            old[:] = 60 + i
        make, _, _ = self.bind(items=769, tile=tile)
        x, y, en = self.arrays(769)
        en[256:512] = 0
        plan = make(x, y, en, window_tiles=2)
        src, dst = np.zeros(8193, np.uint16), np.full(769, 13, np.uint16)
        task = asyncio.create_task(plan.execute(None, src, dst))
        observations = []
        for i in range(3):
            await entered[i].wait()
            observations.append(plan.completed_ranges)
            self.assertEqual(len(observations[-1]), i)
            self.assertIsNone(plan.last_report)
            gates[i].set()
        report = await task
        self.assertEqual(observations, [(), ((0, 256),), ((0, 256), (512, 768))])
        with self.assertRaises(ValueError):
            await plan.execute(None, src[:-1], dst)
        self.assertEqual(len(calls), 3)
        self.assertEqual(plan.completed_ranges, report['completed_ranges'])
        self.assertEqual(plan.last_report, report)
        plan.close()

    async def test_resource_diagnostics_and_preparation_failure_boundary(self):
        make, _, _ = self.bind(items=769)
        arrays = self.arrays(769)
        with patch.object(self.module, 'MAX_PLAN_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'resource bound'):
                make(*arrays)
        with patch.object(self.module, '_immutable', side_effect=MemoryError):
            with self.assertRaisesRegex(MemoryError, 'allocation failed'):
                make(*arrays)
        self.assertEqual(self.calls, [])
        arrays[2][256:512] = 0
        plan = make(*arrays, window_tiles=1)
        src, dst = np.ones(8193, np.uint16), np.full(769, 77, np.uint16)
        take = self.module.np.take
        preparations = []
        def exhausted(*args, **kwargs):
            preparations.append(None)
            if len(preparations) == 2:
                raise MemoryError('bounded gather allocation failed')
            return take(*args, **kwargs)
        with patch.object(self.module.np, 'take', side_effect=exhausted):
            with self.assertRaises(self.module.PlanExecutionError) as caught:
                await plan.execute(None, src, dst)
        self.assertEqual(caught.exception.completed_ranges, ((0, 256),))
        self.assertEqual(caught.exception.failed_range, (512, 768))
        self.assertFalse(caught.exception.poisoned)
        self.assertFalse(plan.poisoned)
        self.assertEqual(len(self.calls), 1)
        np.testing.assert_array_equal(dst[256:], np.full(513, 77))
        plan.close()

    async def test_unscheduled_window_retains_frames_after_namespace_gc(self):
        class Suspended:
            def __await__(self):
                yield 'pending window'
        class Runtime:
            pass
        namespace = {'__name__': 'temporary_embedded_window_support'}
        exec(MODULE.read_text(), namespace)
        async def tile(runtime, *buffers):
            await Suspended()
        make, execute, once = namespace['bind_prepared_entry'](tile, descriptor(items=769))
        plan = make(*self.arrays(769), window_tiles=2)
        runtime = Runtime()
        src, dst = np.zeros(8193, np.uint16), np.zeros(769, np.uint16)
        references = [weakref.ref(value) for value in (runtime, src, dst, *plan._scratch)]
        coroutine = plan.execute(runtime, src, dst)
        self.assertEqual(coroutine.send(None), 'pending window')
        registry = namespace['_INFLIGHT_PLANS']
        plan_id = id(plan)
        del plan, runtime, src, dst, namespace, make, execute, once
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))
        retained = next(value for value in registry if id(value) == plan_id)
        self.assertTrue(retained.pending)
        self.assertEqual(retained.completed_ranges, ())
        coroutine.close()
        del coroutine, retained
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))

    async def test_wide_offsets_readonly_source_and_closed_snapshot(self):
        desc = dict(descriptor(), width=32767, height=3, source_elements=98301)
        make, _, _ = self.bind(desc=desc)
        x = np.array([32766, 32767, 32766, -1], np.int16)
        y = np.array([2, 0, 3, 0], np.int16)
        plan = make(x, y, np.ones(4, np.int16))
        self.assertEqual(plan._windows[0][4].dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(plan._windows[0][4], [98300])
        src = np.arange(98301, dtype=np.uint16)
        src[98300] = 65535
        src.flags.writeable = False
        dst = np.full(4, 123, np.uint16)
        report = await plan.execute(None, src, dst)
        np.testing.assert_array_equal(dst, [65535, 123, 123, 123])
        plan.close()
        self.assertEqual(plan.completed_ranges, report['completed_ranges'])
        self.assertEqual(plan.last_report, report)


if __name__ == '__main__':
    unittest.main()
