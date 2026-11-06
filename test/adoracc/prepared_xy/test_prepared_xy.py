"""Host protocol tests: the tile stand-in is not a hardware correctness claim."""
import asyncio
import gc
import weakref
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
sys.argv = sys.argv[:1]
MODULE = ROOT / 'tools/adoracc/prepared_xy.py'


def descriptor(items=4, nonzero=False):
    return dict(schema='adora.guarded-inplace-large.v1', support_version=1,
                module_identity='a' * 64, entry='overlay', width=8193, height=1,
                items=items, source_elements=8193, tile_items=256,
                value_nonzero=nonzero, source_argument=0, x_argument=1,
                y_argument=2, enable_argument=3, destination_argument=4)


class HostProtocol(unittest.IsolatedAsyncioTestCase):
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
        for changes in [dict(support_version=2), dict(schema='external'),
                        dict(module_identity='unverified'), dict(width=True),
                        dict(items=2**63), dict(width=2**63),
                        dict(source_elements=0), dict(tile_items=128),
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
        self.assertEqual(self.calls[0], self.calls[1])
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


if __name__ == '__main__':
    unittest.main()
