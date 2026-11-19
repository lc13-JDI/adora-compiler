"""Owned canvas host protocol tests; dispatch stand-ins do not validate RTL."""
import asyncio
import gc
import importlib.util
from pathlib import Path
import sys
import unittest
import weakref

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
sys.argv = sys.argv[:1]
MODULE = ROOT / 'tools/adoracc/prepared_canvas.py'


def descriptor(items=353):
    return dict(schema='adora.guarded-canvas-mask.v1', support_version=2,
                canvas_variant='A', packing_layout='indexed', value_logic='canvas',
                module_identity='a' * 64, entry='canvas', width=8193, height=1,
                items=items, source_elements=8193, tile_items=256,
                range_lo=32768, range_hi=60000, source_argument=0, x_argument=1,
                y_argument=2, enable_argument=3, old_argument=4,
                destination_argument=5, ownership='copy',
                regular_entry='_adora_canvas_regular_tile',
                old_only_entry='_adora_canvas_old_tile')


class HostProtocol(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'production canvas host support is missing')
        spec = importlib.util.spec_from_file_location('prepared_canvas', MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.calls = []

    def arrays(self, items=353):
        x, y, en = (np.zeros(items, np.int16) for _ in range(3))
        x[:6] = [8192, 0, -1, 8193, 0, 0]
        y[5] = 1
        en[:6] = [-1, 2, 1, 1, 0, 1]
        return x, y, en

    def bind(self, desc=None, normal=None, old_only=None):
        async def regular(runtime, source, index, enabled, old, mask):
            self.calls.append(('regular', tuple(a.copy() for a in (source, index, enabled, old, mask))))
            valid = (index >= 0) & (enabled != 0)
            mask[:] = np.where(old != 0, 255, 0)
            values = source[index[valid]]
            mask[valid] = np.where((values >= 32768) & (values <= 60000), 255, 0)
        async def old(runtime, previous, mask):
            self.calls.append(('old', (previous.copy(), mask.copy())))
            mask[:] = np.where(previous != 0, 255, 0)
        return self.module.bind_canvas_entry(normal or regular, old_only or old, desc or descriptor())

    async def test_private_same_slot_packing_and_regular_tail(self):
        # Two active, non-adjacent slots span the depth threshold. Swapping
        # their values and supplying matching indices preserves indexed reads
        # but changes both streaming reads. The plan constructs its own layout.
        x, y, en = (np.zeros(353, np.int16) for _ in range(3))
        x[[0, 255, 352]] = [8192, 0, 8192]
        en[[0, 255, 352]] = [-1, 1, -32768]
        src = np.zeros(8193, np.uint16)
        src[8192] = 32768
        old = np.full(353, 65535, np.uint16)
        old[1::2] = 0
        public = np.full(355, 181, np.uint16)
        seen = []
        async def streamed(runtime, packed, enabled, previous, output):
            seen.append((packed.copy(), enabled.copy(), previous.copy()))
            self.assertEqual(packed.dtype, np.uint16)
            self.assertEqual(enabled.dtype, np.int16)
            output[:] = 255 * np.where(enabled != 0, (packed >= 32768) & (packed <= 60000), previous != 0)
        for variant in ('C', 'D'):
            seen.clear()
            desc = dict(descriptor(), canvas_variant=variant, packing_layout='same-slot',
                        value_logic='canvas' if variant == 'C' else 'predicate')
            plan = self.bind(desc=desc, normal=streamed)[0](x, y, en)
            self.assertIsNone(plan._index)
            await plan.execute(None, src, old, public[1:-1])
            expected = np.where(old != 0, 255, 0)
            expected[[0, 255, 352]] = [255, 0, 255]
            np.testing.assert_array_equal(public[1:-1], expected)
            np.testing.assert_array_equal(public[[0,-1]], 181)
            self.assertEqual(len(seen), 2)
            np.testing.assert_array_equal(np.flatnonzero(seen[0][1]), [0, 255])
            np.testing.assert_array_equal(np.flatnonzero(seen[1][1]), [96])
            for buffer in seen[1]:
                self.assertTrue(np.all(buffer[97:] == 0))
            plan.close()

    async def test_sparse_same_slot_plan_handles_last_window_slot_and_wide_offset(self):
        items = 65537
        desc = dict(descriptor(items), height=9, source_elements=8193 * 9,
                    canvas_variant='C', packing_layout='same-slot')
        x, y, en = (np.zeros(items, np.int16) for _ in range(3))
        x[[65535, 65536]], y[[65535, 65536]], en[[65535, 65536]] = 8192, 8, -3
        source = np.zeros(desc['source_elements'], np.uint16)
        source[-1] = 47612
        writes = []

        async def capture(runtime, packed, enabled, previous, out):
            slots = np.flatnonzero(enabled)
            writes.append((slots.tolist(), packed[slots].tolist()))
            out[:] = packed

        async def empty(runtime, previous, out):
            out.fill(0)

        make, _, _ = self.bind(desc=desc, normal=capture, old_only=empty)
        plan = make(x, y, en, window_tiles=256, probes=False)
        # Two active slot/offset pairs suffice beside snapshots and tile roles.
        self.assertLess(plan.stats['plan_bytes'], 7 * items)
        en[:] = 0
        mask = np.empty(items, np.uint16)
        await plan.execute(None, source, np.zeros_like(mask), mask)
        self.assertEqual(writes, [([255], [47612]), ([0], [47612])])
        np.testing.assert_array_equal(np.flatnonzero(mask), [65535, 65536])
        plan.close()

    async def test_owned_geometry_fresh_frames_and_old_only_tail(self):
        make, _, _ = self.bind()
        x, y, en = self.arrays()
        plan = make(x, y, en)
        x[:] = y[:] = en[:] = 0
        src = np.zeros(8193, np.uint16)
        src[8192], src[0] = 32768, 60001
        old = np.zeros(353, np.uint16)
        old[1:6] = [1, 32768, 65535, 1, 0]
        old[256] = 34304
        base = np.full(355, 97, np.uint16)
        src.flags.writeable = old.flags.writeable = False
        report = await plan.execute(None, src, old, base[1:-1])
        expected = np.where(old != 0, 255, 0)
        expected[:2] = [255, 0]
        np.testing.assert_array_equal(base[1:-1], expected)
        np.testing.assert_array_equal(base[[0, -1]], [97, 97])
        self.assertEqual([c[0] for c in self.calls], ['regular', 'old'])
        self.assertEqual(report['completed_ranges'], ((0, 256), (256, 353)))
        self.assertEqual(report['device_calls'], 2)
        self.assertEqual(report['regular_tiles'], 1)
        self.assertEqual(report['old_only_tiles'], 1)
        self.assertTrue(np.all(self.calls[1][1][0][97:] == 0))
        next_src = np.full(8193, 65535, np.uint16)
        next_old = np.zeros(353, np.uint16)
        next_old[2], next_old[256] = 65535, 0
        await plan.execute(None, next_src, next_old, base[1:-1])
        expected = np.where(next_old != 0, 255, 0)
        expected[:2] = 0
        np.testing.assert_array_equal(base[1:-1], expected)
        self.assertEqual([c[0] for c in self.calls], ['regular', 'old'] * 2)
        plan.close()

    async def test_all_true_out_of_range_never_becomes_old_only(self):
        make, _, _ = self.bind()
        x, y, en = (np.zeros(353, np.int16) for _ in range(3))
        en[:] = -1
        plan = make(x, y, en)
        old = np.full(353, 65535, np.uint16)
        output = np.full(353, 17, np.uint16)
        await plan.execute(None, np.full(8193, 60001, np.uint16), old, output)
        self.assertEqual([c[0] for c in self.calls], ['regular', 'regular'])
        self.assertTrue(np.all(output == 0))
        self.assertTrue(np.all(old == 65535))
        plan.close()

    async def test_all_returned_valid_slots_copied_without_host_mask_or_merge(self):
        async def normal(runtime, src, idx, en, old, mask):
            mask[:] = 42000
        async def old_only(runtime, old, mask):
            mask[:] = 43000
        make, _, _ = self.bind(normal=normal, old_only=old_only)
        plan = make(*self.arrays())
        base = np.full(355, 91, np.uint16)
        await plan.execute(None, np.zeros(8193, np.uint16), np.zeros(353, np.uint16), base[1:-1])
        np.testing.assert_array_equal(base[1:257], np.full(256, 42000))
        np.testing.assert_array_equal(base[257:-1], np.full(97, 43000))
        np.testing.assert_array_equal(base[[0, -1]], [91, 91])
        plan.close()

    async def test_rejects_invalid_arrays_aliases_foreign_plans_and_metadata(self):
        make, execute, once = self.bind()
        coords = self.arrays()
        plan = make(*coords)
        source, old, mask = np.zeros(8193, np.uint16), np.zeros(353, np.uint16), np.full(353, 73, np.uint16)
        readonly = mask.copy()
        readonly.flags.writeable = False
        for args in [(source[:-1], old, mask), (source.astype(np.int16), old, mask),
                     (source, old[:-1], mask), (source, old, readonly),
                     (source, old, old), (source, source[:353], mask),
                     (source, old, source[:353]), (source, old, np.zeros(706, np.uint16)[::2]),
                     (source, old, mask.byteswap().view('>u2'))]:
            with self.subTest(args=[a.shape for a in args]), self.assertRaises(ValueError):
                await plan.execute(None, *args)
        with self.assertRaises(ValueError):
            make(coords[0], coords[0], coords[2])
        foreign = self.bind()[0](*self.arrays())
        with self.assertRaises(ValueError):
            await execute(None, foreign, source, old, mask)
        with self.assertRaises(ValueError):
            await once(None, source, *coords, old, coords[0].view(np.uint16))
        for changes in [dict(schema='adora.guarded-inplace-large.v1'), dict(support_version=1),
                        dict(support_version=True), dict(module_identity='stale'),
                        dict(canvas_variant='C'), dict(packing_layout='permuted'),
                        dict(value_logic='predicate'),
                        dict(width=8192), dict(items=0), dict(items=2**63),
                        dict(tile_items=128), dict(old_argument=5), dict(range_lo=0),
                        dict(range_hi=65536), dict(range_lo=True), dict(range_lo=60001),
                        dict(ownership='borrow'), dict(old_only_entry='_adora_canvas_regular_tile')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.bind(desc=dict(descriptor(), **changes))
        self.assertEqual(self.calls, [])
        self.assertTrue(np.all(mask == 73))
        plan.close()
        foreign.close()

    async def test_once_original_argument_order_and_snapshot_detachment(self):
        d = descriptor()
        roles = ('source_argument', 'x_argument', 'y_argument', 'enable_argument', 'old_argument', 'destination_argument')
        d.update(zip(roles, reversed(range(6))))
        _, _, once = self.bind(desc=d)
        source, old, mask = np.full(8193, 32768, np.uint16), np.ones(353, np.uint16), np.zeros(353, np.uint16)
        x, y, en = self.arrays()
        await once(None, mask, old, en, y, x, source)
        self.assertTrue(np.all(mask == 255))

    async def test_bounded_window_scratch_and_serial_completion(self):
        sizes = []
        for items in (353, 256 * 65 + 97):
            make, _, _ = self.bind(desc=descriptor(items))
            plan = make(*self.arrays(items))
            sizes.append(plan.stats['scratch_bytes'])
            self.assertLessEqual(plan.stats['scratch_bytes'], 32 * 256 * 10)
            source, old, mask = np.zeros(8193, np.uint16), np.ones(items, np.uint16), np.zeros(items, np.uint16)
            result = await plan.execute(None, source, old, mask)
            self.assertEqual(result['device_calls'], (items + 255) // 256)
            self.assertEqual(len(plan.completed_ranges), (items + 255) // 256)
            self.assertLessEqual(plan.stats['plan_bytes'], 16 * items + 1024)
            plan.close()
        self.assertLess(sizes[0], sizes[1])

    async def test_failure_retains_inflight_and_only_completed_prefix(self):
        entered, resume = asyncio.Event(), asyncio.Event()
        async def normal(runtime, src, idx, en, old, mask):
            mask[:] = 53
        async def old_only(runtime, old, mask):
            mask[:] = 88
            entered.set()
            await resume.wait()
            raise RuntimeError('completion unknown')
        make, _, _ = self.bind(normal=normal, old_only=old_only)
        plan = make(*self.arrays())
        source, old, mask = np.zeros(8193, np.uint16), np.ones(353, np.uint16), np.full(353, 73, np.uint16)
        task = asyncio.create_task(plan.execute(object(), source, old, mask))
        await entered.wait()
        self.assertTrue(plan.pending)
        with self.assertRaises(ValueError):
            plan.close()
        with self.assertRaises(ValueError):
            await plan.execute(None, source, old, mask)
        resume.set()
        with self.assertRaises(self.module.PlanExecutionError) as failure:
            await task
        self.assertEqual(failure.exception.completed_ranges, ((0, 256),))
        self.assertEqual(failure.exception.failed_range, (256, 353))
        self.assertTrue(plan.poisoned)
        np.testing.assert_array_equal(mask[:256], np.full(256, 53))
        np.testing.assert_array_equal(mask[256:], np.full(97, 73))
        self.assertTrue(np.all(old == 1))
        with self.assertRaises(ValueError):
            plan.close()
        references = [weakref.ref(a) for a in (source, old, mask)]
        del source, old, mask, task, plan, make
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in references))

    async def test_cancellation_preserves_type_prefix_and_untouched_output(self):
        for variant, layout in [('A', 'indexed'), ('C', 'same-slot')]:
            with self.subTest(variant=variant):
                await self.cancel_second_tile(dict(descriptor(), canvas_variant=variant, packing_layout=layout))

    async def cancel_second_tile(self, desc):
        entered = asyncio.Event()
        async def normal(runtime, *buffers):
            buffers[-1][:] = 11
        async def old_only(runtime, old, mask):
            mask[:] = 19
            entered.set()
            await asyncio.Event().wait()
        make, _, _ = self.bind(desc=desc, normal=normal, old_only=old_only)
        plan = make(*self.arrays())
        mask = np.full(353, 71, np.uint16)
        task = asyncio.create_task(plan.execute(None, np.zeros(8193, np.uint16), np.ones(353, np.uint16), mask))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await task
        self.assertEqual(caught.exception.completed_ranges, ((0, 256),))
        self.assertEqual(caught.exception.failed_range, (256, 353))
        self.assertTrue(caught.exception.poisoned)
        self.assertIs(plan._retained[-1], mask)
        np.testing.assert_array_equal(mask[:256], np.full(256, 11))
        np.testing.assert_array_equal(mask[256:], np.full(97, 71))

    async def test_scheduler_unschedule_and_generator_close_keep_storage_alive(self):
        class Suspended:
            def __await__(self):
                yield 'pending device'
        async def tile(runtime, *arrays):
            await Suspended()
        namespace = {'__name__': 'temporary_canvas_support'}
        exec(MODULE.read_text(), namespace)
        make, execute, once = namespace['bind_canvas_entry'](tile, tile, descriptor())
        plan = make(*self.arrays())
        storage = weakref.ref(plan._scratch[-1])
        coroutine = plan.execute(object(), np.zeros(8193, np.uint16),
                                 np.ones(353, np.uint16), np.full(353, 47, np.uint16))
        self.assertEqual(coroutine.send(None), 'pending device')
        self.assertIn(plan, namespace['_INFLIGHT_PLANS'])
        with self.assertRaises(ValueError):
            plan.close()
        coroutine.close()
        self.assertTrue(plan.poisoned)
        self.assertFalse(plan.pending)
        del coroutine, plan, make, execute, once, namespace
        gc.collect()
        self.assertIsNotNone(storage())

    async def test_metadata_detached_probes_disabled_and_closed_plan_rejected(self):
        d = descriptor()
        make, _, _ = self.bind(desc=d)
        plan = make(*self.arrays(), window_tiles=1, probes=False)
        d['items'] = 1
        plan.descriptor['items'] = 1
        plan.stats['old_only_tiles'] = 0
        source, old, mask = np.zeros(8193, np.uint16), np.ones(353, np.uint16), np.full(353, 73, np.uint16)
        report = await plan.execute(None, source, old, mask)
        report['timings_ns']['total'] = -1
        self.assertEqual(plan.descriptor['items'], 353)
        self.assertEqual(plan.stats['old_only_tiles'], 1)
        self.assertGreaterEqual(plan.last_report['timings_ns']['total'], 0)
        self.assertIsNone(report['timings_ns']['device'])
        plan.close()
        plan.close()
        self.assertEqual(plan.completed_ranges, ((0, 256), (256, 353)))
        with self.assertRaises(ValueError):
            await plan.execute(None, source, old, mask)
        for bad in (0, 257, True, 1.5):
            with self.assertRaises(ValueError):
                make(*self.arrays(), window_tiles=bad)

    async def test_mask_storage_is_never_read_by_host(self):
        class WriteOnly(np.ndarray):
            def __getitem__(self, key):
                raise AssertionError('caller mask must not be read')
            def __setitem__(self, key, value):
                # ndarray's subclass assignment obtains a destination view via
                # __getitem__; bypass that view-selection hook, not a data read.
                self.view(np.ndarray)[key] = value
        make, _, _ = self.bind()
        plan = make(*self.arrays())
        mask = np.full(353, 73, np.uint16).view(WriteOnly)
        await plan.execute(None, np.full(8193, 32768, np.uint16), np.ones(353, np.uint16), mask)
        self.assertTrue(np.all(mask.view(np.ndarray) == 255))
        plan.close()


if __name__ == '__main__':
    unittest.main()
