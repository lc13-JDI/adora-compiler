"""Owned host plans for compiler-proved private-canvas mask fusion.

Embedded verbatim in generated modules. Descriptors describe compiled transport;
only the typed compiler proof establishes the original two-loop C semantics.
The host packs source values but never computes canvas values or output masks.
"""
from __future__ import annotations

import re
import sys
import time
import types

import numpy as np

SUPPORT_VERSION = 2
SCHEMA = 'adora.guarded-canvas-mask.v1'
MAX_ELEMENTS = 16 * 1024 * 1024
MAX_PLAN_BYTES = 768 * 1024 * 1024
MAX_WINDOW_TILES = 256
_TILE = 256

# A failed await does not establish DMA completion. This process root survives
# loss of the generated module, coroutine, exception, and user's plan reference.
_new_registry = types.ModuleType('_adora_canvas_lifetimes_v1')
_new_registry.poisoned = []
_new_registry.inflight = set()
_registry = sys.modules.setdefault(_new_registry.__name__, _new_registry)
del _new_registry
_POISONED_PLANS = _registry.poisoned
_INFLIGHT_PLANS = _registry.inflight


class PlanExecutionError(RuntimeError):
    """Only completed_ranges have been copied to the caller's output."""

    def __init__(self, message, completed_ranges, failed_range, poisoned):
        super().__init__(message)
        self.completed_ranges = tuple(completed_ranges)
        self.failed_range = failed_range
        self.poisoned = poisoned


def _validate_descriptor(value):
    if not isinstance(value, dict):
        raise ValueError('canvas entry: expected compiler-produced descriptor')
    result = dict(value)
    if (result.get('schema') != SCHEMA or
            type(result.get('support_version')) is not int or
            result['support_version'] != SUPPORT_VERSION):
        raise ValueError('canvas entry: incompatible descriptor/support version')
    if not isinstance(result.get('module_identity'), str) or not re.fullmatch('[0-9a-f]{64}', result['module_identity']):
        raise ValueError('canvas entry: missing compiled module identity')
    for key in ('entry', 'regular_entry', 'old_only_entry'):
        if not isinstance(result.get(key), str) or not result[key].isidentifier():
            raise ValueError(f'canvas entry: invalid {key} identity')
    if result['regular_entry'] == result['old_only_entry']:
        raise ValueError('canvas entry: kernel roles must be distinct')
    variant = result.get('canvas_variant')
    if (variant not in ('A', 'B', 'C', 'D') or
            result.get('packing_layout') != ('same-slot' if variant in ('C', 'D') else 'indexed') or
            result.get('value_logic') != ('predicate' if variant in ('B', 'D') else 'canvas')):
        raise ValueError('canvas entry: incompatible private kernel variant')
    if result.get('ownership') != 'copy':
        raise ValueError('canvas entry: map ownership must be copy')
    for key in ('width', 'height', 'items', 'source_elements'):
        if type(result.get(key)) is not int or not 0 < result[key] <= MAX_ELEMENTS:
            raise ValueError(f'canvas entry: {key} exceeds positive element bound {MAX_ELEMENTS}')
    if result['width'] * result['height'] != result['source_elements']:
        raise ValueError('canvas entry: source extent contradicts dimensions')
    if type(result.get('tile_items')) is not int or result['tile_items'] != _TILE:
        raise ValueError('canvas entry: only 256-slot kernels are supported')
    lo, hi = result.get('range_lo'), result.get('range_hi')
    if type(lo) is not int or type(hi) is not int or not 0 < lo <= hi <= 65535:
        raise ValueError('canvas entry: invalid positive uint16 inclusive range')
    roles = [result.get(key) for key in ('source_argument', 'x_argument', 'y_argument',
                                        'enable_argument', 'old_argument', 'destination_argument')]
    if any(type(role) is not int for role in roles) or sorted(roles) != list(range(6)):
        raise ValueError('canvas entry: object roles must identify all six arguments')
    return result


def _array(value, elements, dtype, name, writable=False):
    if not isinstance(value, np.ndarray):
        raise ValueError(f'canvas entry {name}: expected numpy.ndarray')
    if value.shape != (elements,) or value.dtype != np.dtype(dtype) or not value.dtype.isnative:
        raise ValueError(f'canvas entry {name}: expected native-endian {dtype}[{elements}]')
    if not value.flags.c_contiguous:
        raise ValueError(f'canvas entry {name}: expected contiguous storage')
    if writable and not value.flags.writeable:
        raise ValueError(f'canvas entry {name}: output must be writable')


def _disjoint(buffers):
    for left in range(len(buffers)):
        for right in range(left + 1, len(buffers)):
            if np.shares_memory(buffers[left], buffers[right]):
                raise ValueError('canvas entry: buffers must not alias')


def _immutable(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _CanvasPlan:
    __slots__ = ('_index', '_identity', '_descriptor', '_normal', '_old_only', '_snapshots',
                 '_offsets', '_windows', '_active', '_scratch', '_ranges', '_capacity', '_probes',
                 '_cursor', '_closed', '_busy', '_poisoned', '_retained', '_stats',
                 '_last_report', '_completed_ranges')

    def __init__(self, identity, descriptor, normal, old_only, x, y, enabled, *,
                 window_tiles=32, probes=True):
        start = time.perf_counter_ns()
        if type(window_tiles) is not int or not 1 <= window_tiles <= MAX_WINDOW_TILES:
            raise ValueError(f'canvas plan: window_tiles must be an integer in 1..{MAX_WINDOW_TILES}')
        if type(probes) is not bool:
            raise ValueError('canvas plan: probes must be a boolean')
        items = descriptor['items']
        tiles = (items + _TILE - 1) // _TILE
        payload_bound = 14 * items + tiles + min(window_tiles, tiles) * _TILE * 10
        if payload_bound > MAX_PLAN_BYTES or payload_bound > np.iinfo(np.intp).max:
            raise ValueError('canvas plan: immutable payload exceeds resource bound')
        for name, value in (('x', x), ('y', y), ('enable', enabled)):
            _array(value, items, 'int16', name)
        _disjoint((x, y, enabled))
        self._identity, self._descriptor = identity, descriptor
        self._normal, self._old_only = normal, old_only
        self._snapshots = tuple(_immutable(value) for value in (x, y, enabled))
        x, y, enabled = self._snapshots
        valid = ((enabled != 0) & (x >= 0) & (y >= 0) &
                 (x < descriptor['width']) & (y < descriptor['height']))
        offsets = np.full(items, -1, np.int64)
        # Widen only after coordinate bounds are proved. A global address is
        # never narrowed into the device's signed 16-bit local index domain.
        offsets[valid] = y[valid].astype(np.int64) * descriptor['width'] + x[valid].astype(np.int64)
        self._ranges = tuple((begin, min(begin + _TILE, items)) for begin in range(0, items, _TILE))
        self._active = _immutable(np.array([np.any(valid[begin:end]) for begin, end in self._ranges], np.bool_))
        self._capacity = min(window_tiles, tiles)
        shape = (self._capacity, _TILE)
        self._index = np.empty(shape, np.int16) if descriptor['packing_layout'] == 'indexed' else None
        self._windows = None
        self._offsets = None
        if self._index is None:
            # A window contains at most 65,536 slots, and source_elements is
            # bounded by MAX_ELEMENTS (2**24). Both casts are lossless after
            # computing validated source offsets in int64.
            window_items = self._capacity * _TILE
            windows = []
            for first in range(0, items, window_items):
                section = slice(first, first + window_items)
                slots = np.flatnonzero(valid[section])
                windows.append((_immutable(slots.astype(np.uint16)),
                                _immutable(offsets[section][slots].astype(np.uint32))))
            self._windows = tuple(windows)
            geometry_bytes = sum(a.nbytes for pair in windows for a in pair)
        else:
            self._offsets = _immutable(offsets)
            geometry_bytes = self._offsets.nbytes
        self._scratch = tuple(np.zeros(shape, dtype) for dtype in
                              (np.uint16, np.int16, np.uint16, np.uint16))
        self._probes = probes
        self._cursor = 0
        self._closed = self._busy = self._poisoned = False
        self._retained = self._last_report = None
        self._completed_ranges = ()
        regular = int(np.count_nonzero(self._active))
        self._stats = dict(plan_bytes=sum(a.nbytes for a in self._snapshots) + geometry_bytes + self._active.nbytes,
                           scratch_bytes=sum(a.nbytes for a in self._scratch) + (self._index.nbytes if self._index is not None else 0),
                           execution_scratch_bound_bytes=32 * self._capacity * _TILE,
                           regular_tiles=regular, old_only_tiles=tiles - regular,
                           logical_tiles=tiles, window_tiles=self._capacity, probes=probes,
                           plan_build_ns=time.perf_counter_ns() - start, executions=0)

    @property
    def descriptor(self):
        return dict(self._descriptor)

    @property
    def stats(self):
        """Persistent ndarray payload sizes; Python object headers are excluded."""
        return dict(self._stats)

    @property
    def closed(self):
        return self._closed

    @property
    def poisoned(self):
        return self._poisoned

    @property
    def pending(self):
        return self._busy

    @property
    def completed_ranges(self):
        return self._completed_ranges if self._closed else self._ranges[:self._cursor]

    @property
    def last_report(self):
        if self._last_report is None:
            return None
        return dict(self._last_report, timings_ns=dict(self._last_report['timings_ns']))

    def close(self):
        if self._busy:
            raise ValueError('canvas plan: execution pending; cannot release borrowed storage')
        if self._poisoned:
            raise ValueError('canvas plan: completion unknown; poisoned storage retained until runtime/process teardown')
        if self._closed:
            return
        self._completed_ranges = self.completed_ranges
        self._closed = True
        self._snapshots = self._offsets = self._windows = self._active = self._scratch = self._index = None
        self._ranges = ()

    def _prepared_tiles(self, src, old, times):
        packed, enabled, previous, output = self._scratch
        for first in range(0, len(self._ranges), self._capacity):
            if self._probes:
                stamp = time.perf_counter_ns()
            ranges = self._ranges[first:first + self._capacity]
            packed.fill(0)
            if self._index is not None:
                self._index.fill(-1)
            enabled.fill(0)
            previous.fill(0)
            # This initializes private output storage, never reads caller mask.
            output.fill(0)
            if self._windows is not None:
                slots, offsets = self._windows[first // self._capacity]
                begin, end = ranges[0][0], ranges[-1][1]
                previous.ravel()[:end - begin] = old[begin:end]
                enabled.ravel()[slots] = 1
            else:
                for row, (begin, end) in enumerate(ranges):
                    previous[row, :end - begin] = old[begin:end]
                    if self._active[first + row]:
                        slots = np.flatnonzero(self._offsets[begin:end] >= 0)
                        self._index[row, slots] = slots
                        enabled[row, slots] = 1
            if self._probes:
                times['prepare'] += time.perf_counter_ns() - stamp
                stamp = time.perf_counter_ns()
            # Gathering depends only on the immutable G/address snapshot.
            if self._windows is not None:
                packed.ravel()[slots] = src[offsets]
            else:
                for row, (begin, end) in enumerate(ranges):
                    if self._active[first + row]:
                        slots = np.flatnonzero(enabled[row])
                        packed[row, slots] = src[self._offsets[begin:end][slots]]
            if self._probes:
                times['pack'] += time.perf_counter_ns() - stamp
            for row, (begin, end) in enumerate(ranges):
                yield begin, end, bool(self._active[first + row]), row

    async def execute(self, runtime, src, old, mask):
        """Borrow arrays through completion; callers must not mutate aliases.

        Source and old are read-only inputs; mask is an independent write-only
        output. Each logical tile gets exactly one serial device call. Only its
        successful return permits copying real (non-padding) slots to mask.
        """
        start = time.perf_counter_ns()
        if self._closed or self._poisoned or self._busy:
            raise ValueError('canvas plan: closed, poisoned, or already executing')
        _array(src, self._descriptor['source_elements'], 'uint16', 'src')
        _array(old, self._descriptor['items'], 'uint16', 'old')
        _array(mask, self._descriptor['items'], 'uint16', 'mask', writable=True)
        _disjoint((src, old, mask))
        times = dict.fromkeys(('precheck', 'prepare', 'pack', 'device', 'copyback'), 0 if self._probes else None)
        if self._probes:
            times['precheck'] = time.perf_counter_ns() - start
        self._busy = True
        self._cursor = 0
        self._last_report = None
        failed_range, in_flight = None, False
        packed, enabled, previous, output = self._scratch
        try:
            for begin, end, regular, row in self._prepared_tiles(src, old, times):
                failed_range = (begin, end)
                if self._probes:
                    stamp = time.perf_counter_ns()
                self._retained = (runtime, src, old, mask)
                _INFLIGHT_PLANS.add(self)
                in_flight = True
                if regular:
                    if self._index is None:
                        await self._normal(runtime, packed[row], enabled[row], previous[row], output[row])
                    else:
                        await self._normal(runtime, packed[row], self._index[row], enabled[row], previous[row], output[row])
                else:
                    await self._old_only(runtime, previous[row], output[row])
                in_flight = False
                _INFLIGHT_PLANS.discard(self)
                self._retained = None
                if self._probes:
                    times['device'] += time.perf_counter_ns() - stamp
                    stamp = time.perf_counter_ns()
                mask[begin:end] = output[row, :end - begin]
                self._cursor += 1
                if self._probes:
                    times['copyback'] += time.perf_counter_ns() - stamp
            self._stats['executions'] += 1
            times['total'] = time.perf_counter_ns() - start
            self._last_report = dict(completed_ranges=self.completed_ranges,
                                     regular_tiles=self._stats['regular_tiles'],
                                     old_only_tiles=self._stats['old_only_tiles'],
                                     logical_tiles=len(self._ranges), device_calls=self._cursor,
                                     timings_ns=times)
            return self.last_report
        except BaseException as error:
            if not in_flight:
                failed_range = self._ranges[self._cursor] if self._cursor < len(self._ranges) else None
            else:
                self._poisoned = True
                _POISONED_PLANS.append(self)
                _INFLIGHT_PLANS.discard(self)
            if not isinstance(error, Exception):
                error.completed_ranges = self.completed_ranges
                error.failed_range = failed_range
                error.poisoned = self._poisoned
                raise
            raise PlanExecutionError(str(error), self.completed_ranges, failed_range, self._poisoned) from error
        finally:
            self._busy = False


def bind_canvas_entry(normal_dispatch, old_dispatch, descriptor):
    """Internal compiler binding; supplying a descriptor grants no source proof."""
    descriptor = _validate_descriptor(descriptor)
    if not callable(normal_dispatch) or not callable(old_dispatch):
        raise ValueError('canvas entry: both generated tile callables are required')
    identity = object()

    def make_plan(x, y, enabled, *, window_tiles=32, probes=True):
        try:
            return _CanvasPlan(identity, descriptor, normal_dispatch, old_dispatch,
                               x, y, enabled, window_tiles=window_tiles, probes=probes)
        except MemoryError as error:
            raise MemoryError('canvas plan: allocation failed within resource bounds; reduce query count or window_tiles') from error

    async def execute_plan(runtime, plan, src, old, mask):
        if type(plan) is not _CanvasPlan or plan._identity is not identity:
            raise ValueError('canvas entry: foreign plan/module identity')
        return await plan.execute(runtime, src, old, mask)

    async def once_entry(runtime, *buffers):
        if len(buffers) != 6:
            raise ValueError('canvas entry: original C ABI requires six buffers')
        src, x, y, enabled, old, mask = (buffers[descriptor[key]] for key in
            ('source_argument', 'x_argument', 'y_argument', 'enable_argument', 'old_argument', 'destination_argument'))
        for value, name, size, dtype, writable in (
                (src, 'src', descriptor['source_elements'], 'uint16', False),
                (x, 'x', descriptor['items'], 'int16', False),
                (y, 'y', descriptor['items'], 'int16', False),
                (enabled, 'enable', descriptor['items'], 'int16', False),
                (old, 'old', descriptor['items'], 'uint16', False),
                (mask, 'mask', descriptor['items'], 'uint16', True)):
            _array(value, size, dtype, name, writable)
        _disjoint(buffers)
        plan = make_plan(x, y, enabled)
        try:
            return await execute_plan(runtime, plan, src, old, mask)
        finally:
            if not plan.poisoned:
                plan.close()

    once_entry.__name__ = descriptor['entry']
    return make_plan, execute_plan, once_entry
