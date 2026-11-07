"""Owned P plans for compiler-proved guarded-inplace-large entries.

This source is embedded into generated modules. Descriptors are compiler output,
not an external proof/import format. Binding checks transport compatibility;
only the compiler's typed SSA analysis establishes source semantics.
"""
from __future__ import annotations

import re
import sys
import time
import types

import numpy as np

SUPPORT_VERSION = 2
SCHEMA = 'adora.guarded-inplace-large.v1'
# Conservative allocation limits; checked before any user-sized allocation.
MAX_ELEMENTS = 16 * 1024 * 1024
_TILE = 256
MAX_WINDOW_TILES = 256
MAX_PLAN_BYTES = 768 * 1024 * 1024
# A failed async runtime call does not prove DMA has stopped. Keep strong roots,
# including the runtime and frame borrow, until process/runtime teardown. There
# is intentionally no unsafe "force close" or unproved recovery operation.
# Embedded support dictionaries can themselves become unreachable. A private
# interpreter module supplies a process root even after callers drop a generated
# module, plan, task, and the exception traceback. Initialize before setdefault
# so concurrent imports never replace an already-live registry.
_new_registry = types.ModuleType('_adora_prepared_lifetimes_v1')
_new_registry.poisoned = []
_new_registry.inflight = set()
_registry = sys.modules.setdefault(_new_registry.__name__, _new_registry)
del _new_registry
_POISONED_PLANS = _registry.poisoned
_INFLIGHT_PLANS = _registry.inflight


class PlanExecutionError(RuntimeError):
    """A frame failed; only completed_ranges have been copied back to the caller."""

    def __init__(self, message, completed_ranges, failed_range):
        super().__init__(message)
        self.completed_ranges = tuple(completed_ranges)
        self.failed_range = failed_range
        self.poisoned = True


def _validate_descriptor(value):
    if not isinstance(value, dict):
        raise ValueError('prepared entry: descriptor must be a compiler-produced object')
    result = dict(value)
    if result.get('schema') != SCHEMA or type(result.get('support_version')) is not int or result['support_version'] != SUPPORT_VERSION:
        raise ValueError('prepared entry: incompatible descriptor/support version')
    if not isinstance(result.get('module_identity'), str) or not re.fullmatch('[0-9a-f]{64}', result['module_identity']):
        raise ValueError('prepared entry: missing compiled module identity')
    if not isinstance(result.get('entry'), str) or not result['entry'].isidentifier():
        raise ValueError('prepared entry: invalid entry identity')
    for key in ('width', 'height', 'items', 'source_elements'):
        if type(result.get(key)) is not int or not 0 < result[key] <= MAX_ELEMENTS:
            raise ValueError(f'prepared entry: {key} exceeds supported positive element bound {MAX_ELEMENTS}')
    if result['width'] * result['height'] != result['source_elements']:
        raise ValueError('prepared entry: source extent contradicts dimensions')
    if type(result.get('tile_items')) is not int or result['tile_items'] != _TILE:
        raise ValueError('prepared entry: only M=N=256 is supported')
    if type(result.get('value_nonzero')) is not bool:
        raise ValueError('prepared entry: value_nonzero must be a compiler-proved boolean')
    roles = [result.get(key) for key in ('source_argument', 'x_argument', 'y_argument',
                                        'enable_argument', 'destination_argument')]
    if any(type(role) is not int for role in roles) or sorted(roles) != list(range(5)):
        raise ValueError('prepared entry: object roles must identify all five arguments')
    return result


def _array(value, elements, dtype, name, writable=False):
    if not isinstance(value, np.ndarray):
        raise ValueError(f'prepared entry {name}: expected numpy.ndarray')
    if value.shape != (elements,) or value.dtype != np.dtype(dtype) or not value.dtype.isnative:
        raise ValueError(f'prepared entry {name}: expected native-endian {dtype}[{elements}]')
    if not value.flags.c_contiguous:
        raise ValueError(f'prepared entry {name}: expected contiguous storage')
    if writable and not value.flags.writeable:
        raise ValueError(f'prepared entry {name}: output must be writable')


def _disjoint(buffers):
    for left in range(len(buffers)):
        for right in range(left + 1, len(buffers)):
            if np.shares_memory(buffers[left], buffers[right]):
                raise ValueError('prepared entry: buffers must not alias')


def _immutable(array):
    """An immutable byte owner, including for internal static slot templates."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _PreparedPlan:
    __slots__ = ('_identity', '_descriptor', '_tile', '_snapshots', '_offsets',
                 '_active', '_scratch', '_positions', '_closed', '_busy',
                 '_poisoned', '_retained', '_stats', '_last_report', '_completed_ranges',
                 '_windows', '_ranges', '_cursor', '_preparation', '_bookkeeping', '_probes')

    def __init__(self, identity, descriptor, tile, x, y, enabled, *,
                 preparation='window', window_tiles=32, bookkeeping='cursor', probes=True):
        start = time.perf_counter_ns()
        items = descriptor['items']
        if preparation not in ('window', 'tile') or bookkeeping not in ('cursor', 'eager'):
            raise ValueError('prepared plan: unsupported preparation or bookkeeping mode')
        if type(window_tiles) is not int or not 1 <= window_tiles <= MAX_WINDOW_TILES:
            raise ValueError(f'prepared plan: window_tiles must be an integer in 1..{MAX_WINDOW_TILES}')
        if type(probes) is not bool:
            raise ValueError('prepared plan: probes must be a boolean')
        tiles = (items + _TILE - 1) // _TILE
        # Wide arithmetic before allocations. Worst-case ndarray payload uses
        # all queries active; Python headers and construction peaks are separate.
        payload_bound = (22 * items + 12 * tiles * _TILE if preparation == 'window'
                         else 14 * items + tiles)
        if payload_bound > MAX_PLAN_BYTES or payload_bound > np.iinfo(np.intp).max:
            raise ValueError(f'prepared plan: immutable payload exceeds {MAX_PLAN_BYTES} byte resource bound')
        for name, value in (('x', x), ('y', y), ('enable', enabled)):
            _array(value, items, 'int16', name)
        _disjoint((x, y, enabled))
        self._identity = identity
        self._descriptor = descriptor
        self._tile = tile
        self._preparation, self._bookkeeping, self._probes = preparation, bookkeeping, probes
        # bytes owns an immutable copy, unlike ndarray writeable=False on an
        # owned allocation (which can be toggled back). No public mutable view.
        self._snapshots = tuple(np.frombuffer(value.tobytes(), dtype=np.int16)
                                for value in (x, y, enabled))
        x, y, enabled = self._snapshots
        valid = ((enabled != 0) & (x >= 0) & (y >= 0) &
                 (x < descriptor['width']) & (y < descriptor['height']))
        offsets = np.full(items, -1, np.int64)
        # Arithmetic is explicitly wide after bounds checking. No global index
        # is ever narrowed into the device's 16-bit local address domain.
        offsets[valid] = y[valid].astype(np.int64) * descriptor['width'] + x[valid].astype(np.int64)
        active = np.array([np.any(valid[start:start + _TILE])
                           for start in range(0, items, _TILE)], dtype=np.bool_)
        self._ranges = tuple((int(i) * _TILE, min((int(i) + 1) * _TILE, items))
                             for i in np.flatnonzero(active))
        self._cursor = 0
        capacity = min(window_tiles, len(self._ranges)) if preparation == 'window' else 1
        self._windows = ()
        if preparation == 'tile':
            # Retain the original preparation algorithm as an explicit fallback
            # and control; it does not use precomputed window slot metadata.
            self._offsets, self._active = _immutable(offsets), _immutable(active)
            self._scratch = (np.zeros(_TILE, np.uint16), np.full(_TILE, -1, np.int16),
                             np.zeros(_TILE, np.int16), np.zeros(_TILE, np.uint16))
            self._positions = np.arange(_TILE, dtype=np.int16)
            static_bytes = self._offsets.nbytes + self._active.nbytes
        else:
            windows = []
            for first in range(0, len(self._ranges), window_tiles):
                ranges = self._ranges[first:first + window_tiles]
                output_indices = np.array([begin for begin, _ in ranges], np.intp)[:, None] + np.arange(_TILE, dtype=np.intp)
                count = ranges[-1][1] - ranges[-1][0]
                output_indices[-1, count:] = 0  # Safe gather, then zero padded old output.
                local_offsets = offsets[output_indices]
                valid_slots = local_offsets >= 0
                valid_slots[-1, count:] = False
                idx = np.where(valid_slots, np.arange(_TILE, dtype=np.int16), -1).astype(np.int16)
                enabled_slots = valid_slots.astype(np.int16)
                positions = np.flatnonzero(valid_slots)
                source_offsets = local_offsets.ravel()[positions]
                windows.append((ranges, *(_immutable(a) for a in
                    (idx, enabled_slots, positions, source_offsets, output_indices))))
            self._windows = tuple(windows)
            self._offsets = self._active = self._positions = None
            self._scratch = (np.zeros((capacity, _TILE), np.uint16),
                             np.zeros((capacity, _TILE), np.uint16))
            static_bytes = sum(a.nbytes for window in self._windows for a in window[1:])
        self._closed = self._busy = self._poisoned = False
        self._retained = None
        self._last_report = None
        self._completed_ranges = ()
        self._stats = dict(plan_bytes=sum(v.nbytes for v in self._snapshots) +
                           static_bytes,
                           scratch_bytes=sum(v.nbytes for v in self._scratch) +
                           (self._positions.nbytes if self._positions is not None else 0),
                           active_tiles=int(np.count_nonzero(active)),
                           skipped_tiles=int(active.size - np.count_nonzero(active)),
                           preparation=preparation, window_tiles=capacity,
                           bookkeeping=bookkeeping, probes=probes,
                           # Conservative ndarray payload bound includes gather,
                           # np.take's checked out buffer, and legacy masks/indexing.
                           execution_scratch_bound_bytes=(16 if preparation == 'window' else 24) * capacity * _TILE,
                           completion_metadata_bytes=sys.getsizeof(self._ranges) + sum(
                               sys.getsizeof(pair) + sum(sys.getsizeof(v) for v in pair)
                               for pair in self._ranges),
                           completion_snapshot_bound_bytes=sys.getsizeof(()) + len(self._ranges) * np.dtype(np.intp).itemsize,
                           plan_build_ns=time.perf_counter_ns() - start,
                           executions=0)

    @property
    def descriptor(self):
        """Detached metadata, never a writable access/offset schema."""
        return dict(self._descriptor)

    @property
    def stats(self):
        """Persistent ndarray payload bytes, excluding Python headers/temporaries."""
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
        if self._closed or self._bookkeeping == 'eager':
            return self._completed_ranges
        return self._ranges[:self._cursor]

    @property
    def last_report(self):
        if self._last_report is None:
            return None
        return dict(self._last_report, timings_ns=dict(self._last_report['timings_ns']))

    def close(self):
        if self._busy:
            raise ValueError('prepared plan: execution pending; close cannot release borrowed storage')
        if self._poisoned:
            raise ValueError('prepared plan: completion unknown; poisoned storage retained until runtime/process teardown')
        if self._closed:
            return
        self._completed_ranges = self.completed_ranges
        self._closed = True
        self._snapshots = self._offsets = self._active = self._scratch = self._positions = None
        self._windows = self._ranges = ()

    def _prepared_tiles(self, src, dst, times):
        """Prepare bounded batches; yield only serial 256-slot device calls."""
        probes = self._probes
        if self._preparation == 'window':
            packed, old = self._scratch
            for ranges, idx, enabled, positions, offsets, outputs in self._windows:
                if probes:
                    stamp = time.perf_counter_ns()
                size = len(ranges)
                current_packed, current_old = packed[:size], old[:size]
                current_packed.fill(0)
                np.take(dst, outputs, out=current_old)
                count = ranges[-1][1] - ranges[-1][0]
                current_old[-1, count:] = 0
                if probes:
                    times['prepare'] += time.perf_counter_ns() - stamp
                    stamp = time.perf_counter_ns()
                current_packed.ravel()[positions] = src[offsets]
                if probes:
                    times['pack'] += time.perf_counter_ns() - stamp
                for row, (begin, end) in enumerate(ranges):
                    yield begin, end, packed[row], idx[row], enabled[row], old[row]
        else:
            packed, idx, enabled, old = self._scratch
            for tile_number, active in enumerate(self._active):
                if not active:
                    continue
                begin = tile_number * _TILE
                end = min(begin + _TILE, self._descriptor['items'])
                count = end - begin
                if probes:
                    stamp = time.perf_counter_ns()
                packed.fill(0)
                idx.fill(-1)
                enabled.fill(0)
                old.fill(0)
                offsets = self._offsets[begin:end]
                valid = offsets >= 0
                idx[:count][valid] = self._positions[:count][valid]
                enabled[:count] = valid
                old[:count] = dst[begin:end]
                if probes:
                    times['prepare'] += time.perf_counter_ns() - stamp
                    stamp = time.perf_counter_ns()
                packed[:count][valid] = src[offsets[valid]]
                if probes:
                    times['pack'] += time.perf_counter_ns() - stamp
                yield begin, end, packed, idx, enabled, old

    async def execute(self, runtime, src, dst):
        """Borrow src/dst exclusively through completion; reject concurrent reuse.

        The caller must not mutate either array or aliases while this coroutine
        is pending. Only generated tile results are copied into dst. A failure
        reports completed output ranges and retains uncertain in-flight storage.
        """
        start = time.perf_counter_ns()
        if self._closed or self._poisoned or self._busy:
            raise ValueError('prepared plan: closed, poisoned, or already executing')
        _array(src, self._descriptor['source_elements'], 'uint16', 'src')
        _array(dst, self._descriptor['items'], 'uint16', 'dst', writable=True)
        _disjoint((src, dst))
        probes = self._probes
        times = dict.fromkeys(('precheck', 'pack', 'prepare', 'device', 'copyback'),
                              0 if probes else None)
        if probes:
            times['precheck'] = time.perf_counter_ns() - start
        self._busy = True
        self._cursor = 0
        self._completed_ranges = ()
        # Diagnostic preparation-only ablation: preserve R010's list growth,
        # new range-pair allocation, and tuple copy at every completion.
        completed = [] if self._bookkeeping == 'eager' else None
        failed_range = None
        in_flight = False
        try:
            for begin, end, packed, idx, enabled, old_dst in self._prepared_tiles(src, dst, times):
                count = end - begin
                failed_range = (begin, end)
                if probes:
                    stamp = time.perf_counter_ns()
                in_flight = True
                self._retained = (runtime, src, dst)
                _INFLIGHT_PLANS.add(self)
                await self._tile(runtime, packed, idx, enabled, old_dst)
                in_flight = False
                _INFLIGHT_PLANS.discard(self)
                self._retained = None
                if probes:
                    times['device'] += time.perf_counter_ns() - stamp
                    stamp = time.perf_counter_ns()
                # Copy every real returned slot; loaded-value decisions and
                # conditional stores belong exclusively to the device.
                dst[begin:end] = old_dst[:count]
                self._cursor += 1
                if completed is not None:
                    completed.append((begin, end))
                    self._completed_ranges = tuple(completed)
                if probes:
                    times['copyback'] += time.perf_counter_ns() - stamp
            self._stats['executions'] += 1
            times['total'] = time.perf_counter_ns() - start
            report = dict(completed_ranges=(tuple(completed) if completed is not None
                                            else self.completed_ranges),
                          active_tiles=self._stats['active_tiles'],
                          skipped_tiles=self._stats['skipped_tiles'], timings_ns=times)
            self._last_report = report
            return self.last_report
        except BaseException as error:
            # Preparation can raise while advancing the generator, before it
            # yields the next tile. The cursor identifies that uncopied range.
            if not in_flight:
                failed_range = self._ranges[self._cursor] if self._cursor < len(self._ranges) else None
            if in_flight:
                self._poisoned = True
                self._retained = (runtime, src, dst)
                _POISONED_PLANS.append(self)
                _INFLIGHT_PLANS.discard(self)
            # Preserve cancellation/control-flow exception type for both asyncio
            # and cocotb. Enrich it with the same partial-result contract.
            if not isinstance(error, Exception):
                error.completed_ranges = self.completed_ranges
                error.failed_range = failed_range
                error.poisoned = self._poisoned
                raise
            failure = PlanExecutionError(str(error), self.completed_ranges, failed_range)
            failure.poisoned = self._poisoned
            raise failure from error
        finally:
            self._busy = False


def bind_prepared_entry(unchecked, descriptor):
    """Compiler-internal binding, not a public descriptor/proof import API.

    Generated modules embed this version and an identity derived from their
    proved compiler artifacts. Each binding also creates a process-local opaque
    identity, so even identically compiled independent modules reject foreign
    plans. Calling this internal function manually grants no compiler trust.
    """
    descriptor = _validate_descriptor(descriptor)
    if not callable(unchecked):
        raise ValueError('prepared entry: generated tile callable is required')
    identity = object()

    def make_plan(x, y, enabled, *, preparation='window', window_tiles=32,
                  bookkeeping='cursor', probes=True):
        try:
            return _PreparedPlan(identity, descriptor, unchecked, x, y, enabled,
                                 preparation=preparation, window_tiles=window_tiles,
                                 bookkeeping=bookkeeping, probes=probes)
        except MemoryError as error:
            raise MemoryError('prepared plan: allocation failed within checked resource bounds; reduce query count or window_tiles') from error

    async def execute_plan(runtime, plan, src, dst):
        if type(plan) is not _PreparedPlan or plan._identity is not identity:
            raise ValueError('prepared entry: foreign plan/module identity')
        return await plan.execute(runtime, src, dst)

    async def once_entry(runtime, *buffers):
        if len(buffers) != 5:
            raise ValueError('prepared entry: original C ABI requires five buffers')
        src, x, y, enabled, dst = (buffers[descriptor[key]] for key in
                                  ('source_argument', 'x_argument', 'y_argument',
                                   'enable_argument', 'destination_argument'))
        for value, name, size, dtype, writable in (
                (src, 'src', descriptor['source_elements'], 'uint16', False),
                (x, 'x', descriptor['items'], 'int16', False),
                (y, 'y', descriptor['items'], 'int16', False),
                (enabled, 'enable', descriptor['items'], 'int16', False),
                (dst, 'dst', descriptor['items'], 'uint16', True)):
            _array(value, size, dtype, name, writable)
        _disjoint(buffers)
        plan = make_plan(x, y, enabled)
        try:
            return await execute_plan(runtime, plan, src, dst)
        finally:
            if not plan.poisoned:
                plan.close()

    once_entry.__name__ = descriptor['entry']
    return make_plan, execute_plan, once_entry
