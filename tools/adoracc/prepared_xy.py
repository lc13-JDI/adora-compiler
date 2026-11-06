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

SUPPORT_VERSION = 1
SCHEMA = 'adora.guarded-inplace-large.v1'
# Conservative allocation limits; checked before any user-sized allocation.
MAX_ELEMENTS = 16 * 1024 * 1024
_TILE = 256
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


class _PreparedPlan:
    __slots__ = ('_identity', '_descriptor', '_tile', '_snapshots', '_offsets',
                 '_active', '_scratch', '_positions', '_closed', '_busy',
                 '_poisoned', '_retained', '_stats', '_last_report', '_completed_ranges')

    def __init__(self, identity, descriptor, tile, x, y, enabled):
        start = time.perf_counter_ns()
        items = descriptor['items']
        for name, value in (('x', x), ('y', y), ('enable', enabled)):
            _array(value, items, 'int16', name)
        _disjoint((x, y, enabled))
        self._identity = identity
        self._descriptor = descriptor
        self._tile = tile
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
        self._offsets = np.frombuffer(offsets.tobytes(), dtype=np.int64)
        active = np.array([np.any(valid[start:start + _TILE])
                           for start in range(0, items, _TILE)], dtype=np.bool_)
        self._active = np.frombuffer(active.tobytes(), dtype=np.bool_)
        self._scratch = (np.zeros(_TILE, np.uint16), np.full(_TILE, -1, np.int16),
                         np.zeros(_TILE, np.int16), np.zeros(_TILE, np.uint16))
        self._positions = np.arange(_TILE, dtype=np.int16)
        self._closed = self._busy = self._poisoned = False
        self._retained = None
        self._last_report = None
        self._completed_ranges = ()
        self._stats = dict(plan_bytes=sum(v.nbytes for v in self._snapshots) +
                           self._offsets.nbytes + self._active.nbytes,
                           scratch_bytes=sum(v.nbytes for v in self._scratch) + self._positions.nbytes,
                           active_tiles=int(np.count_nonzero(active)),
                           skipped_tiles=int(active.size - np.count_nonzero(active)),
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
        return self._completed_ranges

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
        self._closed = True
        self._snapshots = self._offsets = self._active = self._scratch = self._positions = None

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
        times = dict(precheck=time.perf_counter_ns() - start, pack=0, prepare=0,
                     device=0, copyback=0, total=0)
        self._busy = True
        completed = []
        self._completed_ranges = ()
        failed_range = None
        in_flight = False
        try:
            packed, idx, enabled, old_dst = self._scratch
            for tile_number, active in enumerate(self._active):
                if not active:
                    continue
                begin = tile_number * _TILE
                end = min(begin + _TILE, self._descriptor['items'])
                count = end - begin
                failed_range = (begin, end)
                stamp = time.perf_counter_ns()
                packed.fill(0)
                idx.fill(-1)
                enabled.fill(0)
                old_dst.fill(0)
                offsets = self._offsets[begin:end]
                valid = offsets >= 0
                idx[:count][valid] = self._positions[:count][valid]
                enabled[:count] = valid
                old_dst[:count] = dst[begin:end]
                times['prepare'] += time.perf_counter_ns() - stamp
                stamp = time.perf_counter_ns()
                packed[:count][valid] = src[offsets[valid]]
                times['pack'] += time.perf_counter_ns() - stamp
                stamp = time.perf_counter_ns()
                in_flight = True
                self._retained = (runtime, src, dst)
                _INFLIGHT_PLANS.add(self)
                await self._tile(runtime, packed, idx, enabled, old_dst)
                in_flight = False
                _INFLIGHT_PLANS.discard(self)
                self._retained = None
                times['device'] += time.perf_counter_ns() - stamp
                stamp = time.perf_counter_ns()
                # Copy every real returned slot; loaded-value decisions and
                # conditional stores belong exclusively to the device.
                dst[begin:end] = old_dst[:count]
                completed.append((begin, end))
                self._completed_ranges = tuple(completed)
                times['copyback'] += time.perf_counter_ns() - stamp
            self._stats['executions'] += 1
            times['total'] = time.perf_counter_ns() - start
            report = dict(completed_ranges=tuple(completed),
                          active_tiles=self._stats['active_tiles'],
                          skipped_tiles=self._stats['skipped_tiles'], timings_ns=times)
            self._last_report = report
            return self.last_report
        except BaseException as error:
            if in_flight:
                self._poisoned = True
                self._retained = (runtime, src, dst)
                _POISONED_PLANS.append(self)
                _INFLIGHT_PLANS.discard(self)
            # Preserve cancellation/control-flow exception type for both asyncio
            # and cocotb. Enrich it with the same partial-result contract.
            if not isinstance(error, Exception):
                error.completed_ranges = tuple(completed)
                error.failed_range = failed_range
                error.poisoned = self._poisoned
                raise
            failure = PlanExecutionError(str(error), completed, failed_range)
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

    def make_plan(x, y, enabled):
        return _PreparedPlan(identity, descriptor, unchecked, x, y, enabled)

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
