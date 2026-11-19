"""Cocotb tests of actual generated code, host destinations and device traffic."""
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import struct
import sys

import cocotb
from cocotb.triggers import Event, RisingEdge, Timer
import numpy as np
from rtl_device import start_device
import test_runif
from test_runif import DeviceEvent, DeviceRuntime
from workload import frame, reference


def now():
    return int(cocotb.utils.get_sim_time(units='ns'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quotas(rt):
    def regions(spans):
        return [[int(s.address), int(s.size)] for s in spans]
    return [[m.tile_num_idle, m.cfg_spm_idle, regions(m.tile_free_list), regions(m.cfg_free_list)]
            for m in rt._device_managers]


class RecordingRuntime(DeviceRuntime):
    """Observe real completion, device configuration, effective release and D2H."""
    async def finish_execution(self, handler):
        await super().finish_execution(handler)
        reg = self._device_exereg[handler.device_id]
        mask = ((1 << handler.total_tiles) - 1) << handler.offset_starting_tile
        done = int.from_bytes(reg.exe_reg, 'little')
        assert mask and done & mask == mask and reg.resp_time > handler.exes_time
        self.done.append(dict(stream=handler.stream_id, ns=now(), mask=mask, status=done))

    async def _axi_read(self, address, total_bytes):
        start = now()
        data = await super()._axi_read(address, total_bytes)
        assert len(data) == total_bytes
        self.reads.append((address, start, now(), bytes(data)))
        return data

    async def _axi_write(self, address, data_bytes):
        await super()._axi_write(address, data_bytes)
        if address >= self._device_managers[0].device.config_start_address:
            # Single device, no other concurrent device leases. Its config offset is zero.
            assert data_bytes == self.bit_payload, 'runtime configuration differs from final emitted payload'
            self.config_writes.append(dict(ns=now(), address=address, bytes=len(data_bytes),
                                            sha256=hashlib.sha256(data_bytes).hexdigest()))

    async def release_config(self, handler):
        valid = handler.valid
        await super().release_config(handler)
        if valid:
            self.releases += 1


class Destination(np.ndarray):
    """Record successful slice assignments to this public array in fault tests."""
    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if hasattr(self, 'writes'):
            self.writes.append([index.start, index.stop])


class Scene:
    @classmethod
    async def start(cls, dut, name, many=False, mapped=None, bit=None):
        self = cls()
        self.dut, self.name = dut, name
        self.work = Path(os.environ['ADORA_RTL_OUTPUT'])
        mapped = Path(mapped or os.environ['ADORA_RTL_MANY' if many else 'ADORA_RTL_MAPPED'])
        bit = Path(bit or os.environ['ADORA_RTL_MANY_BIT' if many else 'ADORA_RTL_BIT'])
        self.module = load(mapped, 'adora_' + name)
        self.rt = await start_device(dut, runtime_class=RecordingRuntime)
        words = [tuple(int(v, 16) for v in line.split()) for line in bit.read_text().splitlines() if line.strip()]
        # The mapper dumps config.bit before the emitter assigns scratchpad addresses.
        # GenerateCGRAConfig then emits the executable post-allocation table.
        raw_python = bit.parents[1] / 'unchecked.py'
        assigns = [s.value for s in ast.parse(raw_python.read_bytes()).body if isinstance(s, ast.Assign)
                   and getattr(s.targets[0], 'id', '') == 'cfgbit__adora_large_tile']
        table, = [ast.literal_eval(rhs) for rhs in assigns]
        loaded = [(addr, lo | hi << 16) for lo, hi, addr in zip(table[::3], table[1::3], table[2::3])]
        assert len(table) == 3 * len(words) and [a for a, _ in words] == [a for a, _ in loaded]
        allocations = [{'address': a, 'mapped': old, 'emitted': new}
                       for (a, old), (_, new) in zip(words, loaded) if old != new]
        self.rt.bit_payload = b''.join(struct.pack('<IH', data, addr) for addr, data in loaded)
        self.rt.done, self.rt.reads, self.rt.config_writes, self.rt.releases = [], [], [], 0
        self.initial = quotas(self.rt)
        self.arrays, self.frames = {}, []
        assert Path(test_runif.__file__).resolve() == Path(__file__).resolve().parents[3] / 'runtime/python/test_runif.py'
        self.identity = {label: dict(path=str(p), sha256=digest(p)) for label, p in
                         [('runtime', Path(test_runif.__file__)), ('generated', mapped), ('bit', bit), ('post_allocation_emitter', raw_python)]}
        self.short = not many
        self.plan = self.module.make_plan(*frame(0, self.short)[1:4])
        self.record = dict(scenario=name, identity=self.identity, initial_reset_count=1,
                           final_allocation_changes=allocations)
        # The AXI master and scheduler normally log every transfer.
        for name in logging.root.manager.loggerDict.copy():
            if name.startswith(('test_runif', 'DeviceManager', 'StreamScheduler', 'cocotb.tb',
                                'cocotb.test_cgra_cstore.axi')):
                logging.getLogger(name).setLevel(logging.WARNING)
        return self

    def buffers(self, number=0, track=False):
        src, x, y, en, old = frame(number, self.short)
        self.old = old
        self.expected = reference(src, x, y, en, old)
        self.guards = np.arange(17, dtype=np.uint16) + 53117
        self.storage = np.concatenate([old, self.guards])
        self.dst = self.storage[:old.size]
        if track:
            self.dst = self.dst.view(Destination)
            self.dst.writes = []
        self.src = src

    async def run(self):
        return await self.module.execute_plan(self.rt, self.plan, self.src, self.dst)

    def check(self, number, count, key):
        np.testing.assert_array_equal(self.dst[:count], self.expected[:count])
        np.testing.assert_array_equal(self.dst[count:], self.old[count:])
        np.testing.assert_array_equal(self.storage[self.old.size:], self.guards)
        self.arrays[key + '_host'] = np.array(self.dst)
        self.arrays[key + '_guards'] = self.storage[self.old.size:].copy()
        self.frames.append(dict(frame=number, array=key + '_host', checked_prefix=count,
                                published_items=sum(b - a for a, b in self.plan.completed_ranges),
                                ranges=self.plan.completed_ranges, guards=17,
                                src_sha256=hashlib.sha256(self.src.tobytes()).hexdigest(),
                                old_sha256=hashlib.sha256(self.old.tobytes()).hexdigest()))

    def save(self, **extra):
        records = []
        for index, (address, start, end, data) in enumerate(self.rt.reads):
            key = 'd2h_' + str(index)
            self.arrays[key] = np.frombuffer(data, dtype='<u2')
            records.append(dict(address=address, start_ns=start, end_ns=end, array=key))
        np.savez_compressed(self.work / (self.name + '.npz'), **self.arrays)
        record = dict(self.record, frames=self.frames, d2h=records, completions=self.rt.done,
                      configurations=self.rt.config_writes, valid_lease_returns=self.rt.releases,
                      initial_quotas=self.initial, final_quotas=quotas(self.rt),
                      registered_streams=list(self.rt._streams), **extra)
        (self.work / (self.name + '.json')).write_text(json.dumps(record, indent=2) + '\n')

    def quiescent(self, blocks):
        assert len(self.rt.reads) == len(self.rt.done) == len(self.rt.config_writes) == self.rt.releases == blocks
        assert not self.rt._streams and quotas(self.rt) == self.initial


async def monitor_tail(dut, events, address, bank_bytes):
    """Sample accepted AXI reads and active output SRAM writes at rising edges."""
    bank = address // bank_bytes
    signals = {n: getattr(dut.uut, f'cgra_io_srams_iob_{bank}_{n}') for n in ('en','we','addr','din')}
    while True:
        await RisingEdge(dut.clk)
        assert not int(dut.rst.value)
        if int(dut.axi_arvalid.value) and int(dut.axi_arready.value):
            events.append(dict(kind='AR', ns=now(), address=int(dut.axi_araddr.value),
                               length=int(dut.axi_arlen.value), size=int(dut.axi_arsize.value)))
        if int(dut.axi_rvalid.value) and int(dut.axi_rready.value):
            events.append(dict(kind='R', ns=now(), data=int(dut.axi_rdata.value).to_bytes(
                len(dut.axi_rdata)//8, 'little').hex(), last=int(dut.axi_rlast.value),
                resp=int(dut.axi_rresp.value)))
        if int(signals['en'].value) and int(signals['we'].value):
            events.append(dict(kind='SRAM_WRITE', ns=now(), bank=bank, word=int(signals['addr'].value),
                               byte_enable=int(signals['we'].value), value=int(signals['din'].value)))


@cocotb.test(timeout_time=250000, timeout_unit='ns')
async def short_tail(dut):
    scene = await Scene.start(dut, 'short-tail')
    rt = scene.rt
    events = []
    real_retire = rt.retire_completed_stream
    bank_bytes = json.loads(Path(os.environ['ADORA_RTL_ADG']).read_text())['iob_spad_bank_size']
    monitor = None

    async def observe_second_block(stream):
        nonlocal monitor
        await real_retire(stream)
        if len(rt.reads) == 1:
            monitor = cocotb.start_soon(monitor_tail(dut, events, rt.reads[0][0], bank_bytes))
        elif len(rt.reads) == 2:
            monitor.kill()
    rt.retire_completed_stream = observe_second_block
    for number in range(2):
        scene.buffers(number)
        report = await scene.run()
        assert report['completed_ranges'] == ((0, 256), (256, 353))
        scene.check(number, 353, f'frame{number}')
        first, last = rt.reads[number * 2:number * 2 + 2]
        np.testing.assert_array_equal(np.frombuffer(first[3], '<u2'), scene.dst[:256])
        raw_tail = np.frombuffer(last[3], '<u2')
        np.testing.assert_array_equal(raw_tail[:97], scene.dst[256:])
        np.testing.assert_array_equal(raw_tail[97:], np.zeros(159, np.uint16))
    address, start, end, raw = rt.reads[1]
    ar = [e for e in events if e['kind'] == 'AR']
    beats = [e for e in events if e['kind'] == 'R']
    assert ar and ar[0]['address'] == address
    offset = 0
    for entry in ar:
        assert entry['address'] == address + offset
        offset += (entry['length'] + 1) << entry['size']
    assert offset == len(raw) and sum(e['last'] for e in beats) == len(ar)
    assert all(e['resp'] == 0 and start <= e['ns'] <= end for e in beats)
    assert b''.join(bytes.fromhex(e['data']) for e in beats) == raw
    writes = [e for e in events if e['kind'] == 'SRAM_WRITE']
    assert [e['word'] for e in writes] == list(range(address % bank_bytes // 2, address % bank_bytes // 2 + 97))
    assert all(e['byte_enable'] == 3 for e in writes)
    assert [e['value'] for e in writes] == np.frombuffer(raw, '<u2')[:97].tolist()
    assert writes[-1]['ns'] <= rt.done[1]['ns'] <= start
    scene.quiescent(4)
    scene.plan.close()
    scene.save(tail_frame=0, tail_block=1, events=events)


async def until(predicate):
    while not predicate():
        await Timer(2, units='ns')


@cocotb.test(timeout_time=100000, timeout_unit='ns')
async def snapshot_wait(dut):
    scene = await Scene.start(dut, 'snapshot-wait')
    rt = scene.rt
    early = [rt.create_stream(), rt.create_stream()]
    gates = [DeviceEvent(), DeviceEvent(), DeviceEvent()]
    for stream, gate in zip(early, gates):
        await stream.wait_event(gate)
    await until(lambda: all(g._event._pending for g in gates[:2]))
    reached, release = Event(), Event()
    retire = rt.retire_completed_stream

    async def suspend(stream):
        if not reached.is_set():
            reached.set()
            await release.wait()
        await retire(stream)
    rt.retire_completed_stream = suspend
    scene.buffers()
    private = cocotb.start_soon(scene.run())
    await reached.wait()
    snapshot = list(rt._streams)
    wait = cocotb.start_soon(rt.synchronize_all())
    await until(lambda: all(s._fences for s in early))
    late = rt.create_stream()
    await late.wait_event(gates[2])
    await until(lambda: gates[2]._event._pending)
    release.set()
    await private
    scene.check(0, 353, 'frame')
    assert not wait.done()
    gates[0].set()
    await early[0].synchronize()
    assert not wait.done() and not gates[1].is_set()
    gates[1].set()
    await wait
    assert not gates[2].is_set() and late._running
    gates[2].set()
    for stream in [*early, late]:
        await rt.destory_stream(stream)
    scene.quiescent(2)
    scene.save(snapshot=snapshot, early=[s.id for s in early], late=late.id,
               checked_wait_after_E=True, completed_with_L_pending=True)


@cocotb.test(timeout_time=100000, timeout_unit='ns')
async def cleanup_retry(dut):
    scene = await Scene.start(dut, 'cleanup-retry')
    scene.buffers(track=True)
    rt = scene.rt
    retire = rt.retire_completed_stream

    async def reject_second(stream):
        if len(rt.reads) == 2:
            assert len(rt.done) == 2 and scene.plan.completed_ranges == ((0, 256), (256, 353))
            raise RuntimeError('injected failure after second D2H')
        await retire(stream)
    rt.retire_completed_stream = reject_second
    try:
        await scene.run()
    except Exception as error:
        assert type(error).__name__ == 'PlanCleanupError' and error.block == (256, 353)
        assert error.plan is scene.plan and str(error.__cause__) == 'injected failure after second D2H'
    else:
        raise AssertionError('missing cleanup error')
    scene.check(0, 353, 'at_error')
    assert scene.plan.cleanup_pending and rt._streams[scene.plan.pending_stream.id] is scene.plan.pending_stream
    assert scene.dst.writes == [[0, 256], [256, 353]]
    assert rt.releases == 2
    del rt.retire_completed_stream
    await scene.plan.retry_cleanup()
    await scene.plan.retry_cleanup()
    assert not scene.plan.cleanup_pending and scene.plan._retained is None
    assert scene.dst.writes == [[0, 256], [256, 353]]
    scene.quiescent(2)
    scene.save(public_writes=scene.dst.writes, error='PlanCleanupError', retries=2)


async def incomplete(dut, cancel):
    scene = await Scene.start(dut, 'cancel' if cancel else 'unknown')
    scene.buffers(track=True)
    rt, plan = scene.rt, scene.plan
    original = plan._tile
    gate, reached = Event(), Event()
    child = None

    async def boundary(runtime, *args, _adora_stream):
        nonlocal child
        if not plan.completed_ranges:
            return await original(runtime, *args, _adora_stream=_adora_stream)
        # Hold the real worker after accepted inputs and before physical execution.
        # The parent can lose its wait while the independently scheduled child survives.
        start = _adora_stream.execution_start
        device_gate = DeviceEvent()
        async def gated_start():
            await _adora_stream.wait_event(device_gate)
            await start()
        _adora_stream.execution_start = gated_start
        child = cocotb.start_soon(original(runtime, *args, _adora_stream=_adora_stream))
        await until(lambda: device_gate._event._pending)
        assert _adora_stream._handler.valid and not _adora_stream._handler.running
        assert len(rt.reads) == len(rt.done) == 1
        gates.append(device_gate)
        reached.set()
        if cancel:
            await gate.wait()
        raise RuntimeError('lost second block completion')
    plan._tile = boundary
    gates = []
    errors = []
    async def call():
        try:
            await scene.run()
        except Exception as exc:
            errors.append(exc)
    parent = cocotb.start_soon(call())
    await reached.wait()
    if cancel:
        parent.cancel('cancel while second device stream is held')
        await Timer(2, units='ns')
        assert parent.done() and parent.cancelled() and plan.pending
    else:
        await parent
        assert len(errors) == 1 and type(errors[0]).__name__ == 'PlanExecutionError'
        assert errors[0].failed_range == (256, 353) and plan.poisoned
    assert tuple(map(id, plan._retained)) == tuple(map(id, (rt, scene.src, scene.dst, plan.pending_stream)))
    assert rt._streams[plan.pending_stream.id] is plan.pending_stream
    scene.check(0, 256, 'at_boundary')
    gates[0].set()
    await child
    assert len(rt.reads) == len(rt.done) == rt.releases == 2
    assert plan.completed_ranges == ((0, 256),) and plan._retained is not None
    assert scene.dst.writes == [[0, 256]]
    scene.check(0, 256, 'after_late_d2h')
    np.testing.assert_array_equal(np.frombuffer(rt.reads[1][3], '<u2')[:97], scene.expected[256:])
    try:
        plan.close()
    except ValueError as exc:
        rejection = str(exc)
    else:
        raise AssertionError('uncertain ownership released')
    scene.save(public_writes=scene.dst.writes, cancelled=cancel, poisoned=plan.poisoned,
               pending=plan.pending, close_rejection=rejection, borrowed=True)


@cocotb.test(timeout_time=100000, timeout_unit='ns')
async def unknown_completion(dut):
    await incomplete(dut, False)


@cocotb.test(timeout_time=100000, timeout_unit='ns')
async def task_cancel(dut):
    await incomplete(dut, True)


@cocotb.test(timeout_time=4000000, timeout_unit='ns')
async def steady(dut):
    scene = await Scene.start(dut, 'steady', many=True)
    for i in range(8):
        scene.buffers(i)
        report = await scene.run()
        assert len(report['completed_ranges']) == 16
        scene.check(i, scene.dst.size, f'frame{i}')
        scene.quiescent(16 * (i + 1))
    scene.save()


@cocotb.test(timeout_time=200000, timeout_unit='ns')
async def old_module(dut):
    scene = await Scene.start(dut, 'old-module', mapped=os.environ['ADORA_LEGACY_MAPPED'],
                              bit=os.environ['ADORA_LEGACY_BIT'])
    scene.buffers()
    await scene.run()
    scene.check(0, 353, 'frame')
    assert len(scene.rt._streams) == len(scene.rt.reads) == scene.rt.releases == 2
    assert quotas(scene.rt) == scene.initial
    scene.save(legacy_streams_registered=2)


@cocotb.test(timeout_time=20000, timeout_unit='ns')
async def old_runtime(dut):
    # Isolated historical pairing: the fresh module imports actual baseline classes.
    import unittest.mock
    legacy = load(os.environ['ADORA_LEGACY_RUNTIME'], 'legacy_runtime')
    with unittest.mock.patch.dict(sys.modules, test_runif=legacy):
        mapped = load(os.environ['ADORA_RTL_MAPPED'], 'new_entry_with_legacy_runtime')
        rt = await start_device(dut, runtime_class=legacy.DeviceRuntime)
        src, x, y, enabled, old = frame(0)
        dst = old.copy()
        plan = mapped.make_plan(x, y, enabled)
        assert not hasattr(rt, 'PRIVATE_STREAM_OWNERSHIP_VERSION')
        calls = []
        # Observe real transfer entry points. Any call invalidates early rejection.
        for name in ('_axi_read','_axi_write','create_stream'):
            fn = getattr(rt, name)
            def observe(*args, operation=name, original=fn, **kw):
                calls.append(operation)
                return original(*args, **kw)
            setattr(rt, name, observe)
        try:
            await mapped.execute_plan(rt, plan, src, dst)
        except ValueError as error:
            message = str(error)
            assert 'ownership version 1' in message
        else:
            raise AssertionError('new module accepted incompatible runtime')
        np.testing.assert_array_equal(dst, old)
        assert not rt._streams and not calls
        plan.close()
        (Path(os.environ['ADORA_RTL_OUTPUT']) / 'old-runtime.json').write_text(json.dumps(dict(
            error=message, transfers=calls, registered_streams=list(rt._streams),
            runtime_sha256=digest(legacy.__file__), module_sha256=digest(os.environ['ADORA_RTL_MAPPED'])), indent=2))
