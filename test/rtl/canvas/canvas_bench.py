"""Exercise one-reset warm canvas plans and capture passive pre-edge wire data."""
import ctypes
import functools
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import time

import cocotb
from cocotb.triggers import RisingEdge, with_timeout
import numpy as np
from rtl_device import start_device
from test_runif import DeviceRuntime

from canvas_prepare import daily_frames, oracle


def cycle():
    return int(cocotb.utils.get_sim_time(units='ns')) // 2


def frames(d, contract, path):
    if path:
        with np.load(path, allow_pickle=False) as source:
            for i in range(int(source['frames'])):
                arrays = tuple(
                    source[f'{name}_{i}'] for name in ('src', 'x', 'y', 'enabled', 'old')
                )
                yield arrays, source[f'gold_{i}']
    else:
        for arrays in daily_frames(contract):
            yield arrays, oracle(arrays, d)[0]


class WaitingRuntime(DeviceRuntime):
    async def finish_execution(self, handler):
        await super().finish_execution(handler)
        reg = self._device_exereg[handler.device_id]
        mask = ((1 << handler.total_tiles) - 1) << handler.offset_starting_tile
        status = int.from_bytes(reg.exe_reg, 'little')
        assert status & mask == mask and reg.resp_time > handler.exes_time
        self.wires.canvas_wait(cycle(), mask, status, int(reg.resp_time) // 2)


@cocotb.test()
async def canvas_mask(dut):
    work = Path(os.environ['ADORA_RTL_OUTPUT'])
    spec = importlib.util.spec_from_file_location(
        'mapped_canvas', os.environ['ADORA_CANVAS_MAPPED']
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = json.loads((work / 'bindings.json').read_text())
    contract = meta['contract']
    d = contract['defines']
    wires = ctypes.CDLL(str(work / 'observer.so'))
    wires.canvas_open.restype = ctypes.c_char_p
    missing = wires.canvas_open(
        os.fsencode(work / 'wire.bin'),
        os.environ['TOPLEVEL'].encode(),
        meta['banks'],
        meta['tiles'],
    )
    assert not missing, f'missing passive VPI signal {missing}'

    async def monitor():
        while True:
            await RisingEdge(dut.clk)
            wires.canvas_tick(cycle())

    task = cocotb.start_soon(monitor())
    runtime = await start_device(dut, runtime_class=WaitingRuntime)
    runtime.wires = wires
    for name in list(logging.root.manager.loggerDict):
        if name.startswith(
            (
                'test_runif',
                'DeviceManager',
                'StreamScheduler',
                'cocotb.tb',
                'cocotb.test_cgra_cstore.axi',
            )
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    current = {}
    records = []
    sampled = set()

    def observe(fn, kernel):
        @functools.wraps(fn)
        async def wrapped(rt, *arrays):
            i, block = current['frame'], current['block']
            guard = current['guard'][block * 256 : (block + 1) * 256]
            mask = current['gold'][block * 256 : (block + 1) * 256]
            assert kernel == ('regular' if guard.any() else 'old-only')
            bank = meta['kernels'][kernel]['roles']['mask']['bank']
            sample_key = (i, kernel, len(mask))
            detail = block < 2 or sample_key not in sampled
            sampled.add(sample_key)
            wires.canvas_call(cycle(), 1, i, block, bank, int(detail))
            await with_timeout(fn(rt, *arrays), 10000, 'ns')
            np.testing.assert_array_equal(arrays[-1][: len(mask)], mask)
            assert np.all(arrays[-1][len(mask) :] == 0)
            wires.canvas_call(cycle(), 0, i, block, bank, int(detail))
            current['block'] += 1
            if block % 50 == 0:
                (work / 'progress.json').write_text(
                    json.dumps(dict(frame=i, completed=block + 1, cycle=cycle(), wall=time.time()))
                )

        return wrapped

    module._regular_tile = observe(module._regular_tile, 'regular')
    module._old_tile = observe(module._old_tile, 'old-only')
    plan = None
    try:
        for i, (arrays, gold) in enumerate(frames(d, contract, os.environ['ADORA_CANVAS_FRAMES'])):
            src, x, y, en, old = arrays
            expected, guard = oracle(arrays, d)
            np.testing.assert_array_equal(expected, gold)
            if i % 2 == 0:
                if plan:
                    plan.close()
                plan = module.make_plan(x, y, en)
                geometry = (x.copy(), y.copy(), en.copy())
            else:
                for previous, now in zip(geometry, (x, y, en)):
                    np.testing.assert_array_equal(previous, now)
            public = np.full(len(x) + 8, 0x5AA5, np.uint16)
            current.update(frame=i, block=0, guard=guard, gold=gold)
            start = cycle()
            stamp = time.monotonic_ns()
            report = await module.execute_plan(runtime, plan, src, old, public[4:-4])
            np.testing.assert_array_equal(public[4:-4], gold)
            assert np.all(public[:4] == 0x5AA5) and np.all(public[-4:] == 0x5AA5)
            assert list(plan.completed_ranges) == [
                (b, min(b + 256, len(x))) for b in range(0, len(x), 256)
            ]
            records.append(
                dict(
                    frame=i,
                    cycle_begin=start,
                    cycle_end=cycle(),
                    wall_ns=time.monotonic_ns() - stamp,
                    calls=current['block'],
                    regular=report['regular_tiles'],
                    old_only=report['old_only_tiles'],
                    mask_sha256=hashlib.sha256(public[4:-4].astype('<u2').tobytes()).hexdigest(),
                )
            )
            (work / 'frames.json').write_text(json.dumps(records, indent=2) + '\n')
        plan.close()
    finally:
        task.kill()
        wires.canvas_close()
