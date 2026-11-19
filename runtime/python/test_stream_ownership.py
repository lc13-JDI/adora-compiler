"""Real cocotb queues/tasks; hardware resource/status interfaces are minimal doubles."""
import logging
import os
import sys
from types import SimpleNamespace

import cocotb
import cocotb.queue
from cocotb.triggers import Timer, with_timeout
from cocotb.result import SimTimeoutError
if os.environ.get("RUNTIME_UNDER_TEST"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(os.environ["RUNTIME_UNDER_TEST"])))
import test_runif as rt


def runtime():
    obj = rt.DeviceRuntime.__new__(rt.DeviceRuntime)
    obj._streams = {}
    obj._stream_ids = 0
    obj._stream_scheduler_queue_num = 3
    obj.log = logging.getLogger("runtime-unit")
    obj.releases = []
    class Manager:
        tile_free_list = []
        cfg_free_list = []
        def release_config(self, handler):
            assert handler.valid
            obj.releases.append(handler)
    obj._device_managers = [Manager()]
    obj._stream_scheduler = SimpleNamespace(device_state_update=lambda _: None)
    return obj


async def tick():
    await Timer(1, units="ns")


async def expect_error(awaitable, expected):
    # Catch inside the task so cocotb 1.9's immediate-error scheduling cannot
    # classify a deliberately asserted API error as an unhandled worker error.
    async def capture():
        try:
            await awaitable
        except Exception as error:
            return error
        return None
    error = await with_timeout(capture(), 20000, "ns")
    assert error is not None, "operation incorrectly succeeded"
    assert not isinstance(error, SimTimeoutError)
    assert isinstance(error, expected), repr(error)
    return error


@cocotb.test()
async def existing_owner_close_during_global_wait_does_not_mutate_iteration(dut):
    """H's normal tail retirement reproduced through the existing owner API."""
    r = runtime()
    external, private = r.create_stream(), r.create_stream()
    gate = rt.DeviceEvent()
    await external.wait_event(gate)
    await private.set_event(rt.DeviceEvent())
    g = cocotb.start_soon(r.synchronize_all())
    await tick()
    assert not g.done()
    await r.destory_stream(private)
    assert not gate.is_set() and not g.done()
    gate.set()
    await with_timeout(g, 50, "ns")
    await r.destory_stream(external)


@cocotb.test()
async def snapshot_members_and_work_before_first_await(dut):
    r = runtime()
    e, e2, private = (r.create_stream() for _ in range(3))
    gate, gate2, later = (rt.DeviceEvent() for _ in range(3))
    await e.wait_event(gate)
    await e2.wait_event(gate2)
    await private.set_event(rt.DeviceEvent())
    g = cocotb.start_soon(r.synchronize_all())
    await tick()
    assert not g.done()
    await e2.wait_event(later)  # same member, accepted AFTER G's first suspension
    late = r.create_stream()
    await late.wait_event(later)  # new member likewise excluded
    await r.destory_stream(private)  # normal retirement during G's stable snapshot
    gate.set()
    await tick()
    assert not g.done(), "G omitted entry work on E2"
    gate2.set()
    await with_timeout(g, 50, "ns")
    assert not later.is_set()
    assert e2._current_task is not None
    later.set()
    await e.set_event(rt.DeviceEvent())  # explicit external stream remains reusable
    await r.destory_stream(e)
    await r.destory_stream(e2)
    await r.destory_stream(late)
    assert not r.is_device_busy()


@cocotb.test()
async def idle_synchronize_does_not_restart_worker(dut):
    r = runtime()
    stream = r.create_stream()
    starts = []
    original = stream._process_queue
    async def counted():
        starts.append(True)
        await original()
    stream._process_queue = counted
    await stream.synchronize()
    await tick()
    assert starts == [], "empty synchronize started a worker"
    await r.destory_stream(stream)
    await stream.synchronize()  # already retired, completion remains stable
    await tick()
    assert starts == []


@cocotb.test()
async def original_worker_error_propagates_without_cleanup_or_restart(dut):
    r = runtime()
    stream = r.create_stream()
    cause = ValueError("original worker failure")
    class BrokenEvent:
        async def wait(self):
            raise cause
    await stream.wait_event(BrokenEvent())
    error = await expect_error(stream.synchronize(), ValueError)
    assert error is cause
    error = await expect_error(r.synchronize_all(), ValueError)
    assert error is cause
    assert r._streams[stream.id] is stream
    await expect_error(r.retire_completed_stream(stream), ValueError)
    assert r.releases == []


@cocotb.test()
async def retire_rejects_inflight_and_exact_identity_mismatch(dut):
    r, other = runtime(), runtime()
    stream, foreign = r.create_stream(), other.create_stream()
    assert stream.id == foreign.id
    await expect_error(r.destory_stream(foreign), ValueError)
    impostor = rt.DeviceStream(r, stream.id)
    await expect_error(r.destory_stream(impostor), ValueError)
    assert r._streams[stream.id] is stream
    gate = rt.DeviceEvent()
    await stream.wait_event(gate)
    await tick()
    await expect_error(r.retire_completed_stream(stream), RuntimeError)
    assert r._streams[stream.id] is stream
    gate.set()
    await stream.synchronize()
    await r.retire_completed_stream(stream)
    await expect_error(stream.set_event(rt.DeviceEvent()), RuntimeError)
    assert r.PRIVATE_STREAM_OWNERSHIP_VERSION == 1


@cocotb.test()
async def concurrent_retirement_releases_valid_resource_once(dut):
    r = runtime()
    stream = r.create_stream()
    stream._handler.valid = True
    stream._handler.device_id = 0
    await stream.set_event(rt.DeviceEvent())
    await stream.synchronize()
    original = r.release_config
    entered, permit = rt.DeviceEvent(), rt.DeviceEvent()
    async def release(handler):
        entered.set()
        await permit.wait()
        await original(handler)
    r.release_config = release
    first = cocotb.start_soon(r.retire_completed_stream(stream))
    await entered.wait()
    second = cocotb.start_soon(r.retire_completed_stream(stream))
    await tick()
    permit.set()
    await with_timeout(first, 50, "ns")
    await with_timeout(second, 50, "ns")
    await r.retire_completed_stream(stream)
    assert len(r.releases) == 1
    assert not r._streams
    # An old close cannot touch a later object accidentally reusing the ID.
    replacement = rt.DeviceStream(r, stream.id)
    r._streams[stream.id] = replacement
    await r.destory_stream(stream)
    assert r._streams[stream.id] is replacement
    await r.destory_stream(replacement)


@cocotb.test()
async def cleanup_failure_can_retry_after_release_has_taken_effect(dut):
    r = runtime()
    stream = r.create_stream()
    stream._handler.valid, stream._handler.device_id = True, 0
    original = r.release_config
    cause = OSError("post-release return failure")
    async def failed_return(handler):
        await original(handler)
        raise cause
    r.release_config = failed_return
    error = await expect_error(r.retire_completed_stream(stream), OSError)
    assert error is cause
    assert r._streams[stream.id] is stream
    r.release_config = original
    await r.retire_completed_stream(stream)
    assert len(r.releases) == 1
    assert not r._streams


@cocotb.test()
async def transport_queue_error_reaches_stream_fence(dut):
    r = runtime()
    cause = OSError("D2H original failure")
    async def broken_copy(*args):
        raise cause
    queue = rt.AsyncTaskQueue(broken_copy)
    r.add_read_task = lambda *args: queue.add_task(args)
    stream = r.create_stream()
    await stream.memcpyDeviceToHost(rt.DeviceData(0, 4), bytearray(), 4)
    error = await expect_error(stream.synchronize(), OSError)
    assert error is cause
    assert stream._handler.valid is False
    assert r._streams[stream.id] is stream


@cocotb.test()
async def scheduler_fit_error_reaches_stream_fence(dut):
    r = runtime()
    scheduler = rt.StreamScheduler()
    cause = ValueError("invalid configuration in allocator")
    def broken_fit(*args):
        raise cause
    scheduler.add_device(0, broken_fit)
    class ScheduledWork:
        async def wait(self):
            event = await scheduler.put(0, ())
            await event.wait()
    stream = r.create_stream()
    await stream.wait_event(ScheduledWork())
    error = await expect_error(stream.synchronize(), ValueError)
    assert error is cause


@cocotb.test()
async def notification_error_after_allocator_release_does_not_release_twice(dut):
    r = runtime()
    stream = r.create_stream()
    stream._handler.valid, stream._handler.device_id = True, 0
    cause = OSError("scheduler notification failure after free")
    def broken_notify(*args):
        raise cause
    r._stream_scheduler.device_state_update = broken_notify
    error = await expect_error(r.retire_completed_stream(stream), OSError)
    assert error is cause
    assert len(r.releases) == 1
    assert r._streams[stream.id] is stream
    r._stream_scheduler.device_state_update = lambda _: None
    await r.retire_completed_stream(stream)
    assert len(r.releases) == 1, "valid lease returned a second time"


@cocotb.test()
async def queued_transport_errors_wake_all_accepted_waiters(dut):
    cause = OSError("queued DMA failed")
    gate = rt.DeviceEvent()
    async def transfer(*args):
        await gate.wait()
        raise cause
    queue = rt.AsyncTaskQueue(transfer, task_queue_size=3)
    events = [await queue.add_task((index,)) for index in range(3)]
    await tick()
    gate.set()
    for event in events:
        assert await expect_error(event.wait(), OSError) is cause
    assert await expect_error(queue.wait_all(), OSError) is cause
    assert await expect_error(queue.add_task((4,)), OSError) is cause


@cocotb.test()
async def blocked_transport_producers_cannot_bypass_poison(dut):
    cause = OSError("active DMA failed with blocked producers")
    gate = rt.DeviceEvent()
    calls, starts = [], []
    async def transfer(label):
        calls.append(label)
        if label == "A":
            await gate.wait()
            raise cause
    queue = rt.AsyncTaskQueue(transfer, task_queue_size=1)
    original_worker = queue._process_queue
    async def counted_worker():
        starts.append(True)
        await original_worker()
    queue._process_queue = counted_worker
    a = await queue.add_task(("A",))
    await tick()  # A has been removed from the queue and is running behind gate.
    b = await queue.add_task(("B",))
    async def blocked_submit(label):
        try:
            return await queue.add_task((label,))
        except Exception as error:
            return error
    c = cocotb.start_soon(blocked_submit("C"))
    d = cocotb.start_soon(blocked_submit("D"))
    await tick()
    assert calls == ["A"] and queue.size() == 1
    assert not c.done() and not d.done()
    assert len(queue._queue._putters) == 2, "C and D must both suspend in bounded put"
    gate.set()
    assert await expect_error(a.wait(), OSError) is cause
    assert await expect_error(b.wait(), OSError) is cause
    c_result = await with_timeout(c, 50, "ns")
    d_result = await with_timeout(d, 50, "ns")
    await tick()
    cocotb.log.info("bounded poison: DMA=%s worker_starts=%d queued=%d C_original_error=%s D_original_error=%s",
                    calls, len(starts), queue.size(), c_result is cause, d_result is cause)
    assert c_result is cause and d_result is cause
    assert await expect_error(queue.wait_all(), OSError) is cause
    await tick()
    assert calls == ["A"], "poisoned queue launched additional DMA"
    assert len(starts) == 1, "poisoned queue restarted a worker"
    assert queue.size() == 0 and not queue._queue._putters
    assert queue._current_task is None and not queue._running


@cocotb.test()
async def queued_release_waits_for_true_execution_completion(dut):
    r = runtime()
    stream = r.create_stream()
    handler = stream._handler
    handler.valid, handler.device_id, handler.running = True, 0, True
    entered, done = rt.DeviceEvent(), rt.DeviceEvent()
    async def finish(h):
        entered.set()
        await done.wait()
        h.running = False
    r.finish_execution = finish
    await stream.release()
    await tick()
    assert entered.is_set(), "release skipped running execution"
    assert r.releases == [], "resource freed before confirmed done"
    done.set()
    await stream.synchronize()
    assert len(r.releases) == 1
    await r.retire_completed_stream(stream)
    assert len(r.releases) == 1


@cocotb.test()
async def execution_timeout_without_fresh_status_is_failure(dut):
    r = runtime()
    r._device_managers[0].device = SimpleNamespace(reg_cfg_en_tile_reglength=1)
    r._device_exereg = [rt.DeviceExeRegInfo()]
    r._device_fetch_exereg_interval = 1
    handler = rt.ResourceMappingHandler(0)
    handler.device_id, handler.total_tiles, handler.running = 0, 1, True
    await expect_error(r.finish_execution(handler), TimeoutError)
    assert handler.running, "unknown execution incorrectly marked completed"
    assert r._device_exereg[0].need_count == 0


@cocotb.test()
async def execution_timeout_with_fresh_but_absent_done_is_failure(dut):
    r = runtime()
    r._device_managers[0].device = SimpleNamespace(reg_cfg_en_tile_reglength=1)
    reg = rt.DeviceExeRegInfo(exe_reg=b"\x00")
    r._device_exereg = [reg]
    r._device_fetch_exereg_interval = 1
    handler = rt.ResourceMappingHandler(0)
    handler.device_id, handler.total_tiles, handler.running = 0, 1, True
    async def poll():
        while True:
            await Timer(1, units="ns")
            reg.resp_time = cocotb.utils.get_sim_time(units="ns")
    poller = cocotb.start_soon(poll())
    await expect_error(r.finish_execution(handler), TimeoutError)
    poller.kill()
    assert handler.running
    assert reg.need_count == 0


@cocotb.test()
async def completed_stream_is_collectible_without_history_registry(dut):
    import weakref
    import gc
    r = runtime()
    references = []
    for _ in range(128):
        stream = r.create_stream()
        await stream.set_event(rt.DeviceEvent())
        await stream.synchronize()
        references.append(weakref.ref(stream))
        await r.retire_completed_stream(stream)
    del stream
    # GC here is only measuring roots; retirement above must already be done.
    gc.collect()
    assert not r._streams
    assert all(ref() is None for ref in references)


if __name__ == "__main__":
    # Run in a disposable directory: no RTL changes or source-tree build output.
    import pathlib
    import subprocess
    import tempfile
    import xml.etree.ElementTree as ET
    source_dir = pathlib.Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="stream-ownership-unit-") as build:
        build_dir = pathlib.Path(build)
        top = build_dir / "unit_top.v"
        top.write_text("`timescale 1ns/1ps\nmodule unit_top; endmodule\n")
        makefiles = subprocess.check_output(
            ["cocotb-config", "--makefiles"], text=True).strip()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = ["make", "-f", str(pathlib.Path(makefiles) / "Makefile.sim"),
                   "SIM=icarus", "TOPLEVEL_LANG=verilog", "TOPLEVEL=unit_top",
                   "MODULE=test_stream_ownership", "VERILOG_SOURCES=" + str(top)]
        result = subprocess.run(command, cwd=build, env=env)
        report = build_dir / "results.xml"
        if result.returncode or not report.is_file():
            raise SystemExit(result.returncode or 1)
        tree = ET.parse(report)
        # cocotb 1.9 make can return zero even when a regression fails.
        raise SystemExit(1 if tree.findall(".//failure") or tree.findall(".//error") else 0)
