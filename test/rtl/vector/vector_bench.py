"""The pinned IntVecAdd module and original C++ wire client run on the same RTL."""
import json
import logging
import os
from pathlib import Path
import re
import subprocess

import cocotb
import numpy as np

from adora_tcp.server import LoopbackServer
import test_runif
from rtl_device import start_device
from IntVecAdd import IntVecAdd


def record(name, **values):
    directory = Path(os.environ["ADORA_RTL_OUTPUT"])
    (directory / (name + ".json")).write_text(json.dumps({
        "runtime": str(Path(test_runif.__file__).resolve()), **values}, indent=2) + "\n")


@cocotb.test()
async def vector_add(dut):
    runtime = await start_device(dut)
    a = 19 - np.arange(20, dtype=np.int16) * 5
    b = np.arange(20, dtype=np.int16) * 3 - 8
    c = np.full(20, 0x5a5a, dtype=np.int16)
    await IntVecAdd(runtime, a, b, c)
    await runtime.synchronize_all()
    record("vector-add", a=a.tolist(), b=b.tolist(), readback=c.tolist())
    np.testing.assert_array_equal(c, a + b)


@cocotb.test()
async def tcp_client(dut):
    logging.getLogger("cocotb.regression").setLevel(logging.INFO)
    runtime = await start_device(dut)
    directory = Path(os.environ["ADORA_RTL_OUTPUT"])
    root = Path(__file__).resolve().parents[3]
    cpp = root / "runtime" / "cpp"
    binary = directory / "intvecadd-client"
    compile_cmd = [os.environ.get("CXX", "c++"), "-std=c++17", "-I", str(cpp),
                   str(root / "test/rtl/vector/intvecadd.cpp"),
                   *(str(cpp / name) for name in ("CocotbClient.cpp", "api_stream.cpp", "api_axi.cpp")),
                   "-o", str(binary)]
    subprocess.run(compile_cmd, check=True)
    with LoopbackServer(runtime, int(os.environ.get("ADORA_RTL_PORT", 0))) as server:
        port = server.server_address[1]
        with (directory / "client.log").open("w") as log:
            proc = subprocess.Popen([str(binary), str(port)], stdout=log, stderr=subprocess.STDOUT)
            def finished():
                status = proc.poll()
                if status:
                    raise RuntimeError(f"C++ client exited {status}, see {log.name}")
                return status == 0 and server.closed_sessions == 1
            try:
                await server.run_until(finished, timeout=60)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
        assert not runtime._streams
    output = (directory / "client.log").read_text(encoding="utf-8")
    vectors = {}
    for name in ("A", "B", "C"):
        line = re.search(rf"^Vec {name}: (.*)$", output, re.MULTILINE)
        if not line:
            raise AssertionError(f"missing Vec {name} from C++ client")
        vectors[name] = [int(x) for x in re.findall(r"\d+", line[1])]
    record("tcp-client", port=port, client_exit=proc.returncode,
           closed_sessions=server.closed_sessions, compiler=compile_cmd,
           a=vectors["A"], b=vectors["B"], readback=vectors["C"])
    assert len(vectors["C"]) == 20
    np.testing.assert_array_equal(vectors["C"], np.add(vectors["A"], vectors["B"]))


@cocotb.test()
async def tcp_server(dut):
    """A separate executable may connect to this server; the last disconnect ends the run."""
    runtime = await start_device(dut)
    with LoopbackServer(runtime, int(os.environ.get("ADORA_RTL_PORT", 0))) as server:
        host, port = server.server_address
        record("server", host=host, port=port)
        await server.run_until(lambda: server.closed_sessions > 0)
    assert not runtime._streams
