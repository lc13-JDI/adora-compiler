"""Construct one simulation device from the verified target selected by rtl_sim."""
import os
from pathlib import Path

import test_runif

expected_runtime = Path(__file__).resolve().with_name("test_runif.py")
if Path(test_runif.__file__).resolve() != expected_runtime:
    raise ImportError(f"Expected {expected_runtime}, imported {test_runif.__file__}")


async def start_device(dut, runtime_class=test_runif.DeviceRuntime):
    bus = test_runif.Axi4LiteTb(dut)
    await bus.cycle_reset()
    device = test_runif.create_device_info_factory(
        reg_json_path=os.environ["ADORA_RTL_AXIL"], adg_json_path=os.environ["ADORA_RTL_ADG"])
    runtime = runtime_class(dut=dut, axi=bus.axi, axil=bus.axil,
                            axi_size=bus.axi.write_if.max_burst_size)
    runtime.add_device(device)
    return runtime
