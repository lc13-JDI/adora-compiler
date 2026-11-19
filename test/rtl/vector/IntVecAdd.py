"""
Copyright (c) 2025 ADORA
All rights reserved.
Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
Generated on: 2025-11-27 20:09:07

"""

from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime
from typing import List
from numpy import ndarray


async def aux_stream(
    stream: DeviceStream,
    config: List[DeviceConfig],
    iptrs: List[DeviceData],
    idata: List,
    optrs: List[DeviceData],
    odata: List,
    olen: List,
):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------
    await stream.apply(config)
    await stream.config(config_id=0)
    # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        await stream.memcpyHostToDevice(
            d_data=iptrs[i], h_data=idata[i], size=len(idata[i])
        )
    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    # await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i])

    # await stream.release()
    return


def DeviceData_Pong(ptr: DeviceData) -> DeviceData:
    new_ptr = DeviceData(ptr.address + ptr.size, ptr.size)
    return new_ptr


async def aux_stream_pingpong(
    stream: DeviceStream,
    # config: List[DeviceConfig],
    config_id: int,
    iptrs: List[DeviceData],
    idata: List[ndarray],
    optrs: List[DeviceData],
    odata: List,
    olen: List,
    pingpong: bool,
):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    pingpong : bool
        Indicates the pingpong phase(ping-phase or pong-phase)
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------
    await stream.config(config_id=config_id)
    # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        if pingpong == 0:
            await stream.memcpyHostToDevice(
                d_data=iptrs[i], h_data=idata[i], size=len(idata[i]), depend_type=2
            )
        else:
            await stream.memcpyHostToDevice(
                DeviceData_Pong(iptrs[i]),
                h_data=idata[i],
                size=len(idata[i]),
                depend_type=2,
            )

    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        if pingpong == 0:
            await stream.memcpyDeviceToHost(
                d_data=optrs[i], h_data=odata[i], size=olen[i]
            )
        else:
            await stream.memcpyDeviceToHost(
                DeviceData_Pong(optrs[i]), h_data=odata[i], size=olen[i]
            )

    # await stream.synchronize()

    # await stream.release()
    return


async def aux_stream_pingpong_init(stream: DeviceStream, config: List[DeviceConfig]):
    """
    Apply stream configuration
    """
    cfg_copy = list(config)
    await stream.apply(cfg_copy)
    await stream.config(config_id=0)

    # await stream.release()
    return


## ===----------------------------------------------------------------------===//
## Configuration Data
## ===----------------------------------------------------------------------===//
""" kernel: IntVecAdd,  cfgNum: 23"""
cfgbit_IntVecAdd = [
    0x9000,
    0x4000,
    0x0008,
    0x0001,
    0x0000,
    0x0009,
    0x0000,
    0x0000,
    0x000A,
    0x0000,
    0x0008,
    0x000B,
    0x8000,
    0x4000,
    0x0010,
    0x0001,
    0x0000,
    0x0011,
    0x0000,
    0x0000,
    0x0012,
    0x0000,
    0x0008,
    0x0013,
    0x0000,
    0x0000,
    0x0020,
    0x0000,
    0x0001,
    0x0030,
    0x0900,
    0x0000,
    0x0031,
    0x0000,
    0x0000,
    0x0041,
    0x0800,
    0x0000,
    0x0061,
    0x0800,
    0x0000,
    0x0081,
    0x0800,
    0x0000,
    0x00A1,
    0x0800,
    0x0000,
    0x00C1,
    0x2000,
    0x0000,
    0x00E0,
    0x8000,
    0x4000,
    0x00E8,
    0x0001,
    0x0000,
    0x00E9,
    0x0000,
    0x0000,
    0x00EA,
    0x0000,
    0x2308,
    0x00EB,
    0x0080,
    0x0000,
    0x00EC,
]


async def IntVecAdd(
    runtime: DeviceRuntime, arg_0: ndarray, arg_1: ndarray, arg_2: ndarray
):
    # runtime.log.info("[ADORA] Starting CGRA call (IntVecAdd)")
    iptrs, idata = [], []
    optrs, odata, olen = [], [], []
    configs, data_ptr = [], []
    stream = runtime.create_stream()

    ## %0 = ADORA.BlockLoad %arg0 [0] : memref<?xi16> -> memref<20xi16>  {Id = "0", KernelName = "IntVecAdd"}
    idata.append(arg_0[0 : 0 + 20])
    iptrs.append(DeviceData(0x0, 40))

    ## %1 = ADORA.BlockLoad %arg1 [0] : memref<?xi16> -> memref<20xi16>  {Id = "1", KernelName = "IntVecAdd"}
    idata.append(arg_1[0 : 0 + 20])
    iptrs.append(DeviceData(0x2000, 40))

    ## %2 = ADORA.LocalMemAlloc memref<20xi16>  {Id = "2", KernelName = "IntVecAdd"}
    data_ptr.append(DeviceData(0x4000, 40))

    ### IntVecAdd
    data_ptr.append(iptrs)
    config_IntVecAdd = DeviceConfig(
        config_values=cfgbit_IntVecAdd, iob_en=[0x07], tile_en=[0x01], data_ptr=data_ptr
    )
    configs.append(config_IntVecAdd)

    ## ADORA.BlockStore %2, %arg2 [0] : memref<20xi16> -> memref<?xi16>  {Id = "2", KernelName = "IntVecAdd"}
    odata.append(arg_2[0 : 0 + 20])
    optrs.append(DeviceData(0x4000, 40))
    olen.append(40)

    await aux_stream(
        stream=stream,
        config=configs,
        iptrs=iptrs,
        idata=idata,
        optrs=optrs,
        odata=odata,
        olen=olen,
    )

    configs.clear()
    iptrs.clear(), idata.clear()
    optrs.clear(), odata.clear(), olen.clear()

    await stream.synchronize()

    await runtime.destory_stream(stream)
