"""Stream operations for the CGRA-Cocotb-Sim client wire protocol."""
import json
import logging
import struct
from enum import IntEnum
from typing import Dict, Tuple

from test_runif import DeviceStream, DeviceRuntime, DeviceData
import test_runif


class ResponseCode(IntEnum):
    SUCCESS = 0
    ERROR = 1
    INVALID = 2


def build_response_message(response_code, data=b"", extra_info=None):
    info = json.dumps(extra_info or {}).encode("utf-8")
    return struct.pack("<BII", response_code, len(data), len(info)) + data + info


class UserEntry:
    def __init__(self, user_id: int, runtime: DeviceRuntime, logger: logging.Logger):
        self.user_id: int = user_id
        self.runtime = runtime
        self.streams: Dict[int, DeviceStream] = {}
        # stream->rtnid:sz,data
        self.rtn_data: Dict[int, Dict[int, Tuple[int, bytearray]]] = {}
        self.stream_count = int(0)
        self.rtn_count = int(0)

        self.logger = logger

    async def clear(self):
        for stream in self.streams.values():
            await self.runtime.destory_stream(stream)
        self.streams.clear()

    async def createStream(self, data: bytes) -> bytes:
        # param [4 priority]
        # ret   [4 stream_id]
        priority = struct.unpack('<I', data[0:4])[0]
        s_id = self.stream_count
        stream = self.runtime.create_stream(priority=priority)
        self.streams[s_id] = stream
        self.stream_count += 1
        self.rtn_data[s_id] = {}
        resp = build_response_message(
            response_code=ResponseCode.SUCCESS,
            data=struct.pack('<I', s_id),
            extra_info={},
        )
        return resp

    async def memcpyH2D(self, data: bytes) -> bytes:
        # param [4 stream_id][depend_type][data_ptr][data]
        # ret
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()

        depend_type = struct.unpack('<I', data[4:8])[0]
        addr = struct.unpack('<I', data[8:12])[0]
        sz = struct.unpack('<I', data[12:16])[0]
        data_ptr = DeviceData(address=addr, size=sz)
        await self.streams[s_id].memcpyHostToDevice(
            d_data=data_ptr, h_data=data[16:], size=sz, depend_type=depend_type
        )
        return self.rtnSuccess()

    async def memcpyD2H(self, data: bytes) -> bytes:
        # param [4 stream_id][depend_type][data_ptr]
        # ret   [4 ret_id]
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()

        depend_type = struct.unpack('<I', data[4:8])[0]
        addr = struct.unpack('<I', data[8:12])[0]
        sz = struct.unpack('<I', data[12:16])[0]
        data_ptr = DeviceData(address=addr, size=sz)
        buffer = bytearray()
        self.rtn_count += 1
        self.rtn_data[s_id][self.rtn_count] = (sz, buffer)
        await self.streams[s_id].memcpyDeviceToHost(
            d_data=data_ptr, h_data=buffer, size=sz, depend_type=depend_type
        )

        resp = build_response_message(
            response_code=ResponseCode.SUCCESS,
            data=struct.pack('<I', self.rtn_count),
            extra_info={},
        )
        return resp

    async def apply(self, data: bytes) -> bytes:
        # param [4 stream_id][4 cfg_num][cfg]
        # [cfg]:[iob_en][tile_en][data_ptrs][cfg_value]
        # ret
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()
        cfg_num = struct.unpack('<I', data[4:8])[0]
        cfgs = []
        offset = 8
        for _ in range(cfg_num):
            len = struct.unpack('<I', data[offset : offset + 4])[0]
            offset += 4
            iob_en = struct.unpack(f'<{len}B', data[offset : offset + len])  # uint8
            offset += len

            len = struct.unpack('<I', data[offset : offset + 4])[0]
            offset += 4
            tile_en = struct.unpack(f'<{len}B', data[offset : offset + len])  # uint8
            offset += len

            len = struct.unpack('<I', data[offset : offset + 4])[0]
            offset += 4
            data_ptr = []
            for i in range(len):
                addr = struct.unpack('<I', data[offset : offset + 4])[0]
                offset += 4
                sz = struct.unpack('<I', data[offset : offset + 4])[0]
                offset += 4
                data_ptr.append(DeviceData(address=addr, size=sz))

            len = struct.unpack('<I', data[offset : offset + 4])[0]
            offset += 4
            cfg_value = list(
                struct.unpack(f'<{len//2}H', data[offset : offset + len])
            )  # uint16
            offset += len
            cfgs.append(
                test_runif.DeviceConfig(
                    config_values=cfg_value,
                    iob_en=iob_en,
                    tile_en=tile_en,
                    data_ptr=data_ptr,
                )
            )

        await self.streams[s_id].apply(cfgs)
        return self.rtnSuccess()

    async def config(self, data: bytes) -> bytes:
        # param [4 stream_id][4 cfg_id]
        # ret
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()

        cfg_id = struct.unpack('<I', data[4:8])[0]
        await self.streams[s_id].config(config_id=cfg_id)
        return self.rtnSuccess()

    async def exe_start(self, data: bytes) -> bytes:
        # param [4 stream_id]
        # ret
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()

        await self.streams[s_id].execution_start()
        return self.rtnSuccess()

    async def synchronize(self, data: bytes) -> bytes:
        # param [4 stream_id]
        # ret   [4 ret_num][[4 ret_id][4 ret_len][ret_data]]
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()
        await self.streams[s_id].synchronize()

        ret_num = len(self.rtn_data[s_id])

        self.logger.debug(f"User {self.user_id} sync D2H task nums {ret_num}")

        ret = bytearray()
        ret.extend(struct.pack('<I', ret_num))
        for item in self.rtn_data[s_id].items():
            ret.extend(struct.pack('<I', item[0]))
            ret.extend(struct.pack('<I', item[1][0]))
            ret.extend(item[1][1])
            self.logger.debug(
                f"D2H id {item[0]} len {item[1][0]} data {item[1][1].hex()}"
            )
        self.rtn_data[s_id].clear()

        resp = build_response_message(
            response_code=ResponseCode.SUCCESS,
            data=ret,
            extra_info={},
        )
        return resp

    async def memcpyFence(self, data: bytes) -> bytes:
        # param [4 stream_id]
        # ret
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()

        await self.streams[s_id].memcpyFence()
        return self.rtnSuccess()

    async def release(self, data: bytes) -> bytes:
        # param [4 stream_id]
        # ret
        s_id = struct.unpack('<I', data[0:4])[0]
        if s_id not in self.streams:
            return self.rtnInvalidParam()

        await self.streams[s_id].release()
        return self.rtnSuccess()

    def rtnInvalidParam(
        self, extra_info: Dict = {"Error": "Invalid stream id"}
    ) -> bytes:
        resp = build_response_message(
            response_code=ResponseCode.INVALID,
            data=b'',
            extra_info=extra_info,
        )
        return resp

    def rtnError(self, extra_info: Dict = {}) -> bytes:
        resp = build_response_message(
            response_code=ResponseCode.ERROR,
            data=b'',
            extra_info=extra_info,
        )
        return resp

    def rtnSuccess(self) -> bytes:
        resp = build_response_message(
            response_code=ResponseCode.SUCCESS,
            data=b'',
            extra_info={},
        )
        return resp
