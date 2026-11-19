"""Serve the original C++ wire protocol on a loopback port chosen by the caller."""
from concurrent.futures import Future
import logging
import queue
import socketserver
import struct
import threading
import time

import cocotb
from cocotb.triggers import Timer

from .commands import ResponseCode, UserEntry, build_response_message


_COMMANDS = {
    "createStream": "createStream", "memcpyH2D": "memcpyH2D", "memcpyD2H": "memcpyD2H",
    "apply": "apply", "config": "config", "exeStart": "exe_start",
    "synchronize": "synchronize", "memcpyFence": "memcpyFence", "release": "release",
}


class _Connection(socketserver.StreamRequestHandler):
    def exchange(self, command, data):
        reply = Future()
        self.server.requests.put((self, command, data, reply))
        return reply.result()

    def handle(self):
        try:
            while True:
                header = self.rfile.read(8)
                if len(header) != 8:
                    return
                name_size, data_size = struct.unpack("<II", header)
                name = self.rfile.read(name_size)
                data = self.rfile.read(data_size)
                if len(name) != name_size or len(data) != data_size:
                    return
                reply = self.exchange(name.decode("ascii"), data)
                self.wfile.write(reply)
        finally:
            self.exchange(None, b"")


class LoopbackServer(socketserver.ThreadingTCPServer):
    """Requests run on cocotb's thread. Each connection owns its streams."""
    daemon_threads = True

    def __init__(self, runtime, port=0):
        super().__init__(("127.0.0.1", port), _Connection)
        self.runtime = runtime
        self.requests = queue.Queue()
        self.users = {}
        self._session_lock = threading.Lock()
        self._active_sessions = 0
        self._accepting = True
        self.closed_sessions = 0
        self.cleanup_error = None
        self.logger = logging.getLogger("adora_tcp")
        self.thread = threading.Thread(target=self.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)

    def get_request(self):
        # Register the connection in the accept thread, before a worker can
        # block on its first byte or another client can finish.
        with self._session_lock:
            connection, address = super().get_request()
            if not self._accepting:
                connection.close()
                raise OSError("server is shutting down")
            self._active_sessions += 1
        return connection, address

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            with self._session_lock:
                self._active_sessions -= 1
                self.closed_sessions += 1

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()
        self.thread.join()

    async def dispatch(self, connection, command, data, reply):
        try:
            if command is None:
                entry = self.users.get(connection)
                if entry is not None:
                    await entry.clear()
                    del self.users[connection]
                response = b""
            elif command == "echo":
                response = build_response_message(ResponseCode.SUCCESS, data, {"success": "echo"})
            elif command in _COMMANDS:
                if connection not in self.users:
                    self.users[connection] = UserEntry(connection.client_address[1],
                                                       self.runtime, self.logger)
                response = await getattr(self.users[connection], _COMMANDS[command])(data)
            else:
                response = build_response_message(ResponseCode.INVALID,
                                                  extra_info={"error": "unknown command"})
        except Exception as exc:
            self.logger.exception("TCP command %r failed", command)
            if command is None:
                self.cleanup_error = exc
            response = build_response_message(ResponseCode.ERROR, extra_info={"error": str(exc)})
        reply.set_result(response)

    async def run_until(self, finished, timeout=None):
        """Pump device work until the caller signals completion and sessions drain."""
        start = time.monotonic()
        while True:
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError("timed out waiting for TCP device/client completion")
            if self.cleanup_error:
                raise self.cleanup_error
            if finished():
                with self._session_lock:
                    if not self._active_sessions:
                        self._accepting = False
                        return
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                if not self.runtime.is_device_busy():
                    time.sleep(0.001)
            else:
                cocotb.start_soon(self.dispatch(*request))
            await Timer(50, units="ns")
