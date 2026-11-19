#!/usr/bin/env python3
"""Check a separately running TCP server against two overlapping client connections."""
import argparse
import json
from pathlib import Path
import select
import socket
import struct
import subprocess
import time


def check(endpoint, client):
    address = (endpoint["host"], endpoint["port"])
    with socket.create_connection(address, timeout=10) as survivor:
        # The server must have accepted survivor before replying to this later connection.
        with socket.create_connection(address, timeout=10) as first:
            first.sendall(struct.pack("<II", 4, 3) + b"echo" + b"one")
            first.shutdown(socket.SHUT_WR)
            reply = b"".join(iter(lambda: first.recv(4096), b""))
            code, length, info_length = struct.unpack_from("<BII", reply)
            assert (code, length, reply[9:12]) == (0, 3, b"one")
            assert len(reply) == 9 + length + info_length
        # Observe the idle interval and detect early EOF before starting any commands.
        ready, _, _ = select.select([survivor], [], [], 0.25)
        if ready:
            raise AssertionError("accepted idle connection was closed after its peer disconnected")
        # Relay the real C++ example over the already accepted socket. No commands
        # have been sent on survivor, so it has not created a stream or user record.
        with socket.socket() as relay:
            relay.bind(("127.0.0.1", 0))
            relay.listen(1)
            relay.settimeout(10)
            with subprocess.Popen([str(client), str(relay.getsockname()[1])],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
                try:
                    local, _ = relay.accept()
                    with local:
                        peers = {local: survivor, survivor: local}
                        while peers:
                            readable, _, _ = select.select(list(peers), [], [], 10)
                            if not readable:
                                raise TimeoutError("C++ client or simulation stopped responding")
                            for source in readable:
                                data = source.recv(65536)
                                if data:
                                    peers[source].sendall(data)
                                else:
                                    peers.pop(source).shutdown(socket.SHUT_WR)
                    output, _ = proc.communicate(timeout=10)
                    print(output.decode("utf-8"), end="")
                    if proc.returncode:
                        raise RuntimeError(f"C++ client exited {proc.returncode}")
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", type=Path, help="server.json written by the tcp-server case")
    parser.add_argument("client", type=Path, help="intvecadd-client from the tcp-client case")
    args = parser.parse_args()
    deadline = time.monotonic() + 30
    while not args.endpoint.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    check(json.loads(args.endpoint.read_text()), args.client.resolve())
