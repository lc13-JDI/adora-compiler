# Python runtime for full RTL simulation

The generated mapper module, the cocotb bench and the TCP server import
`test_runif` from this directory. Keep this module on the simulator's Python
path under that name so all three users share the same DeviceData, DeviceStream
and DeviceRuntime classes. Use the repository launcher to select the interpreter,
target and module path. The CPU asynchronous C library is built separately.

The initial source import is from `lc13-JDI/CGRA-Cocotb-Sim` at
`c48135347c8ea98b4c5b5da28ecdd16a67d5fef8`. The files
`test_runif.py`, `test_stream_ownership.py` and `STREAM_OWNERSHIP.md`
correspond to the same basenames under `server/` at that revision. The original
history includes zevick's simulation runtime and lc13-JDI's ownership changes.
Subsequent runtime maintenance belongs here. Consult the source notices for
individual files. The pinned simulator tree contains no project-wide license
file. Redistribution of material without an applicable grant requires
confirmation from its rights holders.

From the ADORA root with the simulation environment active, run the original
software regressions with `python -B runtime/python/test_stream_ownership.py`.
The suite runs 16 tests on an Icarus timer-only top. Full device cases and
hardware inputs are described in the [project README](../../README.md#optional-full-rtl-simulation).

## Existing generated modules

Regenerate a large-plan Python module to enable private-stream ownership.
Existing modules can run with this runtime using their original lifetime rules.
New modules check for ownership v1 before creating a stream or transferring
data. Publication and unknown-completion rules are documented in
[private stream ownership](../../docs/development/private-stream-ownership.md).

The historical `private-stream-ownership.patch` was issued for an external
`server/test_runif.py` with SHA-256
`0421394a6c1292bb859b18c7c7021acc9cb27703550167dee5e0e89ea0de0ca0`.
The repository runtime already includes that protocol.

The IntVecAdd reference module comes from `workspace/IntVecAdd.py`. The C++
example in `test/rtl/vector/intvecadd.cpp` is based on `workspace/test.cpp`.
The public C++ client sources in `runtime/cpp/` correspond to `client/` at the
same revision. The dispatcher in `adora_tcp/commands.py` is based on
`server/CommandHandler.py`. Those original client and example sources are
attributed to zevick. Use the named `vector-add` and `tcp-client` cases to run
the two paths with the locked IntVecAdd hardware.
