# Shared Python runtime compatibility patch

`private-stream-ownership.patch` updates the external CGRA Cocotb server runtime
used by compiler-generated `guarded-inplace-large` calls. It also adds its
standalone regression test and ownership API documentation under `server/`.
The external simulation tree is not maintained as an ADORA submodule; this
applicable delta records the runtime half of the compiler change.

Apply from the root of the shared simulation tree after reviewing its local
changes:

```sh
git apply --check /path/to/adora/runtime/python/private-stream-ownership.patch
git apply /path/to/adora/runtime/python/private-stream-ownership.patch
```

The patch expects the original `server/test_runif.py` SHA-256
`0421394a6c1292bb859b18c7c7021acc9cb27703550167dee5e0e89ea0de0ca0`.
Do not overwrite a different local runtime; reconcile that change first.
The patch does not modify RTL, wrappers or hardware specifications.

Run `python -B server/test_stream_ownership.py` with cocotb 1.9.2, NumPy,
Icarus Verilog and `cocotb-config` available in that Python environment's PATH.
It creates a temporary minimal simulation for software concurrency tests;
these tests do not substitute for full VITRA hardware validation.

New large-plan modules require runtime ownership protocol version 1 before
creating a stream. Existing generated modules remain supported, but retain
their previous ownership behavior until regenerated. See
`docs/development/private-stream-ownership.md` for publication, cleanup retry,
and unknown-completion borrowing rules.
