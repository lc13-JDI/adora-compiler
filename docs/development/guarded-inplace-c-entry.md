# Conditional in-place C entry

This profile is experimental. Compiler and mapping regressions pass, but the
frozen VITRA integration used for R006 does **not** pass repeated execution.
On a warm launch its tile controller sends start before the GPE execution enable
is active. The fixed-Latency=0 index ACC then aborts and can write repeatedly to
destination element zero. A successful first call is not evidence that a later
call is safe. Do not use this profile for workloads requiring repeated calls on
that integration; R006 hardware acceptance remains partial.

The index-ACC contract requires enable to be held high before start through the
last value. The existing runtime has no separate pre-start GPE enable API.
This compiler change does not modify RTL, insert dummy executions, reset between
calls, or change that fixed timing contract. A target integration must satisfy
the launch prerequisite before the semantics below can be relied on for
repeated execution.

`adoracc.py --profile guarded-inplace` extends the checked ordinary C pipeline
with a single partially written destination. The separate `guarded-stream`
profile continues to require complete output coverage. Neither profile changes
the default driver pipeline.

For example:

```c
void overlay(const unsigned short *src, const short *idx,
             const short *enabled, unsigned short *dst) {
  for (int i = 0; i < M; ++i) {
    int k = idx[i];
    if (enabled[i] != 0 && k >= 0 && k < N)
      dst[i] = src[k];
  }
}
```

The source and destination must already contain initialized data. The destination
is transferred to the device before execution. A true predicate writes the
sampled pixel to `dst[i]`; a false predicate suppresses that hardware write,
preserving the transferred old value. The generated call waits for normal
completion and copies the whole destination back into the same host array.
There is no host mask merge or CPU kernel fallback.

Use the existing contract schema with exactly one `inout` object:

```json
{
  "entry": "overlay",
  "defines": {"N": 4, "M": 4},
  "objects": [
    {"argument_position": 0, "elements": 4, "host_dtype": "uint16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 1, "elements": 4, "host_dtype": "int16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 2, "elements": 4, "host_dtype": "int16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 3, "elements": 4, "host_dtype": "uint16", "access_assertion": "inout", "contiguous": true}
  ]
}
```

`inout` declares preservation of unwritten elements. It does not authorize
arbitrary reads of `dst` in the C kernel. The IR bridge checks the actual effects:
one destination, one conditional assignment to `dst[i]`, and no reads or feedback
from that destination. The same local object connects its initializing
`ADORA.BlockLoad`, kernel CSTORE, and final `ADORA.BlockStore`.

With the selected ADORA build and cgeist on `PATH`, compile to a new directory:

```bash
RUN="$(mktemp -d /tmp/adora-inplace.XXXXXX)"
python3 tools/adoracc/adoracc.py overlay.c \
  --profile guarded-inplace --contract overlay.json \
  --adg-path test/spec/cgra_cstore_vitra/artifacts/adg.json \
  --op-file test/spec/cgra_cstore_vitra/artifacts/operations.json \
  --work-dir "$RUN" -o "$RUN/mapped.py" \
  --seed 7 --max-iters 1000 --timeout 120000
```

The public async API follows the C argument order:

```python
await mapped.overlay(runtime, src_uint16, idx_int16, enabled_int16, dst_uint16)
```

Before creating any stream or issuing transfers, it checks exact one-dimensional
shape, native NumPy dtype, contiguous storage, writable destination and pairwise
nonaliasing (including overlapping views). Read-only inputs are accepted. Equal
width `int16` transport views preserve all `uint16` pixel bits and the original
destination storage. Initialization cannot be inferred from arbitrary host
memory and remains a caller requirement. Changing dimensions requires a new
compilation.

The supported subset is a fixed positive zero-based unit-stride loop, a single
dynamic read-only source, sequential coordinate/mask inputs, and one conditional
destination element per iteration. Flattened two-dimensional coordinates retain
separate x/y bounds. A pixel `!= 0` condition may further suppress writing after
the read; it does not determine whether its own load executes. Noncanonical
nonzero masks such as `2` and `-1` are normalized to boolean conditions.

Safe-address LOAD runs once per slot, using source element zero for an invalid
read guard. Therefore the source must be nonempty, initialized, ordinary memory;
volatile/MMIO, atomic and strong no-access contracts are rejected. Conditional
writing uses the existing CSTORE and loop-index ACC. It is independent of the
number of true predicates; false slots still advance and complete normally.

The bridge proves signed coordinate promotions and every address intermediate
before narrowing. General int32 coordinates, unproven wide arithmetic, floating
point, dynamic trip counts, carry reductions, scatter, duplicate destination
writes, source/destination feedback, and multi-bank objects are outside this
profile. Target bank capacity and representable execution fields remain checked;
a count no greater than 4095 is necessary but does not guarantee mapping.

Artifacts and failure provenance follow the [checked C entry documentation](guarded-stream-c-entry.md).
Raw frontend IR is retained; only a working copy loses incompatible module
metadata. Every required compile/map stage is fatal on failure. Compile/map
success alone does not claim RTL or hardware execution.
