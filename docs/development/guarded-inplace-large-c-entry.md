# Owned plans for large guarded gathers

`guarded-inplace-large` is an explicit capacity backend for a restricted ordinary
C gather into a sequential inout destination. The compiler proves the original
wide global access and effects, then emits a four-buffer M=N=256 local kernel
through the existing LOAD/CSTORE, mapper, and generated runtime path. The source
image and destination can exceed one hardware bank. This is not a claim of
hardware speedup or support for general scatter, aliasing, or dynamic device
loops. Existing `guarded-stream` and `guarded-inplace` profiles retain their
small-object contracts.

## Compile a complete example

Run from the repository root with the built ADORA tools and `clang`/`cgeist`
on PATH. The existing CSTORE-capable hardware contract is required; an arbitrary
fp32 example specification does not establish CSTORE support.

```sh
ROOT="$(pwd)"
RUN="$(mktemp -d /tmp/adora-large-example.XXXXXX)"
export PATH="$ROOT/build/bin:$PATH"
export GeneralOpNameFile="$ROOT/lib/DFG/Documents/GeneralOpName.txt"
cat > "$RUN/overlay.c" <<'C'
#include <stddef.h>
void large_overlay(const unsigned short *src, const short *x,
                   const short *y, const short *enabled, unsigned short *dst) {
  for (int i = 0; i < T; ++i) {
    int xx = x[i], yy = y[i];
    if (enabled[i] != 0 && xx >= 0 && xx < W && yy >= 0 && yy < H) {
      unsigned short value = src[(size_t)yy * W + (size_t)xx];
#if VALUE_NONZERO
      if (value != 0)
#endif
        dst[i] = value;
    }
  }
}
C
cat > "$RUN/contract.json" <<'JSON'
{
  "entry": "large_overlay",
  "defines": {"W": 8193, "H": 1, "T": 4, "VALUE_NONZERO": 0},
  "objects": [
    {"argument_position": 0, "elements": 8193, "host_dtype": "uint16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 1, "elements": 4, "host_dtype": "int16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 2, "elements": 4, "host_dtype": "int16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 3, "elements": 4, "host_dtype": "int16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 4, "elements": 4, "host_dtype": "uint16", "access_assertion": "inout", "contiguous": true}
  ],
  "host_plan": {"strategy": "per-query", "tile_items": 256,
                "bind_arguments": [1, 2, 3], "ownership": "copy"}
}
JSON
python3 "$ROOT/tools/adoracc/adoracc.py" "$RUN/overlay.c" \
  --profile guarded-inplace-large --contract "$RUN/contract.json" \
  --adg-path "$ROOT/test/spec/cgra_cstore_vitra/artifacts/adg.json" \
  --op-file "$ROOT/test/spec/cgra_cstore_vitra/artifacts/operations.json" \
  --work-dir "$RUN" --output "$RUN/overlay.py" \
  --seed 7 --max-iters 100 --timeout 10000 --mapper-wall-timeout 120
```

The output must be a new `.py` path. Compilation preserves the original C,
preprocessed C, AST, raw MLIR, verified host descriptor, local IR, DFG, mapper
output, and provenance under `adora-c-entry/` in the selected work directory.
An optional `--kernel-cache PATH` reuses a compatible local kernel from a trusted
compiler cache, with a checked artifact inventory; it does not reuse frame data
or map snapshots. The cache contains executable generated code and is not an
untrusted descriptor import mechanism.

Import the resulting module by file path in an asynchronous application that
already has a matching ADORA runtime instance (including a cocotb application):

```python
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("overlay", generated_python_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

async def run_example(runtime):
    x = np.array([8192, 0, 8192, -1], dtype=np.int16)
    y = np.zeros(4, dtype=np.int16)
    en = np.array([2, -1, 0, 1], dtype=np.int16)
    src = np.zeros(8193, dtype=np.uint16)
    src[8192], src[0] = 30, 10
    dst = np.array([7, 8, 9, 10], dtype=np.uint16)
    plan = module.make_plan(x, y, en)
    await plan.execute(runtime, src, dst)
    np.testing.assert_array_equal(dst, [30, 10, 9, 10])
    plan.close()
    return dst
```

The runtime instance owns hardware/simulation transport and must match the
compiled ADG and operations. Constructing that external runtime is outside this
host support library. Do not use `asyncio.run` to drive a cocotb-owned runtime.

## Public execution API

A generated Python module provides these public interfaces:

```python
# x/y/en are contiguous native int16 arrays with the compiled target length.
plan = module.make_plan(x, y, en)
# src is contiguous native uint16 with the compiled source length.
# dst is a writable contiguous native uint16 array initialized with old pixels.
report0 = await plan.execute(runtime, src0, dst0)
report1 = await plan.execute(runtime, src1, dst1)
plan.close()

# Equivalent module entry checks that the plan belongs to this module instance.
plan = module.make_plan(x, y, en)
await module.execute_plan(runtime, plan, src0, dst0)
plan.close()

# The original C function name is also a once-only entry. Its five buffers use
# the original C parameter order, including compiler-proved parameter reordering.
await module.large_overlay(runtime, src0, x, y, en, dst0)
```

The once-only entry validates all five objects, constructs a plan, executes it,
and releases it on safe completion. Its cost includes plan creation every time.
Changing coordinates or enable requires a new plan. Changing source or old
destination values takes effect on the next execution without rebuilding.

## Ownership and validation

A plan owns immutable byte copies of all three bound int16 arrays. Setting a
caller's array read-only is unnecessary; a surviving writable alias cannot alter
the plan. Public metadata and statistics are detached copies, and public APIs
never expose mutable internal coordinates or offsets. `make_plan` checks shape,
dtype, native byte order, contiguity, and pairwise disjoint bound buffers before
allocation. Callers must not modify bound arrays while `make_plan` copies them.

`execute` validates source/destination shape, dtype, native byte order,
contiguity, destination writeability, pairwise non-overlap, and plan lifecycle
before invoking any runtime operation. The once-only entry checks all five
original buffers for overlap before making a plan. The caller grants an
exclusive borrow of source/destination and their aliases until execution
completes; concurrent external mutation is unsupported. A plan rejects concurrent
execution. Distinct plans do not promise serialization on a shared runtime;
callers must sequence them unless their runtime independently supports it.

The descriptor schema is `adora.guarded-inplace-large.v1`, host support version
1. Source and target each have a conservative maximum of 16,777,216 elements,
checked before user-sized allocations. The compiler additionally restricts width and height to 1..32767.
Dimensions and their product are checked in Python's wide integer domain; accepted offsets are stored as int64. All
hardware indices remain 0..255 or -1. Positive signed-int16 coordinates limit
the reachable portion of any dimension, independently of allocation capacity.

The compiler embeds host support and the generated local implementation into
the public module, together with a compiled artifact identity. Descriptor JSON
is not an input proof format and cannot authorize a new source program. The
internal binder checks compatibility, not C semantics. Each generated module
instance also has an opaque process-local identity: even a plan from an
identically compiled independent module is rejected by `module.execute_plan`.
Python private names are implementation details, not a security sandbox against
a caller modifying generated executable code.

## Packing and output

Each 256-query output tile has four completely initialized 512-byte buffers:
packed source, local index, enable, and old destination. Query slot j gets the
current global source pixel at its proven offset, local index j, and enable 1.
Invalid slots get index -1 and enable 0. Source dummy values and padded old
output slots are zero. Real old destination slots come from the current frame.
Only a tile with no valid read guard is skipped. Pixel values never select tasks,
so the optional source-value-nonzero predicate remains on the device.

After the generated call completes, the first `count` returned destination
slots are copied back without a host predicate or host result merge. Padding
is never copied beyond the real output range. Scratch buffers are reused only
after completion. The immutable plan payload is `14*T + ceil(T/256)` bytes:
three 16-bit snapshots, one 64-bit offset per query, and one activity byte per
tile. Persistent scratch payload is 2,560 bytes (four tile arrays and local slot
indices). Plan construction uses additional O(T) temporary arrays; execution
uses bounded tile temporaries. Python object headers, runtime-owned resources,
caller frames, and transient allocation peaks are excluded from these payload
figures and require separate process-memory measurement.

`plan.stats` reports `plan_bytes`, `scratch_bytes`, `active_tiles`,
`skipped_tiles`, `plan_build_ns`, and completed `executions`. Every successful
execution returns `completed_ranges` (half-open output intervals), tile counts,
and `timings_ns` for precheck, prepare, pack, device, copyback, and total.
`device` measures awaited generated-call wall time, including runtime and
simulation overhead; it is not hardware execution time. Timing calls are part
of production host overhead. `last_report` returns a detached copy of the last
successful report. `completed_ranges` reports progress of the current/latest
attempt; skipped unchanged ranges are not listed as device-completed ranges.

## Failure and cancellation

`close()` is idempotent after safe completion. It rejects pending executions and
poisoned plans and never frees uncertain in-flight storage. Closing a healthy
plan releases its owned snapshots, offsets, and scratch; future execution fails
before any runtime effect.

An exception during an awaited generated call may leave device activity alive.
The plan is poisoned, rejects all further execution/close, and strongly retains
its scratch, runtime, and frame borrow. Ordinary exceptions become
`PlanExecutionError` with `completed_ranges`, `failed_range`, and `poisoned`.
Cancellation/control-flow exceptions keep their original type and receive the
same fields. Previously copied ranges remain visible; no whole-frame rollback
is promised. The failed tile has not been copied to the global destination.

The implementation uses scheduler-neutral awaits and does not require an
asyncio loop. Cocotb 1.9 `Task.kill()` can unschedule a started coroutine without
throwing into it. An in-flight registry therefore roots storage *before* each
await, not only in an exception handler. A hard-unscheduled plan remains
`pending` and retained, with `completed_ranges` available for diagnosis; closing
or reusing it is rejected. If the coroutine subsequently closes, it becomes
poisoned. Do not treat a scheduler's hard-kill return value as successful frame
completion.

There is no recovery/free API for uncertain completion: the current runtime
has no generated-call recovery barrier that proves all outstanding DMA has
stopped. Retention ends only with process teardown (or interpreter teardown
after the runtime has been stopped). Restart the runtime/process after such a
failure. This intentionally trades bounded exceptional resource retention for
avoiding use-after-free; it does not modify shared runtime behavior.

## Verification

The standalone protocol regression is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 test/adoracc/prepared_xy/test_prepared_xy.py
```

Its semantic tile stand-in checks real production packing, copyback, snapshots,
prechecks, and lifecycle behavior; it is HOST_ONLY evidence. The lit wrapper
also includes it in compiler regression discovery. Hardware claims additionally
require newly generated configuration, real runtime calls, and full original
RTL results. A stand-in's success is never a substitute for those checks.
