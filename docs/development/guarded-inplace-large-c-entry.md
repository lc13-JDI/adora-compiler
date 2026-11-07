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

The default host preparation uses bounded NumPy windows and cursor completion
accounting. Diagnostic choices are explicit keyword-only plan options:

```python
plan = module.make_plan(x, y, en, preparation="window", window_tiles=32,
                        bookkeeping="cursor", probes=True)
```

`window_tiles` accepts integers from 1 through 256; allocated capacity is further
limited to the number of active tiles. `preparation="tile"` retains the original
per-tile preparation algorithm as a fallback/control. `bookkeeping="eager"`
retains the original `completed.append((begin, end))` and `tuple(completed)`
on each completion, plus the final report's tuple copy, permitting a
preparation-only comparison against the cursor implementation. These choices never move value predicates
or output computation onto the host, change the 256-slot device ABI, or allow
multiple outstanding calls. `probes=False` disables fine-grained clocks;
unmeasured phase values are `None`, while `total` remains measured. The once-only
five-array entry uses default options.

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
2. Version 2 rejects stale version 1 descriptors. Previously generated modules
embed their own version 1 support and remain independently executable.
Source and target each have a conservative maximum of 16,777,216 elements,
checked before user-sized allocations. The compiler additionally restricts width and height to 1..32767.
Dimensions and their product are checked in Python's wide integer domain; accepted offsets are stored as int64. All
hardware indices remain 0..255 or -1. Positive signed-int16 coordinates limit
the reachable portion of any dimension, independently of allocation capacity.
Before copying input arrays the plan also checks its conservative immutable
ndarray payload estimate against a 768 MiB limit and the platform index size.
Allocation failures retain `MemoryError` with a plan-specific diagnostic; these
checks do not guarantee that the process has enough available memory.

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

The window path precomputes immutable local indices, enable masks, valid packed
positions, int64 source offsets, and safe destination gather indices. A batch
contains up to `window_tiles` active tiles in global order; skipped tiles consume
no window slots. Each execution gathers current source and old destination
values for just that batch. Invalid source offsets are never gathered; padded
old destinations are explicitly zeroed. Prefetching the old values of later
tiles is safe because their destination ranges are disjoint and the caller's
exclusive borrow covers the entire execution.

After the generated call completes, the first `count` returned destination
slots are copied back without a host predicate or host result merge. Padding
is never copied beyond the real output range. The device still sees one serial
256-slot call; no window is refilled until all its calls have completed.

On a 64-bit host, with T queries, V valid queries, A active tiles and
B=min(window_tiles,A), immutable window ndarray payload is `6*T + 16*V +
12*256*A` bytes. Persistent dynamic scratch is `4*256*B` bytes (packed and old
uint16 arrays). `execution_scratch_bound_bytes` conservatively allows
`16*256*B` bytes for these arrays and NumPy gather temporaries. Thus the default
32-tile capacity uses at most 32 KiB persistent dynamic scratch, with a 128 KiB
execution payload bound; the maximum configurable capacity has a 1 MiB bound.
All-invalid plans allocate zero dynamic payload. The tile fallback retains
`14*T + ceil(T/256)` immutable bytes and 2,560 persistent scratch bytes, with a
6,144-byte execution payload bound. These execution bounds exclude the existing
static plan and runtime-owned allocations.

Plan construction uses additional O(T) temporary arrays and O(window_tiles*256)
window construction buffers. Python headers, caller frames, allocator overhead,
runtime resources, and construction peaks are excluded from ndarray payload
figures and require separate memory measurement. `completion_metadata_bytes`
counts the immutable active-range tuple, its range tuples and integer endpoints
using `sys.getsizeof`; additional window-container headers are excluded.
`completion_snapshot_bound_bytes` bounds one observation tuple's container,
whose range entries share immutable objects. Eager bookkeeping additionally
allocates a growing list and one new pair tuple per completed tile, and may
temporarily hold two prefix containers. Those Python allocations are excluded
from the static metadata count and ndarray scratch bound; measure their peak
separately. Callers can also retain arbitrarily many snapshots.

`plan.stats` reports `plan_bytes`, `scratch_bytes`, `active_tiles`,
`skipped_tiles`, `plan_build_ns`, and completed `executions`, together with the
actual `preparation`, `window_tiles`, `bookkeeping`, `probes` and the memory
fields above. Every successful
execution returns `completed_ranges` (half-open output intervals), tile counts,
and `timings_ns` for precheck, prepare, pack, device, copyback, and total.
Window `prepare` covers clearing packed storage and gathering/zeroing old output;
`pack` covers valid source gathers. Immutable template construction belongs to
`plan_build_ns`, not execution. `copyback` includes cursor/prefix updates.
`total` retains the old boundary: entry through final copyback, before final
report construction and its detached return copy. It also includes loop and
bookkeeping work outside the phase clocks. Measure outer coroutine wall time
when accounting for report construction or comparing full host overhead.
`device` measures awaited generated-call wall time, including runtime and
simulation overhead; it is not hardware execution time. Timing calls are part
of production host overhead. `last_report` returns a detached copy of the last
successful report, even while a later attempt is pending or fails.
`completed_ranges` materializes an immutable snapshot of the current/latest
attempt from a completion cursor; reading it while a call awaits observes every
previously copied tile. Existing snapshots never change. Skipped unchanged
ranges are not listed as device-completed ranges. A precheck rejection leaves
previous progress/report untouched.

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
Failure during host preparation, before a new call begins, reports the next
uncopied active range without poisoning the plan. Already completed output is
still retained. This is distinct from a failure inside a generated await, where
completion is unknown and the entire window owner is retained by the rooted
plan alongside runtime, source and destination.

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
