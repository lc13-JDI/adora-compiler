# Checked private canvas-to-mask C entry

`adoracc.py --profile guarded-canvas-mask` accepts a restricted ordinary C
producer loop followed by a pointwise mask loop. It is opt-in; existing
`guarded-inplace-large` entries retain their original partial-write semantics
and skip wholly inactive tiles.

The producer initializes a private `uint16_t canvas[T]` from an independent
read-only `old`, then updates valid enabled coordinates with a depth in a
positive inclusive uint16 range or zero. The consumer fully defines an
independent uint16 mask as `canvas[i] != 0 ? 255 : 0`. The typed bridge proves
the two equal static iteration domains, complete initialization, unique
same-slot consumption, every allocation use, absence of observable intermediate
effects, the guarded wide source address and both uint16 value functions.
Parameter names and contract macros do not establish these facts. Renaming and
an equivalent SSA merge with an unconditional final private store are supported.
Views, escapes, additional consumers, altered indices, incomplete initialization,
volatile accesses and unproved effects are rejected.

The explicit object contract covers all six pointers, shapes, uint16/int16 host
types, read/write access and contiguous storage. `host_plan` uses
`strategy: per-query`, `tile_items: 256`, `ownership: copy`, and the three bound
coordinate/enable argument positions. It cannot assert that fusion is legal.
The compiler infers object roles and the positive inclusive depth interval from
typed SSA. Source address arithmetic uses the target 64-bit `size_t` domain.

```bash
python3 tools/adoracc/adoracc.py input.c --profile guarded-canvas-mask \
  --contract objects.json --adg-path adg.json --op-file operations.json \
  --work-dir /new/private/run -o /new/private/run/mapped.py \
  --seed 7 --max-iters 30 --timeout 30000 --mapper-wall-timeout 120
```

Set `PATH` to the matching cgeist, Clang and ADORA tools and set
`GeneralOpNameFile` to `lib/DFG/Documents/GeneralOpName.txt`. A new output is
required. `adora-c-entry/frontend/` retains original/preprocessed C, Clang AST,
promotion normalization and cgeist raw MLIR. `regular/` and `old-only/` each
retain their typed proof, local contract, IR, DFG, map and generated call.
`host-proof.json` and `provenance.json` record the checked public ABI and hashes.

The regular tile reads packed source/index/enable and independent old values,
then writes all 256 device mask slots. The old-only tile reads only old and
writes all mask slots. Classification uses only the immutable geometry and
enable snapshot; zero or out-of-range pixels never make a valid tile old-only.
The unsigned depth range uses a proven 16-bit bias and signed comparisons.
The final nonzero select uses its proven-zero value as the false operand, so
only one literal is needed. The driver rejects multiple immediate inputs and
checks mapped operand/constant relationships before publishing any callable.

```python
plan = mapped.make_plan(x, y, enabled)  # owns immutable copies
try:
    await plan.execute(runtime, src_frame0, old_frame0, mask_frame0)
    await mapped.execute_plan(runtime, plan, src_frame1, old_frame1, mask_frame1)
finally:
    if not plan.poisoned:
        plan.close()
```

The original six-argument function name is also an async single-call wrapper.
Both APIs check host shapes, dtypes, contiguous storage, output writeability
and nonoverlap before device submission. Plans are bound to their generated
module identity. Original coordinate arrays may change after plan creation;
execution uses owned snapshots. Source and old are fresh borrowed read-only
inputs, and callers must not mutate their aliases until completion.

One serial device call completes each logical tile, including wholly inactive
ones. A bounded window (32 tiles by default) holds packed buffers; there is no
host-sized intermediate canvas. Only valid tail elements are copied back.
Private device padding still executes and writes, but cannot update adjacent
host output. A successful call commits one completed prefix tile. Failure or
cancellation leaves every uncompleted mask byte as supplied by the caller.
Unknown completion poisons the plan and retains runtime, borrowed arrays and
scratch buffers for runtime/process teardown; `close()` cannot release them.
No automatic cancellation recovery or general buffer ownership is claimed.

Optional `--kernel-cache` entries have a separate canvas namespace and include
role, geometry, range, proof, tools, hardware, compiler/support and mapping
options. Both kernels and their inventories must validate; missing, changed or
mixed components are rejected. Publishing the public module happens only after
both mappings and final provenance are complete, without overwriting an existing
output. Shared runtime, ADG/operations and VITRA are unchanged.

Software tests under `test/adoracc/canvas_mask/` cover typed proof boundaries,
operand preservation, publication, cache identity, owned plans and failure
lifetimes. Fake transport tests establish host protocol only; hardware claims
require the separate R014 original-RTL transaction and readback evidence.
