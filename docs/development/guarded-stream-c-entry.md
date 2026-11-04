# Checked ordinary C entry for guarded streams

`adoracc.py --profile guarded-stream` compiles a restricted ordinary C function
through cgeist, the checked C-entry bridge, production CDFG generation, the real
mapper, and the pytest/runtime emitter. The generated Python exports the C
function name as an async checked entry. The default driver pipeline is unchanged.

This profile supports fixed positive trip counts over one-dimensional contiguous
16-bit buffers: signed `short` coordinates, `unsigned short` pixels, read-only
input objects, and a distinct output written once per iteration. Typedef aliases
of those element types are supported; ambiguous shadowed
aliases are conservatively rejected. Guarded gather
and nearest-coordinate remap can use nested `if` or equivalent conjunctions.
Object sizes are explicit runtime contracts; C array parameter declarations do
not prove allocation sizes. The compiler derives read/write effects from IR and
requires them to agree with the contract. It proves guard-linked address ranges
before any narrowing. Invalid coordinates use a safe source address on hardware
and preserve the C fallback value. This requires ordinary nonvolatile memory
where a safe extra read is permitted.

A minimal input and matching contract are:

```c
#ifndef N
#define N 4
#endif
#ifndef M
#define M 4
#endif
void gather(const unsigned short *src, const short *idx, unsigned short *dst) {
  for (int i = 0; i < M; ++i) {
    unsigned short value = 60000;
    if (idx[i] >= 0 && idx[i] < N)
      value = src[idx[i]];
    dst[i] = value;
  }
}
```

```json
{
  "entry": "gather",
  "defines": {"N": 4, "M": 4},
  "objects": [
    {"argument_position": 0, "elements": 4, "host_dtype": "uint16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 1, "elements": 4, "host_dtype": "int16", "access_assertion": "read", "contiguous": true},
    {"argument_position": 2, "elements": 4, "host_dtype": "uint16", "access_assertion": "write", "contiguous": true}
  ]
}
```

`defines` contains integer preprocessing constants. Every argument position must
appear exactly once, starting at zero. An optional object `name` and optional
root `id` and `source` are informational; names never select compiler templates.
Unknown fields are rejected, including strong no-speculative-access or permitted
alias assertions. These memory models are outside this profile.

Put the desired build's `bin` and cgeist on `PATH`; provide `clang` for semantic
source checks. The source driver prefers adjacent compiler tools, then `PATH`.
The `GeneralOpNameFile` environment variable can select the compiler opcode name
table; otherwise the driver searches its repository ancestors. This is distinct
from the hardware operations JSON.

```bash
RUN="$(mktemp -d /tmp/adora-c-entry.XXXXXX)"
python3 tools/adoracc/adoracc.py gather.c \
  --profile guarded-stream --contract gather.json \
  --adg-path test/spec/cgra_cstore_vitra/artifacts/adg.json \
  --op-file test/spec/cgra_cstore_vitra/artifacts/operations.json \
  --work-dir "$RUN" -o "$RUN/mapped.py" \
  --seed 7 --max-iters 1000 --timeout 120000
```

The hardware must declare a 16-bit datapath, SRAM-mode dynamic LOAD, and bank
capacity sufficient for every complete object. The bridge receives the actual
ADG bank size. The mapper checks the resulting graph against the selected
hardware. Mapping budgets limit search and are not a proof of hardware
impossibility. This single fixed kernel omits task scheduling and generic
footprint/unrolling passes deliberately.

The target is explicitly `x86_64-unknown-linux-gnu`, with a forced compile-time
check for 8-bit bytes, 16-bit `short`, and 32-bit `int`. The driver first
preprocesses the original C plus macros and retains that output. Both Clang's
semantic AST check and cgeist consume this identical expanded C, so different
frontend versions cannot choose different preprocessing branches. cgeist then
runs `-O2 -S --function=<entry>`. The original C, target-width assertions, AST,
frontend diagnostics, and raw MLIR remain unchanged in the work directory.
Source qualifiers/annotations are checked before optimized IR can lose them.
Because Polygeist and ADORA can use different LLVM versions, only a separate
`frontend/bridge-input.mlir` work copy has module metadata removed using the
existing driver helper. Raw IR and the original target triple/data layout are
retained in provenance. Arithmetic operation attributes are left for the bridge
to prove or reject.

The public API is:

```python
await mapped.gather(runtime, source_uint16, coordinates_int16, output_uint16)
```

Every array must have exactly the compiled one-dimensional shape, matching
native NumPy dtype and contiguous storage. Outputs must be writable and no pair
of arrays may alias. These checks finish before the generated entry creates a
stream, transfers data, or starts execution, and remain enabled with `python -O`.
The wrapper uses equal-width `view(np.int16)` for transport, preserving all pixel
bits and output writeback without a numeric cast. Input coordinates are neither
clipped nor sanitized on the host. Empty arrays are rejected for positive-trip
compiled configurations; changing dimensions requires another compilation.
The generated implementation runs in private Python globals. Its entry is
identified by the emitted ABI signature and renamed structurally using Python's
AST, so valid names such as `np`, `len`, or `aux_stream` cannot shadow runtime
helpers or builtins. The public module exports the original C entry name.

Artifacts under `<work-dir>/adora-c-entry/` are:

| Path | Contents |
| --- | --- |
| `frontend/input.c`, `preprocessed.c`, `source.ast.json`, `raw.mlir` | Source snapshot and actual frontend artifacts |
| `contract.original.json`, `contract.json` | User contract and normalized argument order |
| `adg.json`, `operations.json`, `GeneralOpName.txt` | Exact target and opcode tables |
| `ir/bridged.mlir`, `ir/final.mlir` | Verified bridge IR and DFG-pass output |
| `dfg/*_CDFG.dot` | Fresh production DFGs |
| `map/unchecked.py` | Raw generated runtime implementation; use the checked public output instead |
| `map/generated.py` | Readable private implementation with the checked ABI wrapper |
| `map/*_map_result/` | Actual mapping graphs, routes, IO bindings, and `config.bit` |
| `provenance.json`, `*.log` | Commands, return codes, source/tool/artifact SHA256s and failure diagnostics |

The final `--output` is a new standalone `.py` file embedding the real generated
implementation and exporting its checked public entry. Existing output files and an existing `adora-c-entry/`
directory are rejected to prevent old artifacts being mistaken for success.
Every required frontend, proof, DFG, and mapping stage must succeed; there is no
fallback to earlier IR or CPU execution. A completed compile/map does not itself
claim RTL execution or readback validation.

Runtime sizes, general int32 buffers/arithmetic, float interpolation, state
carried across iterations, aliasing writes, volatile/MMIO/atomic memory,
no-access annotations, dependent loads, branch stores, and `break`/`continue`
are unsupported and must be diagnosed. Bounds, unsupported operations, and
read/write assertions are checked by the C++ IR bridge. Frontend annotations
are conservatively rejected instead of assuming they are irrelevant.

Run the focused contract and checked-call tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 test/adoracc/c_entry/test_c_entry.py .
PYTHONDONTWRITEBYTECODE=1 python3 -O test/adoracc/c_entry/test_c_entry.py .
```

The corresponding lit entry is `test/adoracc/c_entry/checks.mlir`. Actual C to
mapper and RTL validation additionally requires the corresponding frontend,
hardware package, and runtime.
