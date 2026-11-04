# Bounded dynamic LOAD execution

ADORA supports a restricted MLIR kernel → CDFG → mapping → checked pytest
execution path for ordinary dynamic `LOAD`. It uses existing IOB address inputs,
SRAM banks, routes and ACC configuration; it does not require a hardware change.

## Supported contract

- Read-only, static rank-one, contiguous signless i16 `ADORA.BlockLoad` objects
  copied in full from function arguments. Inputs and outputs must not alias.
- Dynamic addresses derived from constant or sequential affine input reads,
  i16-to-index casts and bounded index addition/multiplication. Index input
  objects declare `adora.value_range = [lo : i64, hi : i64]`; the compiler checks
  every address against the object and representable arithmetic domain.
- A single-invocation affine body, the fixed-role reduction below, or the
  guarded/sequential stream contract described below. Output allocations and
  host output copies must cover the full result.
- The pytest emitter checks buffer shape, dtype, contiguity, aliasing, index
  ranges and writable outputs before starting transfers. C/SDK emission of
  these dynamic LOAD kernels is rejected because it lacks this checked input
  contract. This is not automatic range inference from arbitrary C input.

The fixed-role reduction has one direct affine loop with constant positive
length, lower bound zero and unit step. It reads `ids[iv]`, performs two dynamic
LOADs, updates two independent zero-initialized i16 additive ACCs and writes
their final results once. The induction variable may only index the sequential
input. The loop body must belong to those address/reduction producer slices.
See [the loop fixture](../../test/cgra-opt/cdfggen/dynamic_load_loop.mlir).

The loop retains five I/O roles: one sequential INPUT, two LOADs and two final
OUTPUTs. LOADs preserve object identity, byte-address operands and `(0,L)`
patterns. The final OUTPUT starts `(L-1)*II` later than the ordinary routed
partial-sum arrival. ACC, IOB timing and address fields must be representable;
each object must fit an available reachable SRAM bank. A nominal trip-count
limit alone does not establish legal execution at that length.

## Guarded gather and sequential outputs

A separate fixed stream contract supports one zero-based, unit-step affine
loop with a constant positive trip count (at most 4095 before physical field
checks), no carried state, sequential `affine.load input[iv]`, ordinary dynamic
LOADs, and exactly one `affine.store output[iv]` per output object. Output
allocations and host copies cover the complete trip-count-sized array. INPUT
and OUTPUT patterns are `(2,M)` in bytes; LOAD uses `(0,M)` with its explicit
byte address. OUTPUTs start at their routed latency, without a reduction's
final-sample delay. Every physical IOB count, stride, latency, II and SRAM
capacity must still be representable.

Nested or merged `scf.if` reads can be legalized when signed i16 coordinate
leaves come from sequential INPUTs and actual `sge`/`slt` guard conjunctions
prove their bounds. The compiler selects each coordinate or zero **before**
index casts and address multiplication/addition, then issues an ordinary LOAD
and selects the original branch result or pass-through value. It validates this
disposable candidate before the existing strict preflights; failure preserves
the original kernel. The proof binds predicates to the selected SSA leaves,
checks every address intermediate and the final object extent, and trusts no
"already safe" annotation. Re-ingestion of normalized swapped-SLE predicates
retains the same proof. NE/EQ and predicate XOR use equivalent signed
comparisons, AND/OR and bounded boolean subtraction supported by the paired
target. Unknown or dependent address loads, division, branch
writes and integer overflow flags are rejected by this specialization.

These safe-address streams accept the complete signed int16 coordinate domain
without declaring or scanning a narrower `adora.value_range`. Existing kernels
that declare a range retain their checked host range contract. Buffer shape,
contiguity, writable output and pairwise non-aliasing checks still run before
DMA. Image pixels use the existing int16 transport as bits: pass a contiguous
uint16 image with `.view(np.int16)` and interpret output with
`.view(np.uint16)`. This preserves values such as 32768, 60000 and 65535 without
changing signed coordinate interpretation or claiming general uint16 arithmetic.
A zero-length application returns outside the kernel without dispatch.

This strategy is **safe-address ordinary LOAD**, not physical predication:
false iterations still read a valid element of the initialized read-only
object (element zero for the remap/gather forms). Requests marked
`adora.require_no_access` or `adora.no_access` on the function, kernel,
BlockLoad or dynamic load are rejected. No CLOAD opcode or hardware change is
introduced. The software graph/configuration tests do not constitute RTL or
board execution.

## Rejection boundaries

Unknown/out-of-range addresses, input writes or aliasing, unsupported host
control flow, dynamic termination, nested loops, extra loop-carried state,
nonzero reduction initial values, incomplete outputs and unrepresentable
configuration fields fail explicitly. This does not add CLOAD, arbitrary
LOAD/STORE role switching, multi-bank objects or a general loop scheduler.
SELECT-based contribution masking does not suppress the physical LOAD and
cannot make an invalid address safe.

I/O allocation failure propagates through both ordinary and tensor-kernel
mapping. A failed invocation must not produce an executable call file.

## Validation

Run `cmake --build build --target check-adora -j2`. Focused coverage is in
`test/cgra-mapper/dynamic_load/`, the two `dynamic_load_*.mlir` CDFG fixtures,
and the GEMM I/O failure test under `test/cgra-mapper/ADORATensor/gemm_funccall/`.
These tests check lowering, rejection, mapping, final configuration and emitted
input validation; they do not execute RTL.

Separate local experiments executed compiler-generated configurations on the
unchanged complete VITRA RTL: plain/expanded dynamic reads and L1/L4 fixed-role
reductions, followed by L16/D2 with a four-row table and L4/D2 with a 512-row
table, using seeds 7 and 19. Traces checked SRAM requests/returns, iteration
alignment, ACC initialization/final state, output writes and controller/runtime
completion. The numeric validation domain used i16 values without partial-sum
overflow. These are RTL simulation results, not board measurements, and do not
establish arbitrary lengths, f32/int64 execution or the L16/N512 combination.
