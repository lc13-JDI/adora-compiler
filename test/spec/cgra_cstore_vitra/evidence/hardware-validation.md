# Hardware validation evidence

This record summarizes the completed VITRA-CGRA Task 1–4 evidence that defines the artifacts in this package. The authoritative source is branch `feature/cstore-hardware`, commit `15e432f83427fc3121c50e2d4832a76b86d7a64f`.

## Task 1 — baseline and failure localization

- Verified the lab Chipyard project list contains `fdra` and the README VITRA generation command works using the isolated writable build copy.
- Recorded Chipyard `27981d26675e7fd2e41d772375b5632951ece968`, fdra `0cb9381fa231b4d26881779a80a5f0c78ec4c5ff`, sbt 1.8.2, and OpenJDK 11.0.27.
- SRAM_MODE baseline elaborated successfully.
- Pre-fix COND_LS_MODE failed at the real first user frame `tram.dsa.IOB.$anonfun$new$6(IOB.scala:132)` with `IndexOutOfBoundsException: 2`, exposing the three-to-two operand discontinuity.

## Task 2 — truthful three-operand structure

- Established real logical operands data 0, byte address 1, and enable/predicate 2 through independent muxes, three SharedDelayPipe lanes, controller inputs, interconnect, and generated ADG.
- Conditional capability was narrowed to exactly `INPUT`, `OUTPUT`, `LOAD`, `STORE`, `CSTORE`; no CLOAD claim.
- Predicate encoding was fixed by source and tests to uniform-width operand 2 word bit 0. Values 0 and 2 are false; value 1 is true.

## Task 3 — write gating and cycle proof

- Added conditional-only static `UseEn` as config ID 19 at aggregate bit 127, without shifting legacy FIFO/SRAM fields.
- Runtime write permission is `!UseEn || operand2(0)`. It gates the final conditional SRAM request enable and write mask, not `wValid`, FSM, II counter, cycle counter, address generation, or `done`.
- Production timing uses `addRegSram=2`; gated valid/mask, address, and data remain aligned as one transaction.
- True oracle: word address 3 changed from `OLD=0x0bad` to `NEW=0x55aa` via exactly one `en=1`, `we=0b11` request.
- False oracle: OLD remained `0x0bad`, with zero write requests, `en=0`, `we=0`, and `done=1`.
- Alternating II=1 trace:

| iteration | data | byte address | predicate | observed result |
| --- | --- | --- | --- | --- |
| 0 | `0x1011` | 0 | 1 | `en=1`, `we=0x3`, word address 0, write |
| 1 | `0x2022` | 2 | 0 | `en=0`, `we=0x0`, no write |
| 2 | `0x3033` | 4 | 1 | `en=1`, `we=0x3`, word address 2, write |
| 3 | `0x4044` | 6 | 0 | `en=0`, `we=0x0`, no write, `done=1` |

- Normal STORE regression: `UseEn=0` with predicate 0 still created exactly one normal write. Normal LOAD remained independent of predicate.

## Task 4 — production artifacts and regression

- Added the separate production entrypoint `tram.vitra.CStoreVerilogGen`; default `tram.vitra.VerilogGen` remains SRAM_MODE.
- Generator-owned CSTORE record is `OPC=4`, `numOperands=3`, `numRes=0`, `latency=1`.
- Generated conditional IOB is mode 2, has three logical operands and six physical inputs, exposes `IsStore`/`UseAddr`/`UseEn`, and uses `cgra_iob_sram_add_reg=2`.
- Generated operation catalog and ADG are mutually consistent and omit CLOAD.
- Final full fdra regression: 35 tests, 5 suites, 35 succeeded, 0 failed, 233 seconds.
- Existing ADORA schema parse smoke succeeded with 240 GPE, 48 IOB, and 8 tiles. It was a read-only compatibility check, not Mapper development.

## Final commit-bound generation

After the final code/documentation commit was pushed, commit `15e432f83427fc3121c50e2d4832a76b86d7a64f` was exported with `git archive` and freshly generated once into `/tmp/vitra-cstore-handoff-final.H1PHq5`. The command exited 0 in 90 seconds and the generator's complete artifact audit passed.

- `operations.json`: `0eee215afdbed65fed6bd7773f40a783189d67a66accb6e7bc86aaac582bae8d`
- `adg.json`: `e34fcef5b15f47718762812513d3cd7db35f06e60a17097e50429dfc41e26cf7`
- resolved `vitra_spec.json`: `3155ec148eee0efd4dbe8cae0c58359bd4deba4a140c9a64ed2bf5d0490d832d`
- generated RTL, retained by hash only: `96e709d80a6c8a007fd650af934ceab69df3d25fb99beb62315628993fc46752`

The two delivered semantic JSON files and resolved generator input are raw byte copies from that same run.

## Schema-correct redelivery

ADORA Task 0 found that the first physical package represented the semantic
contract as an object but omitted the required top-level identifier
`contract="adora-cstore-v1"` and generation timestamp. The producer rebuilt
the package instead of editing incoming payload in place. A fresh generation
from the same clean, pushed commit completed at `2026-08-19T21:19:16Z` in 104
seconds with exit 0 and a passing built-in artifact audit. Its RTL,
operations, ADG, and resolved generator-input hashes are identical to the
final-commit generation above. The replacement manifest uses the v1 schema,
records the timestamp, structured hardware results, source blob provenance,
and the retained RTL hash.

## Scope boundary for ADORA Task 0

CLOAD is out of scope, not implemented, and absent from both delivered capability artifacts. At the previously inspected ADORA reader, `mapper/src/ir/adg_ir.cpp:148-154` inferred CLOAD and CSTORE from conditional mode rather than trusting the artifact's exact operations list. ADORA Task 0 must validate and preserve the delivered contract instead of treating mode 2 as evidence of CLOAD. No ADORA tracked file was modified during this handoff.
