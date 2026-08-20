# VITRA CSTORE hardware handoff

This package is the immutable `adora-cstore-v1` hardware-contract input for ADORA Compiler Task 0. It was produced from VITRA-CGRA branch `feature/cstore-hardware` at full commit `15e432f83427fc3121c50e2d4832a76b86d7a64f`, which was verified at `origin/feature/cstore-hardware` before delivery.

## Verify first

From this directory, run:

```sh
sha256sum -c SHA256SUMS
jq empty manifest.json artifacts/operations.json artifacts/adg.json artifacts/generator-input/vitra_spec.json
```

`artifacts/operations.json` and `artifacts/adg.json` are raw byte-for-byte copies from one fresh generation. `artifacts/generator-input/vitra_spec.json` is the resolved generator input emitted by that same run. Their provenance and SHA-256 values are machine-readable in `manifest.json`.

## Source and generation

- Repository: `https://github.com/MIONkb/VITRA-CGRA.git`
- Branch: `feature/cstore-hardware`
- Commit: `15e432f83427fc3121c50e2d4832a76b86d7a64f`
- Generator: `tram.vitra.CStoreVerilogGen`
- Fresh output ID: `vitra-cstore-redelivery.xeNXdq`
- Lab Chipyard commit: `27981d26675e7fd2e41d772375b5632951ece968`
- fdra integration commit: `0cb9381fa231b4d26881779a80a5f0c78ec4c5ff`
- Toolchain: sbt 1.8.2, OpenJDK 11.0.27

The exact final commit was exported with `git archive` into a clean temporary source tree and synchronized into the established writable `/tmp` Chipyard build copy. `/home/jhlou/chipyard` was only the external build/elaboration environment and was not modified.

Generation command:

```sh
env XDG_RUNTIME_DIR=/tmp/vitra-task1-sbt/runtime TMPDIR=/tmp/vitra-task1-sbt/runtime COURSIER_CACHE=/tmp/vitra-task1-sbt/coursier JAVA_TOOL_OPTIONS='-Djava.io.tmpdir=/tmp/vitra-task1-sbt/runtime -Dsbt.global.base=/tmp/vitra-task1-sbt/global -Dsbt.boot.directory=/tmp/vitra-task1-sbt/boot -Dsbt.ivy.home=/tmp/vitra-task1-sbt/ivy -Dsbt.coursier.home=/tmp/vitra-task1-sbt/coursier -Dsbt.server.autostart=false -Dsbt.server.forcestart=true' /home/jhlou/chipyard/.conda-env/bin/sbt -java-home /usr/lib/jvm/java-11-openjdk-amd64 -batch 'project fdra' 'runMain tram.vitra.CStoreVerilogGen -td /tmp/vitra-cstore-redelivery.xeNXdq'
```

The schema-correct redelivery command exited 0 in 104 seconds and its built-in CSTORE artifact audit passed. It completed at `2026-08-19T21:19:16Z`; the generated semantic JSON and RTL hashes are identical to the final-commit generation recorded below.

## Contract for ADORA

- Logical operands are `data=0`, `address=1`, `enable=2`.
- Address operand 1 is a byte address/byte offset. For this 16-bit datapath, hardware converts it to an SRAM word address by dividing by 2.
- `UseEn` is a static conditional-IOB configuration field: config ID 19, aggregate bit 127.
- Runtime enable is operand 2; predicate truth is word bit 0. Thus values 0 and 2 are false and value 1 is true.
- `UseEn=0` preserves normal STORE and ignores operand 2. `UseEn=1` permits a write only when operand 2 bit 0 is set.
- Production `cgra_iob_sram_add_reg` is 2. Data, byte address, predicate, request enable, write mask, address, and data were validated as one aligned transaction through this registered path.
- CSTORE is `OPC=4`, `numOperands=3`, `numRes=0`, `latency=1`.
- The generated IOB capability is exactly `INPUT`, `OUTPUT`, `LOAD`, `STORE`, `CSTORE`.
- CLOAD is out of scope, is not implemented, and is not advertised by either delivered JSON. ADORA must not infer CLOAD merely from conditional IOB mode.

## Recorded hardware proof

The 16-bit memory oracle used `OLD=0x0bad` and `NEW=0x55aa`: true CSTORE produced exactly one `en=1`, `we=0b11` write and stored NEW; false CSTORE preserved OLD with zero requests (`en=0`, `we=0`) while `done` still asserted. Alternating predicates `1,0,1,0` at II=1 wrote only iterations 0 and 2 with distinct data/address pairs. Normal STORE with `UseEn=0` and predicate 0 still wrote normally. The final full fdra regression passed 35/35 tests in 5 suites with zero failures in 233 seconds.

See `evidence/hardware-validation.md` for the Task 1–4 audit trail. This handoff does not start or modify Mapper development.
