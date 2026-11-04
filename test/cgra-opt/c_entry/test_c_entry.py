"""Structural C-entry normalization and fail-closed contract checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

tool, fixture = sys.argv[1:]
source = Path(fixture).read_text()
contract = {"entry": "remap_u16", "objects": [
    {"argument_position": i, "elements": n, "host_dtype": dtype,
     "access_assertion": access, "contiguous": True}
    for i, n, dtype, access in [(0, 64, "uint16", "read"),
                               (1, 16, "int16", "read"),
                               (2, 16, "int16", "read"),
                               (3, 16, "uint16", "write")]]}

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    def run(text, specification, expected=None):
        (root / "input.mlir").write_text(text)
        (root / "contract.json").write_text(json.dumps(specification))
        result = subprocess.run([tool, str(root / "input.mlir"),
            "--adora-lower-c-entry=contract=" + str(root / "contract.json")],
            capture_output=True, text=True, cwd=root)
        if expected is None:
            assert result.returncode == 0, result.stderr
            assert "ADORA.kernel" in result.stdout
            assert "affine.for" in result.stdout
            assert "arith.extsi" not in result.stdout
            assert "scf.if" in result.stdout
            assert "memref<64xi16>" in result.stdout
            assert "arith.muli" in result.stdout and " : index" in result.stdout
            (root / "bridged.mlir").write_text(result.stdout)
            graph = root / (specification["entry"] + "_CDFG.dot")
            graph.unlink(missing_ok=True)
            env = dict(os.environ, GeneralOpNameFile=str(
                Path(fixture).resolve().parents[3] / "lib/DFG/Documents/GeneralOpName.txt"))
            dfg = subprocess.run([tool, str(root / "bridged.mlir"),
                                  "--adora-kernel-dfg-gen"], cwd=root,
                                 env=env, capture_output=True, text=True)
            assert dfg.returncode == 0, dfg.stderr
            assert graph.is_file() and graph.stat().st_size > 0
            assert "undefined" not in graph.read_text()
            return result.stdout
        assert result.returncode != 0, result.stdout
        assert expected in result.stderr, result.stderr

    run(source, contract)
    run(source.replace("scf.for %arg4 = %c0 to %c16 step %c1", "affine.for %arg4 = 0 to 16"), contract)
    nested = source.replace("%11 = scf.if %10 -> (i16) {", "%11 = scf.if %4 -> (i16) {\n        %inner = scf.if %9 -> (i16) {")
    nested = nested.replace("scf.yield %15 : i16\n      } else {", "scf.yield %15 : i16\n        } else {\n          scf.yield %c-5536_i16 : i16\n        }\n        scf.yield %inner : i16\n      } else {")
    run(nested, contract)
    renamed = json.loads(json.dumps(contract)); renamed["entry"] = "unrelated_name"
    run(source.replace("remap_u16", "unrelated_name"), renamed)
    run(source.replace("arith.cmpi sge, %1, %c0_i32", "arith.cmpi sle, %c0_i32, %1"), contract)
    run(source.replace("arith.muli %3, %c8_i32 : i32", "arith.muli %3, %c8_i32 overflow<nsw, nuw> : i32")
              .replace("arith.addi %12, %1 : i32", "arith.addi %12, %1 overflow<nsw, nuw> : i32"), contract)
    missing = json.loads(json.dumps(contract)); missing["objects"].pop()
    run(source, missing, "object contract")
    bad = json.loads(json.dumps(contract)); bad["objects"][0]["elements"] = 0
    run(source, bad, "elements")
    bad = json.loads(json.dumps(contract)); bad["objects"][1]["access_assertion"] = "write"
    run(source, bad, "effects")
    wide = source.replace("%arg1: memref<?xi16>", "%arg1: memref<?xi32>")
    wide = wide.replace("%arg1[%arg4] : memref<?xi16>", "%arg1[%arg4] : memref<?xi32>")
    wide = wide.replace("arith.extsi %0 : i16 to i32", "arith.addi %0, %c0_i32 : i32")
    run(wide, contract, "i16")
    run(source.replace("arith.extsi %0 : i16 to i32", "arith.constant 65536 : i32"), contract, "i16")
    run(source.replace("sge, %1, %c0_i32", "uge, %1, %c0_i32"), contract, "unsigned comparison")
    run(source.replace("memref<?xi16>", "memref<16xi16>"), contract, "contradicts")
    bad = json.loads(json.dumps(contract)); bad["objects"][0]["elements"] = 4097
    run(source, bad, "target bank")
    run(source.replace("arith.muli %3, %c8_i32 : i32", "arith.muli %3, %c8_i32 {unmodeled} : i32"), contract, "attributes")
    run(source.replace("%c16 = arith.constant 16", "%c16 = arith.constant 0"), contract, "positive")
    run(source.replace("arith.cmpi sge, %1, %c0_i32", "arith.cmpi sge, %3, %c0_i32"), contract, "guard")
    run(source.replace("scf.if %10", "scf.if %4"), contract, "guard")
    runtime = source.replace("scf.for %arg4 = %c0 to %c16 step %c1", "%runtime = memref.dim %arg0, %c0 : memref<?xi16>\n    scf.for %arg4 = %c0 to %runtime step %c1")
    run(runtime, contract, "unsupported entry operation")
    run(source.replace("%c8_i32 = arith.constant 8", "%c8_i32 = arith.constant 32768"), contract, "i16")
    run(source.replace("%c8_i32 = arith.constant 8", "%c8_i32 = arith.constant 32767"), contract, "guard")
    run(source.replace("%15 = memref.load %arg0[%14]", "%15 = memref.load %arg0[%14] {adora.require_no_access}"), contract, "attributes")
    for key, value in [("no_speculative_access", True), ("may_alias_with", [3]), ("unknown", True)]:
        bad = json.loads(json.dumps(contract)); bad["objects"][0][key] = value
        run(source, bad, "unsupported object contract key")
    for endpoint in (-1, -32768, 32767):
        # Changing the fallback tests exact i16 bits, independently of coordinate
        # sign checks. Runtime endpoint behavior is covered by RTL/C oracles.
        result = run(source.replace("-5536 : i16", str(endpoint) + " : i16"), contract)
        assert str(endpoint) + " : i16" in result
print("C-entry structural tests passed")
