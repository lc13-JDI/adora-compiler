"""Check the one-bit recurrence and guarded address on direct MLIR input."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

opt, fixture = sys.argv[1:]
original = Path(fixture).read_text()
objects = [dict(argument_position=i, elements=8, contiguous=True, host_dtype=dtype,
                access_assertion=access) for i, (dtype, access) in enumerate([
                    ("uint16", "read"), ("int16", "read"),
                    ("int16", "read"), ("uint16", "inout")])]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    contract = root / "contract.json"
    contract.write_text(json.dumps(dict(entry="paint", objects=objects)))

    def run(mlir, diagnostic=None, required=False):
        src = root / "input.mlir"
        src.write_text(mlir)
        options = f"contract={contract} allow-partial-writes=true"
        if required:
            options += " require-prefix-exit=true"
        result = subprocess.run([opt, str(src), "--verify-each", "--adora-lower-c-entry=" + options],
            capture_output=True, text=True)
        if diagnostic:
            assert result.returncode and diagnostic in result.stderr, result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert result.stdout.count("ADORA.prefix_count") == 1, result.stdout
            assert result.stdout.count("memref.store ") == 1
            assert result.stdout.count("memref.load ") == 1
            assert result.stdout.count("affine.load ") == 2
            assert result.stdout.count("scf.if ") == 2
            assert "arith.cmpi slt," in result.stdout and "arith.cmpi sle," in result.stdout
            assert "iter_args" not in result.stdout
            graph = subprocess.run([opt, "--adora-kernel-dfg-gen", "--verify-each"],
                                   input=result.stdout, capture_output=True, cwd=root, text=True)
            assert graph.returncode == 0, graph.stderr
            dot = (root / "paint_CDFG.dot").read_text()
            acc, = re.findall(r'^(\w+)\[opcode = "ACC", acc_params="0,8,1,1", acc_first=1', dot, re.M)
            # Only the signed comparison and the inclusive-zero test connect
            # to the counter. A memory-order edge here corrupts the LOAD port.
            incident = [line for line in dot.splitlines() if " -> " in line and
                        re.search(rf"\b{acc}\b", line)]
            assert len(incident) == 2 and all("operand = 0," in line for line in incident), incident

    run(original)
    run(original, required=True)
    plain = (Path(fixture).parent / "guarded_inplace.mlir.inc").read_text()
    run(plain.replace("@guarded_inplace", "@paint"), "prefix exit", required=True)
    # A pre-constructed count in source input is never a source proof.
    forged = original.replace("scf.if %6 {", "%fake = ADORA.prefix_count %6\n        scf.if %6 {")
    run(forged, "requires a direct eight-slot kernel loop")
    for before, after in [
        ("= %true)", "= %false)"), ("scf.yield %false", "scf.yield %true"),
        ("scf.yield %6", "scf.yield %arg5"), ("scf.if %arg5 ->", "scf.if %true ->"),
        ("sge, %5,", "uge, %5,"), ("to %c8 step", "to %c1 step"),
        ("load %arg2[%arg4]", "load %arg2[%c1]"),
        ("ne, %8,", "ne, %3,"), ("slt, %3,", "slt, %8,"),
        ("scf.if %13 {", "scf.if %12 {"), ("store %15,", "store %4,"),
        ("store %15, %arg3[%arg4]", "store %15, %arg3[%14]"),
        ("scf.if %6 {", "memref.store %2, %arg3[%arg4] : memref<?xi16>\n        scf.if %6 {"),
        ("arith.extsi %4", "arith.extui %4"),
        ("load %arg2[%arg4] :", "load %arg2[%arg4] {adora.require_no_access} :"),
    ]:
        assert before in original
        run(original.replace(before, after), "prefix exit")

print("prefix exit bridge tests passed")
