"""Structural partial-write bridge; no case names influence production lowering."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

tool, fixture = sys.argv[1:]
fixtures = Path(fixture).resolve().parent

def contract(entry, sizes):
    return {"entry": entry, "objects": [
        {"argument_position": i, "elements": n,
         "host_dtype": "uint16" if i in (0, len(sizes) - 1) else "int16",
         "access_assertion": "inout" if i == len(sizes) - 1 else "read",
         "contiguous": True} for i, n in enumerate(sizes)]}

overlay = Path(fixture).read_text()
base = (fixtures / "guarded_inplace.mlir.inc").read_text()
tiny = (fixtures / "guarded_inplace_singleton.mlir.inc").read_text()
remap = (fixtures / "transparent_remap.mlir.inc").read_text()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    def run(text, spec, expected=None, enabled=True):
        (root / "input.mlir").write_text(text)
        (root / "contract.json").write_text(json.dumps(spec))
        options = "contract=" + str(root / "contract.json")
        if enabled:
            options += " allow-partial-writes=true"
        result = subprocess.run([tool, str(root / "input.mlir"),
            "--adora-lower-c-entry=" + options], cwd=root, capture_output=True, text=True)
        if expected is not None:
            assert result.returncode != 0, result.stdout
            assert expected in result.stderr, result.stderr
            return
        assert result.returncode == 0, result.stderr
        assert "ADORA.LocalMemAlloc" not in result.stdout
        assert "arith.extui" not in result.stdout and "arith.extsi" not in result.stdout
        assert "scf.if" in result.stdout and "memref.store" in result.stdout
        output = len(spec["objects"]) - 1
        moved = re.search(r'(%\w+) = ADORA.BlockLoad %arg' + str(output) + r'\b', result.stdout)
        assert moved, result.stdout
        local = moved.group(1)
        assert re.search(r'memref.store .*?, ' + re.escape(local) + r'\[', result.stdout)
        assert "ADORA.BlockStore " + local + ", %arg" + str(output) in result.stdout
        assert "affine.for" in result.stdout
        if "arith.extui" in text:
            pixel = re.search(r'(%\w+) = memref.load ', result.stdout)
            assert pixel, result.stdout
            assert re.search(r'arith.cmpi ne, ' + re.escape(pixel.group(1)) + r', .* : i16', result.stdout)
        # Backend cooperation is covered by a separate RUN after its joint
        # legalization lands; this test isolates bridge structure and effects.
        return result.stdout

    spec = contract("nonzero_overlay", [4, 6, 6, 6])
    run(overlay, spec)
    run(base, contract("guarded_inplace", [4, 4, 4, 4]))
    nested = base.replace("scf.if %8 {", "scf.if %4 {\n        scf.if %5 {\n          scf.if %6 {")
    nested = nested.replace("memref.store %10, %arg3[%arg4] : memref<?xi16>\n      }",
                            "memref.store %10, %arg3[%arg4] : memref<?xi16>\n          }\n        }\n      }")
    run(nested, contract("guarded_inplace", [4, 4, 4, 4]))
    run(tiny, contract("guarded_inplace", [4, 1, 1, 1]))
    run((fixtures / "guarded_inplace_s1.mlir.inc").read_text(),
        contract("guarded_inplace", [1, 1, 1, 1]))
    run(remap, contract("transparent_remap", [6, 8, 8, 8, 8]))
    renamed = dict(spec, entry="copy_when_selected")
    run(overlay.replace("nonzero_overlay", "copy_when_selected"), renamed)
    run(overlay.replace("ne, %11, %c0_i32", "ne, %c0_i32, %11"), spec)
    run(overlay, spec, "inout", enabled=False)
    wrong = json.loads(json.dumps(spec)); wrong["objects"][-1]["access_assertion"] = "write"
    run(overlay, wrong, "inout")
    wrong = json.loads(json.dumps(spec)); wrong["objects"][-1]["access_assertion"] = "read"
    run(overlay, wrong, "effects")
    run(overlay.replace("%arg3[%arg4]", "%arg3[%9]"), spec, "same iteration")
    store = "memref.store %10, %arg3[%arg4] : memref<?xi16>"
    run(overlay.replace(store, store + "\n          " + store), spec, "one conditional")
    run(overlay.replace(store, "%old = memref.load %arg3[%arg4] : memref<?xi16>\n          " + store), spec, "reads")
    run(overlay.replace("sge, %1, %c0_i32", "sge, %3, %c0_i32"), spec, "guard")
    late_bounds = overlay.replace("scf.if %8 {", "scf.if %4 {")
    late_bounds = late_bounds.replace("scf.if %12 {", "%late = arith.andi %7, %12 : i1\n        scf.if %late {")
    run(late_bounds, spec, "guard")
    run(overlay.replace("ne, %11, %c0_i32", "sgt, %11, %c0_i32"), spec, "zero comparison")
    run(overlay.replace("memref.store %10,", "memref.store %0,"), spec, "loaded value")
    run(overlay.replace("%10 = memref.load %arg0[%9]", "%10 = memref.load %arg0[%9] {adora.no_access}"), spec, "attributes")
print("C-entry partial-write structural tests passed")
