"""cgeist's folded M=1 form retains symbolic affine source addressing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

tool, fixture = sys.argv[1:]
source = Path(fixture).read_text()
contract = {"entry": "remap_u16", "objects": [
    {"argument_position": i, "elements": 1,
     "host_dtype": "int16" if i in (1, 2) else "uint16",
     "access_assertion": "write" if i == 3 else "read", "contiguous": True}
    for i in range(4)]}
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    def run(text, spec, expected=None):
        (root / "input.mlir").write_text(text)
        (root / "contract.json").write_text(json.dumps(spec))
        result = subprocess.run([tool, str(root / "input.mlir"),
            "--adora-lower-c-entry=contract=" + str(root / "contract.json")],
            capture_output=True, text=True, cwd=root)
        if expected is not None:
            assert result.returncode != 0 and expected in result.stderr, result.stderr
            return
        assert result.returncode == 0, result.stderr
        assert "affine.for" in result.stdout and "to 1" in result.stdout
        assert "scf.if" in result.stdout and "arith.extsi" not in result.stdout
        assert "-1 : i16" in result.stdout
        (root / "bridged.mlir").write_text(result.stdout)
        graph = root / (spec["entry"] + "_CDFG.dot")
        graph.unlink(missing_ok=True)
        env = dict(os.environ, GeneralOpNameFile=str(
            Path(fixture).resolve().parents[3] / "lib/DFG/Documents/GeneralOpName.txt"))
        dfg = subprocess.run([tool, str(root / "bridged.mlir"), "--adora-kernel-dfg-gen"],
                             capture_output=True, text=True, cwd=root, env=env)
        assert dfg.returncode == 0, dfg.stderr
        assert graph.is_file() and graph.stat().st_size and "undefined" not in graph.read_text()

    run(source, contract)
    renamed = dict(contract, entry="tiny_image")
    run(source.replace("remap_u16", "tiny_image"), renamed)
    bad = json.loads(json.dumps(contract)); bad["objects"][3]["elements"] = 2
    run(source, bad, "one-element output")
    run(source.replace("sge, %1, %c0_i32", "sge, %4, %c0_i32"), contract, "guard")
print("C-entry singleton tests passed")
