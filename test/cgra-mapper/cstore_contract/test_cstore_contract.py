#!/usr/bin/env python3
"""Black-box ADG and DFG contract checks for CSTORE."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


CASES = (
    "no-cstore",
    "missing-use-en",
    "two-operands",
    "missing-enable-path",
    "overlapping-input-paths",
    "missing-intra-endpoint",
    "cyclic-intra-connect",
    "invalid-cstore-operands",
    "invalid-cstore-results",
)

HARDWARE_CASES = frozenset(CASES[1:7])


def load(path):
    with path.open() as stream:
        return json.load(stream)


def dump(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def cstore_iob(adg):
    for module in adg["sub_modules"]:
        attrs = module.get("attributes", {})
        if module.get("type") == "IOB" and "CSTORE" in attrs.get("operations", []):
            return attrs
    raise AssertionError("pinned ADG has no CSTORE IOB")


def mutate(case, adg, operations):
    attrs = cstore_iob(adg)
    if case == "no-cstore":
        attrs["operations"].remove("CSTORE")
        return "No legal ADG node supports CSTORE", "required=1", "available=0"
    if case == "missing-use-en":
        del attrs["io_controller_cfg_id"]["UseEn"]
        return "missing UseEn",
    if case == "two-operands":
        attrs["num_operands"] = 2
        return "expected num_operands=3",
    if case == "missing-enable-path":
        del attrs["connections"]["12"]
        return "operand 2 has no physical input path",
    if case == "overlapping-input-paths":
        for connection in attrs["connections"].values():
            if connection[0] == 0 and connection[2] == 4:
                connection[3] = 3
                connection[5] = 0
                attrs["connections"][str(max(map(int, attrs["connections"])) + 1)] = \
                    [0, "This", 4, 5, "Muxn", 0]
                return "physical input sets overlap",
        raise AssertionError("pinned CSTORE IOB has no physical input 4")
    if case == "missing-intra-endpoint":
        attrs["connections"]["12"][3] = 99
        return "malformed endpoint", "iob_index 0", "edge 12"
    if case == "cyclic-intra-connect":
        attrs["connections"]["11"][3] = 5
        attrs["connections"]["11"][4] = "Muxn"
        attrs["connections"]["11"][5] = 1
        return "operand 2 has no physical input path",
    for operation in operations["Operations"]:
        if operation["name"] != "CSTORE":
            continue
        if case == "invalid-cstore-operands":
            operation["numOperands"] = 2
        elif case == "invalid-cstore-results":
            operation["numRes"] = 1
        else:
            raise AssertionError(f"unknown case: {case}")
        return "Invalid CSTORE DFG node",
    raise AssertionError("pinned operations has no CSTORE entry")


def store_kernel():
    return """module {
  func.func @store_only(%value: i32, %output: memref<8xi32>) {
    %index = arith.constant 0 : index
    ADORA.kernel {
      memref.store %value, %output[%index] : memref<8xi32>
      ADORA.terminator
    } {KernelName = \"store_only\"}
    return
  }
}
"""


def run(mapper, adg_path, operations_path, kernel, output, environment, cwd):
    command = [
        str(mapper), "--seed=7", f"--adg={adg_path}",
        f"--op-file={operations_path}", "--output-type=pytest",
        "--obj-opt=false", "--max-iters=30", "--timeout=30000",
        str(kernel), f"--output={output}",
    ]
    return subprocess.run(command, cwd=cwd, text=True, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=60, check=False)


def run_case(case, workdir, mapper, adg_source, operations_source, kernel):
    case_dir = workdir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    adg_path = case_dir / "adg.json"
    operations_path = case_dir / "operations.json"
    shutil.copyfile(adg_source, adg_path)
    shutil.copyfile(operations_source, operations_path)
    adg = load(adg_path)
    operations = load(operations_path)
    expected = mutate(case, adg, operations)
    dump(adg_path, adg)
    dump(operations_path, operations)

    environment = dict(os.environ)
    environment["GeneralOpNameFile"] = str(
        kernel.parents[3] / "lib/DFG/Documents/GeneralOpName.txt")
    result = run(mapper, adg_path, operations_path, kernel,
                 case_dir / "mapped.py", environment, case_dir)
    if result.returncode != 1:
        raise AssertionError(
            f"{case}: expected exit status 1, got {result.returncode}\n{result.stdout}")
    for message in expected:
        if message not in result.stdout:
            raise AssertionError(
                f"{case}: missing diagnostic {message!r}\n{result.stdout}")
    if case in HARDWARE_CASES and "Invalid CSTORE IOB" not in result.stdout:
        raise AssertionError(f"{case}: missing CSTORE IOB identity\n{result.stdout}")
    if case == "no-cstore":
        normal_store = case_dir / "normal-store.mlir"
        normal_store.write_text(store_kernel())
        normal_result = run(mapper, adg_path, operations_path, normal_store,
                            case_dir / "normal-store.py", environment, case_dir)
        if normal_result.returncode != 0:
            raise AssertionError(
                f"no-cstore normal STORE failed with {normal_result.returncode}\n"
                f"{normal_result.stdout}")
        if not (case_dir / "normal-store.py").is_file():
            raise AssertionError("no-cstore normal STORE produced no output")
        cdfg = case_dir / "store_only_map_result" / "before_map_store_only_CDFG.dot"
        if not cdfg.is_file():
            raise AssertionError("no-cstore normal STORE produced no before-map CDFG")
        if 'opcode = "store"' not in cdfg.read_text():
            raise AssertionError("no-cstore normal workload CDFG was not memref.store")
        mapped_dfgio = case_dir / "store_only_map_result" / "mapped_dfgio.txt"
        if not mapped_dfgio.is_file() or "STORE_" not in mapped_dfgio.read_text():
            raise AssertionError("no-cstore normal workload was not mapped as STORE")


def run_dfg_contract_tests(mapper, operations):
    unit = mapper.with_name("cstore-dfg-contract-test")
    expected = {
        "valid": (0, ()),
        "missing": (1, ("Invalid CSTORE DFG node", "logical operand 2")),
        "duplicate": (1, ("Invalid CSTORE DFG node", "logical operand 0")),
        "out-of-range": (1, ("Invalid CSTORE DFG node", "out of range")),
        "memory-ignored": (0, ()),
    }
    for case, (status, messages) in expected.items():
        result = subprocess.run([str(unit), case, str(operations)], text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=60, check=False)
        if result.returncode != status:
            raise AssertionError(
                f"dfg-{case}: expected exit status {status}, got {result.returncode}\n"
                f"{result.stdout}")
        for message in messages:
            if message not in result.stdout:
                raise AssertionError(
                    f"dfg-{case}: missing diagnostic {message!r}\n{result.stdout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("mapper", type=Path)
    parser.add_argument("adg", type=Path)
    parser.add_argument("operations", type=Path)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--case", choices=("all",) + CASES, default="all")
    args = parser.parse_args()
    args.mapper = args.mapper.resolve()
    args.adg = args.adg.resolve()
    args.operations = args.operations.resolve()
    args.kernel = args.kernel.resolve()
    args.workdir = args.workdir.resolve()
    run_dfg_contract_tests(args.mapper, args.operations)
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case, args.workdir, args.mapper, args.adg, args.operations,
                 args.kernel)
    print("CSTORE contract variants passed:", ", ".join(cases))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
