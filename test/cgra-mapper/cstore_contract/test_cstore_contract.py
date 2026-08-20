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
        return "operand 2 has no physical input path",
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

    command = [
        str(mapper), "--seed=7", f"--adg={adg_path}",
        f"--op-file={operations_path}", "--output-type=pytest",
        "--obj-opt=false", "--max-iters=30", "--timeout=30000",
        str(kernel), f"--output={case_dir / 'mapped.py'}",
    ]
    environment = dict(os.environ)
    environment["GeneralOpNameFile"] = str(
        kernel.parents[3] / "lib/DFG/Documents/GeneralOpName.txt")
    result = subprocess.run(command, cwd=case_dir, text=True, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=60, check=False)
    if result.returncode != 1:
        raise AssertionError(
            f"{case}: expected exit status 1, got {result.returncode}\n{result.stdout}")
    for message in expected:
        if message not in result.stdout:
            raise AssertionError(
                f"{case}: missing diagnostic {message!r}\n{result.stdout}")


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
