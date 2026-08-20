#!/usr/bin/env python3
"""Real VITRA placement, routing, and input-alignment checks."""

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HEADER = (
    "edge_id", "src_dfg", "dst_dfg", "dst_operation", "logical_operand",
    "dst_adg", "dst_physical_input", "src_latency", "route_latency",
    "rdu_delay", "arrival_latency", "target_latency",
)
PHYSICAL_INPUTS = {0: {0, 1}, 1: {2, 3}, 2: {4, 5}}
SEEDS = (7, 19, 101)


def run(command, cwd, environment, expected=0):
    result = subprocess.run(
        [str(value) for value in command], cwd=cwd, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=60, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"expected status {expected}, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def assert_pre_mapping_cdfg(cgra_opt, kernel, workdir, environment):
    preflight = workdir / "cdfg-preflight"
    preflight.mkdir()
    run((cgra_opt, "--adora-kernel-dfg-gen", kernel), preflight, environment)
    dot_path = preflight / "cstore_route_CDFG.dot"
    if not dot_path.is_file():
        raise AssertionError("CDFG preflight produced no cstore_route_CDFG.dot")

    node_pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\[opcode = "([^"]+)"')
    edge_pattern = re.compile(
        r'^([A-Za-z_][A-Za-z0-9_]*) -> ([A-Za-z_][A-Za-z0-9_]*)'
        r'\[.*operand = (-?[0-9]+)')
    opcodes = {}
    incoming = {}
    for line in dot_path.read_text().splitlines():
        node = node_pattern.match(line)
        if node:
            opcodes[node.group(1)] = node.group(2)
        edge = edge_pattern.match(line)
        if edge:
            incoming.setdefault(edge.group(2), []).append(
                (edge.group(1), int(edge.group(3))))

    cstores = [node for node, opcode in opcodes.items() if opcode == "CSTORE"]
    if len(cstores) != 1:
        raise AssertionError(f"expected exactly one CSTORE in CDFG, got {cstores}")
    cstore = cstores[0]
    logical_edges = [(src, port) for src, port in incoming.get(cstore, [])
                     if port >= 0]
    if sorted(port for _, port in logical_edges) != [0, 1, 2]:
        raise AssertionError(f"CSTORE logical inputs are not exactly 0/1/2: {logical_edges}")

    def producer_depth(node, active):
        if node in active:
            raise AssertionError(f"cycle in CDFG producer chain at {node}")
        opcode = opcodes.get(node)
        if opcode in ("Input", "CONST"):
            return 0
        predecessors = [src for src, port in incoming.get(node, []) if port >= 0]
        if not predecessors:
            raise AssertionError(f"dynamic producer {node} has no predecessor")
        return 1 + max(producer_depth(src, active | {node}) for src in predecessors)

    depths = {port: producer_depth(src, set()) for src, port in logical_edges}
    if depths != {0: 1, 1: 2, 2: 3}:
        raise AssertionError(f"unexpected CSTORE producer depths: {depths}")


def mapper_command(mapper, seed, adg, operations, kernel, output):
    return (
        mapper, f"--seed={seed}", f"--adg={adg}",
        f"--op-file={operations}", "--output-type=pytest",
        "--obj-opt=false", "--max-iters=30", "--timeout=30000",
        kernel, f"--output={output}",
    )


def run_mapping(label, seed, workdir, mapper, adg, operations, kernel,
                kernel_name, environment, expected=0):
    run_dir = workdir / label
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    log = run(mapper_command(mapper, seed, adg, operations, kernel, output),
              run_dir, environment, expected=expected)
    if log.count(f"Random seed: {seed}\n") != 1:
        raise AssertionError(f"{label}: mapper did not log seed {seed} exactly once")
    if expected != 0:
        return log, None
    if not output.is_file():
        raise AssertionError(f"{label}: mapper produced no output")
    manifest = run_dir / f"{kernel_name}_map_result" / "mapped_routes.tsv"
    if not manifest.is_file():
        raise AssertionError(f"{label}: mapper produced no route manifest")
    return log, manifest


def read_manifest(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != HEADER:
            raise AssertionError(f"{path}: unexpected manifest header {reader.fieldnames}")
        return list(reader)


def validate_cstore_manifest(path):
    rows = [row for row in read_manifest(path)
            if row["dst_operation"] == "CSTORE"]
    if len(rows) != 3:
        raise AssertionError(f"{path}: expected three CSTORE rows, got {len(rows)}")
    by_port = {int(row["logical_operand"]): row for row in rows}
    if set(by_port) != {0, 1, 2} or len(by_port) != len(rows):
        raise AssertionError(f"{path}: CSTORE logical operands are not exactly 0/1/2")
    physical = []
    for logical, row in by_port.items():
        physical_input = int(row["dst_physical_input"])
        if physical_input not in PHYSICAL_INPUTS[logical]:
            raise AssertionError(
                f"{path}: logical {logical} used physical input {physical_input}")
        physical.append(physical_input)
        if int(row["arrival_latency"]) != int(row["target_latency"]):
            raise AssertionError(f"{path}: CSTORE logical {logical} is misaligned")
    if len(set(physical)) != 3:
        raise AssertionError(f"{path}: CSTORE physical inputs alias: {physical}")
    return rows


def validate_store_manifest(path):
    rows = read_manifest(path)
    store_rows = [row for row in rows if row["dst_operation"] == "STORE"]
    if not store_rows:
        raise AssertionError(f"{path}: normal STORE has no route rows")
    if any(int(row["logical_operand"]) == 2 for row in rows):
        raise AssertionError(f"{path}: normal STORE unexpectedly uses logical operand 2")


def make_no_cstore_adg(source, destination):
    with source.open() as stream:
        adg = json.load(stream)
    removed = 0
    for module in adg.get("sub_modules", []):
        operations = module.get("attributes", {}).get("operations", [])
        while "CSTORE" in operations:
            operations.remove("CSTORE")
            removed += 1
    if removed == 0:
        raise AssertionError("pinned ADG has no CSTORE capability to remove")
    with destination.open("w") as stream:
        json.dump(adg, stream, indent=2)
        stream.write("\n")


def main():
    if len(sys.argv) != 8:
        raise AssertionError(
            "usage: test_cstore_alignment.py WORKDIR CGRA_OPT MAPPER ADG "
            "OPERATIONS CSTORE_KERNEL STORE_KERNEL")
    workdir, cgra_opt, mapper, adg, operations, cstore_kernel, store_kernel = (
        Path(value).resolve() for value in sys.argv[1:])
    workdir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)

    assert_pre_mapping_cdfg(cgra_opt, cstore_kernel, workdir, environment)

    manifests = []
    for seed in SEEDS:
        _, manifest = run_mapping(
            f"cstore-seed-{seed}", seed, workdir, mapper, adg, operations,
            cstore_kernel, "cstore_route", environment)
        manifests.append(manifest)
    _, repeated = run_mapping(
        "cstore-seed-7-repeat", 7, workdir, mapper, adg, operations,
        cstore_kernel, "cstore_route", environment)

    cstore_rows = []
    for manifest in manifests + [repeated]:
        cstore_rows.extend(validate_cstore_manifest(manifest))
    if not any(int(row["rdu_delay"]) > 0 for row in cstore_rows):
        raise AssertionError("all deterministic CSTORE inputs have zero RDU delay")
    if manifests[0].read_bytes() != repeated.read_bytes():
        raise AssertionError("seed 7 route manifests are not byte-identical")

    _, store_manifest = run_mapping(
        "normal-store", 7, workdir, mapper, adg, operations, store_kernel,
        "normal_store_route", environment)
    validate_store_manifest(store_manifest)

    no_cstore_adg = workdir / "no-cstore-adg.json"
    make_no_cstore_adg(adg, no_cstore_adg)
    rejection, _ = run_mapping(
        "no-cstore-reject", 7, workdir, mapper, no_cstore_adg, operations,
        cstore_kernel, "cstore_route", environment, expected=1)
    for message in ("No legal ADG node supports CSTORE", "required=1", "available=0"):
        if message not in rejection:
            raise AssertionError(f"no-CSTORE rejection missing {message!r}")
    _, no_cstore_store = run_mapping(
        "no-cstore-normal-store", 7, workdir, mapper, no_cstore_adg,
        operations, store_kernel, "normal_store_route", environment)
    validate_store_manifest(no_cstore_store)

    print("CSTORE alignment passed for seeds 7, 19, 101; normal STORE passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
