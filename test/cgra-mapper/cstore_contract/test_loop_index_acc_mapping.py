#!/usr/bin/env python3
"""Decode the emitted VITRA config for loop-index ACC CSTORE mappings."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


SEEDS = (7, 19)
HEADER = (
    "edge_id", "src_dfg", "dst_dfg", "dst_operation", "logical_operand",
    "dst_adg", "dst_physical_input", "src_latency", "route_latency",
    "rdu_delay", "arrival_latency", "target_latency",
)


def load_alignment_helpers(test_dir):
    helper_path = test_dir / "test_cstore_alignment.py"
    spec = importlib.util.spec_from_file_location("cstore_alignment", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command, cwd, environment):
    result = subprocess.run([str(value) for value in command], cwd=cwd,
                            env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=90, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"expected status 0, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def mapper_command(mapper, seed, adg, operations, source, output):
    return (mapper, f"--seed={seed}", f"--adg={adg}",
            f"--op-file={operations}", "--output-type=pytest",
            "--obj-opt=false", "--max-iters=30", "--timeout=30000",
            source, f"--output={output}")


def read_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != HEADER:
            raise AssertionError(f"{path}: unexpected manifest header")
        return list(reader)


def mapped_gpe_for_acc(mapped_adg, mapped_dfg, require_immediate=True):
    mapped_dfg_text = mapped_dfg.read_text()
    accs = [match.group(1) for line in mapped_dfg_text.splitlines()
            for match in [re.match(r'^"(ACC[^"]+)"\[label', line)]
            if match and (not require_immediate or "immIdx=0" in line)]
    if len(accs) != 1:
        raise AssertionError(f"expected one mapped loop-index ACC, got {accs}")
    match = re.search(r'GPE(\d+)\[label = "GPE\d+\\nDFG:' +
                      re.escape(accs[0]) + r'", color = red\];',
                      mapped_adg.read_text())
    if not match:
        raise AssertionError("mapped loop-index ACC has no GPE placement")
    return int(match.group(1)), accs[0]


def find_gpe(adg, node_id):
    instances = [item for item in adg["instances"]
                 if item.get("id") == node_id and item.get("type") == "GPE"]
    if len(instances) != 1:
        raise AssertionError(f"expected GPE instance {node_id}")
    modules = [item for item in adg["sub_modules"]
               if item.get("id") == instances[0].get("module_id") and
               item.get("type") == "GPE"]
    if len(modules) != 1:
        raise AssertionError(f"expected GPE module for instance {node_id}")
    return instances[0], modules[0]["attributes"]


def decode_gpe_field(helpers, packet_map, instance, attributes, adg, field):
    config_ids = attributes["affine_ctrl_reg_cfg_id"]
    config_id = config_ids[field]
    low, high = helpers.config_range(attributes, config_id, field)
    return helpers.decode_range(packet_map, instance, adg, low, high)


def decode_gpe_opcode(helpers, packet_map, instance, attributes, adg):
    connections = list(attributes["connections"].values())
    alu_ids = {int(edge[3]) for edge in connections if edge[4] == "ALU"}
    if len(alu_ids) != 1:
        raise AssertionError("expected one GPE ALU configuration")
    low, high = helpers.config_range(attributes, alu_ids.pop(), "ALU")
    return helpers.decode_range(packet_map, instance, adg, low, high)


def assert_acc_route(adg, mapped_dfg, mapped_adg, step):
    text = mapped_dfg.read_text()
    if f"imm={step}\\nimmIdx=0" not in text:
        raise AssertionError("positive loop step is not folded at ACC operand 0")
    gpe, _ = mapped_gpe_for_acc(mapped_adg, mapped_dfg)
    _, attributes = find_gpe(adg, gpe)
    connections = {tuple(edge) for edge in attributes["connections"].values()}
    route = {(1, "Const", 0, 5, "Muxn", 0),
             (5, "Muxn", 0, 4, "DelayPipe", 0),
             (4, "DelayPipe", 0, 2, "ALU", 0)}
    if not route.issubset(connections):
        raise AssertionError("ACC operand 0 is not the physical routed step path")


def assert_cstore_routes(helpers, adg, packet_map, output, result_dir, kernel):
    rows = [row for row in read_rows(result_dir / "mapped_routes.tsv")
            if row["dst_operation"] == "CSTORE"]
    if len(rows) != 3:
        raise AssertionError(f"{kernel}: expected data/address/predicate CSTORE routes")
    by_port = {int(row["logical_operand"]): row for row in rows}
    if set(by_port) != {0, 1, 2}:
        raise AssertionError(f"{kernel}: CSTORE logical ports changed")
    if any(int(row["arrival_latency"]) != int(row["target_latency"])
           for row in rows):
        raise AssertionError(f"{kernel}: CSTORE route alignment changed")
    helpers.validate_emitted_configuration(
        adg, result_dir / "mapped_routes.tsv", output, kernel, "CSTORE", rows)
    return rows


def assert_loop_configuration(helpers, adg_data, contract, result_dir, output,
                              kernel, init, step, trips):
    packets = helpers.read_config_packets(result_dir / "config.bit",
                                          int(adg_data["cfg_data_width"]))
    if packets != helpers.read_pytest_packets(output, kernel, adg_data):
        raise AssertionError(f"{kernel}: emitted pytest config differs from config.bit")
    packet_map = dict(packets)
    gpe, _ = mapped_gpe_for_acc(result_dir / "mapped_adg.dot",
                                 result_dir / "mapped_dfg.dot")
    instance, attributes = find_gpe(adg_data, gpe)
    if decode_gpe_opcode(helpers, packet_map, instance, attributes, adg_data) != \
            contract["operations"]["ACC"]["opcode"]:
        raise AssertionError(f"{kernel}: ACC opcode did not decode from the operation catalog")
    expected = {"InitVal": init, "WI": 1, "Latency": 0,
                "Cycles": trips, "Repeats": 1, "SkipFirst": 1}
    actual = {field: decode_gpe_field(helpers, packet_map, instance, attributes,
                                      adg_data, field)
              for field in expected}
    if actual != expected:
        raise AssertionError(f"{kernel}: loop-index ACC fields expected {expected}, got {actual}")
    if "loop_index_acc=1" not in (result_dir / "mapped_dfg.dot").read_text():
        raise AssertionError(f"{kernel}: mapper lost the loop-index ACC marker")
    assert_acc_route(adg_data, result_dir / "mapped_dfg.dot",
                     result_dir / "mapped_adg.dot", step)
    return packet_map


def check_case(helpers, workdir, mapper, adg, operations, source, kernel,
               contract, init, step, trips, seed):
    run_dir = workdir / f"{kernel}-seed-{seed}"
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    environment = dict(os.environ)
    log = run(mapper_command(mapper, seed, adg, operations, source, output),
              run_dir, environment)
    if log.count(f"Random seed: {seed}\n") != 1:
        raise AssertionError(f"{kernel}: mapper did not log deterministic seed {seed}")
    result_dir = run_dir / f"{kernel}_map_result"
    with adg.open() as stream:
        adg_data = json.load(stream)
    packet_map = assert_loop_configuration(
        helpers, adg_data, contract, result_dir, output, kernel, init, step, trips)
    rows = assert_cstore_routes(helpers, adg, packet_map, output, result_dir, kernel)
    if kernel == "loop_index_b":
        route_text = (result_dir / "mapped_dfg.dot").read_text()
        if '"ACC' not in route_text or '"MUL' not in route_text or \
                '"CSTORE' not in route_text:
            raise AssertionError("Case B lost ACC, MUL, or CSTORE")
        if not re.search(r'^"MUL\w+"\[label = "\\N\\nlat=\d+\\nimm=2\\nimmIdx=1"',
                         route_text, flags=re.MULTILINE):
            raise AssertionError("Case B byte-scaling MUL is not an immediate ×2")
        edges = re.findall(r'^"(\w+)"->"(\w+)"\[label = '
                           r'"lat=(\d+)\\nop=(\d+)', route_text,
                           flags=re.MULTILINE)
        if not any(src.startswith("ACC") and dst.startswith("MUL") and
                   latency == "0" and port == "0"
                   for src, dst, latency, port in edges) or \
                not any(src.startswith("MUL") and dst.startswith("CSTORE") and
                        port == "1" for src, dst, _, port in edges):
            raise AssertionError("Case B lost explicit byte-scaling MUL")
    return (result_dir / "mapped_routes.tsv").read_bytes(), \
        (result_dir / "config.bit").read_bytes(), rows


def check_normal_store(helpers, workdir, mapper, adg, operations, test_dir):
    source = workdir / "normal-store.mlir"
    shutil.copyfile(test_dir / "normal_store.mlir.in", source)
    run_dir = workdir / "normal-store"
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    run(mapper_command(mapper, 7, adg, operations, source, output), run_dir,
        dict(os.environ))
    result_dir = run_dir / "normal_store_route_map_result"
    with adg.open() as stream:
        adg_data = json.load(stream)
    rows = [row for row in read_rows(result_dir / "mapped_routes.tsv")
            if row["dst_operation"] == "STORE"]
    helpers.validate_emitted_configuration(
        adg, result_dir / "mapped_routes.tsv", output, "normal_store_route",
        "STORE", rows)


def check_generic_acc(helpers, workdir, mapper, adg, operations, test_dir):
    source = test_dir.parents[1] / "cgra-opt/cdfggen/mmul_relu/mmul_relu_opt.mlir"
    run_dir = workdir / "generic-acc"
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    run(mapper_command(mapper, 7, adg, operations, source, output), run_dir,
        dict(os.environ))
    result_dir = run_dir / "mmul_relu_map_result"
    with adg.open() as stream:
        adg_data = json.load(stream)
    packets = helpers.read_config_packets(result_dir / "config.bit",
                                          int(adg_data["cfg_data_width"]))
    gpe, _ = mapped_gpe_for_acc(result_dir / "mapped_adg.dot",
                                 result_dir / "mapped_dfg.dot",
                                 require_immediate=False)
    instance, attributes = find_gpe(adg_data, gpe)
    packet_map = dict(packets)
    expected = {"InitVal": 0, "WI": 1, "Latency": 11,
                "Cycles": 25, "Repeats": 625, "SkipFirst": 0}
    actual = {field: decode_gpe_field(helpers, packet_map, instance, attributes,
                                      adg_data, field)
              for field in expected}
    if actual != expected:
        raise AssertionError(f"generic ACC config changed: expected {expected}, got {actual}")
    if "loop_index_acc=1" in (result_dir / "mapped_dfg.dot").read_text():
        raise AssertionError("ordinary ACC was marked as a loop-index ACC")


def main():
    if len(sys.argv) != 8:
        raise AssertionError("usage: test_loop_index_acc_mapping.py WORKDIR CGRA_OPT "
                             "MAPPER ADG OPERATIONS CONTRACT TESTDIR")
    workdir, cgra_opt, mapper, adg, operations, contract_path, test_dir = \
        (Path(value).resolve() for value in sys.argv[1:])
    del cgra_opt
    helpers = load_alignment_helpers(test_dir)
    contract = json.loads(contract_path.read_text())
    if contract["configuration_formula"]["WI"] != 1 or \
            contract["configuration_formula"]["Latency"] != 0:
        raise AssertionError("pinned loop-index contract has unexpected ACC formula")
    chunks = (test_dir / "loop_index_acc_mapping.mlir").read_text().split("// -----\n")
    if len(chunks) != 2:
        raise AssertionError("expected Case A and Case B fixture chunks")
    sources = []
    for name, chunk in zip(("loop_index_a", "loop_index_b"), chunks):
        source = workdir / f"{name}.mlir"
        source.write_text(chunk)
        sources.append(source)

    for seed in SEEDS:
        check_case(
            helpers, workdir, mapper, adg, operations, sources[0], "loop_index_a",
            contract, 0, 1, 4, seed)
        check_case(
            helpers, workdir, mapper, adg, operations, sources[1], "loop_index_b",
            contract, 3, 2, 5, seed)
    check_normal_store(helpers, workdir, mapper, adg, operations, test_dir)
    check_generic_acc(helpers, workdir, mapper, adg, operations, test_dir)
    print("loop-index ACC config, byte-scaled CSTORE routing, and STORE regression passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
