#!/usr/bin/env python3
"""Focused physical-CDFG contract checks for affine CSTORE loop indices."""

import pathlib
import re
import subprocess
import sys
import tempfile


def run(command, cwd, expected=0):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"expected status {expected}, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def nodes_and_edges(dot):
    nodes = {}
    edges = []
    for line in dot.read_text().splitlines():
        node = re.match(r'^(\w+)\[opcode = "([^"]+)"(.*)\];$', line)
        if node:
            nodes[node.group(1)] = (node.group(2), node.group(3))
            continue
        edge = re.match(r'^(\w+) -> (\w+)\[.*operand = (-?\d+)', line)
        if edge:
            edges.append((edge.group(1), edge.group(2), int(edge.group(3))))
    return nodes, edges


def check_supported(dot, init, step, trips):
    nodes, edges = nodes_and_edges(dot)
    if any(op == "for" for op, _ in nodes.values()):
        raise AssertionError(f"{dot}: supported CSTORE loop still has physical FOR")
    accs = [(name, attrs) for name, (op, attrs) in nodes.items() if op == "ACC"]
    if len(accs) != 1:
        raise AssertionError(f"{dot}: expected exactly one loop-index ACC, got {accs}")
    acc, attrs = accs[0]
    expected = f'acc_params="{init},{trips},1,1"'
    if expected not in attrs or 'loop_index_acc="1"' not in attrs or \
       'skip_first="1"' not in attrs:
        raise AssertionError(f"{dot}: missing loop-index ACC semantics: {attrs}")
    step_nodes = []
    for name, (op, attrs) in nodes.items():
        value = re.search(r'value="0x([0-9A-Fa-f]+)"', attrs)
        if op == "CONST" and value and int(value.group(1), 16) == step:
            step_nodes.append(name)
    if not step_nodes:
        raise AssertionError(f"{dot}: expected a routed step CONST {step}")
    if not any((step_node, acc, 0) in edges for step_node in step_nodes):
        raise AssertionError(f"{dot}: positive step is not ACC operand 0")


def main():
    cgra_opt = pathlib.Path(sys.argv[1]).resolve()
    mapper = pathlib.Path(sys.argv[2]).resolve()
    adg = pathlib.Path(sys.argv[3]).resolve()
    operations = pathlib.Path(sys.argv[4]).resolve()
    test_dir = pathlib.Path(sys.argv[5]).resolve()
    fixture = test_dir / "cstore_loop_index_acc.mlir"
    chunks = fixture.read_text().split("// -----\n")
    if len(chunks) != 3:
        raise AssertionError("expected three loop-index fixture chunks")

    with tempfile.TemporaryDirectory(prefix="adora-loop-index-acc-") as temp:
        workdir = pathlib.Path(temp)
        (workdir / "lib").symlink_to(cgra_opt.parents[2] / "lib", target_is_directory=True)
        for index, chunk in enumerate(chunks[:2]):
            source = workdir / f"supported-{index}.mlir"
            source.write_text(chunk)
            run([cgra_opt, "--adora-kernel-dfg-gen", source], workdir)

        check_supported(workdir / "loop_index_a_CDFG.dot", 0, 1, 4)
        check_supported(workdir / "loop_index_b_CDFG.dot", 3, 2, 5)

        run([mapper, "--seed=7", f"--adg={adg}", f"--op-file={operations}",
             "--output-type=pytest", "--obj-opt=false", "--max-iters=30",
             "--timeout=30000", workdir / "supported-1.mlir",
             "--output=mapped.py"], workdir)
        mapped_dfg = (workdir / "loop_index_b_map_result" / "mapped_dfg.dot")
        mapped_text = mapped_dfg.read_text()
        if '"ACC' not in mapped_text or '"FOR' in mapped_text:
            raise AssertionError("mapper did not preserve the loop-index ACC")

        unsupported = workdir / "unsupported.mlir"
        unsupported.write_text(chunks[2])
        output = run([cgra_opt, "--adora-kernel-dfg-gen", unsupported], workdir,
                     expected=1)
        if "unsupported loop-index CSTORE" not in output:
            raise AssertionError(f"missing fail-closed diagnostic:\n{output}")

        run([cgra_opt, "--adora-kernel-dfg-gen",
             test_dir / "mmul_relu/mmul_relu_opt.mlir"], workdir)
        generic_dot = workdir / "mmul_relu_CDFG.dot"
        if 'opcode = "ACC"' not in generic_dot.read_text() or \
           'loop_index_acc="1"' in generic_dot.read_text():
            raise AssertionError("ordinary ACC behavior was changed")

        run([cgra_opt, "--adora-kernel-dfg-gen",
             test_dir / "control_flow_paths/conditional_store.mlir"], workdir)
        if 'opcode = "CSTORE"' not in (workdir / "cf_direct_i8_CDFG.dot").read_text():
            raise AssertionError("existing loop-free CSTORE no longer lowers")

        run([cgra_opt, "--adora-kernel-dfg-gen",
             test_dir.parent.parent / "cgra-mapper/cstore_contract/normal_store.mlir.in"],
            workdir)


if __name__ == "__main__":
    main()
