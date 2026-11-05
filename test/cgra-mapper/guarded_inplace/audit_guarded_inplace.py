#!/usr/bin/env python3
"""Audit mapped/emitted LOAD+CSTORE configuration; does not replace RTL execution."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re


def audit(entry, adg_path, operations, count):
    helper_dir = Path(__file__).resolve().parents[1] / 'cstore_contract'
    spec = importlib.util.spec_from_file_location('loop_config', helper_dir / 'test_loop_index_acc_mapping.py')
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    h = loop.load_alignment_helpers(helper_dir)
    results = list((entry / 'map').glob('*_map_result'))
    assert len(results) == 1, results
    result = results[0]
    kernel = result.name.removesuffix('_map_result')
    output = entry / 'map/unchecked.py'
    adg = json.loads(adg_path.read_text())
    packets = dict(h.read_pytest_packets(output, kernel, adg))
    raw = dict(h.read_config_packets(result / 'config.bit', adg['cfg_data_width']))
    nodes, edges = loop.parse_mapped_dfg(result / 'mapped_dfg.dot')
    loop.assert_no_physical_for(nodes, result / 'mapped_adg.dot')
    opcodes = loop.read_operation_opcodes(operations)
    op_latencies = {op['name']: op['latency'] for op in json.loads(operations.read_text())['Operations']}
    acc = loop.assert_acc_route(h, adg, packets, result / 'mapped_adg.dot', nodes, 1, opcodes['ACC'])
    scheduled_acc_latency = int(re.search(r'\\nlat=(\d+)', nodes[acc])[1])
    assert scheduled_acc_latency == op_latencies['ACC'], (
        'autonomous index ACC cannot be postponed by the mapper', scheduled_acc_latency)
    gpe = loop.mapped_gpe_for_node(result / 'mapped_adg.dot', acc)
    inst, attrs = loop.find_gpe(adg, gpe)
    expected = dict(InitVal=0, WI=1, Latency=0, Cycles=count, Repeats=1, SkipFirst=1)
    actual = {f: loop.decode_gpe_field(h, packets, inst, attrs, adg, f) for f in expected}
    assert actual == expected, (actual, expected)
    assert 'loop_index_acc=1' in nodes[acc]
    store = loop.one_node(nodes, 'CSTORE')
    loads = [name for name in nodes if loop.node_operation(name).upper() == 'LOAD']
    assert len(loads) == 1, loads
    load = loads[0]
    incoming = {port: src for src, dst, _, port in edges if dst == store}
    rows = loop.read_rows(result / 'mapped_routes.tsv')
    def origin(name):
        seen = set()
        while loop.node_operation(name).upper() == 'PASS':
            assert name not in seen
            seen.add(name)
            ins = [(src, port) for src, dst, _, port in edges if dst == name]
            assert len(ins) == 1 and ins[0][1] == 0, (name, ins)
            physical = loop.mapped_gpe_for_node(result / 'mapped_adg.dot', name)
            instance, attributes = loop.find_gpe(adg, physical)
            assert loop.decode_gpe_opcode(h, packets, instance, attributes, adg) == opcodes['PASS']
            routed = [row for row in rows if int(row['dst_dfg']) == loop.node_id(name)]
            assert len(routed) == 1 and int(routed[0]['src_dfg']) == loop.node_id(ins[0][0])
            h.validate_route_configuration(packets, instance, attributes, adg, routed)
            _, _, connection = loop.external_path_to_alu_operand(attributes, int(routed[0]['dst_physical_input']))
            assert connection[5] == 0, 'PASS must forward physical operand zero'
            name = ins[0][0]
        return name
    assert set(incoming) == {0, 1, 2} and origin(incoming[0]) == load, incoming
    origin(incoming[2])  # Validate any inserted identity route on the predicate.
    mul = origin(incoming[1])
    assert loop.node_operation(mul) == 'MUL' and 'imm=2\\nimmIdx=1' in nodes[mul]
    mul_inputs = [(src, port) for src, dst, _, port in edges if dst == mul]
    assert len(mul_inputs) == 1 and mul_inputs[0][1] == 0 and origin(mul_inputs[0][0]) == acc
    physical = loop.mapped_gpe_for_node(result / 'mapped_adg.dot', mul)
    instance, attributes = loop.find_gpe(adg, physical)
    assert loop.decode_gpe_opcode(h, packets, instance, attributes, adg) == opcodes['MUL']
    mul_rows = [row for row in rows if int(row['dst_dfg']) == loop.node_id(mul)]
    assert len(mul_rows) == 1 and int(mul_rows[0]['src_dfg']) == loop.node_id(mul_inputs[0][0])
    _, _, connection = loop.external_path_to_alu_operand(attributes, int(mul_rows[0]['dst_physical_input']))
    loop.selected_constant_operand(h, adg, packets, instance, attributes, 2, int(connection[5]), active_operands=2)
    h.validate_route_configuration(packets, instance, attributes, adg, mul_rows)
    for row in rows:
        assert int(row['src_latency']) + int(row['route_latency']) + int(row['rdu_delay']) == int(row['arrival_latency']) == int(row['target_latency']), row
    crows = [r for r in rows if r['dst_operation'] == 'CSTORE']
    assert len(crows) == 3 and {int(r['logical_operand']) for r in crows} == {0, 1, 2}
    assert len({r['target_latency'] for r in crows}) == 1
    for row in crows:
        assert int(row['src_dfg']) == loop.node_id(incoming[int(row['logical_operand'])])
        assert int(row['dst_dfg']) == loop.node_id(store)

    tree = ast.parse(output.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == kernel)
    transfers = {name: [] for name in ('idata', 'iptrs', 'odata', 'optrs')}
    mask = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.attr == 'append' and node.func.value.id in transfers:
            target, value = node.func.value.id, node.args[0]
            transfers[target].append([ast.literal_eval(a) for a in value.args] if target.endswith('ptrs') else dict(arg=value.value.id, slice=ast.unparse(value.slice)))
        if isinstance(node, ast.keyword) and node.arg == 'iob_en':
            mask = ast.literal_eval(node.value)
    assert len(transfers['optrs']) == len(transfers['odata']) == 1
    def constant(expr):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return constant(expr.left) + constant(expr.right)
        raise AssertionError(('unexpected transfer bound', ast.dump(expr)))
    for direction in ('i', 'o'):
        assert len(transfers[direction + 'ptrs']) == len(transfers[direction + 'data'])
        for pointer, data in zip(transfers[direction + 'ptrs'], transfers[direction + 'data']):
            span = ast.parse('buffer[' + data['slice'] + ']', mode='eval').body.slice
            assert isinstance(span, ast.Slice) and span.step is None
            assert constant(span.lower) == 0 and 2 * constant(span.upper) == pointer[1]
    destination = transfers['optrs'][0]
    assert destination[1] == 2 * count
    copies = [i for i, ptr in enumerate(transfers['iptrs']) if ptr == destination]
    assert len(copies) == 1, 'inout must initialize the exact readback allocation'
    assert transfers['idata'][copies[0]] == transfers['odata'][0]
    bindings, allowed_relocation = [], {}
    for line in (result / 'mapped_dfgio.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        tag, name, physical, index, latency = [v.strip() for v in line.split(',')]
        physical, index, latency = int(physical), int(index), int(latency)
        operation = tag.rsplit('_', 1)[0]
        assert operation in ('INPUT', 'LOAD', 'CSTORE')
        inst, attrs = h.find_iob(adg, physical)
        fields = {f: h.decode_controller_field(packets, inst, attrs, adg, f) for f in
                  ('BaseAddr', 'UseAddr', 'UseEn', 'IsStore', 'Latency', 'II', 'Cycles0', 'Stride0')}
        assert fields['UseAddr'] == int(operation != 'INPUT')
        assert fields['UseEn'] == fields['IsStore'] == int(operation == 'CSTORE')
        assert fields['Cycles0'] == count and fields['Stride0'] == int(operation == 'INPUT')
        # The IO manifest records store completion; input roles already report
        # input arrival. Controller latency always denotes operand arrival.
        arrival = latency - op_latencies[operation] if operation == 'CSTORE' else latency
        assert fields['Latency'] == arrival and fields['II'] > 0, (name, fields, latency)
        routed = [r for r in rows if int(r['dst_adg']) == physical]
        assert len(routed) == {'INPUT': 0, 'LOAD': 1, 'CSTORE': 3}[operation]
        if routed:
            h.validate_route_configuration(packets, inst, attrs, adg, routed)
            assert all(int(r['target_latency']) == fields['Latency'] for r in routed)
        banks = adg['iob_to_spad_banks'][str(index)]
        size = adg['iob_spad_bank_size']
        base = min(banks) * size + 2 * fields['BaseAddr']
        assert base // size in banks
        if operation == 'CSTORE':
            assert [base, 2 * count] == destination
        else:
            assert sum(ptr[0] == base for ptr in transfers['iptrs']) == 1
        low, high = h.config_range(attrs, attrs['io_controller_cfg_id']['BaseAddr'], 'BaseAddr')
        for bit in range(low, high + 1):
            address = (inst['cfg_blk_index'] << adg['cfg_blk_offset']) | (bit // adg['cfg_data_width'])
            allowed_relocation[address] = allowed_relocation.get(address, 0) | (1 << (bit % adg['cfg_data_width']))
        bindings.append(dict(operation=operation, node=name, iob=index, base=base, fields=fields))
    assert set(raw) == set(packets)
    assert all((raw[a] ^ packets[a]) & ~allowed_relocation.get(a, 0) == 0 for a in raw), 'emitter changed a field other than buffer base relocation'
    assert mask is not None and {i for i in range(8 * len(mask)) if mask[i // 8] >> (i % 8) & 1} == {b['iob'] for b in bindings}
    assert len({b['fields']['II'] for b in bindings}) == 1
    assert len({b['base'] for b in bindings}) == len(bindings)
    return dict(kernel=kernel, count=count, index_acc=actual, cstore_routes=crows,
                bindings=bindings, transfers=transfers, config_packets=len(packets),
                route_count=len(rows), status='PASS', scope='mapped routes and emitted packets; RTL checked separately')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('entry', type=Path)
    parser.add_argument('--adg', required=True, type=Path)
    parser.add_argument('--operations', required=True, type=Path)
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.entry, args.adg, args.operations, args.count)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)
