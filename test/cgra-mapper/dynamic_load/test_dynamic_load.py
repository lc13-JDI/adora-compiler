#!/usr/bin/env python3
"""Check final execution configuration, not RTL. Never import generated runtime."""
import ast
import csv
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def load_helpers():
    path = Path(__file__).parents[1] / 'cstore_contract/test_cstore_alignment.py'
    spec = importlib.util.spec_from_file_location('alignment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph(path):
    nodes, incoming = {}, {}
    for line in path.read_text().splitlines():
        m = re.match(r'(\w+)\[opcode\s*=\s*"([^"]+)"(.*)', line)
        if m:
            attrs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', line))
            nodes[m[1]] = attrs
        m = re.match(r'(\w+) -> (\w+)\[.*operand\s*=\s*(-?\d+)', line)
        if m:
            incoming.setdefault(m[2], {})[int(m[3])] = m[1]
    return nodes, incoming


def check_graph(nodes, incoming, kernel):
    loads = [n for n, a in nodes.items() if a['opcode'].lower() == 'load']
    assert len(loads) == (0 if kernel == 'regular_sum' else 4), loads
    for n in loads:
        a = nodes[n]
        assert a['ref_name'] == kernel + ':w' and a['size'] == '16', a
        assert a['offset'] == '0,0' and a['pattern'] == '0,1', a
        assert set(incoming[n]) == {0}
    # Independent evaluation of the actual generated graph and byte addresses.
    tables = [list(range(-4, 4)), [8,-8,7,-7,1,-1,0,2], [0]*8]
    index_pairs = [(0,1)] if kernel == 'regular_sum' else list(itertools.product(range(4), repeat=2))
    for w, ids in itertools.product(tables, index_pairs):
        values, addresses = {}, []
        def evaluate(n):
            if n in values: return values[n]
            a = nodes[n]; op = a['opcode'].upper()
            args = {p: evaluate(src) for p, src in incoming.get(n, {}).items()}
            if op == 'CONST': value = int(a['value'], 0)
            elif op == 'INPUT': value = (w if a['ref_name'].endswith(':w') else ids)[int(a['offset'].split(',')[1]) // 2]
            elif op == 'MUL': value = args[0] * args[1]
            elif op == 'ADD': value = args[0] + args[1]
            elif op == 'SLT': value = int(args[0] < args[1])
            elif op == 'SEL': value = args[1] if args[2] else args[0]
            elif op == 'LOAD':
                address = args[0]; addresses.append(address)
                assert 0 <= address <= 14 and address % 2 == 0, address
                value = w[address // 2]
            elif op == 'OUTPUT': value = args[0]
            else: raise AssertionError(op)
            values[n] = value
            return value
        actual = {}
        for n, a in nodes.items():
            if a['opcode'].upper() == 'OUTPUT':
                actual[int(a['ref_name'][-1])] = evaluate(n)
        expected = {d: sum(w[2*i+d] for i in ids if kernel != 'ebag_padding' or i != 3) for d in (0, 1)}
        assert actual == expected, (w, ids, actual, expected)
        if loads: assert sorted(addresses) == sorted(2*(2*i+d) for i in ids for d in (0,1))
    return len(tables)*len(index_pairs)


def inspect_call(path, kernel):
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == kernel)
    transfers, outputs, mask = [], [], None
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            target = node.func.value.id
            if node.func.attr == 'append' and target in ('idata', 'iptrs', 'odata', 'optrs'):
                if target in ('idata', 'odata'):
                    value = node.args[0]
                    assert isinstance(value, ast.Subscript)
                    item = (value.value.id, ast.unparse(value.slice))
                else:
                    item = tuple(ast.literal_eval(a) for a in node.args[0].args)
                (transfers if target in ('idata', 'iptrs') else outputs).append((target, item))
        if isinstance(node, ast.keyword) and node.arg == 'iob_en': mask = ast.literal_eval(node.value)
    data = [v for k,v in transfers if k == 'idata']; ptrs = [v for k,v in transfers if k == 'iptrs']
    out = [v for k,v in outputs if k == 'odata']; outptr = [v for k,v in outputs if k == 'optrs']
    assert len(data) == len(ptrs) == (4 if kernel == 'regular_sum' else 6) and len(out) == len(outptr) == 2
    assert data.count(('arg_0', '0:0 + 8')) == 4, data
    assert data.count(('arg_1', '0:0 + 2')) == (0 if kernel == 'regular_sum' else 2), data
    assert out == [('arg_2', '0:0 + 1'), ('arg_2', '1:1 + 1')], out
    assert [size for _,size in ptrs] == [16]*4+([] if kernel == 'regular_sum' else [4]*2), ptrs
    assert [size for _,size in outptr] == [2,2]
    return mask, dict(zip([n for n,_ in data], ptrs)), ptrs, outptr, function


def check_host_guards(function):
    # Execute only actual generated entry checks. A sentinel runtime proves
    # illegal inputs fail before create_stream, DMA or execution.
    import numpy as np
    body = []
    for statement in function.body:
        if not isinstance(statement, ast.If): break
        body.append(statement)
    assert body, 'missing host preflight'
    code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<generated-entry-checks>', 'exec', optimize=2)
    def run(w, ids, y):
        exec(code, {'np':np, 'arg_0':w, 'arg_1':ids, 'arg_2':y})
    w = np.arange(8,dtype=np.int16); y = np.zeros(2,dtype=np.int16)
    for pair in itertools.product(range(4), repeat=2): run(w,np.array(pair,dtype=np.int16),y)
    bad = [(w,np.array([-1,0],dtype=np.int16),y), (w,np.array([0,4],dtype=np.int16),y),
           (w,np.array([0,1],dtype=np.int32),y), (w,np.array([0],dtype=np.int16),y),
           (w[::2],np.array([0,1],dtype=np.int16),y), (w,w[:2],y)]
    readonly = y.copy(); readonly.flags.writeable=False
    bad.append((w,np.array([0,1],dtype=np.int16),readonly))
    for args in bad:
        try: run(*args)
        except ValueError: continue
        raise AssertionError('invalid host input accepted')


def check_result(work, output, kernel, adg, helpers):
    mapping = work / (kernel+'_map_result')
    nodes, incoming = graph(mapping / ('before_map_'+kernel+'_CDFG.dot'))
    oracle_cases = check_graph(nodes,incoming,kernel)
    mask, _, inputs, outputs, function = inspect_call(output,kernel)
    if kernel != 'regular_sum': check_host_guards(function)
    packets = dict(helpers.read_pytest_packets(output,kernel,adg))
    with (mapping/'mapped_routes.tsv').open() as stream:
        routes = list(csv.DictReader(stream, delimiter='\t'))
    bindings=[]; used=set()
    for line in (mapping/'mapped_dfgio.txt').read_text().splitlines():
        if line.startswith('#'): continue
        tag,name,physical,index,lat = [x.strip() for x in line.split(',')]
        physical,index,lat = int(physical),int(index),int(lat)
        opcode = tag.split('_')[0]
        assert mask[index//8] & (1 << (index%8)), (tag,index,mask)
        # CDFG numbering is zero-based; mapper names are one-based.
        match = re.match(r'(.*?)(\d+)$',name)
        n = match[1] + str(int(match[2])-1); attrs=nodes[n]
        instance,module = helpers.find_iob(adg,physical)
        fields = {field: helpers.decode_controller_field(packets,instance,module,adg,field)
                  for field in ('BaseAddr','UseAddr','UseEn','IsStore','Latency','II','Cycles0','Stride0')}
        assert fields['UseAddr'] == int(opcode == 'LOAD') and fields['UseEn'] == 0
        assert fields['IsStore'] == int(opcode == 'OUTPUT') and fields['Cycles0'] == 1 and fields['Stride0'] == 0
        assert fields['II'] >= 1
        banks=adg['iob_to_spad_banks'][str(index)]; bank_size=adg['iob_spad_bank_size']
        bytebase=fields['BaseAddr']*2-int(attrs['offset'].split(',')[1])
        bank=min(banks)+bytebase//bank_size
        assert bank in banks and bytebase%bank_size == 0 and bank not in used
        used.add(bank)
        address=bank*bank_size
        assert (address,int(attrs['size'])) in (outputs if opcode=='OUTPUT' else inputs)
        if opcode=='LOAD':
            assert fields['Latency']==lat, (fields,lat)
            address_routes = [r for r in routes if int(r['dst_adg']) == physical]
            assert len(address_routes) == 1 and address_routes[0]['logical_operand'] == '0'
            route = address_routes[0]
            assert int(route['dst_physical_input']) in (0,1)
            assert int(route['arrival_latency']) + int(route['rdu_delay']) == int(route['target_latency'])
            helpers.validate_route_configuration(packets,instance,module,adg,address_routes)
        bindings.append(dict(node=n,opcode=opcode,iob=index,bank=bank,host_address=address,fields=fields))
    assert len(bindings)==(6 if kernel == 'regular_sum' else 8)
    assert sum(b['opcode']=='LOAD' for b in bindings)==(0 if kernel == 'regular_sum' else 4)
    assert sum(x.bit_count() for x in mask)==len(bindings)
    result={'bindings':bindings,'input_bytes_including_copies':sum(s for _,s in inputs),'output_bytes':sum(s for _,s in outputs),'oracle_cases':oracle_cases,'rtl':'NOT_RUN'}
    (work/'contract.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    mapper,adg_path,ops,source,work = [Path(x).resolve() for x in sys.argv[1:]]
    work.mkdir(parents=True,exist_ok=True)
    adg=json.loads(adg_path.read_text()); helpers=load_helpers(); records=[]
    kernel=re.search(r'func.func @(\w+)',source.read_text())[1]
    for seed in (7,19):
        run=work/f'seed{seed}';run.mkdir(); output=run/'mapped.py'
        mapper_input = source
        if seed == 19:
            mapper_input = run/'post-dfg.mlir'
            command = [str(mapper.with_name('cgra-opt')), str(source),
                       '--adora-kernel-dfg-gen', '--verify-each', '-o', str(mapper_input)]
            p = subprocess.run(command,cwd=run,capture_output=True,text=True,timeout=60)
            (run/'dfg.log').write_text(p.stdout+p.stderr)
            (run/'dfg-command.json').write_text(json.dumps(dict(argv=command,rc=p.returncode),indent=2))
            assert p.returncode == 0, f'DFG pass failed: {run}/dfg.log'
        command=[str(mapper),str(mapper_input),'--adg='+str(adg_path),'--op-file='+str(ops),'--output-type=pytest','--output='+str(output),'--seed='+str(seed),'--max-iters=30','--timeout=10000','--obj-opt=false']
        p=subprocess.run(command,cwd=run,capture_output=True,text=True,timeout=60)
        (run/'mapper.log').write_text(p.stdout+p.stderr)
        (run/'command.json').write_text(json.dumps(dict(argv=command,rc=p.returncode),indent=2))
        assert p.returncode == 0, f'mapper failed: {run}/mapper.log'
        records.append(check_result(run,output,kernel,adg,helpers))
    (work/'results.json').write_text(json.dumps(records,indent=2))
    print(f"{kernel}: 2 seeds, {len(records[0]['bindings'])} IOBs each, final config and {records[0]['oracle_cases']} graph oracle cases PASS; RTL NOT_RUN")

if __name__=='__main__': main()
