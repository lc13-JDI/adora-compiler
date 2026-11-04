#!/usr/bin/env python3
"""Independent source oracle, emitted-CDFG word model and final IOB packet audit.

This does not simulate routing, controller cycles, SRAM or RTL. Configuration
checks are static; functional results are generated-CDFG model results.
"""
import argparse
import ast
import collections
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


def make_remap(w, h, count, border=60000, nested=True, manual=False):
    """Source has runtime signed coordinates and independently guarded axes."""
    n = w * h
    border = border if border < 32768 else border - 65536
    declarations = f'''module {{
 func.func @remap(%src: memref<{n}xi16>, %mx: memref<{count}xi16>, %my: memref<{count}xi16>, %dst: memref<{count}xi16>) {{
  %s = ADORA.BlockLoad %src [0] : memref<{n}xi16> -> memref<{n}xi16> {{Id = "src", KernelName = "remap"}}
  %xmap = ADORA.BlockLoad %mx [0] : memref<{count}xi16> -> memref<{count}xi16> {{Id = "mx", KernelName = "remap"}}
  %ymap = ADORA.BlockLoad %my [0] : memref<{count}xi16> -> memref<{count}xi16> {{Id = "my", KernelName = "remap"}}
  %out = ADORA.LocalMemAlloc memref<{count}xi16> {{Id = "dst", KernelName = "remap"}}
  %z = arith.constant 0 : i16
  %w = arith.constant {w} : i16
  %h = arith.constant {h} : i16
  %wi = arith.constant {w} : index
  %border = arith.constant {border} : i16
  ADORA.kernel {{
   affine.for %i = 0 to {count} {{
    %x = affine.load %xmap[%i] : memref<{count}xi16>
    %y = affine.load %ymap[%i] : memref<{count}xi16>
    %xlo = arith.cmpi sge, %x, %z : i16
    %xhi = arith.cmpi slt, %x, %w : i16
    %xok = arith.andi %xlo, %xhi : i1
    %ylo = arith.cmpi sge, %y, %z : i16
    %yhi = arith.cmpi slt, %y, %h : i16
    %yok = arith.andi %ylo, %yhi : i1
'''
    address = f'''      %xi = arith.index_cast %x : i16 to index
      %yi = arith.index_cast %y : i16 to index
      %row = arith.muli %yi, %wi : index
      %index = arith.addi %row, %xi : index
      %pixel = memref.load %s[%index] : memref<{n}xi16>
'''
    if manual:
        body = '''    %ok = arith.andi %xok, %yok : i1
    %sx = arith.select %ok, %x, %z : i16
    %sy = arith.select %ok, %y, %z : i16
''' + address.replace('%x :', '%sx :').replace('%y :', '%sy :') + '''    %result = arith.select %ok, %pixel, %border : i16
'''
    elif nested:
        body = '''    %result = scf.if %xok -> (i16) {
     %inner = scf.if %yok -> (i16) {
''' + address + '''      scf.yield %pixel : i16
     } else {
      scf.yield %border : i16
     }
     scf.yield %inner : i16
    } else {
     scf.yield %border : i16
    }
'''
    else:
        body = '''    %ok = arith.andi %xok, %yok : i1
    %result = scf.if %ok -> (i16) {
''' + address + '''     scf.yield %pixel : i16
    } else {
     scf.yield %border : i16
    }
'''
    return declarations + body + f'''    affine.store %result, %out[%i] : memref<{count}xi16>
   }}
   ADORA.terminator
  }} {{KernelName = "remap"}}
  ADORA.BlockStore %out, %dst [0] : memref<{count}xi16> -> memref<{count}xi16> {{Id = "dst", KernelName = "remap"}}
  return
 }}
}}
'''


def make_gather(n, count):
    return f'''module {{
 func.func @gather(%src: memref<{n}xi16>, %ids: memref<{count}xi16>, %enabled: memref<{count}xi16>, %fallback: memref<{count}xi16>, %dst: memref<{count}xi16>) {{
  %s = ADORA.BlockLoad %src [0] : memref<{n}xi16> -> memref<{n}xi16> {{Id = "src", KernelName = "gather"}}
  %idx = ADORA.BlockLoad %ids [0] : memref<{count}xi16> -> memref<{count}xi16> {{Id = "ids", KernelName = "gather"}}
  %en = ADORA.BlockLoad %enabled [0] : memref<{count}xi16> -> memref<{count}xi16> {{Id = "enabled", KernelName = "gather"}}
  %fb = ADORA.BlockLoad %fallback [0] : memref<{count}xi16> -> memref<{count}xi16> {{Id = "fallback", KernelName = "gather"}}
  %out = ADORA.LocalMemAlloc memref<{count}xi16> {{Id = "dst", KernelName = "gather"}}
  %z = arith.constant 0 : i16
  %n = arith.constant {n} : i16
  ADORA.kernel {{
   affine.for %i = 0 to {count} {{
    %id = affine.load %idx[%i] : memref<{count}xi16>
    %e = affine.load %en[%i] : memref<{count}xi16>
    %f = affine.load %fb[%i] : memref<{count}xi16>
    %on = arith.cmpi ne, %e, %z : i16
    %lo = arith.cmpi sge, %id, %z : i16
    %hi = arith.cmpi slt, %id, %n : i16
    %bound = arith.andi %lo, %hi : i1
    %ok = arith.andi %on, %bound : i1
    %v = scf.if %ok -> (i16) {{
     %ix = arith.index_cast %id : i16 to index
     %pixel = memref.load %s[%ix] : memref<{n}xi16>
     scf.yield %pixel : i16
    }} else {{
     scf.yield %f : i16
    }}
    affine.store %v, %out[%i] : memref<{count}xi16>
   }}
   ADORA.terminator
  }} {{KernelName = "gather"}}
  ADORA.BlockStore %out, %dst [0] : memref<{count}xi16> -> memref<{count}xi16> {{Id = "dst", KernelName = "gather"}}
  return
 }}
}}
'''


def read_graph(path):
    nodes, edges = {}, {}
    for line in path.read_text().splitlines():
        match = re.match(r'(\w+)\[opcode\s*=', line)
        if match:
            nodes[match[1]] = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', line))
        match = re.match(r'(\w+) -> (\w+)\[.*operand\s*=\s*(-?\d+)', line)
        if match and int(match[3]) >= 0:
            port = int(match[3]); incoming = edges.setdefault(match[2], {})
            assert port not in incoming, ('duplicate operand', line)
            incoming[port] = match[1]
    assert nodes and not any(a['opcode'].lower() == 'undefined' for a in nodes.values())
    return nodes, edges


def signed(word):
    word &= 65535
    return word if word < 32768 else word - 65536


def model(nodes, edges, arrays, count):
    outputs, accesses = {}, []
    for iteration in range(count):
        values, active = {}, set()
        def evaluate(node):
            if node in values:
                return values[node]
            assert node not in active, ('unsupported recurrence', node)
            active.add(node)
            attrs = nodes[node]; opcode = attrs['opcode'].upper()
            args = {port: evaluate(src) for port, src in edges.get(node, {}).items()}
            if opcode == 'CONST':
                value = int(attrs['value'], 0)
            elif opcode in ('INPUT', 'LOAD'):
                obj = attrs['ref_name'].split(':')[-1]
                offset = int(attrs['offset'].split(',')[1])
                stride, cycles = map(int, attrs['pattern'].split(','))
                assert cycles == count
                address = offset + (args[0] if opcode == 'LOAD' else iteration * stride)
                assert address % 2 == 0 and 0 <= address < len(arrays[obj]) * 2, (node, iteration, address)
                value = arrays[obj][address // 2]
                if opcode == 'LOAD':
                    accesses.append(dict(node=node, iteration=iteration, object=obj, byte_offset=address))
            elif opcode == 'ADD': value = args[0] + args[1]
            elif opcode == 'SUB': value = args[0] - args[1]
            elif opcode == 'MUL': value = args[0] * args[1]
            elif opcode == 'AND': value = args[0] & args[1]
            elif opcode == 'OR': value = args[0] | args[1]
            elif opcode == 'XOR': value = args[0] ^ args[1]
            elif opcode == 'SLT': value = int(signed(args[0]) < signed(args[1]))
            elif opcode == 'SLE': value = int(signed(args[0]) <= signed(args[1]))
            elif opcode == 'SGE': value = int(signed(args[0]) >= signed(args[1]))
            elif opcode == 'ULT': value = int(args[0] < args[1])
            elif opcode == 'EQ': value = int(args[0] == args[1])
            elif opcode == 'NE': value = int(args[0] != args[1])
            elif opcode in ('SEL', 'SELECT'): value = args[1] if args[2] else args[0]
            elif opcode == 'OUTPUT':
                value = args[0]
                obj = attrs['ref_name'].split(':')[-1]
                stride, cycles = map(int, attrs['pattern'].split(','))
                assert cycles == count and (stride == 2 or count == 1)
                address = int(attrs['offset'].split(',')[1]) + iteration * stride
                result = outputs.setdefault(obj, {})
                assert address // 2 not in result
                result[address // 2] = value
            else: raise AssertionError(('unmodeled opcode', opcode))
            active.remove(node)
            values[node] = value & 65535
            return values[node]
        # All graph nodes execute, including both SELECT producers.
        for node in nodes:
            evaluate(node)
    assert len(outputs) == 1
    output = next(iter(outputs.values()))
    assert sorted(output) == list(range(count))
    return [output[i] for i in range(count)], accesses


def oracle(case):
    if case['kind'] == 'guarded_gather':
        valid = [bool(e) and 0 <= i < case['N'] for i, e in zip(case['idx'], case['enabled'])]
        indices = [i if v else 0 for i, v in zip(case['idx'], valid)]
        values = [case['source'][i] if v else f for i, v, f in zip(case['idx'], valid, case['fallback'])]
        arrays = dict(src=case['source'], ids=case['idx'], enabled=[int(v) for v in case['enabled']], fallback=case['fallback'])
    else:
        valid = [0 <= x < case['W'] and 0 <= y < case['H'] for x, y in zip(case['map_x'], case['map_y'])]
        indices = [y*case['W']+x if v else 0 for x, y, v in zip(case['map_x'], case['map_y'], valid)]
        values = [case['src'][i] if v else case['border'] for i, v in zip(indices, valid)]
        arrays = dict(src=case['src'], mx=case['map_x'], my=case['map_y'])
    if case.get('dual_read'):
        bounded = [0 <= i < case['N'] for i in case['idx']]
        values = [(a + (case['source'][i] if b else 12345)) & 65535 for a,i,b in zip(values,case['idx'],bounded)]
        indices = [j for a,i,b in zip(indices,case['idx'],bounded) for j in (a,i if b else 0)]
        valid = [v for a,b in zip(valid,bounded) for v in (a,b)]
    if 'expected' in case: assert values == case['expected']
    return arrays, values, indices, valid


def host_checks(fn, arrays, count):
    import numpy as np
    statements = []
    for statement in fn.body:
        if not isinstance(statement, ast.If): break
        statements.append(statement)
    assert statements, 'missing checked entry'
    code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), '<emitted-preflight>', 'exec', optimize=2)
    buffers = [np.asarray(data, dtype=np.int64).astype(np.uint16).view(np.int16) for data in arrays.values()]
    buffers.append(np.zeros(count, dtype=np.int16))
    def execute(args):
        exec(code, dict(np=np, **{f'arg_{i}': value for i, value in enumerate(args)}))
    execute(buffers)  # Full signed coordinate domain, including invalid values.
    invalid = []
    for i in range(len(buffers)):
        for bad in (buffers[i].astype(np.int32), buffers[i][:-1], np.repeat(buffers[i], 2)[::2]):
            args = buffers.copy(); args[i] = bad; invalid.append(args)
    readonly = buffers[-1].copy(); readonly.flags.writeable = False
    invalid.append(buffers[:-1] + [readonly])
    for i in range(len(buffers)-1):
        backing = np.zeros(max(len(buffers[i]), count), dtype=np.int16)
        args = buffers.copy(); args[i] = backing[:len(buffers[i])]; args[-1] = backing[:count]; invalid.append(args)
    rejected = 0
    for args in invalid:
        # A length-one NumPy strided view remains C contiguous by definition.
        if all(a.dtype == np.int16 and a.flags.c_contiguous and a.shape == b.shape and a.flags.writeable
               for a, b in zip(args, buffers)) and not any(np.shares_memory(a, b) for i,a in enumerate(args) for b in args[i+1:]):
            continue
        try: execute(args)
        except ValueError: rejected += 1
        else: raise AssertionError('invalid host buffers accepted')
    words = np.array([0,32767,32768,60000,65535], dtype=np.uint16)
    assert words.view(np.int16).tobytes() == words.tobytes()
    assert np.array_equal(np.frombuffer(words.view(np.int16).tobytes(), dtype=np.int16).view(np.uint16), words)
    return dict(rejected_inputs=rejected, u16_words=words.tolist(), transport='NumPy equal-width view and byte roundtrip; runtime DMA not executed')


def helper_module(root):
    path = root/'test/cgra-mapper/cstore_contract/test_cstore_alignment.py'
    spec = importlib.util.spec_from_file_location('alignment', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def audit_configuration(root, run, kernel, count, nodes, output, adg, arrays):
    h = helper_module(root)
    tree = ast.parse(output.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == kernel)
    transfers = {name: [] for name in ('idata','iptrs','odata','optrs')}; mask = None
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.attr == 'append' and n.func.value.id in transfers:
            target = n.func.value.id; value = n.args[0]
            transfers[target].append([ast.literal_eval(a) for a in value.args] if target.endswith('ptrs') else dict(arg=value.value.id, slice=ast.unparse(value.slice)))
        if isinstance(n, ast.keyword) and n.arg == 'iob_en': mask = ast.literal_eval(n.value)
    packets = dict(h.read_pytest_packets(output, kernel, adg))
    mapping = run/(kernel+'_map_result')
    routes = list(csv.DictReader((mapping/'mapped_routes.tsv').open(), delimiter='\t'))
    bindings, used = [], set()
    for line in (mapping/'mapped_dfgio.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip(): continue
        tag, name, physical, index, latency = [v.strip() for v in line.split(',')]
        physical, index, latency = int(physical), int(index), int(latency)
        match = re.match(r'(.*?)(\d+)$', name); node = match[1] + str(int(match[2])-1)
        attrs = nodes[node]; op = attrs['opcode'].upper()
        inst, module = h.find_iob(adg, physical)
        fields = {f:h.decode_controller_field(packets, inst, module, adg, f) for f in ('BaseAddr','UseAddr','UseEn','IsStore','Latency','II','Cycles0','Stride0')}
        assert fields['UseAddr'] == int(op == 'LOAD') and fields['UseEn'] == 0 and fields['IsStore'] == int(op == 'OUTPUT')
        stride, cycles = map(int, attrs['pattern'].split(','))
        assert cycles == count and fields['Cycles0'] == count and fields['Stride0'] == stride//2
        assert stride == (0 if op == 'LOAD' else 2)
        banks = adg['iob_to_spad_banks'][str(index)]; size = adg['iob_spad_bank_size']
        bytebase = fields['BaseAddr']*2 - int(attrs['offset'].split(',')[1])
        bank = min(banks)+bytebase//size
        assert bank in banks and bank not in used and bytebase % size == 0
        used.add(bank)
        pointers = transfers['optrs' if op == 'OUTPUT' else 'iptrs']
        entry = pointers.index([bank*size, int(attrs['size'])])
        data = transfers['odata' if op == 'OUTPUT' else 'idata'][entry]
        obj = attrs['ref_name'].split(':')[-1]
        argument = len(arrays) if op == 'OUTPUT' else list(arrays).index(obj)
        assert data['arg'] == f'arg_{argument}'
        expected_elements = count if op == 'OUTPUT' else len(arrays[obj])
        parsed = ast.parse('buffer[' + data['slice'] + ']', mode='eval').body.slice
        def constant(expr):
            if isinstance(expr, ast.Constant): return expr.value
            if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add): return constant(expr.left)+constant(expr.right)
            raise AssertionError(('unexpected transfer slice', data))
        assert isinstance(parsed, ast.Slice) and parsed.step is None
        assert constant(parsed.lower) == 0 and constant(parsed.upper) == expected_elements
        incoming = [r for r in routes if int(r['dst_adg']) == physical]
        if op in ('LOAD','OUTPUT'):
            assert len(incoming) == 1 and incoming[0]['logical_operand'] == '0'
            assert fields['Latency'] == int(incoming[0]['target_latency']), 'stream output incorrectly delayed as final reduction'
            h.validate_route_configuration(packets, inst, module, adg, incoming)
        assert fields['II'] > 0
        bindings.append(dict(node=node, opcode=op, object=attrs['ref_name'], bank=bank, iob=index, fields=fields))
    assert mask is not None and {i for i in range(len(mask)*8) if mask[i//8] >> (i%8) & 1} == {b['iob'] for b in bindings}
    assert all(int(r['src_latency']) + int(r['route_latency']) + int(r['rdu_delay']) == int(r['arrival_latency']) == int(r['target_latency']) for r in routes)
    assert len(transfers['optrs']) == 1 and transfers['optrs'][0][1] == 2*count
    return fn, dict(bindings=bindings, transfers=transfers, input_bytes=sum(p[1] for p in transfers['iptrs']), output_bytes=2*count, route_count=len(routes), logical_operations=dict(collections.Counter(a['opcode'].upper() for a in nodes.values())), II=bindings[0]['fields']['II'], max_iob_latency=max(b['fields']['Latency'] for b in bindings), config_bytes=len(packets)*6, config_text_bytes=(mapping/'config.bit').stat().st_size)


def run_case(root, tools, work, case, seed=7, nested=True, manual=False):
    name = case['id'] + ('-manual' if manual else '-nested' if nested else '-merged') + f'-s{seed}'
    run = work/name; run.mkdir(parents=True, exist_ok=False)
    kernel = 'gather' if case['kind'] == 'guarded_gather' else 'remap'
    ir = make_gather(case['N'], case['M']) if kernel == 'gather' else make_remap(case['W'],case['H'],case['M'],case['border'],nested,manual)
    if case.get('dual_read'):
        second = f'''    %v2 = scf.if %bound -> (i16) {{
     %ix2 = arith.index_cast %id : i16 to index
     %pixel2 = memref.load %s[%ix2] : memref<{case['N']}xi16>
     scf.yield %pixel2 : i16
    }} else {{
     %different = arith.constant 12345 : i16
     scf.yield %different : i16
    }}
    %sum = arith.addi %v, %v2 : i16
'''
        ir = ir.replace('    affine.store %v,', second + '    affine.store %sum,')
    source = run/'input.mlir'; source.write_text(ir)
    env = dict(os.environ, GeneralOpNameFile=str(root/'lib/DFG/Documents/GeneralOpName.txt'))
    result = dict(id=name, M=case['M'], seed=seed, kind=case['kind'], L4=dict(status='not_run',reason='R003 forbids new RTL/SoC runs'))
    commands = []
    def invoke(command, log):
        start = time.monotonic()
        proc = subprocess.run(list(map(str, command)), cwd=run, env=env, capture_output=True, text=True, timeout=180)
        (run/log).write_text(proc.stdout+proc.stderr)
        commands.append(dict(argv=list(map(str,command)),rc=proc.returncode,seconds=time.monotonic()-start))
        (run/'commands.json').write_text(json.dumps(commands,indent=2))
        return proc
    proc = invoke([tools/'cgra-opt',source,'--adora-kernel-dfg-gen','--verify-each','-o',run/'lowered.mlir'],'dfg.log')
    result['L1'] = dict(status='passed' if proc.returncode == 0 else 'failed', reason=None if proc.returncode == 0 else 'see dfg.log')
    if proc.returncode:
        result['L2'] = result['L3'] = dict(status='not_run',reason='DFG generation failed')
    else:
        nodes, edges = read_graph(run/(kernel+'_CDFG.dot'))
        arrays, expected, indices, valid = oracle(case)
        actual, accesses = model(nodes,edges,arrays,case['M'])
        assert actual == expected, (name,actual,expected)
        assert len(accesses) == len(indices) and [a['byte_offset']//2 for a in accesses] == indices, (name,accesses,indices)
        result['model'] = dict(status='passed',expected=expected,actual=actual,source_accesses=accesses,false_source_reads=sum(not v for v in valid),source_reads=len(accesses),layer='generated CDFG word model, not hardware transactions')
        adg_path = root/'test/spec/cgra_cstore_vitra/artifacts/adg.json'
        ops = root/'test/spec/cgra_cstore_vitra/artifacts/operations.json'
        output = run/'mapped.py'
        mapper_source = run/'lowered.mlir' if seed == 19 else source
        result['mapper_input'] = mapper_source.name
        proc = invoke([tools/'cgra-mapper',mapper_source,'--adg='+str(adg_path),'--op-file='+str(ops),'--output-type=pytest','--output='+str(output),'--seed='+str(seed),'--max-iters=30','--timeout=30000','--obj-opt=false'],'mapper.log')
        result['L2'] = dict(status='passed' if proc.returncode == 0 else 'failed', reason=None if proc.returncode == 0 else 'see mapper.log; mapping/config failure is not proof of impossibility')
        if proc.returncode == 0:
            mapped_nodes,mapped_edges = read_graph(run/(kernel+'_map_result')/('before_map_'+kernel+'_CDFG.dot'))
            mapped_values, mapped_accesses = model(mapped_nodes,mapped_edges,arrays,case['M'])
            assert mapped_values == expected
            assert len(mapped_accesses) == len(indices) and [a['byte_offset']//2 for a in mapped_accesses] == indices
            result['model']['mapper_source_accesses'] = mapped_accesses
            fn, contract = audit_configuration(root,run,kernel,case['M'],mapped_nodes,output,json.loads(adg_path.read_text()),arrays)
            result['contract'] = contract
            source_bindings = [b for b in contract['bindings'] if b['opcode'] == 'LOAD']
            result['model']['mapped_absolute_source_accesses'] = [dict(access, bank=b['bank'], absolute_byte_address=b['bank']*json.loads(adg_path.read_text())['iob_spad_bank_size']+access['byte_offset']) for access in mapped_accesses for b in source_bindings if b['node'] == access['node']]
            result['host'] = host_checks(fn, arrays, case['M'])
            result['L3'] = dict(status='passed',reason='CDFG functional model, actual emitted entry checks and static packet/transfer/route audit; no cycle-accurate execution')
        else:
            assert not output.exists(), 'failed mapping left executable call'
            result['L3'] = dict(status='not_run',reason='no valid mapped invocation')
    result['commands'] = commands
    (run/'result.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--tools',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--workloads',type=Path)
    args = parser.parse_args()
    if args.workloads:
        cases = json.loads(args.workloads.read_text())['fixtures']
    else:
        cases = [dict(id='remap_smoke',kind='nearest_remap_constant_border',W=2,H=2,M=6,src=[0,32767,32768,65535],map_x=[0,1,0,1,-1,2],map_y=[0,0,1,1,1,0],border=60000,expected=[0,32767,32768,65535,60000,60000])]
        gather = dict(id='gather_passthrough',kind='guarded_gather',N=8,M=8,
                      source=[0,17,65535,123,500,42,32768,9],
                      idx=[-32768,-1,0,7,8,32767,2,4],
                      enabled=[1,1,1,1,1,1,0,1],
                      fallback=[60000,12345,7,8,9,10,60000,12])
        cases.extend([gather, dict(gather,id='distinct_guard_passthrough',dual_read=True)])
    args.work.mkdir(parents=True,exist_ok=True)
    results = [run_case(args.root.resolve(),args.tools.resolve(),args.work.resolve(),case) for case in cases]
    (args.work/'results.json').write_text(json.dumps(results,indent=2))
    assert all(r['L1']['status']=='passed' and r['L2']['status']=='passed' for r in results), 'compiler/mapping failed; see results.json'
    print(f'{len(results)} guarded stream cases: generated graph model, host checks and static configuration PASS; RTL NOT_RUN')

if __name__ == '__main__':
    main()
