"""R006 production DFG regression. Word model is not an RTL substitute."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

opt, fixture = sys.argv[1:]
base = Path(fixture).read_text()
manual_body = '''    %safe = arith.select %read_guard, %k, %zero : i16
    %address = arith.index_cast %safe : i16 to index
    %value = memref.load %s[%address] : memref<4xi16>
    ADORA.cond_store %value, %old[%i] if %read_guard : memref<4xi16>'''
guarded_body = '''    scf.if %read_guard {
     %address = arith.index_cast %k : i16 to index
     %value = memref.load %s[%address] : memref<4xi16>
     memref.store %value, %old[%i] : memref<4xi16>
    }'''
guarded = base.replace(manual_body, guarded_body)
assert guarded != base
overlay = guarded.replace('     memref.store %value, %old[%i] : memref<4xi16>', '''     %nonzero = arith.cmpi ne, %value, %zero : i16
     scf.if %nonzero {
      memref.store %value, %old[%i] : memref<4xi16>
     }''')


def parse(dot):
    nodes = {name: dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', attrs))
             for name, attrs in re.findall(r'^(\w+)\[(.*)\];$', dot, re.M)}
    edges = {}
    for src, dst, port in re.findall(r'^(\w+) -> (\w+)\[.*operand = (-?\d+)', dot, re.M):
        if int(port) >= 0:
            assert int(port) not in edges.setdefault(dst, {}), (dst, port)
            edges[dst][int(port)] = src
    return nodes, edges


def check_graph(dot, nonzero):
    nodes, edges = parse(dot)
    byop = lambda op: [n for n, attrs in nodes.items() if attrs['opcode'].upper() == op]
    assert not byop('OUTPUT') and not byop('CLOAD') and 'undefined' not in dot
    assert 'operand = -1' not in dot, 'plain certified read/read pairs must not become unroutable physical operands'
    assert len(byop('CSTORE')) == len(byop('LOAD')) == len(byop('ACC')) == 1
    assert len(byop('INPUT')) == 2
    load, store, acc = (byop(op)[0] for op in ('LOAD', 'CSTORE', 'ACC'))
    assert nodes[acc]['loop_index_acc'] == nodes[acc]['skip_first'] == '1'
    assert nodes[acc]['acc_params'] == '0,4,1,1'
    assert set(edges[store]) == {0, 1, 2} and set(edges[load]) == {0}
    assert nodes[load]['pattern'] == '0,4'
    assert nodes[store]['pattern'] == '2,4'
    assert nodes[store]['cstore_address_mode'] == 'sequential'
    assert nodes[store]['cstore_slots'] == '4'
    assert nodes[store]['cstore_object'] == nodes[store]['ref_name'] == 'b2_inplace:dst'
    assert all(nodes[n]['pattern'] == '2,4' for n in byop('INPUT'))
    def ancestors(node, seen=None):
        seen = set() if seen is None else seen
        for src in edges.get(node, {}).values():
            if src not in seen:
                seen.add(src)
                ancestors(src, seen)
        return seen
    assert load not in ancestors(edges[load][0]), 'write guard must not control its own LOAD'
    assert (load in ancestors(edges[store][2])) == nonzero
    assert acc in ancestors(edges[store][1])
    datasets = [
        ([10,20,30,40], [2,-1,0,4], [1,1,1,1], [7,8,9,10]),
        ([0,32768,60000,65535], [3,1,0,4], [2,-1,1,1], [11,22,33,44]),
        ([0,32768,60000,65535], [-32768,-1,32767,4], [2,-1,1,0], [55,66,77,88]),
        ([0,32768,60000,65535], [0,1,2,3], [0,0,0,0], [65535,32768,60000,0]),
    ]
    signed = lambda x: x if x < 32768 else x - 65536
    for source, ids, enabled, before in datasets:
        objects = {'src': source, 'idx': ids, 'enabled': enabled, 'dst': before.copy()}
        reads, writes = [], []
        for slot in range(4):
            values, active = {}, set()
            def evaluate(n):
                if n in values: return values[n]
                assert n not in active, ('cycle', n)
                active.add(n)
                attrs = nodes[n]; op = attrs['opcode'].upper()
                args = {p: evaluate(src) for p, src in edges.get(n, {}).items()}
                if op == 'CONST': value = int(attrs['value'], 16)
                elif op == 'ACC':
                    assert args[0] == 1
                    value = slot
                elif op == 'INPUT': value = objects[attrs['ref_name'].split(':')[-1]][slot]
                elif op == 'LOAD':
                    assert args[0] % 2 == 0 and 0 <= args[0] < 8
                    reads.append(args[0] // 2)
                    value = objects[attrs['ref_name'].split(':')[-1]][args[0] // 2]
                elif op == 'CSTORE':
                    assert args[1] == 2*slot and args[2] in (0, 1)
                    if args[2]:
                        objects['dst'][slot] = args[0]
                        writes.append(slot)
                    value = 0
                elif op == 'ADD': value = args[0]+args[1]
                elif op == 'SUB': value = args[0]-args[1]
                elif op == 'MUL': value = args[0]*args[1]
                elif op == 'AND': value = args[0]&args[1]
                elif op == 'OR': value = args[0]|args[1]
                elif op == 'SLT': value = int(signed(args[0]) < signed(args[1]))
                elif op == 'SLE': value = int(signed(args[0]) <= signed(args[1]))
                elif op == 'SEL': value = args[1] if args[2] else args[0]
                else: raise AssertionError(('unsupported model op', op))
                active.remove(n)
                values[n] = value & 65535
                return values[n]
            for n in nodes: evaluate(n)
        valid = [enabled[i] != 0 and 0 <= ids[i] < 4 for i in range(4)]
        expected_writes = [i for i in range(4) if valid[i] and (not nonzero or source[ids[i]] != 0)]
        expected = [source[ids[i]] if i in expected_writes else before[i] for i in range(4)]
        assert objects['dst'] == expected, (objects['dst'], expected)
        assert reads == [ids[i] if valid[i] else 0 for i in range(4)], reads
        assert writes == expected_writes, writes


def run(name, text, good=False, nonzero=False):
    with tempfile.TemporaryDirectory(prefix='adora-inplace-') as tmp:
        directory = Path(tmp); source = directory/'input.mlir'; source.write_text(text)
        lowered = directory/'lowered.mlir'
        result = subprocess.run([opt,str(source),'--adora-kernel-dfg-gen','-o',str(lowered)],cwd=tmp,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        logs = Path(os.environ['ADORA_INPLACE_LOG_DIR']) if os.environ.get('ADORA_INPLACE_LOG_DIR') else None
        if logs:
            logs.mkdir(parents=True,exist_ok=True)
            (logs/f'{name}.mlir').write_text(text)
            (logs/f'{name}.log').write_text(f'exit={result.returncode}\n'+result.stdout)
        if not good:
            assert result.returncode != 0 and ('bounded dynamic LOAD' in result.stdout or 'guarded stream LOAD' in result.stdout), (name,result.stdout)
            assert not list(directory.glob('*CDFG.dot')), name
            return
        assert result.returncode == 0, (name,result.stdout)
        dot = (directory/'b2_inplace_CDFG.dot').read_text()
        check_graph(dot, nonzero)
        if logs:
            (logs/f'{name}.dot').write_text(dot)
            (logs/f'{name}.lowered.mlir').write_text(lowered.read_text())
        again = subprocess.run([opt,str(lowered),'--adora-kernel-dfg-gen'],cwd=tmp,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert again.returncode == 0, (name,again.stdout)


for name, text, nonzero in [('manual-b2',base,False),('guarded',guarded,False),('overlay',overlay,True)]:
    run(name,text,True,nonzero)
negative = {
    'uninitialized': base.replace('%old = ADORA.BlockLoad %dst [0] : memref<4xi16> -> memref<4xi16>', '%old = ADORA.LocalMemAlloc memref<4xi16>'),
    'different-readback': base.replace('ADORA.BlockStore %old, %dst', 'ADORA.BlockStore %old, %src'),
    'read-object-is-write-host': base.replace('ADORA.BlockLoad %enabled [0]', 'ADORA.BlockLoad %dst [0]'),
    'scatter': base.replace('ADORA.cond_store %value, %old[%i]', 'ADORA.cond_store %value, %old[%address]'),
    'feedback': base.replace('%value = memref.load %s', '%value = memref.load %old'),
    'duplicate-write': base.replace('    ADORA.cond_store', '    ADORA.cond_store %value, %old[%i] if %read_guard : memref<4xi16>\n    ADORA.cond_store'),
    'missing-readback': '\n'.join(line for line in base.splitlines() if 'ADORA.BlockStore ' not in line),
    'nonsequential-input': base.replace('affine.load %ens[%i]', 'affine.load %ens[0]'),
    'wrong-count': base.replace('= 0 to 4', '= 0 to 3'),
    'zero-count': base.replace('= 0 to 4', '= 0 to 0'),
    'count-overflow': base.replace('= 0 to 4', '= 0 to 4096'),
    'bad-read-bound': guarded.replace('%bounds = arith.andi %lo, %hi', '%bounds = arith.andi %lo, %lo'),
    'volatile-store': guarded.replace('memref.store %value, %old[%i] :', 'memref.store %value, %old[%i] {volatile} :'),
    'no-access': base.replace('} {KernelName = "b2_inplace"}', '} {KernelName = "b2_inplace", adora.require_no_access}'),
    'step-two': base.replace('= 0 to 4', '= 0 to 4 step 2'),
}
claim = ' {adora.cstore_address_mode = "sequential", adora.cstore_slots = 4 : i64, adora.cstore_object = "b2_inplace:dst"}'
claimed = base.replace('if %read_guard :', 'if %read_guard' + claim + ' :')
run('reproved-claim', claimed, True)
for name, replacement in {
    'wrong-mode': claim.replace('"sequential"', '"explicit"'),
    'unknown-mode': claim.replace('"sequential"', '"automatic"'),
    'wrong-slot-claim': claim.replace('4 : i64', '3 : i64'),
    'zero-slot-claim': claim.replace('4 : i64', '0 : i64'),
    'overflow-slot-claim': claim.replace('4 : i64', '4096 : i64'),
    'wrong-slot-type': claim.replace('4 : i64', '4 : i32'),
    'wrong-object-claim': claim.replace('b2_inplace:dst', 'b2_inplace:src'),
    'incomplete-claim': ' {adora.cstore_address_mode = "sequential"}',
    'unknown-claim': claim[:-1] + ', adora.cstore_trusted}',
}.items():
    negative[name] = base.replace('if %read_guard :', 'if %read_guard' + replacement + ' :')
negative['forged-scatter'] = claimed.replace('%old[%i] if', '%old[%address] if')
negative['claim-on-kernel'] = base.replace('} {KernelName = "b2_inplace"}', '} {KernelName = "b2_inplace", adora.cstore_slots = 4 : i64}')
negative['claim-on-module'] = base.replace('module {', 'module attributes {adora.cstore_slots = 4 : i64} {')
negative['claim-on-source-store'] = guarded.replace('memref.store %value, %old[%i] :',
                                                     'memref.store %value, %old[%i]' + claim + ' :')
negative['unproven-legacy-claim'] = '''module {
 func.func @legacy(%value: i16, %enable: i1, %dst: memref<4xi16>) {
  ADORA.kernel {
   affine.for %i = 0 to 4 {
    ADORA.cond_store %value, %dst[%i] if %enable''' + claim + ''' : memref<4xi16>
   }
   ADORA.terminator
  } {KernelName = "legacy"}
  return
 }
}'''
for name,text in negative.items(): run(name,text)
print(f'guarded inout: 4 positive graph/model/re-ingestion checks and {len(negative)} rejections passed')
