"""The large profile proves genuine SSA effects before emitting a bounded tile."""
import copy
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile

tool, fixture = sys.argv[1:]
source = Path(fixture).read_text()
contract = {'entry':'large_overlay', 'objects':[
    dict(argument_position=i,elements=n,host_dtype=d,access_assertion=a,contiguous=True)
    for i,n,d,a in [(0,8193,'uint16','read'),(1,4,'int16','read'),(2,4,'int16','read'),(3,4,'int16','read'),(4,4,'uint16','inout')]],
    'host_plan':dict(strategy='per-query',tile_items=256,bind_arguments=[1,2,3],ownership='copy')}
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    def run(text, c=contract, good=True, large=True, z=True, indexed=False):
        (root/'input.mlir').write_text(text)
        (root/'contract.json').write_text(json.dumps(c))
        descriptor=root/'proof.json'; descriptor.unlink(missing_ok=True)
        opt=f'--adora-lower-c-entry=contract={root}/contract.json allow-partial-writes=true'
        if large: opt+=f' large-objects=true descriptor={descriptor}'
        if large and not indexed: opt+=" large-indexed=false"
        r=subprocess.run([tool,str(root/'input.mlir'),opt,'--verify-each'],capture_output=True,text=True)
        assert (r.returncode==0)==good, r.stderr+r.stdout
        if good:
            proof=json.loads(descriptor.read_text())
            assert proof['width']==8193 and proof['height']==1 and proof['items']==next(o['elements'] for o in c['objects'] if o['access_assertion']=='inout')
            assert proof['tile_items']==256 and proof['value_nonzero']==z
            assert 'memref<256xi16>' in r.stdout and 'ADORA.kernel' in r.stdout
            assert 'i64' not in r.stdout and 'i32' not in r.stdout
            same_slot = proof['value_mode'] == 'direct' and not indexed
            assert proof['packing_layout'] == ('same-slot' if same_slot else 'indexed')
            assert proof['local_roles'] == (['packed', 'enable', 'old'] if same_slot
                                            else ['packed', 'idx', 'enable', 'old'])
            signature = next(line for line in r.stdout.splitlines() if 'func.func' in line)
            assert signature.count('memref<256xi16>') == len(proof['local_roles'])
            if same_slot:
                def dfg(ir, good):
                    lowered = subprocess.run([tool, '-', '--adora-kernel-dfg-gen', '--verify-each'],
                        cwd=root, input=ir, text=True, capture_output=True)
                    assert (lowered.returncode == 0) == good, lowered.stderr
                    return lowered.stdout
                local = dfg(r.stdout, True)
                dot = (root / '_adora_large_tile_CDFG.dot').read_text()
                assert dot.count('opcode = "Input"') == 2 and 'opcode = "load"' not in dot.lower(), dot
                assert 'cstore_address_mode="sequential"' in dot and 'cstore_slots="256"' in dot
                data = re.search(r'^(\w+)\[opcode = "Input", ref_name="_adora_large_tile:arg0"', dot, re.M)
                store = re.search(r'^(\w+)\[opcode = "CSTORE", ref_name="_adora_large_tile:arg2"', dot, re.M)
                assert data and store, dot
                edges = re.findall(r'^(\w+) -> ' + store[1] + r'\[.*?operand = (-?\d+)', dot, re.M)
                assert sorted(port for _, port in edges) == ['0', '2'] and (data[1], '0') in edges, dot
                assert dot.count('pattern="2,256"') == 3 and 'opcode = "output"' not in dot.lower()
                # The mapper reads final MLIR and independently repeats the proof.
                dfg(local, True)
                for before, after in [
                    ('arith.cmpi slt,', 'arith.cmpi sle,'),
                    (', adora.large_direct_same_slot', ''),
                    ('adora.host_dtype = "int16"', 'adora.host_dtype = "uint16"'),
                    ('adora.cstore_slots = 256', 'adora.cstore_slots = 255'),
                    ('ADORA.BlockLoad %arg1', 'ADORA.BlockLoad %arg0'),
                    ('ADORA.BlockStore %result_1, %arg2', 'ADORA.BlockStore %result_1, %arg0'),
                ]:
                    assert before in local, local
                    dfg(local.replace(before, after), False)
            return proof
        assert not descriptor.exists()
    run(source)
    single=copy.deepcopy(contract)
    for obj in single['objects'][1:]: obj['elements']=1
    direct = Path(fixture).with_name('large_single.txt').read_text()
    run(direct, single, z=False)
    # A driver which did not negotiate owned same-slot packing keeps four roles.
    run(direct, single, z=False, indexed=True)
    forged = copy.deepcopy(single)
    forged['host_plan']['packing_layout'] = 'same-slot'
    run(direct, forged, good=False)
    run(direct.replace('scf.if %16 {', 'scf.if %8 {'), single, good=False)
    renamed=copy.deepcopy(contract); renamed['entry']='unrelated'
    run(source.replace('large_overlay','unrelated').replace('%20','%pixel'),renamed)
    # Argument identity follows actual SSA, not a spelling/position heuristic.
    reordered=copy.deepcopy(contract)
    reordered['objects'][0],reordered['objects'][1]=reordered['objects'][1],reordered['objects'][0]
    for i,o in enumerate(reordered['objects']): o['argument_position']=i
    reordered['host_plan']['bind_arguments']=[0,2,3]
    text=source.replace('%arg0: memref<?xi16>, %arg1: memref<?xi16>', '%arg1: memref<?xi16>, %arg0: memref<?xi16>')
    assert run(text,reordered)['source_argument']==1
    run(source.replace('arith.andi %6, %13','arith.andi %13, %6'))
    run(source.replace('sge, %1, %c0_i32','sge, %3, %c0_i32'),good=False)
    run(source.replace('ne, %21, %c0_i32','eq, %21, %c0_i32'),good=False)
    run(source.replace('memref.store %20, %arg4[%arg5]','memref.store %20, %arg4[%19]'),good=False)
    run(source.replace('memref.load %arg0[%19]','memref.load %arg4[%19]'),good=False)
    run(source.replace('memref.store %20, %arg4[%arg5] : memref<?xi16>','memref.store %20, %arg4[%arg5] : memref<?xi16>\n memref.store %20, %arg4[%arg5] : memref<?xi16>'),good=False)
    run(source.replace('8193 : i64','8192 : i64'),good=False)
    run(source.replace('arith.extsi %2 : i16 to i64','arith.extui %2 : i16 to i64'),good=False)
    run(source.replace('scf.if %14','scf.if %6'),good=False)
    run(source.replace('module attributes {}', 'module attributes {adora.unknown}'),good=False)
    run(source.replace('attributes {llvm.linkage', 'attributes {adora.unknown, llvm.linkage'),good=False)
    run(source.replace('%arg0: memref<?xi16>', '%arg0: memref<?xi16> {adora.unknown}'),good=False)
    bad=copy.deepcopy(contract); bad['host_plan']['bind_arguments']=[2,1,3]
    run(source,bad,good=False)
    bad=copy.deepcopy(contract); bad.pop('host_plan')
    run(source,bad,good=False,large=False)
print('large C entry SSA tests passed')
