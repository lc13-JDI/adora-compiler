"""The large profile proves genuine SSA effects before emitting a bounded tile."""
import copy
import json
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
    def run(text, c=contract, good=True, large=True, z=True):
        (root/'input.mlir').write_text(text)
        (root/'contract.json').write_text(json.dumps(c))
        descriptor=root/'proof.json'; descriptor.unlink(missing_ok=True)
        opt=f'--adora-lower-c-entry=contract={root}/contract.json allow-partial-writes=true'
        if large: opt+=f' large-objects=true descriptor={descriptor}'
        r=subprocess.run([tool,str(root/'input.mlir'),opt,'--verify-each'],capture_output=True,text=True)
        assert (r.returncode==0)==good, r.stderr+r.stdout
        if good:
            proof=json.loads(descriptor.read_text())
            assert proof['width']==8193 and proof['height']==1 and proof['items']==next(o['elements'] for o in c['objects'] if o['access_assertion']=='inout')
            assert proof['tile_items']==256 and proof['value_nonzero']==z
            assert 'memref<256xi16>' in r.stdout and 'ADORA.kernel' in r.stdout
            assert 'i64' not in r.stdout and 'i32' not in r.stdout
            return proof
        assert not descriptor.exists()
    run(source)
    single=copy.deepcopy(contract)
    for obj in single['objects'][1:]: obj['elements']=1
    run(Path(fixture).with_name('large_single.txt').read_text(),single,z=False)
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
