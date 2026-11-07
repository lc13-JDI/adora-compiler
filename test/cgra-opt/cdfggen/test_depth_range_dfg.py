"""Range data and address-only write predicate survive real DFG generation."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

tool, fixture = sys.argv[1:]
tool = str(Path(tool).resolve())
source = Path(fixture).read_text()

def run(name, text, good, bounds=(32768,60000)):
    with tempfile.TemporaryDirectory(prefix='adora-range-dfg-') as directory:
        root = Path(directory)
        (root/'input.mlir').write_text(text)
        result = subprocess.run([tool,str(root/'input.mlir'),'--adora-kernel-dfg-gen',
                                 '--verify-each','-o',str(root/'output.mlir')],cwd=root,capture_output=True,text=True)
        assert (result.returncode == 0) == good, name + '\n' + result.stderr
        graphs = list(root.glob('*CDFG.dot'))
        if not good:
            assert not graphs, name
            return
        assert len(graphs) == 1
        dot = graphs[0].read_text()
        assert 'undefined' not in dot
        nodes = {n:dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"',attrs))
                 for n,attrs in re.findall(r'^(\w+)\[(.*)\];$',dot,re.M)}
        edges = {}
        for src,dst,port in re.findall(r'^(\w+) -> (\w+)\[.*operand = (-?\d+)',dot,re.M):
            if int(port)>=0: edges.setdefault(dst,{})[int(port)] = src
        byop = lambda op:[n for n,a in nodes.items() if a['opcode'].upper()==op]
        assert len(byop('LOAD')) == len(byop('CSTORE')) == 1
        load,store = byop('LOAD')[0],byop('CSTORE')[0]
        def ancestors(n,seen=None):
            seen=set() if seen is None else seen
            for parent in edges.get(n,{}).values():
                if parent not in seen:
                    seen.add(parent);ancestors(parent,seen)
            return seen
        assert nodes[edges[store][0]]['opcode'].upper() == 'SEL'
        assert load in ancestors(edges[store][0])
        assert load not in ancestors(edges[store][2]), 'pixel comparison leaked into write predicate'
        assert nodes[store]['cstore_address_mode']=='sequential' and nodes[store]['cstore_slots']=='256'
        # Evaluate actual emitted data slice for every uint16 code. No host oracle
        # value or hand-authored ideal graph is substituted for the DFG.
        def signed(v): return v if v<32768 else v-65536
        for value in range(65536):
            memo={load:value}
            def evaluate(n):
                if n in memo:return memo[n]
                a=nodes[n];op=a['opcode'].upper();args={p:evaluate(v) for p,v in edges.get(n,{}).items()}
                if op=='CONST':v=int(a['value'],16)
                elif op=='ADD':v=args[0]+args[1]
                elif op=='SUB':v=args[0]-args[1]
                elif op=='AND':v=args[0]&args[1]
                elif op=='OR':v=args[0]|args[1]
                elif op=='XOR':v=args[0]^args[1]
                elif op=='SLT':v=int(signed(args[0])<signed(args[1]))
                elif op=='SLE':v=int(signed(args[0])<=signed(args[1]))
                elif op=='SEL':v=args[1] if args[2] else args[0]
                else:raise AssertionError((op,n))
                memo[n]=v&65535;return memo[n]
            assert evaluate(edges[store][0]) == (value if bounds[0]<=value<=bounds[1] else 0)
        again=subprocess.run([tool,str(root/'output.mlir'),'--adora-kernel-dfg-gen','--verify-each'],cwd=root,capture_output=True,text=True)
        assert again.returncode==0,again.stderr
        print(name,'PASS',flush=True)

run('actual_high_range',source,True)
run('low_range', source.replace('%c0_i16_3 = arith.constant 0 : i16',
    '%c0_i16_3 = arith.constant -32268 : i16').replace('arith.constant 27232 : i16',
    'arith.constant -28768 : i16'), True, (500,4000))
for name,before,after in [
    ('nonzero_fallback','scf.yield %c0_i16 : i16','scf.yield %c256_i16 : i16'),
    ('wrong_pixel','scf.yield %8 : i16','scf.yield %0 : i16'),
    ('wrong_bias','arith.constant -32768 : i16','arith.constant -32767 : i16'),
    ('overflow_bias','arith.addi %8, %c-32768_i16 : i16','arith.addi %8, %c-32768_i16 overflow<nsw> : i16'),
    ('feedback','memref.load %result[%7]','memref.load %result_2[%7]'),
    ('wrong_index','%result_2[%arg4]','%result_2[%7]'),
    ('pixel_write_predicate','memref.store %13, %result_2[%arg4] : memref<256xi16>',
        'scf.if %12 { memref.store %13, %result_2[%arg4] : memref<256xi16> }')]:
    assert before in source,name
    run(name,source.replace(before,after),False)
print('depth range DFG tests passed')
