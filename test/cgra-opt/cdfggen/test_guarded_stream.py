"""Guarded reads must become safe LOADs and complete sequential outputs."""
import os
import pathlib
import re
import subprocess
import sys
import tempfile

opt, fixture = sys.argv[1:]
original = pathlib.Path(fixture).read_text()

def run(ir, name, succeeds):
    with tempfile.TemporaryDirectory(prefix='adora-guarded-') as directory:
        path = pathlib.Path(directory) / 'input.mlir'
        path.write_text(ir)
        lowered = pathlib.Path(directory) / 'lowered.mlir'
        result = subprocess.run([opt, str(path), '--adora-kernel-dfg-gen', '-o', str(lowered)], cwd=directory,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if os.environ.get('ADORA_GUARDED_LOG_DIR'):
            logs = pathlib.Path(os.environ['ADORA_GUARDED_LOG_DIR'])
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f'{name}.mlir').write_text(ir)
            (logs / f'{name}.log').write_text(f'exit={result.returncode}\n' + result.stdout)
        if not succeeds:
            assert result.returncode != 0, (name, result.stdout)
            assert ('guarded stream LOAD' in result.stdout or 'bounded dynamic LOAD' in result.stdout), (name, result.stdout)
            assert not list(pathlib.Path(directory).glob('*CDFG.dot')), name
            return
        assert result.returncode == 0, (name, result.stdout)
        dot = (pathlib.Path(directory) / 'guarded_stream_CDFG.dot').read_text()
        assert 'undefined' not in dot and 'CLOAD' not in dot and 'ACC' not in dot, dot
        for unsupported in ('XOR', 'EQ', 'NE', 'SGE'):
            assert f'opcode = "{unsupported}"' not in dot, (unsupported, dot)
        for opcode, count, pattern in [('Input', 2, '2,16'), ('load', 1, '0,16'), ('Output', 1, '2,16')]:
            nodes = [line for line in dot.splitlines() if f'opcode = "{opcode}"' in line]
            assert len(nodes) == count and all(f'pattern="{pattern}"' in line for line in nodes), (opcode, nodes)
        assert dot.count('opcode = "SEL"') >= 3, dot
        again = subprocess.run([opt, str(lowered), '--adora-kernel-dfg-gen'], cwd=directory,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if os.environ.get('ADORA_GUARDED_LOG_DIR'):
            (logs / f'{name}.lowered.mlir').write_text(lowered.read_text())
            (logs / f'{name}.reingest.log').write_text(f'exit={again.returncode}\n' + again.stdout)
        assert again.returncode == 0, (name, 'lowered IR must remain provable', again.stdout)

run(original, 'nested', True)
merged = original.replace('%v = scf.if %gx -> (i16) {\n     %inner = scf.if %gy -> (i16) {',
                          '%guard = arith.andi %gx, %gy : i1\n    %v = scf.if %guard -> (i16) {')
merged = merged.replace('     } else {\n      scf.yield %border : i16\n     }\n     scf.yield %inner : i16\n', '')
run(merged, 'merged', True)
start = merged.index('    %guard = arith.andi')
end = merged.index('    affine.store', start)
manual = merged[:start] + """    %guard = arith.andi %gx, %gy : i1
    %sx = arith.select %guard, %x, %zero : i16
    %sy = arith.select %guard, %y, %zero : i16
    %ix = arith.index_cast %sx : i16 to index
    %iy = arith.index_cast %sy : i16 to index
    %row = arith.muli %iy, %width : index
    %address = arith.addi %row, %ix : index
    %pixel = memref.load %s[%address] : memref<64xi16>
    %v = arith.select %guard, %pixel, %border : i16
""" + merged[end:]
run(manual, 'manual-safe-select', True)
# The true path must imply BOTH lower and upper bounds for each dimension.
negative = {
    'missing-lower': original.replace('%gx = arith.andi %xlo, %xhi', '%gx = arith.andi %xhi, %xhi'),
    'missing-upper': original.replace('%gx = arith.andi %xlo, %xhi', '%gx = arith.andi %xlo, %xlo'),
    'wrong-leaf': original.replace('%xhi = arith.cmpi slt, %x,', '%xhi = arith.cmpi slt, %y,'),
    'unsigned-bound': original.replace('arith.cmpi sge, %x', 'arith.cmpi uge, %x'),
    'short-source': original.replace('64xi16', '63xi16'),
    'empty-source': original.replace('64xi16', '0xi16'),
    'wrong-output-index': original.replace('affine.store %v, %out[%i]', 'affine.store %v, %out[0]'),
    'wrong-input-index': original.replace('affine.load %xs[%i]', 'affine.load %xs[0]'),
    'partial-output': original.replace('= 0 to 16', '= 0 to 15'),
    'empty-loop': original.replace('= 0 to 16', '= 0 to 0'),
    'step-two': original.replace('= 0 to 16', '= 0 to 16 step 2'),
    'physical-no-access': original.replace('%pixel = memref.load %s[%address]', '%pixel = memref.load %s[%address] {adora.require_no_access}'),
    'volatile': original.replace('%pixel = memref.load %s[%address]', '%pixel = memref.load %s[%address] {volatile}'),
    'dependent-read': original.replace('%pixel = memref.load', '%dependent = memref.load %s[%address] : memref<64xi16>\n      %next = arith.index_cast %dependent : i16 to index\n      %pixel = memref.load').replace('%pixel = memref.load %s[%address]', '%pixel = memref.load %s[%next]'),
    'overflow-flags': original.replace('arith.muli %iy, %width : index', 'arith.muli %iy, %width overflow<nsw> : index'),
    'kernel-no-access': original.replace('} {KernelName = \"guarded_stream\"}', '} {KernelName = \"guarded_stream\", adora.require_no_access}'),
    'division': original.replace('%row = arith.muli', '%row = arith.divsi'),
    'write-source': original.replace('      %pixel =', '      affine.store %zero, %s[0] : memref<64xi16>\n      %pixel ='),
    'unknown-object': original.replace('memref.load %s[%address]', 'memref.load %src[%address]'),
    'outer-missing-lower': original.replace('%v = scf.if %gx', '%v = scf.if %xhi'),
}
negative['unguarded-else-load'] = original.replace(
    '%inner = scf.if %gy -> (i16) {',
    '%inner = scf.if %gy -> (i16) {\n      scf.yield %border : i16\n     } else {').replace(
    '      scf.yield %pixel : i16\n     } else {\n      scf.yield %border : i16',
    '      scf.yield %pixel : i16')
negative['loop-overflow-flags'] = original.replace(
    '    %v = scf.if %gx',
    '    %poison = arith.addi %y, %eight overflow<nsw> : i16\n    %v = scf.if %gx').replace(
    '     %inner = scf.if %gy',
    '     %extra = arith.cmpi ne, %poison, %zero : i16\n     %inner_guard = arith.andi %gy, %extra : i1\n     %inner = scf.if %inner_guard')
run(negative['loop-overflow-flags'].replace(' overflow<nsw>', ''), 'defined-extra-predicate', True)
negative['count-overflow'] = original.replace('= 0 to 16', '= 0 to 4096')
negative['function-no-access'] = original.replace('%dst: memref<16xi16>) {', '%dst: memref<16xi16>) attributes {adora.require_no_access} {')
negative['manual-no-access'] = manual.replace('} {KernelName = "guarded_stream"}', '} {KernelName = "guarded_stream", adora.require_no_access}')
negative['nontemporal'] = original.replace('%pixel = memref.load %s[%address]', '%pixel = memref.load %s[%address] {nontemporal = true}')
negative['manual-volatile'] = manual.replace('%pixel = memref.load %s[%address]', '%pixel = memref.load %s[%address] {volatile}')
negative['manual-false-unsafe'] = manual.replace('arith.select %guard, %x, %zero', 'arith.select %guard, %x, %border')
negative['unknown-effect'] = original.replace('module {', 'module {\n func.func private @effect(i16) -> i16').replace(
    '      %pixel =', '      %effect = func.call @effect(%x) : (i16) -> i16\n      %pixel =')
negative['loop-carried-fallback'] = original.replace('  ADORA.kernel {', '  %init = arith.constant 0 : i16\n  ADORA.kernel {').replace(
    'affine.for %i = 0 to 16 {', '%final = affine.for %i = 0 to 16 iter_args(%carry = %init) -> (i16) {').replace(
    'scf.yield %border', 'scf.yield %carry').replace(
    'affine.store %v, %out[%i] : memref<16xi16>', 'affine.store %v, %out[%i] : memref<16xi16>\n    affine.yield %v : i16')
negative['source-no-access'] = original.replace('Id = "src", KernelName', 'adora.require_no_access, Id = "src", KernelName')
for name, ir in negative.items():
    run(ir, name, False)
print(f'guarded stream: nested/merged/manual/extra-predicate positives and {len(negative)} rejection cases passed')
