"""Typed-IR positive and rejection checks, with no fixture-name semantics."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

opt = str(Path(shutil.which(sys.argv[1]) or sys.argv[1]).resolve())
here = Path(__file__).resolve().parent
source = (here / 'two_loops.mlir').read_text()
contract = json.loads((here / 'contract.json').read_text())
with tempfile.TemporaryDirectory(prefix='adora-canvas-proof-') as temporary:
    work = Path(temporary)
    def run(text, accept=True, c=None, role='both'):
        (work/'in.mlir').write_text(text)
        (work/'contract.json').write_text(json.dumps(c or contract))
        proof = work/'proof.json'
        proof.unlink(missing_ok=True)
        p = subprocess.run([opt, str(work/'in.mlir'),
            f'--adora-lower-c-entry=contract={work}/contract.json large-objects=true canvas-mask=true canvas-kernel={role} descriptor={proof}',
            '--verify-each', '-o', str(work/'out.mlir')], capture_output=True, text=True)
        assert p.returncode == (0 if accept else 1), p.stderr
        if not accept:
            assert ': error: canvas C-entry:' in p.stderr or ': error: large C-entry:' in p.stderr, p.stderr
        assert proof.exists() == accept
        if accept:
            d = json.loads(proof.read_text())
            assert d['schema'] == 'adora.guarded-canvas-mask.v1'
            assert d['old_argument'] == 4 and d['destination_argument'] == 5
            output = (work/'out.mlir').read_text()
            assert 'memref<512' not in output
            assert ('@_adora_canvas_regular_tile' in output) == (role != 'old-only')
            assert ('@_adora_canvas_old_tile' in output) == (role != 'regular')
        return p
    run(source)
    run(source.replace('arith.constant 5000 : i32', 'arith.constant 32768 : i32').replace('arith.constant 10000 : i32', 'arith.constant 60000 : i32'))
    run((here / 'ssa_equivalent.mlir.in').read_text())
    for role in ['regular', 'old-only']:
        run(source, role=role)
        if role == 'old-only':
            dfg = subprocess.run([opt, str(work/'out.mlir'), '--adora-kernel-dfg-gen',
                '--verify-each', '-o', str(work/'dfg.mlir')], cwd=work, capture_output=True, text=True)
            assert dfg.returncode == 0, dfg.stderr
            graph = (work/'_adora_canvas_old_tile_CDFG.dot').read_text()
            assert 'opcode = "NE"' not in graph, 'old-only mask needs hardware-supported signed nonzero comparisons'
            # Canonicalization turns signed 0 < v into signed 1 <= v.
            assert graph.count('opcode = "SLT"') == 1 and graph.count('opcode = "SLE"') == 1
            assert 'opcode = "OR"' in graph
    for items, width, height in [(353, 32, 8), (307200, 640, 480)]:
        c = json.loads(json.dumps(contract))
        for obj in c['objects']:
            obj['elements'] = width * height if obj['argument_position'] == 0 else items
        text = source.replace('arith.constant 512 : index', f'arith.constant {items} : index')
        text = text.replace('memref<512xi16>', f'memref<{items}xi16>')
        text = text.replace('arith.constant 32 : i64', f'arith.constant {width} : i64')
        text = text.replace('arith.constant 32 : i32', f'arith.constant {width} : i32')
        text = text.replace('arith.constant 8 : i32', f'arith.constant {height} : i32')
        run(text, c=c)
    c = dict(contract, entry='anything')
    run(source.replace('@depth_canvas_mask', '@anything').replace('%arg', '%different'), c=c)
    # Every local-object use is enumerated. Extra observer, cross index,
    # missing initialization and altered output semantics must reject.
    mutations = [
        source.replace('memref.store %4, %alloca[%arg6]', 'memref.store %c0_i16, %alloca[%arg6]'),
        source.replace('    scf.for %arg6', '    %escape = memref.cast %alloca : memref<512xi16> to memref<?xi16>\n    scf.for %arg6', 1),
        source.replace('      memref.store %4, %alloca[%arg6] : memref<512xi16>', ''),
        source.replace('memref.load %arg4[%arg6] : memref<?xi16>', 'memref.load %arg4[%arg6] {adora.no_access} : memref<?xi16>'),
        source.replace('%0 = memref.load %alloca[%arg6]', '%extra = memref.load %alloca[%arg6] : memref<512xi16>\n      %0 = memref.load %alloca[%arg6]'),
        source.replace('%0 = memref.load %alloca[%arg6]', '%0 = memref.load %alloca[%c1]'),
        source.replace('arith.constant 255 : i32', 'arith.constant 254 : i32'),
        source.replace('memref.store %25, %alloca[%arg6] : memref<512xi16>', 'memref.store %25, %arg5[%arg6] : memref<?xi16>'),
        source.replace('step %c1 {', 'step %c512 {', 1),
        source.replace('to %c512 step %c1 {\n      %0 = memref.load %alloca', 'to %c1 step %c1 {\n      %0 = memref.load %alloca'),
        source.replace('arith.constant 5000 : i32', 'arith.constant 0 : i32'),
        source.replace('%c32_i32 : i32', '%c8_i32 : i32', 1),
        source.replace('arith.select %26, %21, %c0_i16', 'arith.select %26, %c0_i16, %21'),
        source.replace('arith.constant 5000 : i32', 'arith.constant 32768 : i32').replace('arith.constant 10000 : i32', 'arith.constant 60000 : i32').replace('arith.extui %21', 'arith.extsi %21'),
    ]
    for text in mutations:
        run(text, False)
    print('canvas typed proof: positive, SSA variant, high range, tail/full domains, roles, renaming and 14 rejection cases passed')
