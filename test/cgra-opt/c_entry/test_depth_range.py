"""Exact range effects, high-bit semantics, and fail-closed SSA mutations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

tool, fixture = sys.argv[1:]
source = Path(fixture).read_text()
contract = dict(entry='depth_range_overlay', objects=[
    dict(argument_position=i, elements=n, host_dtype=t, access_assertion=a, contiguous=True)
    for i, n, t, a in [(0,9,'uint16','read'), (1,12,'int16','read'),
                        (2,12,'int16','read'), (3,12,'int16','read'), (4,12,'uint16','inout')]],
    host_plan=dict(strategy='per-query', tile_items=256, bind_arguments=[1,2,3], ownership='copy'))
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root/'contract.json').write_text(json.dumps(contract))
    def run(name, text, bounds=(500,4000), good=True):
        (root/'input.mlir').write_text(text)
        proof = root/'proof.json'
        proof.unlink(missing_ok=True)
        result = subprocess.run([tool, str(root/'input.mlir'),
            f'--adora-lower-c-entry=contract={root}/contract.json allow-partial-writes=true large-objects=true descriptor={proof}',
            '--verify-each'], capture_output=True, text=True)
        assert (result.returncode == 0) == good, name + '\n' + result.stderr
        if not good:
            assert not proof.exists(), name
            return
        p = json.loads(proof.read_text())
        assert p['value_mode'] == 'range' and not p['value_nonzero'], p
        assert (p['range_lo'],p['range_hi']) == bounds, p
        assert 'arith.addi' in result.stdout and '-32768' in result.stdout
        assert 'i32' not in result.stdout and 'i64' not in result.stdout
        assert 'overflow<nsw' not in result.stdout and 'overflow<nuw' not in result.stdout
        lo, hi = bounds
        # Exhaustive independent arithmetic model for the emitted modular bias.
        for value in range(65536):
            biased = ((value + 32768) & 65535)
            signed = biased - 65536 if biased >= 32768 else biased
            assert (lo <= value <= hi) == (lo-32768 <= signed <= hi-32768)
        print(name, 'PASS', flush=True)
    run('actual_cgeist_low', source)
    unsigned = source.replace('arith.extsi %20 : i16 to i32', 'arith.extui %20 : i16 to i32')
    high = unsigned.replace('500 : i32', '32768 : i32').replace('4000 : i32', '60000 : i32')
    run('unsigned_high', high, (32768,60000))
    run('reversed_comparisons', unsigned.replace('sge, %21, %c500_i32', 'sle, %c500_i32, %21').replace('sle, %21, %c4000_i32', 'sge, %c4000_i32, %21'))
    run('reversed_unsigned_comparisons', high.replace('sge, %21, %c500_i32', 'ule, %c500_i32, %21').replace('sle, %21, %c4000_i32', 'uge, %c4000_i32, %21'), (32768,60000))
    for name, before, after in [
        ('missing_guard','scf.if %14 {','scf.if %6 {'),
        ('wrong_bound','slt, %1, %c9_i32','sle, %1, %c9_i32'),
        ('narrow_address','arith.extsi %2 : i16 to i64','arith.extui %2 : i16 to i64'),
        ('other_comparison','sge, %21, %c500_i32','sge, %1, %c500_i32'),
        ('wrong_fallback','scf.yield %c0_i32 : i32','scf.yield %c500_i32 : i32'),
        ('wrong_index','memref.store %24, %arg4[%arg5]','memref.store %24, %arg4[%19]'),
        ('feedback','memref.load %arg0[%19]','memref.load %arg4[%19]'),
        ('extra_load','%21 = arith.extsi','%extra = memref.load %arg0[%19] : memref<?xi16>\n        %21 = arith.extsi'),
        ('extra_store','memref.store %24, %arg4[%arg5] : memref<?xi16>','memref.store %24, %arg4[%arg5] : memref<?xi16>\n        memref.store %24, %arg4[%arg5] : memref<?xi16>'),
        ('volatile','memref.load %arg0[%19] :','memref.load %arg0[%19] {volatile} :'),
        ('zero_lower','500 : i32','0 : i32'),
        ('empty_range','500 : i32','5000 : i32')]:
        assert before in source, name
        run(name, source.replace(before, after), good=False)
    run('signed_high_is_not_unsigned', high.replace('arith.extui %20', 'arith.extsi %20'), good=False)
print('depth range SSA tests passed')
