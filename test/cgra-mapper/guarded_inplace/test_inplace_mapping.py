"""Map independent B2 and a constant-placement variant, then decode packets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from audit_guarded_inplace import audit

mapper, fixture, adg, operations, work = [Path(arg).resolve() for arg in sys.argv[1:]]
work.mkdir(parents=True, exist_ok=True)
base = fixture.read_text()
# Equivalent scalar constant placement changes graph IDs and routed placement.
# This form exposes a late address path whose old scheduler shifted the
# autonomous ACC instead of providing enough real route/RDU delay.
late = base.replace('    %zero = arith.constant 0 : i16\n', '').replace(
    '    %n = arith.constant 4 : i16\n', '')
late = late.replace('    %on =', '    %zero = arith.constant 0 : i16\n    %on =')
late = late.replace('    %lo =', '    %lowerzero = arith.constant 0 : i16\n    %lo =').replace(
    'sge, %k, %zero', 'sge, %k, %lowerzero')
late = late.replace('    %hi =', '    %n = arith.constant 4 : i16\n    %hi =')
late = late.replace('    %safe =', '    %safezero = arith.constant 0 : i16\n    %safe =').replace(
    'select %read_guard, %k, %zero', 'select %read_guard, %k, %safezero')
for name, source in [('b2', base), ('late-constants', late)]:
    entry = Path(tempfile.mkdtemp(prefix=name + '-', dir=work))
    (entry / 'map').mkdir()
    (entry / 'input.mlir').write_text(source)
    command = [str(mapper), str(entry / 'input.mlir'), '--adg=' + str(adg),
               '--op-file=' + str(operations), '--output-type=pytest',
               '--output=' + str(entry / 'map/unchecked.py'), '--seed=7',
               '--max-iters=100', '--timeout=30000']
    completed = subprocess.run(command, cwd=entry / 'map', env=os.environ,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=150)
    (entry / 'mapper.log').write_text(completed.stdout)
    (entry / 'command.json').write_text(json.dumps(dict(argv=command, returncode=completed.returncode)))
    assert completed.returncode == 0, (entry, completed.stdout[-3000:])
    report = audit(entry, adg, operations, 4)
    (entry / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print('initialized inout mapping: two independent-B2 packet/route/ACC launch audits passed')
