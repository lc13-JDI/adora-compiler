"""A mapped tensor kernel must not emit a call after I/O allocation fails."""
import json
import subprocess
import sys
from pathlib import Path

mapper, adg, operations, source, output = [Path(p).resolve() for p in sys.argv[1:]]
output.mkdir(parents=True, exist_ok=True)
architecture = json.loads(adg.read_text())
# Preserve the graph/topology, but make each bank too small for the GEMM
# buffers. Mapping still succeeds; the emitter must reject the allocation.
architecture['iob_spad_bank_size'] = 2
small_bank = output / 'small_bank.json'
small_bank.write_text(json.dumps(architecture))
call = output / 'rejected.py'
assert not call.exists(), 'use a fresh test output directory'
command = [str(mapper), str(source), '--adg=' + str(small_bank),
           '--op-file=' + str(operations), '--output-type=pytest',
           '--obj-opt=false', '--seed=7', '--max-iters=30', '--timeout=10000',
           '--output=' + str(call)]
result = subprocess.run(command, cwd=output, capture_output=True, text=True,
                        timeout=120)
log = result.stdout + result.stderr
(output / 'mapper.log').write_text(log)
assert 'incomplete I/O execution contract' in log, log
assert result.returncode != 0, 'I/O contract failure was ignored; mapper returned success'
assert 'cgra-mapper: tensor CDFG generation failed.' in log, log
assert not call.exists() or call.stat().st_size == 0, 'failed tensor emitted a call'
