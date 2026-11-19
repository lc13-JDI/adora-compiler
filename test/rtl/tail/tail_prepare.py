"""Build the C entry and check that the simulator receives its fresh artifacts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from workload import HERE, contract

ROOT = HERE.parents[2]


def compile_case(driver, paths, work, build_dir, cgeist, short):
    work.mkdir()
    compiled = work / 'mapped.py'
    spec = work / 'contract.json'
    spec.write_text(json.dumps(contract(short), indent=2) + '\n')
    cgeist = Path(cgeist or shutil.which('cgeist') or 'cgeist').resolve()
    if not cgeist.is_file() or cgeist.name != 'cgeist':
        raise ValueError('provide --cgeist PATH to the Polygeist cgeist executable')
    env = dict(os.environ, PATH=os.pathsep.join([str(build_dir / 'bin'), str(cgeist.parent), os.environ['PATH']]),
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               GeneralOpNameFile=str(ROOT / 'lib/DFG/Documents/GeneralOpName.txt'))
    env.pop('PYTHONPATH', None)
    for tool in ('cgra-opt', 'cgra-mapper'):
        if Path(shutil.which(tool, path=env['PATH']) or '').resolve() != (build_dir / 'bin' / tool).resolve():
            raise ValueError(f'build {tool} in {build_dir} first')
    command = [sys.executable, '-B', str(driver), str(HERE / 'large_overlay.c'),
               '--profile', 'guarded-inplace-large', '--contract', str(spec),
               '--adg-path', str(paths['adg']), '--op-file', str(paths['operations']),
               '--kernel-cache', str(work / 'kernel-cache'), '--seed', '7',
               '--max-iters', '30', '--timeout', '30000', '--mapper-wall-timeout', '900',
               '--work-dir', str(work), '-o', str(compiled)]
    (work / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (work / 'compiler.log').open('w') as log:
        status = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if status.returncode:
        raise RuntimeError(f'ordinary C compilation failed ({status.returncode}), see {work}/compiler.log')
    artifacts = work / 'adora-c-entry'
    provenance = json.loads((artifacts / 'provenance.json').read_text())
    if provenance['status'] != 'complete' or provenance['kernel_cache']['hit']:
        raise RuntimeError('expected fresh ordinary C mapping')
    bit, = (artifacts / 'map').glob('*_map_result/config.bit')
    for role in ('adg', 'operations'):
        if (artifacts / (role + '.json')).read_bytes() != paths[role].read_bytes():
            raise RuntimeError(f'compiler {role} differs from RTL target')
    return compiled, bit


def prepare(paths, output, build_dir, cgeist, case):
    driver = ROOT / 'tools/adoracc/adoracc.py'
    build_dir = Path(build_dir).resolve()
    mapped, bit = compile_case(driver, paths, output / 'short', build_dir, cgeist, True)
    env = dict(ADORA_RTL_MAPPED=str(mapped), ADORA_RTL_BIT=str(bit), TESTCASE='short_tail')
    if case == 'ownership':
        many, many_bit = compile_case(driver, paths, output / 'many', build_dir, cgeist, False)
        env.update(ADORA_RTL_MANY=str(many), ADORA_RTL_MANY_BIT=str(many_bit),
                   TESTCASE='snapshot_wait,cleanup_retry,unknown_completion,task_cancel,steady')
    return env
