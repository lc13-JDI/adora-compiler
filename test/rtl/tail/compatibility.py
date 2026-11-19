"""Maintainer regression for the two historical ownership-version pairings."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'build_tools'), str(HERE)]
import rtl_sim
from tail_prepare import compile_case

ADORA_BASELINE = 'efc13bb02fdbdf87836e801962afe7fd0440db26'
RUNTIME_BASELINE = '9f9f1e48a196a5933a55b8053adc5215f91f8db5d572226fadc2d25da453a7c0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('assets', 'cache', 'build-dir', 'output', 'cgeist', 'legacy-runtime'):
        parser.add_argument('--' + arg, type=rtl_sim.absolute, required=True)
    parser.add_argument('--adora-git', type=rtl_sim.absolute, default=ROOT)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if hashlib.sha256(args.legacy_runtime.read_bytes()).hexdigest() != RUNTIME_BASELINE:
        parser.error('expected server/test_runif.py from e9560326ee307f2a456c65ea3a7a71e91846d019')
    spec, paths = rtl_sim.validated('cstore', args.assets)
    args.output.mkdir(parents=True)
    driver = args.output / 'legacy-driver'
    driver.mkdir()
    for name in ('adoracc.py', 'prepared_xy.py', 'c_entry.py', 'prepared_canvas.py', 'canvas_entry.py'):
        source = subprocess.check_output(['git', '-C', str(args.adora_git), 'show',
                                          f'{ADORA_BASELINE}:tools/adoracc/{name}'])
        (driver / name).write_bytes(source)
    old, old_bit = compile_case(driver / 'adoracc.py', paths, args.output / 'old', args.build_dir, args.cgeist, True)
    new, new_bit = compile_case(ROOT / 'tools/adoracc/adoracc.py', paths, args.output / 'new', args.build_dir,
                                args.cgeist, True)
    binary, receipt, hit = rtl_sim.build_model(spec, paths, args.cache, args.jobs)
    env = dict(TESTCASE='old_module,old_runtime', ADORA_LEGACY_RUNTIME=str(args.legacy_runtime),
               ADORA_LEGACY_MAPPED=str(old), ADORA_LEGACY_BIT=str(old_bit),
               ADORA_RTL_MAPPED=str(new), ADORA_RTL_BIT=str(new_bit))
    rtl_sim.run(SimpleNamespace(case='compatibility', output=args.output / 'rtl', port=0),
                'cstore', paths, binary, receipt, hit, case_env=env)


if __name__ == '__main__':
    main()
