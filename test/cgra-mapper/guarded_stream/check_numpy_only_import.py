"""Import the whole generated module without optional convolution dependencies.

The default runtime placeholder only supplies import names: this is an import
regression, not kernel execution. --runtime-dir uses an installed real runtime.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('generated', type=Path)
    parser.add_argument('--runtime-dir', type=Path)
    args = parser.parse_args()
    if args.runtime_dir:
        sys.path.insert(0, str(args.runtime_dir))
    else:
        runtime = types.ModuleType('test_runif')
        class ImportOnlyRuntime:
            def __init__(self, *args, **kwargs):
                raise AssertionError('import regression must not execute a runtime')
        for name in ('DeviceData', 'DeviceConfig', 'DeviceStream', 'DeviceRuntime'):
            setattr(runtime, name, ImportOnlyRuntime)
        sys.modules['test_runif'] = runtime
    # Deterministically test an environment without torch even when installed.
    sys.modules['torch'] = None
    spec = importlib.util.spec_from_file_location('generated_call', args.generated)
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    assert any(callable(getattr(generated, name, None)) for name in ('gather', 'remap'))
    array = np.array([10, 20, 30, 40], dtype=np.int16)
    np.testing.assert_array_equal(generated.safe_slice_1d(array, 2, (4,)), [30, 40, 0, 0])
    print('PASS: complete generated module and ordinary slicing work without torch')


if __name__ == '__main__':
    main()
