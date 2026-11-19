"""Ordinary C argument fixtures and an independent full-frame NumPy oracle."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def contract(short):
    spec = json.loads((HERE / 'full_direct.contract.json').read_text())
    if short:
        spec['defines']['T'] = 353
        for obj in spec['objects'][1:]:
            obj['elements'] = 353
    return spec


def frame(number, short=True):
    with np.load(HERE / 'remap_inputs.npz', allow_pickle=False) as inputs:
        src, x, y, old = [inputs[key].ravel().copy() for key in
                          (f'src{number % 2}', 'x', 'y', f'old{number % 2}')]
    if short:
        x = (np.arange(353) % 7).astype(np.int16)
        y = np.zeros_like(x)
        enabled = np.full_like(x, -3)
        x[1], enabled[8] = -1, 0
        src[:7] = np.roll([0, 4999, 5000, 10000, 10001, 65535, 6000], number)
        old = (45000 + 512 * number + np.arange(353)).astype(np.uint16)
    else:
        valid = (x >= 0) & (x < 640) & (y >= 0) & (y < 480)
        blocks = np.unique(np.flatnonzero(valid) // 256)[:16]
        enabled = np.isin(np.arange(x.size) // 256, blocks).astype(np.int16)
    return src, x, y, enabled, old


def reference(src, x, y, enabled, old):
    valid = (x >= 0) & (x < 640) & (y >= 0) & (y < 480) & (enabled != 0)
    result = old.copy()
    result[valid] = src[640 * y[valid].astype(np.int64) + x[valid]]
    return result
