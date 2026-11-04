"""Application adapter for a shape/border-specialized generated remap kernel.

Coordinates stay signed and unchanged. The generated checked entry validates
its exact nonempty shapes; this adapter only exposes the u16 pixel bit domain
and handles an empty output without calling the kernel.
"""
import numpy as np


async def remap_u16(runtime, kernel, src, map_x, map_y, dst):
    for value, dtype in ((src, np.uint16), (map_x, np.int16),
                         (map_y, np.int16), (dst, np.uint16)):
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype)
                or value.ndim != 1 or not value.flags.c_contiguous):
            raise ValueError('remap: invalid buffer dtype, rank or layout')
    if not src.size or map_x.shape != map_y.shape or dst.shape != map_x.shape:
        raise ValueError('remap: nonempty source and matching coordinate/output shapes required')
    if not dst.flags.writeable:
        raise ValueError('remap: output must be writable')
    buffers = (src, map_x, map_y, dst)
    if any(np.shares_memory(a, b) for i, a in enumerate(buffers) for b in buffers[i+1:]):
        raise ValueError('remap: buffers must not alias')
    if not dst.size:
        return
    await kernel(runtime, src.view(np.int16), map_x, map_y, dst.view(np.int16))
